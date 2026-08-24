"""PAL Inference Service — FastAPI microservice for Persona-Aware Alignment inference.

Exposes the two-stage Select-then-Generate PAL inference pipeline (arxiv:2511.10215v1)
as a deployable HTTP API. Defaults to stub LLM clients so the service starts without
any GPU or API key dependency. Production deployments override via environment variables.

Endpoints:
    POST /api/v1/pal/infer       — End-to-end two-stage inference
    POST /api/v1/pal/select      — Stage 1: persona selection only
    POST /api/v1/pal/generate    — Stage 2: response generation only
    GET  /health                 — Service health check
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure the pal_framework package is importable
# ---------------------------------------------------------------------------
_PAL_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "synthetic" / "wrapper" / "pal_framework",
)
if _PAL_PATH not in sys.path:
    sys.path.insert(0, _PAL_PATH)

from inference_wrapper import (  # type: ignore[import-untyped]  # noqa: E402
    DEFAULT_LATENCY_BUDGET_SECONDS,
    JsonLeakageError,
    LatencyExceededError,
    PalInferenceWrapper,
    SelectionParseError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PORT = 8010
DEFAULT_CANDIDATE_PERSONAS_ENV = "PAL_CANDIDATE_PERSONAS"
PAL_SELECTOR_ENDPOINT_ENV = "PAL_SELECTOR_ENDPOINT"
PAL_GENERATOR_ENDPOINT_ENV = "PAL_GENERATOR_ENDPOINT"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PalInferRequest(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=1,
        description="The patient dialogue to infer a persona for.",
        examples=["Patient: I have been feeling very tired lately. Doctor: How long has this been going on?"],
    )


class PalSelectionResponse(BaseModel):
    persona_string: str = Field(..., description="The selected persona rendered as a natural-language string.")
    selected_index: int = Field(
        ..., ge=0, description="Zero-based index of the selected persona in the candidate pool."
    )
    latency_seconds: float = Field(..., ge=0.0, description="Wall-clock seconds for Stage 1.")


class PalGenerationResponse(BaseModel):
    response: str = Field(..., description="The generated assistant response.")
    latency_seconds: float = Field(..., ge=0.0, description="Wall-clock seconds for Stage 2.")


class PalGenerateRequest(BaseModel):
    persona_string: str = Field(
        ..., min_length=1, description="The persona string to use for response generation."
    )
    dialogue_history: str = Field(
        ..., description="The dialogue history up to this point."
    )


class PalInferResponse(BaseModel):
    selection: PalSelectionResponse = Field(..., description="Stage 1 persona selection result.")
    generation: PalGenerationResponse = Field(..., description="Stage 2 response generation result.")
    total_latency_seconds: float = Field(..., ge=0.0, description="Total wall-clock seconds for both stages.")
    dialogue: str = Field(..., description="The input dialogue echoed back.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status.")
    wrapper_initialized: bool = Field(
        ..., description="Whether the PAL inference wrapper is ready."
    )
    n_candidate_personas: int = Field(..., description="Number of loaded candidate personas.")
    latency_budget_seconds: float = Field(..., description="Current latency budget.")
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Stub LLM clients (used when no real endpoint is configured)
# ---------------------------------------------------------------------------


class _StubSelector:
    """Returns option '1' for every request — the first candidate persona."""

    def __call__(self, _messages: list[dict[str, str]]) -> str:
        return "1"


class _StubGenerator:
    """Returns a canned persona-aligned response."""

    def __call__(self, _messages: list[dict[str, str]]) -> str:
        return (
            "I have been feeling this way for a while now. "
            "Thank you for explaining things clearly, doctor."
        )


# ---------------------------------------------------------------------------
# Wrapper factory
# ---------------------------------------------------------------------------


def _load_candidate_personas() -> list[dict[str, Any]]:
    """Load candidate personas from the PAL_CANDIDATE_PERSONAS env var.

    Expects a JSON array of Meddies-shaped persona dicts. Falls back to a
    single default persona so the service starts without configuration.
    """
    raw = os.environ.get(DEFAULT_CANDIDATE_PERSONAS_ENV)
    if raw:
        try:
            personas = json.loads(raw)
            if not isinstance(personas, list) or not personas:
                raise ValueError("PAL_CANDIDATE_PERSONAS must be a non-empty JSON array")
            return personas
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse %s, falling back to default personas: %s",
                DEFAULT_CANDIDATE_PERSONAS_ENV,
                exc,
            )

    # Default single persona so the service can always start
    return [
        {
            "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
            "healthcare_behavior": {
                "health_literacy": "low",
                "preference": "traditional medicine",
            },
        },
        {
            "demographics": {"age": 30, "gender": "male", "location": "HCMC"},
            "healthcare_behavior": {
                "health_literacy": "high",
                "preference": "modern medicine",
            },
        },
    ]


def _build_selector_client() -> Any:
    """Build the selector LLM client.

    Reads PAL_SELECTOR_ENDPOINT from env. When unset, returns the stub selector.
    """
    endpoint = os.environ.get(PAL_SELECTOR_ENDPOINT_ENV)
    if endpoint:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            client = OpenAI(base_url=endpoint)
            model = os.environ.get("PAL_SELECTOR_MODEL", "gpt-4o-mini")

            def selector(messages: list[dict[str, str]]) -> str:
                resp = client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
                return resp.choices[0].message.content or "1"

            return selector
        except ImportError:
            logger.warning("openai not installed; falling back to stub selector")
    return _StubSelector()


def _build_generator_client() -> Any:
    """Build the generator LLM client.

    Reads PAL_GENERATOR_ENDPOINT from env. When unset, returns the stub generator.
    """
    endpoint = os.environ.get(PAL_GENERATOR_ENDPOINT_ENV)
    if endpoint:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            client = OpenAI(base_url=endpoint)
            model = os.environ.get("PAL_GENERATOR_MODEL", "gpt-4o-mini")

            def generator(messages: list[dict[str, str]]) -> str:
                resp = client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
                return resp.choices[0].message.content or ""

            return generator
        except ImportError:
            logger.warning("openai not installed; falling back to stub generator")
    return _StubGenerator()


def _build_latency_budget() -> float:
    raw = os.environ.get("PAL_LATENCY_BUDGET_SECONDS", str(DEFAULT_LATENCY_BUDGET_SECONDS))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LATENCY_BUDGET_SECONDS


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_wrapper: PalInferenceWrapper | None = None
"""Lazily initialised — see lifespan below."""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Initialize the PAL wrapper on startup, log on shutdown."""
    global _wrapper
    try:
        selector = _build_selector_client()
        generator = _build_generator_client()
        personas = _load_candidate_personas()
        budget = _build_latency_budget()
        _wrapper = PalInferenceWrapper(
            selector_client=selector,
            generator_client=generator,
            candidate_personas=personas,
            latency_budget_seconds=budget,
        )
        logger.info(
            "PAL wrapper initialized: %d candidate personas, %.2fs budget",
            len(personas),
            budget,
        )
    except Exception:
        logger.exception("Failed to initialize PAL wrapper")
        _wrapper = None
    yield


app = FastAPI(
    title="PAL Inference Service",
    description="Persona-Aware Alignment (PAL) two-stage inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="healthy" if _wrapper is not None else "degraded",
        wrapper_initialized=_wrapper is not None,
        n_candidate_personas=len(_wrapper.candidate_personas) if _wrapper else 0,
        latency_budget_seconds=(
            _wrapper.latency_budget_seconds if _wrapper else DEFAULT_LATENCY_BUDGET_SECONDS
        ),
    )


@app.post("/api/v1/pal/select", response_model=PalSelectionResponse, tags=["PAL"])
async def select_persona(req: PalInferRequest) -> PalSelectionResponse:
    """Stage 1: Select the best-matching persona for a given dialogue."""
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="PAL wrapper not initialized")

    try:
        result = _wrapper.select_persona(req.dialogue)
        return PalSelectionResponse(
            persona_string=result.persona_string,
            selected_index=result.selected_index,
            latency_seconds=result.latency_seconds,
        )
    except (ValueError, SelectionParseError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/v1/pal/generate", response_model=PalGenerationResponse, tags=["PAL"])
async def generate_response(req: PalGenerateRequest) -> PalGenerationResponse:
    """Stage 2: Generate a persona-aligned response given a persona and dialogue history."""
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="PAL wrapper not initialized")

    try:
        result = _wrapper.generate_response(
            persona_string=req.persona_string,
            dialogue_history=req.dialogue_history,
        )
        return PalGenerationResponse(
            response=result.response,
            latency_seconds=result.latency_seconds,
        )
    except (ValueError, JsonLeakageError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/v1/pal/infer", response_model=PalInferResponse, tags=["PAL"])
async def infer(req: PalInferRequest) -> PalInferResponse:
    """End-to-end two-stage PAL inference: select persona, then generate response."""
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="PAL wrapper not initialized")

    try:
        result = _wrapper.infer(req.dialogue)
        return PalInferResponse(
            selection=PalSelectionResponse(
                persona_string=result.selection.persona_string,
                selected_index=result.selection.selected_index,
                latency_seconds=result.selection.latency_seconds,
            ),
            generation=PalGenerationResponse(
                response=result.generation.response,
                latency_seconds=result.generation.latency_seconds,
            ),
            total_latency_seconds=result.total_latency_seconds,
            dialogue=result.dialogue_history_text,
        )
    except LatencyExceededError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from None
    except (ValueError, SelectionParseError, JsonLeakageError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PAL_API_PORT", str(DEFAULT_PORT)))
    uvicorn.run(
        "pal_inference_service:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("PAL_DEV_MODE", "").lower() in ("1", "true"),
    )
