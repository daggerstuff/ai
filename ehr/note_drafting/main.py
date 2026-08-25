"""FastAPI application for the AI Note Drafting microservice.

Exposes:
- ``POST /draft`` — accept a telehealth transcript and return a SOAP/DAP note draft.
- ``GET /health`` — health check endpoint.

The service enforces a BAA (Business Associate Agreement) gate: if
``NOTE_DRAFTING_BAA_CONFIRMED`` is not set to ``true``, all drafting
requests are rejected with HTTP 403.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import NoteDraftingSettings, get_settings
from .models import DraftRequest, DraftResponse, ErrorResponse
from .phi import sanitize_for_logging
from .service import NoteDraftingService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Type-annotated dependencies (avoids B008) ---

SettingsDep = Annotated[NoteDraftingSettings, Depends(get_settings)]


# --- BAA gate dependency ---


def verify_baa_gate(settings: SettingsDep) -> None:
    """Reject requests if BAA has not been confirmed.

    Args:
        settings: Injected settings singleton.

    Raises:
        HTTPException: 403 if ``NOTE_DRAFTING_BAA_CONFIRMED`` is not True.
    """
    if not settings.baa_confirmed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="BAA not confirmed. Set NOTE_DRAFTING_BAA_CONFIRMED=true to enable note drafting.",
        )


# --- Service factory ---


def get_service(settings: SettingsDep) -> NoteDraftingService:
    """Return a ``NoteDraftingService`` instance bound to current settings."""
    return NoteDraftingService(settings)


ServiceDep = Annotated[NoteDraftingService, Depends(get_service)]
BaaGateDep = Annotated[None, Depends(verify_baa_gate)]

# --- Lifespan ---


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log startup info (no PHI)."""
    settings = get_settings()
    logger.info(
        "note_drafting:startup nim_configured=%s baa_confirmed=%s port=%d",
        settings.is_configured,
        settings.baa_confirmed,
        settings.service_port,
    )
    yield
    logger.info("note_drafting:shutdown")


# --- FastAPI app ---

app = FastAPI(
    title="AI Note Drafting Service",
    description="Converts telehealth transcripts to SOAP/DAP clinical note drafts via NIM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "note-drafting"}


@app.post(
    "/draft",
    response_model=DraftResponse,
    responses={
        403: {"model": ErrorResponse, "description": "BAA not confirmed."},
        422: {"model": ErrorResponse, "description": "Validation error."},
        500: {"model": ErrorResponse, "description": "NIM endpoint error."},
    },
    tags=["drafting"],
)
async def draft_note(
    request: DraftRequest,
    _baa: BaaGateDep,
    service: ServiceDep,
) -> DraftResponse:
    """Draft a clinical note from a telehealth transcript.

    Accepts a transcript with patient/session metadata and returns a
    structured SOAP or DAP note draft. All logging is PHI-sanitized.

    Args:
        request: Validated draft request.
        _baa: BAA gate dependency (rejects if not confirmed).
        service: Note drafting service instance.

    Returns:
        ``DraftResponse`` with the draft note, sections, confidence, and warnings.
    """
    # Log sanitized request metadata (no transcript content in logs)
    logger.info(
        "draft:request format=%s transcript_length=%d transcript_preview=%s",
        request.note_format.value,
        len(request.transcript),
        sanitize_for_logging(request.transcript[:100]),
    )

    try:
        # Use mock service if NIM is not configured
        if not service.settings.is_configured:
            return await service.draft_note_mock(request)
        return await service.draft_note(request)
    except RuntimeError as exc:
        logger.error("draft:error err=%s", sanitize_for_logging(str(exc)))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Note drafting failed: {exc}",
        ) from exc
