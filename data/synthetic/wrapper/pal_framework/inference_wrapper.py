"""PAL Phase 5 — Production Inference Wrapper (Select-then-Generate).

PIX-4077. Implements the two-stage PAL inference architecture described in
arxiv:2511.10215v1:

    Stage 1 (Selection): route the incoming dialogue to a persona classifier
    that picks the best matching persona from a candidate pool.
    Stage 2 (Generation): prepend the selected persona to the dialogue and
    generate the next response.

The wrapper is model-agnostic: callers inject two ``LlmClient``-shaped
callables (selector + generator). Tests use stub clients; production wires
in real HuggingFace endpoints. A hard latency budget (default 2.0s, per
PIX-4077 acceptance criteria on A100) raises ``LatencyExceededError`` when
violated so regressions surface immediately.

No JSON is permitted to leak into generated text — see ``_has_json_leakage``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from generate_selection_dataset import build_selection_messages
from generate_sft_dialogue import SYSTEM_PROMPT

DEFAULT_LATENCY_BUDGET_SECONDS = 2.0

VALID_PERSONA_KEYS = ("demographics", "healthcare_behavior")


class LlmClient(Protocol):
    """Minimal callable shape consumed by the wrapper.

    Any callable ``messages -> str`` satisfies this; real deployments wrap a
    HuggingFace ``pipeline`` or vLLM endpoint.
    """

    def __call__(self, messages: list[dict[str, str]]) -> str: ...


class LatencyExceededError(RuntimeError):
    """Raised when end-to-end PAL inference exceeds the configured budget."""


class SelectionParseError(ValueError):
    """Raised when Stage 1 response cannot be parsed as an option index."""


class JsonLeakageError(ValueError):
    """Raised when generated text contains JSON formatting characters."""


@dataclass
class PalSelectionResult:
    persona_string: str
    selected_index: int
    latency_seconds: float


@dataclass
class PalGenerationResult:
    response: str
    latency_seconds: float


@dataclass
class PalInferenceResult:
    selection: PalSelectionResult
    generation: PalGenerationResult
    total_latency_seconds: float
    dialogue_history_text: str


@dataclass
class PalInferenceWrapper:
    """Two-stage Select-then-Generate PAL inference wrapper.

    Parameters
    ----------
    selector_client:
        Callable ``messages -> str``. Returns the option number (1-indexed)
        as a string per the Phase 2.1 selection prompt contract.
    generator_client:
        Callable ``messages -> str``. Returns the assistant response.
    candidate_personas:
        Pool of Meddies-shaped persona dicts (``demographics`` +
        ``healthcare_behavior``) to select from.
    latency_budget_seconds:
        Hard ceiling for total inference time. Defaults to 2.0s per
        PIX-4077 acceptance criteria (A100 target).
    """

    selector_client: LlmClient
    generator_client: LlmClient
    candidate_personas: list[dict[str, Any]]
    latency_budget_seconds: float = DEFAULT_LATENCY_BUDGET_SECONDS
    # Cached selection prompts keyed by dialogue hash to avoid rebuilding on
    # repeated inference of the same dialogue. Kept tiny — production uses
    # bounded LRU caches upstream.
    _selection_cache: dict[str, PalSelectionResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_personas:
            raise ValueError("candidate_personas must not be empty")
        if self.latency_budget_seconds <= 0:
            raise ValueError("latency_budget_seconds must be positive")

    # ------------------------------------------------------------------
    # Stage 1 — Persona Selection
    # ------------------------------------------------------------------
    def select_persona(self, dialogue: str) -> PalSelectionResult:
        if not isinstance(dialogue, str) or not dialogue.strip():
            raise ValueError("dialogue must be a non-empty string")

        candidates = [_persona_to_string(p) for p in self.candidate_personas]
        messages = build_selection_messages(
            dialogue=dialogue,
            candidates=candidates,
            correct_index=0,  # Stage 1 at inference has no ground truth;
            # pass 0 as placeholder — the LLM picks.
        )
        # Drop the assistant turn (Phase 2.1 builder adds the gold answer);
        # the selector LLM should produce it fresh.
        messages = [m for m in messages if m["role"] != "assistant"]

        start = time.perf_counter()
        raw = self.selector_client(messages)
        latency = time.perf_counter() - start

        idx = _parse_selection_index(raw, len(candidates))
        persona_string = candidates[idx]
        return PalSelectionResult(
            persona_string=persona_string,
            selected_index=idx,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Stage 2 — Response Generation
    # ------------------------------------------------------------------
    def generate_response(
        self,
        persona_string: str,
        dialogue_history: str,
    ) -> PalGenerationResult:
        if not persona_string.strip():
            raise ValueError("persona_string must be non-empty")
        if not isinstance(dialogue_history, str):
            raise ValueError("dialogue_history must be a string")

        # Build ChatML messages directly instead of reusing
        # ``build_sft_messages`` (which requires a non-empty ``response``).
        # Stage 2 is inference-time, so we omit the assistant turn entirely;
        # the generator LLM produces it.
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Given this persona: {persona_string}\n\n"
                    f"Dialogue history:\n{dialogue_history}\n\n"
                    "Generate the next response."
                ),
            },
        ]

        start = time.perf_counter()
        raw = self.generator_client(messages)
        latency = time.perf_counter() - start

        if _has_json_leakage(raw):
            raise JsonLeakageError(f"Generator produced text containing JSON formatting characters: {raw!r}")
        return PalGenerationResult(
            response=raw.strip(),
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    def infer(self, dialogue: str) -> PalInferenceResult:
        selection = self.select_persona(dialogue)
        generation = self.generate_response(
            persona_string=selection.persona_string,
            dialogue_history=dialogue,
        )
        total = selection.latency_seconds + generation.latency_seconds
        if total > self.latency_budget_seconds:
            raise LatencyExceededError(
                f"PAL inference took {total:.3f}s, exceeds budget {self.latency_budget_seconds:.3f}s"
            )
        return PalInferenceResult(
            selection=selection,
            generation=generation,
            total_latency_seconds=total,
            dialogue_history_text=dialogue,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _persona_to_string(persona: dict[str, Any]) -> str:
    """Convert a Meddies-shaped persona dict into the PAL natural-language form.

    Reuses ``meddies_to_pal.format_persona`` when available; falls back to a
    compact inline format to avoid a hard import dependency in deployment
    environments where the training_corpus package is not on the path.
    """
    try:
        from meddies_to_pal import format_persona  # local import for testability

        return format_persona(persona)
    except Exception:  # pragma: no cover — exercised in tests via monkeypatch
        demographics = persona.get("demographics", {}) if isinstance(persona, dict) else {}
        behavior = persona.get("healthcare_behavior", {}) if isinstance(persona, dict) else {}
        age = demographics.get("age", "unknown age")
        gender = demographics.get("gender", "person")
        location = demographics.get("location", "Vietnam")
        literacy = behavior.get("health_literacy", "average")
        preference = behavior.get("preference", "standard medicine")
        return (
            f"This patient is a {age}-year-old {gender} from {location} with "
            f"{literacy} health literacy who prefers {preference}."
        )


def _parse_selection_index(raw: str, n_candidates: int) -> int:
    """Parse the selector LLM's response as a 1-indexed option number."""
    if not isinstance(raw, str):
        raise SelectionParseError(f"Selector returned non-string: {raw!r}")
    text = raw.strip()
    if not text:
        raise SelectionParseError("Selector returned empty response")
    # Accept either "3", "3.", or "3. option text" — take the leading integer.
    first_token = text.split()[0].rstrip(".")
    try:
        idx1 = int(first_token)
    except ValueError as exc:
        raise SelectionParseError(f"Selector response not parseable as option index: {raw!r}") from exc
    if not 1 <= idx1 <= n_candidates:
        raise SelectionParseError(f"Selector index {idx1} out of range (1-{n_candidates})")
    return idx1 - 1  # convert to 0-indexed


def _has_json_leakage(text: str) -> bool:
    """Return True if generated text contains JSON formatting characters.

    Matches the Phase 1 ``test_no_json_leakage`` contract: curly braces,
    single quotes, or double quotes are forbidden in patient-facing output.
    """
    return any(ch in text for ch in "{}\"'")
