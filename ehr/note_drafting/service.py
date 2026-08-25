"""Core note drafting service — NIM client, prompt building, and response parsing.

This module handles:
- Building structured clinical prompts for SOAP/DAP formats.
- Calling the NIM inference endpoint via httpx with retry + backoff.
- Parsing NIM responses into structured ``DraftResponse`` objects.
- Ensuring no PHI is logged or persisted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .config import NoteDraftingSettings
from .models import DraftRequest, DraftResponse, NoteFormat, NoteSections
from .phi import redact_patient_id, redact_session_id

logger = logging.getLogger(__name__)

# --- Prompt templates ---

_SOAP_SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Given a telehealth transcript, \
generate a SOAP note with four clearly labeled sections:

S — Subjective: Patient-reported symptoms, complaints, and history.
O — Objective: Observable findings, vital signs, and examination results.
A — Assessment: Clinical impressions, differential diagnoses, and problem list.
P — Plan: Treatment plan, medications, referrals, patient education, and follow-up.

Output ONLY the four sections as valid JSON with keys: subjective, objective, assessment, plan. \
Each value is a string with the section content. Do not include any text outside the JSON."""

_DAP_SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Given a telehealth transcript, \
generate a DAP note with three clearly labeled sections:

D — Data: Observable facts, patient statements, symptoms, and findings.
A — Assessment: Clinical interpretation, impressions, and problem list.
P — Plan: Treatment plan, interventions, and follow-up.

Output ONLY the three sections as valid JSON with keys: data, assessment, plan. \
Each value is a string with the section content. Do not include any text outside the JSON."""

_SOAP_USER_TEMPLATE = """\
Transcript:
{transcript}

Generate the SOAP note JSON now."""

_DAP_USER_TEMPLATE = """\
Transcript:
{transcript}

Generate the DAP note JSON now."""


def _build_prompt(request: DraftRequest) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the given request.

    Args:
        request: The validated draft request.

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    if request.note_format == NoteFormat.DAP:
        return _DAP_SYSTEM_PROMPT, _DAP_USER_TEMPLATE.format(transcript=request.transcript)
    return _SOAP_SYSTEM_PROMPT, _SOAP_USER_TEMPLATE.format(transcript=request.transcript)


def _parse_nim_response(
    content: str,
    note_format: NoteFormat,
) -> tuple[str, NoteSections, float]:
    """Parse the NIM response content into structured note sections.

    Expects the model to return JSON with section keys. Falls back to
    raw text with a warning-level confidence if parsing fails.

    Args:
        content: Raw text from the NIM model response.
        note_format: The requested note format.

    Returns:
        Tuple of (draft_note, sections, confidence).
    """
    # Attempt to extract JSON from the response
    json_text = content.strip()

    # Handle markdown code fences if present
    if json_text.startswith("```"):
        lines = json_text.splitlines()
        # Remove first and last fence lines
        if len(lines) >= 2:
            json_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        # Fallback: use raw text as draft, low confidence
        sections = NoteSections()
        if note_format == NoteFormat.DAP:
            sections.data = content
        else:
            sections.subjective = content
        return content, sections, 0.3

    sections = NoteSections()
    warnings_text: list[str] = []

    if note_format == NoteFormat.SOAP:
        sections.subjective = str(parsed.get("subjective", "") or "")
        sections.objective = str(parsed.get("objective", "") or "")
        sections.assessment = str(parsed.get("assessment", "") or "")
        sections.plan = str(parsed.get("plan", "") or "")

        missing = [
            name
            for name, val in [
                ("subjective", sections.subjective),
                ("objective", sections.objective),
                ("assessment", sections.assessment),
                ("plan", sections.plan),
            ]
            if not val.strip()
        ]
        if missing:
            warnings_text.append(f"Missing/empty SOAP sections: {', '.join(missing)}")
    else:  # DAP
        sections.data = str(parsed.get("data", "") or "")
        sections.assessment = str(parsed.get("assessment", "") or "")
        sections.plan = str(parsed.get("plan", "") or "")

        missing = [
            name
            for name, val in [
                ("data", sections.data),
                ("assessment", sections.assessment),
                ("plan", sections.plan),
            ]
            if not val.strip()
        ]
        if missing:
            warnings_text.append(f"Missing/empty DAP sections: {', '.join(missing)}")

    # Build full draft note from sections
    if note_format == NoteFormat.SOAP:
        draft_parts = [
            f"S (Subjective):\n{sections.subjective}",
            f"\nO (Objective):\n{sections.objective}",
            f"\nA (Assessment):\n{sections.assessment}",
            f"\nP (Plan):\n{sections.plan}",
        ]
    else:
        draft_parts = [
            f"D (Data):\n{sections.data}",
            f"\nA (Assessment):\n{sections.assessment}",
            f"\nP (Plan):\n{sections.plan}",
        ]
    draft_note = "\n".join(draft_parts)

    # Confidence: higher if all sections present, lower if fallback was used
    confidence = 0.85 if not warnings_text else 0.6

    return draft_note, sections, confidence


class NoteDraftingService:
    """Service that drafts clinical notes from telehealth transcripts via NIM."""

    def __init__(self, settings: NoteDraftingSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> NoteDraftingSettings:
        """Expose settings for configuration checks."""
        return self._settings

    async def draft_note(self, request: DraftRequest) -> DraftResponse:
        """Generate a clinical note draft from a telehealth transcript.

        Args:
            request: Validated draft request containing transcript and metadata.

        Returns:
            ``DraftResponse`` with the structured note draft.

        Raises:
            RuntimeError: If the NIM endpoint is not configured or all retries fail.
        """
        if not self._settings.is_configured:
            raise RuntimeError("NIM endpoint is not configured (NOTE_DRAFTING_NIM_URL or NOTE_DRAFTING_NIM_API_KEY missing).")

        system_prompt, user_prompt = _build_prompt(request)

        logger.info(
            "draft_note:start format=%s pid=%s sid=%s transcript_len=%d",
            request.note_format.value,
            redact_patient_id(request.patient_id),
            redact_session_id(request.session_id),
            len(request.transcript),
        )

        content = await self._call_nim(system_prompt, user_prompt)
        draft_note, sections, confidence = _parse_nim_response(content, request.note_format)

        warnings: list[str] = []
        if confidence < 0.5:
            warnings.append("Low confidence score — manual review strongly recommended.")
        if confidence == 0.3:
            warnings.append("Model response was not valid JSON — raw text returned as fallback.")

        logger.info(
            "draft_note:done format=%s pid=%s confidence=%.2f warnings=%d",
            request.note_format.value,
            redact_patient_id(request.patient_id),
            confidence,
            len(warnings),
        )

        return DraftResponse(
            draft_note=draft_note,
            sections=sections,
            confidence=confidence,
            warnings=warnings,
        )

    async def _call_nim(self, system_prompt: str, user_prompt: str) -> str:
        """Call the NIM endpoint with retry and exponential backoff.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message containing the transcript.

        Returns:
            The model's response content as a string.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        payload: dict[str, Any] = {
            "model": self._settings.nim_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self._settings.nim_api_key}",
            "Content-Type": "application/json",
        }

        last_error: str = ""
        for attempt in range(1, self._settings.nim_max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self._settings.nim_timeout_seconds,
                ) as client:
                    response = await client.post(
                        self._settings.nim_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    # OpenAI-compatible response format
                    choices = data.get("choices", [])
                    if not choices:
                        raise RuntimeError("NIM response contained no choices.")
                    content = choices[0].get("message", {}).get("content", "")
                    if not content.strip():
                        raise RuntimeError("NIM response content was empty.")
                    return content

            except httpx.TimeoutException:
                last_error = f"Request timed out after {self._settings.nim_timeout_seconds}s (attempt {attempt})."
                logger.warning("nim:timeout attempt=%d/%d", attempt, self._settings.nim_max_retries)
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code} from NIM (attempt {attempt})."
                logger.warning(
                    "nim:http_error attempt=%d/%d status=%d",
                    attempt,
                    self._settings.nim_max_retries,
                    exc.response.status_code,
                )
                # Don't retry on 4xx (except 429)
                if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    break
            except httpx.HTTPError as exc:
                last_error = f"HTTP error: {exc} (attempt {attempt})."
                logger.warning("nim:http_error attempt=%d/%d err=%s", attempt, self._settings.nim_max_retries, str(exc))
            except (KeyError, IndexError, RuntimeError) as exc:
                last_error = f"Response parse error: {exc} (attempt {attempt})."
                logger.warning("nim:parse_error attempt=%d/%d err=%s", attempt, self._settings.nim_max_retries, str(exc))

            # Exponential backoff between retries
            if attempt < self._settings.nim_max_retries:
                delay = self._settings.nim_retry_base_delay * (2 ** (attempt - 1))
                logger.info("nim:retry delay=%.1fs next_attempt=%d", delay, attempt + 1)
                await asyncio.sleep(delay)

        raise RuntimeError(f"NIM request failed after {self._settings.nim_max_retries} attempts: {last_error}")

    async def draft_note_mock(self, request: DraftRequest) -> DraftResponse:
        """Generate a mock clinical note draft without calling NIM.

        Used for testing and development when NIM is not available.

        Args:
            request: Validated draft request.

        Returns:
            ``DraftResponse`` with mock content.
        """
        if request.note_format == NoteFormat.SOAP:
            sections = NoteSections(
                subjective="[Mock] Patient reported feeling anxious and described sleep difficulties over the past two weeks.",
                objective="[Mock] Patient appeared alert and oriented. Vital signs within normal limits. No acute distress noted.",
                assessment="[Mock] Generalized anxiety with insomnia. No immediate safety concerns identified.",
                plan="[Mock] Continue therapy sessions weekly. Consider CBT techniques for sleep hygiene. Follow up in two weeks.",
            )
            draft_note = (
                "S (Subjective):\n[Mock] Patient reported feeling anxious and described sleep difficulties.\n\n"
                "O (Objective):\n[Mock] Patient appeared alert. Vitals normal.\n\n"
                "A (Assessment):\n[Mock] Generalized anxiety with insomnia.\n\n"
                "P (Plan):\n[Mock] Weekly therapy. CBT for sleep. Follow-up in 2 weeks."
            )
        else:
            sections = NoteSections(
                data="[Mock] Patient reported anxiety and sleep difficulties. Appealed alert during session.",
                assessment="[Mock] Generalized anxiety with insomnia. No safety concerns.",
                plan="[Mock] Continue weekly therapy. CBT sleep hygiene. Follow-up in 2 weeks.",
            )
            draft_note = (
                "D (Data):\n[Mock] Patient reported anxiety and sleep difficulties.\n\n"
                "A (Assessment):\n[Mock] Generalized anxiety with insomnia.\n\n"
                "P (Plan):\n[Mock] Weekly therapy. CBT sleep hygiene. Follow-up in 2 weeks."
            )

        logger.info(
            "draft_note:mock format=%s pid=%s",
            request.note_format.value,
            redact_patient_id(request.patient_id),
        )

        return DraftResponse(
            draft_note=draft_note,
            sections=sections,
            confidence=0.95,
            warnings=["Mock response — no NIM endpoint was called."],
        )
