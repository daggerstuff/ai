"""Shared Stage 2 persona quality checks.

This module centralizes record validation and text-quality heuristics used by both
the generation pipeline and verification tooling.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ai.core.persona_manager import ROBOTIC_PHRASING_PENALTIES


MIN_USER_CHARS = 10
MIN_ASSISTANT_CHARS = 15
MAX_ASSISTANT_CHARS = 4000
MIN_DIRECTIVE_CHARS = 5
PERSONA_IMBALANCE_FRACTION = 0.35


REFUSAL_PREFIXES = (
    "i cannot generate",
    "i'm not able to",
    "i can't generate",
    "as an ai",
    "i don't want to talk about it right now",
    "i guess i just don't have much to say about that",
)


def _validate_messages(record: dict, index: int) -> list[str]:
    """Validate the messages array and required role/content invariants."""
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(f"line {index}: missing or invalid 'messages' (must be array)")
        return errors
    if len(messages) == 0:
        errors.append(f"line {index}: 'messages' is empty")

    roles = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"line {index}: messages[{i}] is not an object")
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role:
            roles.add(role)
        if content is not None and not isinstance(content, str):
            errors.append(f"line {index}: messages[{i}].content is not string")
        elif isinstance(content, str) and not content.strip():
            errors.append(f"line {index}: messages[{i}].content is empty")

    if "user" not in roles:
        errors.append(f"line {index}: no 'user' message")
    if "assistant" not in roles:
        errors.append(f"line {index}: no 'assistant' message")
    return errors


def _validate_metadata_gs(record: dict, index: int) -> list[str]:
    """Validate metadata.gestalt_simulation fields."""
    errors: list[str] = []
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        errors.append(f"line {index}: missing or invalid 'metadata'")
        return errors
    gs = meta.get("gestalt_simulation")
    if not isinstance(gs, dict):
        errors.append(f"line {index}: missing or invalid metadata.gestalt_simulation")
        return errors
    if not gs.get("persona_id"):
        errors.append(f"line {index}: metadata.gestalt_simulation.persona_id missing")
    if "directive" not in gs:
        errors.append(f"line {index}: metadata.gestalt_simulation.directive missing")
    return errors


def validate_record(record: dict, index: int) -> list[str]:
    """Validate one generated persona turn record."""
    if not isinstance(record, dict):
        return [f"line {index}: not a JSON object"]
    errors = _validate_messages(record, index)
    errors.extend(_validate_metadata_gs(record, index))
    return errors


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _stable_message_hash(text: str) -> str:
    """Stable hash of message content for deduplication."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def last_user_content(record: dict) -> str:
    messages = record.get("messages") or []
    message = next((x for x in reversed(messages) if x.get("role") == "user"), None)
    return (message.get("content") or "").strip() if message else ""


def last_assistant_content(record: dict) -> str:
    messages = record.get("messages") or []
    message = next((x for x in reversed(messages) if x.get("role") == "assistant"), None)
    return (message.get("content") or "").strip() if message else ""


def _is_refusal_or_fallback(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lower = stripped.lower()
    return any(prefix in lower for prefix in REFUSAL_PREFIXES)


def _fails_human_likeness(text: str) -> bool:
    if not text:
        return True
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in ROBOTIC_PHRASING_PENALTIES):
        return True
    return "\n1." in text and "\n2." in text and "\n3." in text


@dataclass
class QualityCounts:
    short_user: int = 0
    short_assistant: int = 0
    long_assistant: int = 0
    empty_directive: int = 0
    short_directive: int = 0
    refusal_or_fallback: int = 0
    robotic: int = 0

