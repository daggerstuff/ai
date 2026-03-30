"""
Shared record formatting helpers for standard therapeutic data.
"""

from __future__ import annotations

from typing import Any


def extract_standard_therapeutic_text(conv: Any) -> str:
    if not isinstance(conv, dict):
        return ""

    text = conv.get("text", "")
    if isinstance(text, str) and text:
        return text

    conversation_array = conv.get("conversation", [])
    if isinstance(conversation_array, list) and conversation_array:
        parts = _parts_from_messages(conversation_array)
        if parts:
            return "\n".join(parts)

    messages = conv.get("messages", [])
    if isinstance(messages, list) and messages:
        parts = _parts_from_messages(messages)
        if parts:
            return "\n".join(parts)

    content = conv.get("content", "")
    return content if isinstance(content, str) else ""


def is_standard_therapeutic_record(conv: Any) -> bool:
    return bool(extract_standard_therapeutic_text(conv))


def _parts_from_messages(messages: list[Any]) -> list[str]:
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(role, str) and isinstance(content, str) and role and content:
            parts.append(f"{role.capitalize()}: {content}")
    return parts


__all__ = [
    "extract_standard_therapeutic_text",
    "is_standard_therapeutic_record",
]
