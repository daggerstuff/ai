"""Therapeutic conversation collection interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TherapeuticConversation:
    conversation_id: str
    messages: list[dict[str, Any]]


class TherapeuticConversations:
    """Simple container for collecting and filtering therapeutic conversations."""

    def __init__(self) -> None:
        self._items: list[TherapeuticConversation] = []

    def add(self, conversation: TherapeuticConversation) -> None:
        self._items.append(conversation)

    def collect(self, source: list[TherapeuticConversation] | list[dict[str, Any]]) -> list[TherapeuticConversation]:
        collected = []
        for item in source:
            if isinstance(item, TherapeuticConversation):
                collected.append(item)
            elif isinstance(item, dict):
                collected.append(
                    TherapeuticConversation(
                        conversation_id=str(item.get("conversation_id", "unknown")),
                        messages=item.get("messages", []),
                    )
                )
        self._items.extend(collected)
        return collected

    def list(self, limit: int | None = None) -> list[TherapeuticConversation]:
        return self._items[:limit] if limit else list(self._items)


__all__ = ["TherapeuticConversation", "TherapeuticConversations"]
