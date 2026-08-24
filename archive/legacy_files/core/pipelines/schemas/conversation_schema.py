"""Canonical Conversation schema used by core pipeline modules."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Single message record."""

    role: str
    content: str


@dataclass
class Conversation:
    """Lightweight conversation container with common helpers."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def get_messages_by_role(self, role: str) -> list[Message]:
        return [message for message in self.messages if message.role == role]

    def get_all_content(self) -> str:
        return " ".join(message.content for message in self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [{"role": message.role, "content": message.content} for message in self.messages],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


__all__ = ["Conversation", "Message"]
