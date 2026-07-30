"""Conversation schema for quality validation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single message in a conversation."""

    role: str
    content: str


@dataclass
class Conversation:
    """
    A conversation consisting of multiple messages.

    Used for quality validation and dataset processing.
    """

    conversation_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))

    def get_messages_by_role(self, role: str) -> list[Message]:
        """Get all messages with a specific role."""
        return [msg for msg in self.messages if msg.role == role]

    def get_all_content(self) -> str:
        """Get all message content concatenated."""
        return " ".join(msg.content for msg in self.messages)


__all__ = ["Conversation", "Message"]
