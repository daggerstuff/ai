"""
Defines the standard conversation schema for the Pixelated Empathy AI dataset pipeline.
Ensures a unified, enterprise-grade data structure for all therapeutic conversations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Message:
    """
    Represents a single message within a conversation.
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

@dataclass(kw_only=True)
class Conversation:
    """
    Represents a complete conversation, adhering to the unified schema.
    """
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str | None = None
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Adds a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the conversation to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "source": self.source,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Creates a Conversation instance from a dictionary."""
        messages = [Message(**msg_data) for msg_data in data.get("messages", [])]
        return cls(
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            source=data.get("source"),
            messages=messages,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )
