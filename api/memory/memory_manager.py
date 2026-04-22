"""
Memory System Integration Module.

Integrates the configured shared memory backend with higher-level helpers for
managing user memory contexts, conversation history, and therapeutic sessions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from ai.memory.manager_factory import get_required_memory_manager as get_backend_memory_manager

logger = logging.getLogger(__name__)


class MemoryType(StrEnum):
    """Types of memory in the therapeutic context."""

    CONVERSATION = "conversation"
    SESSION_SUMMARY = "session_summary"
    THERAPEUTIC_NOTES = "therapeutic_notes"
    EMOTIONAL_STATE = "emotional_state"
    TREATMENT_PLAN = "treatment_plan"
    CRISIS_CONTEXT = "crisis_context"
    PROGRESS_NOTES = "progress_notes"


class MessageRole(StrEnum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    THERAPIST = "therapist"
    SYSTEM = "system"


@dataclass
class MemoryMessage:
    """Single message in memory."""

    content: str
    role: MessageRole
    timestamp: datetime
    message_id: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryContext:
    """Complete memory context for a user."""

    user_id: str
    session_id: str
    messages: list[MemoryMessage]
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    summary: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemoryManager:
    """
    Manages memory persistence and retrieval using the configured backend.
    """

    def __init__(self, memory_client: Any):
        if not memory_client:
            raise ValueError("memory_client is required")
        self.client = memory_client

    def add_message(
        self,
        user_id: str,
        session_id: str,
        content: str,
        role: MessageRole,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        try:
            # Check if client supports legacy CRUD-style storage helpers
            if not hasattr(self.client, "add"):
                logger.error("Memory client does not support 'add'")
                return False

            self.client.add(
                content,
                user_id=user_id,
                metadata={
                    "session_id": session_id,
                    "role": role.value,
                    "memory_type": memory_type.value,
                    **(metadata or {}),
                },
            )
            return True
        except Exception as e:
            logger.error(f"Error adding message to memory backend: {e}")
            return False

    def get_conversation_history(
        self, user_id: str, session_id: str, limit: int = 50
    ) -> list[MemoryMessage]:
        try:
            if not hasattr(self.client, "get_all"):
                return []

            memories = self.client.get_all(user_id=user_id)
            if isinstance(memories, dict):
                memories = memories.get("results", [])

            # Filter by session_id in metadata
            session_messages = [
                m
                for m in memories
                if isinstance(m, dict)
                and m.get("metadata", {}).get("session_id") == session_id
            ]

            # Convert to MemoryMessage
            return [
                MemoryMessage(
                    content=m.get("content", "") or m.get("memory", ""),
                    role=MessageRole(m.get("metadata", {}).get("role", "user")),
                    timestamp=datetime.now(
                        timezone.utc
                    ),
                    message_id=m.get("id"),
                    metadata=m.get("metadata", {}),
                )
                for m in session_messages[:limit]
            ]
        except Exception as e:
            logger.error(f"Error retrieving history from memory backend: {e}")
            return []

    def store_session_summary(self, *args, **kwargs) -> bool:
        # Simplified for now, just adding as a factual memory
        summary = kwargs.get("summary")
        if not summary and len(args) > 2:
            summary = args[2]

        uid = kwargs.get("user_id")
        if not uid and args:
            uid = args[0]

        sid = kwargs.get("session_id")
        if not sid and len(args) > 1:
            sid = args[1]

        summary_text = f"Session Summary: {summary}"
        return self.add_message(
            user_id=uid,
            session_id=sid,
            content=summary_text,
            role=MessageRole.SYSTEM,
            memory_type=MemoryType.SESSION_SUMMARY,
        )

    def get_emotional_state(
        self, user_id: str, session_id: str
    ) -> dict[str, Any] | None:
        if not hasattr(self.client, "search"):
            return None

        results = self.client.search("emotional state", user_id=user_id)
        if isinstance(results, dict):
            results = results.get("results", [])

        # Filter for current session if possible
        return next(
            (
                {
                    "content": r.get("content") or r.get("memory"),
                    "metadata": r.get("metadata"),
                }
                for r in results
                if r.get("metadata", {}).get("session_id") == session_id
            ),
            None,
        )

    def store_emotional_state(
        self,
        user_id: str,
        session_id: str,
        emotions: dict[str, float],
        context: str,
        triggers: list[str] = None,
    ) -> bool:
        content = (
            f"Emotional state: {emotions}. Context: {context}. Triggers: {triggers}"
        )
        return self.add_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
            role=MessageRole.SYSTEM,
            memory_type=MemoryType.EMOTIONAL_STATE,
        )

    def clear_session_memory(self, session_id: str) -> bool:
        logger.warning(
            "Clear session memory for %s is not implemented for the current backend",
            session_id,
        )
        return True

    def get_memory_stats(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "provider": type(self.client).__name__,
        }


_memory_manager_instance: MemoryManager | None = None


def get_memory_manager(memory_client: Any | None = None) -> MemoryManager:
    """
    Get or create the global MemoryManager instance.

    Args:
        memory_client: Optional pre-configured memory client

    Returns:
        Configured MemoryManager instance

    Requires a configured shared local memory backend.
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        if not memory_client:
            memory_client = get_backend_memory_manager()
            logger.info(
                "Initialized backend client for MemoryManager: %s",
                type(memory_client).__name__,
            )

        _memory_manager_instance = MemoryManager(memory_client)
    return _memory_manager_instance
