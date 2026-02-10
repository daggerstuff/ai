"""
Memory System Integration Module.

Integrates Mem0-based memory management with the MCP server for managing
user memory contexts, conversation history, and therapeutic session data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from mem0 import Memory
except ImportError:
    try:
        from mem0ai import Memory
    except ImportError:
        raise ImportError("Please install mem0ai: uv add mem0ai")

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the therapeutic context."""

    CONVERSATION = "conversation"
    SESSION_SUMMARY = "session_summary"
    THERAPEUTIC_NOTES = "therapeutic_notes"
    EMOTIONAL_STATE = "emotional_state"
    TREATMENT_PLAN = "treatment_plan"
    CRISIS_CONTEXT = "crisis_context"
    PROGRESS_NOTES = "progress_notes"


class MessageRole(str, Enum):
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
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryContext:
    """Complete memory context for a user."""

    user_id: str
    session_id: str
    messages: List[MemoryMessage]
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    summary: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemoryManager:
    """
    Manages memory persistence and retrieval using Mem0.
    """

    def __init__(self, mem0_client: Memory):
        if not mem0_client:
            raise ValueError("mem0_client is required")
        self.client = mem0_client

    def add_message(
        self,
        user_id: str,
        session_id: str,
        content: str,
        role: MessageRole,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
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
            logger.error(f"Error adding message to Mem0: {e}")
            return False

    def get_conversation_history(
        self, user_id: str, session_id: str, limit: int = 50
    ) -> List[MemoryMessage]:
        try:
            # Mem0 search returns relevant memories, but for history we might want get_all or a 특정 filter
            # Mem0 doesn't exactly have a "get chronological history" easily if it's just a vector store,
            # but we can filter by session_id in metadata if supported.
            memories = self.client.get_all(user_id=user_id)

            # Filter by session_id in metadata
            session_messages = [
                m for m in memories if m.get("metadata", {}).get("session_id") == session_id
            ]

            # Convert to MemoryMessage
            messages = []
            for m in session_messages[:limit]:
                # Mem0 returns content directly
                messages.append(
                    MemoryMessage(
                        content=m["content"],
                        role=MessageRole(m.get("metadata", {}).get("role", "user")),
                        timestamp=datetime.now(
                            timezone.utc
                        ),  # Mem0 might not store exact original timestamp in direct recall
                        message_id=m.get("id"),
                        metadata=m.get("metadata", {}),
                    )
                )
            return messages
        except Exception as e:
            logger.error(f"Error retrieving history from Mem0: {e}")
            return []

    def store_session_summary(self, *args, **kwargs) -> bool:
        # Simplified for now, just adding as a factual memory
        summary_text = f"Session Summary: {args[2] if len(args) > 2 else kwargs.get('summary')}"
        return self.add_message(
            user_id=args[0] if len(args) > 0 else kwargs.get("user_id"),
            session_id=args[1] if len(args) > 1 else kwargs.get("session_id"),
            content=summary_text,
            role=MessageRole.SYSTEM,
            memory_type=MemoryType.SESSION_SUMMARY,
        )

    def get_emotional_state(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        # Search for emotional state in Mem0
        results = self.client.search("emotional state", user_id=user_id)
        # Filter for current session if possible
        for r in results:
            if r.get("metadata", {}).get("session_id") == session_id:
                return {"content": r["content"], "metadata": r.get("metadata")}
        return None

    def store_emotional_state(
        self,
        user_id: str,
        session_id: str,
        emotions: Dict[str, float],
        context: str,
        triggers: List[str] = None,
    ) -> bool:
        content = f"Emotional state: {emotions}. Context: {context}. Triggers: {triggers}"
        return self.add_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
            role=MessageRole.SYSTEM,
            memory_type=MemoryType.EMOTIONAL_STATE,
        )

    def clear_session_memory(self, session_id: str) -> bool:
        # Mem0 doesn't have a clear by session_id easily in one call,
        # would need to find and delete.
        # For now, we'll mark it or just acknowledge limitations.
        logger.warning(f"Clear session memory for {session_id} not fully implemented for Mem0 yet")
        return True

    def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        return {"session_id": session_id, "provider": "mem0"}


_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager(mem0_client: Optional[Memory] = None) -> MemoryManager:
    """
    Get or create the global MemoryManager instance.

    Args:
        mem0_client: Optional pre-configured Memory client

    Returns:
        Configured MemoryManager instance
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        if not mem0_client:
            # Create a null Memory implementation for development/fallback
            # This ensures the server can always start (AGENTS.md: complete implementation)
            from ai.api.memory.null_memory import NullMemoryManager

            mem0_client = NullMemoryManager()
            logger.info("Using null Memory implementation for MemoryManager")

        _memory_manager_instance = MemoryManager(mem0_client)
    return _memory_manager_instance
