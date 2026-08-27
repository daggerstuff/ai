"""
Memory System Integration Module.

Integrates the configured shared memory backend with higher-level helpers for
managing user memory contexts, conversation history, and therapeutic sessions.

Also provides dream-cycle integration — the DreamManager runs NREM/REM-style
consolidation after session processing to surface themes, patterns, and insights.
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ai.research.manager_factory import (
    create_dream_manager as _create_dream_manager,
    get_required_memory_manager as get_backend_memory_manager,
)

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

    Also manages dream cycles for memory consolidation. The dream manager
    is lazily created on first access and runs NREM/REM-style processing
    to extract themes, patterns, and insights from session memories.
    """

    def __init__(self, memory_client: Any, mongodb_uri: str | None = None):
        if not memory_client:
            raise ValueError("memory_client is required")
        self.client = memory_client
        self._mongodb_uri = mongodb_uri
        self._dream_manager: Any | None = None  # DreamManager, lazily created

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

    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 50) -> list[MemoryMessage]:
        try:
            if not hasattr(self.client, "get_all"):
                return []

            memories = self.client.get_all(user_id=user_id)
            if isinstance(memories, dict):
                memories = memories.get("results", [])

            # Filter by session_id in metadata
            session_messages = [
                m for m in memories if isinstance(m, dict) and m.get("metadata", {}).get("session_id") == session_id
            ]

            # Convert to MemoryMessage
            return [
                MemoryMessage(
                    content=m.get("content", "") or m.get("memory", ""),
                    role=MessageRole(m.get("metadata", {}).get("role", "user")),
                    timestamp=datetime.now(UTC),
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

    def get_emotional_state(self, user_id: str, session_id: str) -> dict[str, Any] | None:
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
        triggers: list[str] | None = None,
    ) -> bool:
        content = f"Emotional state: {emotions}. Context: {context}. Triggers: {triggers}"
        return self.add_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
            role=MessageRole.SYSTEM,
            memory_type=MemoryType.EMOTIONAL_STATE,
        )

    def clear_session_memory(self, session_id: str, user_id: str) -> bool:
        """
        Clear all memories associated with a specific session.

        Fetches every memory for the user, filters by ``session_id`` in
        the metadata, then deletes each matching record individually so
        that memories from other sessions are preserved.

        Args:
            session_id: Session whose memories should be cleared.
            user_id: Owner of the memories (required by the backend).

        Returns:
            True if all session memories were cleared (or none existed),
            False if any deletion failed or the backend lacks the needed
            operations.
        """
        if not session_id:
            logger.error("clear_session_memory: session_id is required")
            return False
        if not user_id:
            logger.error("clear_session_memory: user_id is required")
            return False

        try:
            # --- Fetch all memories for the user ---
            # The backend may expose either the legacy-adapter interface
            # (``get_all`` returning a dict) or the native interface
            # (``get_all_memories`` returning a list).
            memories: list[dict[str, Any]] = []

            if hasattr(self.client, "get_all"):
                raw = self.client.get_all(user_id=user_id)
                if isinstance(raw, dict):
                    memories = raw.get("results", [])
                elif isinstance(raw, list):
                    memories = raw
            elif hasattr(self.client, "get_all_memories"):
                memories = self.client.get_all_memories(user_id=user_id, limit=10000)

            # --- Filter to memories belonging to this session ---
            session_memory_ids: list[str] = []
            for m in memories:
                if not isinstance(m, dict):
                    continue
                meta = m.get("metadata") or {}
                if meta.get("session_id") == session_id:
                    mid = m.get("id")
                    if mid:
                        session_memory_ids.append(str(mid))

            if not session_memory_ids:
                logger.info(
                    "No memories found for session %s (user %s); nothing to clear",
                    session_id,
                    user_id,
                )
                return True

            # --- Delete matching memories ---
            # Prefer batch deletion when available for efficiency.
            if hasattr(self.client, "_delete_memories"):
                deleted = self.client._delete_memories(
                    session_memory_ids, user_id=user_id
                )
                if isinstance(deleted, int):
                    success = deleted == len(session_memory_ids)
                else:
                    success = bool(deleted)
                if not success:
                    logger.warning(
                        "Batch delete cleared %s/%s memories for session %s",
                        deleted,
                        len(session_memory_ids),
                        session_id,
                    )
                return success

            # Fallback: delete one-by-one via whichever interface is available.
            failed: list[str] = []
            for mid in session_memory_ids:
                ok = False
                if hasattr(self.client, "delete"):
                    ok = self.client.delete(mid, user_id=user_id)
                elif hasattr(self.client, "delete_memory"):
                    ok = self.client.delete_memory(mid, user_id=user_id)
                else:
                    logger.error(
                        "Memory client does not support deletion (no "
                        "'delete' or 'delete_memory' method)"
                    )
                    return False
                if not ok:
                    failed.append(mid)

            if failed:
                logger.warning(
                    "Failed to delete %d/%d memories for session %s: %s",
                    len(failed),
                    len(session_memory_ids),
                    session_id,
                    failed,
                )
                return False

            logger.info(
                "Cleared %d memories for session %s (user %s)",
                len(session_memory_ids),
                session_id,
                user_id,
            )
            return True
        except Exception as e:
            logger.error(
                "Error clearing session memory for session %s: %s",
                session_id,
                e,
            )
            return False

    def get_memory_stats(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "provider": type(self.client).__name__,
        }

    # ------------------------------------------------------------------
    # Dream cycle integration
    # ------------------------------------------------------------------

    @property
    def dream_manager(self) -> Any:
        """Lazily-initialized DreamManager instance."""
        if self._dream_manager is None:
            self._dream_manager = _create_dream_manager(mongodb_uri=self._mongodb_uri)
            logger.info(
                "DreamManager initialized (%s)",
                type(self._dream_manager.memory_store).__name__,
            )
        return self._dream_manager

    async def trigger_dream_cycle(
        self,
        user_id: str,
        memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Run a full dream cycle (NREM → REM → consolidation → reflection)
        for the given user's memories.

        Args:
            user_id: User whose memories to process.
            memories: Optional pre-fetched memory list. When omitted the
                      dream manager fetches them from its store.

        Returns:
            Dream cycle result as a serialisable dict.
        """
        result = await self.dream_manager.start_dream_cycle(
            user_id=user_id,
            memories=memories,
        )
        logger.info(
            "Dream cycle %s for user %s: %d themes, %d patterns",
            result.dream_id,
            user_id,
            len(result.themes),
            len(result.patterns),
        )
        return result.to_dict()

    async def get_dream_status(self, dream_id: str) -> dict[str, Any] | None:
        """Return the status of an active or completed dream cycle."""
        status = await self.dream_manager.get_dream_status(dream_id)
        if status is None:
            return None
        return status

    async def close_dream_manager(self) -> None:
        """Release the dream manager resources (connections, tasks)."""
        if self._dream_manager is not None:
            await self._dream_manager.close()
            self._dream_manager = None
            logger.info("DreamManager closed")

    async def close(self) -> None:
        """Release all resources including dream manager."""
        await self.close_dream_manager()


_memory_manager_instance: MemoryManager | None = None


def get_memory_manager(
    memory_client: Any | None = None,
    mongodb_uri: str | None = None,
) -> MemoryManager:
    """
    Get or create the global MemoryManager instance.

    Args:
        memory_client: Optional pre-configured memory client.
        mongodb_uri: Optional MongoDB URI for dream store. Falls back to
                     ``MONGODB_URI`` environment variable.

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

        resolved_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        _memory_manager_instance = MemoryManager(
            memory_client,
            mongodb_uri=resolved_uri,
        )
    return _memory_manager_instance
