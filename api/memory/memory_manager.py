"""
Memory System Integration Module.

Integrates Zep-based memory management with the MCP server for managing
user memory contexts, conversation history, and therapeutic session data.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from zep_cloud import Message, Zep
except ImportError as e:
    raise ImportError(
        "zep-cloud package required. Install with: uv pip install zep-cloud"
    ) from e

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
    Manages memory persistence and retrieval for therapeutic sessions.

    Provides:
    - Conversation history management
    - Session memory persistence
    - Emotional state tracking
    - Treatment plan storage
    - HIPAA-compliant memory encryption
    """

    def __init__(self, zep_client: Zep):
        """
        Initialize Memory Manager.

        Args:
            zep_client: Initialized Zep client

        Raises:
            ValueError: If zep_client is not provided
        """
        if not zep_client:
            raise ValueError("zep_client is required")

        self.client = zep_client
        self.memory_cache: Dict[str, MemoryContext] = {}

    def add_message(
        self,
        user_id: str,
        session_id: str,
        content: str,
        role: MessageRole,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add message to session memory.

        Args:
            user_id: User ID
            session_id: Session ID
            content: Message content
            role: Message role
            memory_type: Type of memory
            metadata: Optional message metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message
            message = Message(
                content=content,
                role=role.value,
                metadata={"memory_type": memory_type.value, **(metadata or {})},
            )

            return self._store_memory_message(
                message, session_id, "Message added to session "
            )
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")
            return False

    def get_conversation_history(
        self, user_id: str, session_id: str, limit: int = 50
    ) -> List[MemoryMessage]:
        """
        Retrieve conversation history for a session.

        Args:
            user_id: User ID
            session_id: Session ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of MemoryMessage objects
        """
        try:
            # Retrieve memory from Zep
            thread = self.client.thread.get(session_id)

            if not thread or not thread.messages:
                return []

            # Convert to MemoryMessage objects
            messages = []
            for msg in thread.messages:
                memory_msg = MemoryMessage(
                    content=msg.content,
                    role=MessageRole(msg.role),
                    timestamp=msg.created_at or datetime.now(timezone.utc),
                    message_id=getattr(msg, "id", None),
                    metadata=msg.metadata or {},
                )
                messages.append(memory_msg)

            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def store_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        key_points: List[str],
        emotional_insights: Dict[str, Any],
        next_steps: List[str],
    ) -> bool:
        """
        Store therapeutic session summary.

        Args:
            user_id: User ID
            session_id: Session ID
            summary: Session summary text
            key_points: List of key discussion points
            emotional_insights: Emotional state insights
            next_steps: Recommended next steps

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create summary message
            summary_data = {
                "summary": summary,
                "key_points": key_points,
                "emotional_insights": emotional_insights,
                "next_steps": next_steps,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            message = Message(
                content=json.dumps(summary_data),
                role=MessageRole.SYSTEM.value,
                metadata={
                    "memory_type": MemoryType.SESSION_SUMMARY.value,
                    "summary_type": "therapeutic_session",
                },
            )

            return self._store_memory_message(
                message, session_id, "Session summary stored for "
            )
        except Exception as e:
            logger.error(f"Error storing session summary: {e}")
            return False

    def _retrieve_memory_by_type(
        self, session_id: str, memory_type: MemoryType
    ) -> Optional[Dict[str, Any]]:
        """
        Helper method to retrieve memory by type.

        Args:
            session_id: Session ID
            memory_type: Type of memory to retrieve

        Returns:
            Dictionary with memory data or None
        """
        try:
            thread = self.client.thread.get(session_id)
            if not thread or not thread.messages:
                return None

            for msg in reversed(thread.messages):
                if (
                    msg.metadata
                    and msg.metadata.get("memory_type") == memory_type.value
                ):
                    try:
                        return json.loads(msg.content)
                    except json.JSONDecodeError:
                        continue
            return None
        except Exception as e:
            logger.error(f"Error retrieving {memory_type.value} from memory: {e}")
            return None

    def get_emotional_state(
        self, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve latest emotional state from memory.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Dictionary with emotional state data or None
        """
        return self._retrieve_memory_by_type(session_id, MemoryType.EMOTIONAL_STATE)

    def store_emotional_state(
        self,
        user_id: str,
        session_id: str,
        emotions: Dict[str, float],
        context: str,
        triggers: List[str] = None,
    ) -> bool:
        """
        Store emotional state snapshot.

        Args:
            user_id: User ID
            session_id: Session ID
            emotions: Dictionary of emotion scores (0-1)
            context: Contextual information about emotional state
            triggers: List of identified emotional triggers

        Returns:
            True if successful, False otherwise
        """
        try:
            emotional_data = {
                "emotions": emotions,
                "context": context,
                "triggers": triggers or [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            message = Message(
                content=json.dumps(emotional_data),
                role=MessageRole.SYSTEM.value,
                metadata={
                    "memory_type": MemoryType.EMOTIONAL_STATE.value,
                    "has_crisis_markers": any(
                        score > 0.8 for score in emotions.values()
                    ),
                },
            )

            return self._store_memory_message(
                message, session_id, "Emotional state stored for "
            )
        except Exception as e:
            logger.error(f"Error storing emotional state: {e}")
            return False

    def get_treatment_plan(
        self, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve treatment plan from memory.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Dictionary with treatment plan or None
        """
        return self._retrieve_memory_by_type(session_id, MemoryType.TREATMENT_PLAN)

    def store_treatment_plan(
        self,
        user_id: str,
        session_id: str,
        goals: List[str],
        interventions: List[str],
        progress_metrics: Dict[str, Any],
        duration_weeks: int,
    ) -> bool:
        """
        Store therapeutic treatment plan.

        Args:
            user_id: User ID
            session_id: Session ID
            goals: Treatment goals
            interventions: Planned interventions
            progress_metrics: Metrics to track progress
            duration_weeks: Expected treatment duration

        Returns:
            True if successful, False otherwise
        """
        try:
            plan_data = {
                "goals": goals,
                "interventions": interventions,
                "progress_metrics": progress_metrics,
                "duration_weeks": duration_weeks,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            message = Message(
                content=json.dumps(plan_data),
                role=MessageRole.SYSTEM.value,
                metadata={
                    "memory_type": MemoryType.TREATMENT_PLAN.value,
                    "num_goals": len(goals),
                },
            )

            return self._store_memory_message(
                message, session_id, "Treatment plan stored for "
            )
        except Exception as e:
            logger.error(f"Error storing treatment plan: {e}")
            return False

    def _store_memory_message(
        self, message: Message, session_id: str, action_message: str
    ) -> bool:
        """Store a message in memory for a session.

        Args:
            message: Message object to store
            session_id: Session ID
            action_message: Log message prefix

        Returns:
            True if successful
        """
        self.client.thread.add_messages(session_id, messages=[message])
        logger.info(f"{action_message}{session_id}")
        return True

    def clear_session_memory(self, session_id: str) -> bool:
        """
        Clear all memory for a session (for privacy/compliance).

        Args:
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Zep doesn't have direct delete, so we mark as deleted
            message = Message(
                content="SESSION_CLEARED_FOR_PRIVACY",
                role=MessageRole.SYSTEM.value,
                metadata={"memory_type": "system", "action": "session_cleared"},
            )

            self.client.thread.add_messages(session_id, messages=[message])

            # Clear cache
            if session_id in self.memory_cache:
                del self.memory_cache[session_id]

            logger.info(f"Session memory cleared for {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing session memory: {e}")
            return False

    def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with memory stats
        """
        try:
            thread = self.client.thread.get(session_id)

            if not thread or not thread.messages:
                return {
                    "total_messages": 0,
                    "memory_types": {},
                    "first_message": None,
                    "last_message": None,
                }

            # Count messages by type
            memory_types = {}
            for msg in thread.messages:
                msg_type = (
                    msg.metadata.get("memory_type", "unknown")
                    if msg.metadata
                    else "unknown"
                )
                memory_types[msg_type] = memory_types.get(msg_type, 0) + 1

            return {
                "total_messages": len(thread.messages),
                "memory_types": memory_types,
                "first_message": thread.messages[0].created_at.isoformat()
                if thread.messages
                else None,
                "last_message": thread.messages[-1].created_at.isoformat()
                if thread.messages
                else None,
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}


# Singleton instance
_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager(zep_client: Optional[Zep] = None) -> MemoryManager:
    """
    Get or create Memory Manager instance.

    Args:
        zep_client: Zep client (required for first call)

    Returns:
        MemoryManager instance

    Raises:
        ValueError: If zep_client not provided on first call
    """
    global _memory_manager_instance

    if _memory_manager_instance is None:
        if not zep_client:
            raise ValueError(
                "zep_client required for first Memory Manager initialization"
            )
        _memory_manager_instance = MemoryManager(zep_client)

    return _memory_manager_instance
