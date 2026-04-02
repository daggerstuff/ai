"""
Subconscious context using contextvars.

This is the core mechanism. A contextvar holds the state,
flowing through async call chains without explicit passing.

Thread-safe. Async-safe. No globals.
"""

import asyncio
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .conversation_manager import ConversationManager
from .constants import MAX_QUERY_LENGTH
from .config import SubconsciousConfig, UserConfig
from .memory_enrichment import enrich_user_message
from .provider import (
    Memory,
    MemoryProvider,
    close_memory_provider,
    create_memory_provider,
    flush_memory_provider,
)
from .reflection import reflect_conversation

__all__ = [
    "SubconsciousState",
    "set_subconscious",
    "get_subconscious",
    "reset_subconscious",
]

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# The context variable. Holds SubconsciousState or None.
subconscious_context: ContextVar[Optional["SubconsciousState"]] = ContextVar(
    "subconscious_context", default=None
)


@dataclass
class SubconsciousState:
    """
    Active subconscious state for a session.

    Created once per session, holds conversation history,
    and triggers reflection on close.
    """

    config: UserConfig
    _provider: Optional[MemoryProvider] = None
    _conversation_manager: ConversationManager = field(default_factory=ConversationManager)
    _closed: bool = False

    async def _ensure_provider(self) -> MemoryProvider:
        """Lazy init of memory provider."""
        if self._provider is None:
            self._provider = create_memory_provider(self.config.base)
        return self._provider

    async def enrich(self, message: str) -> str:
        """
        Enrich a message with relevant memories.

        Returns the original message if:
        - Disabled
        - No memories found
        - Timeout/error (and fail_open=True)

        Otherwise returns: <memories>\\n\\n{message}
        """
        if not self.config.base.enabled:
            logger.debug("Subconscious disabled, returning original message")
            return message

        if self._closed:
            logger.warning("Attempted to enrich after state closed")
            return message

        try:
            provider = await self._ensure_provider()

            # Validate input
            if not message or not message.strip():
                logger.debug("Empty message, skipping enrichment")
                return message

            memories = await self._recall_memories(provider, message)

            if not memories:
                logger.debug("No memories found, returning original message")
                return message

            # Format memories as XML block
            logger.info(f"Enriched message with {len(memories)} memories")
            return enrich_user_message(
                conversation_manager=self._conversation_manager,
                message=message,
                memories=memories,
                max_memories=self.config.base.max_memories,
            )

        except asyncio.TimeoutError:
            logger.warning(
                f"Memory lookup timed out for user={self.config.user_id} "
                f"(timeout={self.config.base.query_timeout_ms}ms)"
            )
            if self.config.base.fail_open:
                return message
            raise

        except Exception as e:
            logger.error(f"Memory lookup failed: {e}", exc_info=True)
            if self.config.base.fail_open:
                return message
            raise

    def _recall_timeout_seconds(self, provider: MemoryProvider) -> float:
        """Give provider-native timeouts room to fail cleanly before outer cancellation."""
        base_timeout = self.config.base.query_timeout_ms / 1000
        provider_timeout_ms = getattr(provider, "timeout_ms", None)
        if isinstance(provider_timeout_ms, int) and provider_timeout_ms > 0:
            return max(base_timeout, (provider_timeout_ms / 1000) + 1.0)
        return base_timeout

    async def _recall_memories(
        self,
        provider: MemoryProvider,
        message: str,
    ) -> List[Memory]:
        """Recall scoped memories with an explicit timeout contract."""
        return await asyncio.wait_for(
            provider.recall(
                query=message[:MAX_QUERY_LENGTH],
                user_id=self.config.user_id,
                limit=self.config.base.max_memories,
            ),
            timeout=self._recall_timeout_seconds(provider),
        )

    def record(self, role: str, content: str) -> None:
        """
        Record a message for later reflection.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        if self._closed:
            logger.warning("Attempted to record after state closed")
            return

        if not role or not content:
            logger.debug(
                f"Empty record: role={role}"
            )
            return

        self._conversation_manager.record_message(role, content)

    async def reflect(self):
        """
        Trigger reflection on the conversation.

        Analyzes the conversation and stores extracted memories.
        """
        if self._closed:
            logger.warning("Attempted to reflect after state closed")
            return

        if not self._conversation_manager.conversation:
            logger.debug("No conversation to reflect on")
            return

        provider = await self._ensure_provider()

        stored_count = await reflect_conversation(
            provider=provider,
            user_id=self.config.user_id,
            conversation=self._conversation_manager.conversation,
            api_key=self.config.base.api_key,
            base_url=self.config.base.base_url,
            model=self.config.base.model,
            focus_prompt=(
                "Extract learnings from conversation.\n\n"
                "Focus on:\n"
                "- User preferences and patterns\n"
                "- Project-specific knowledge\n"
                "- Important decisions or context"
            ),
        )

        logger.info(
            f"Reflected on {len(self._conversation_manager.conversation)} messages, "
            f"stored {stored_count} memories"
        )

    async def close(self):
        """
        Close the state, triggering reflection if configured.

        This is called automatically by reset_subconscious().
        """
        if self._closed:
            logger.debug("State already closed")
            return

        logger.debug(f"Closing subconscious state for user {self.config.user_id}")

        if self.config.base.reflect_on_close and self._conversation_manager.conversation:
            await self.reflect()

        await flush_memory_provider(self._provider)
        # Close provider connection if it has one
        await close_memory_provider(self._provider)

        # Clear conversation
        self._conversation_manager.clear()
        self._closed = True


def set_subconscious(config: SubconsciousConfig, user_id: str) -> Token:
    """
    Set the subconscious context for this async chain.

    Returns a token that must be passed to reset_subconscious().

    Usage:
        token = set_subconscious(config, user_id="alice")
        try:
            # ... do work ...
        finally:
            await reset_subconscious(token)

    Args:
        config: Subconscious configuration
        user_id: User identifier

    Returns:
        ContextVar token for cleanup

    Raises:
        ValueError: If user_id is empty
    """
    user_config = config.with_user(user_id)
    state = SubconsciousState(config=user_config)
    token = subconscious_context.set(state)
    logger.debug(f"Set subconscious context for user {user_id}")
    return token


def get_subconscious() -> Optional[SubconsciousState]:
    """
    Get the current subconscious state, or None if not set.

    Usage:
        state = get_subconscious()
        if state:
            enriched = await state.enrich(message)
    """
    return subconscious_context.get()


async def reset_subconscious(token: Token):
    """
    Reset the subconscious context, triggering cleanup.

    Must be called with the token from set_subconscious().

    Usage:
        token = set_subconscious(config, user_id="alice")
        try:
            # ... work ...
        finally:
            await reset_subconscious(token)

    Args:
        token: Token returned by set_subconscious()
    """
    state = subconscious_context.get()
    if state:
        await state.close()

    try:
        subconscious_context.reset(token)
        logger.debug("Reset subconscious context")
    except ValueError as e:
        # Token from different context - state already cleaned up
        # This is expected in nested context scenarios, log and continue
        logger.debug(f"ContextVar reset with token from different context: {e}")
