"""
Subconscious context using contextvars.

This is the core mechanism. A contextvar holds the state,
flowing through async call chains without explicit passing.

Thread-safe. Async-safe. No globals.
"""

import asyncio
import copy
import json
import logging
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .config import SubconsciousConfig, UserConfig
from .provider import LocalHindsightProvider, Memory, MemoryProvider

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
    _conversation: List[dict] = field(default_factory=list)
    _closed: bool = False

    async def _ensure_provider(self) -> MemoryProvider:
        """Lazy init of memory provider."""
        if self._provider is None:
            if self.config.base.memory_provider == "mock":
                from .provider import MockProvider

                self._provider = MockProvider()
            else:
                self._provider = LocalHindsightProvider(
                    self.config.base.bank_id,
                    max_retries=self.config.base.max_retries,
                    retry_delay_ms=self.config.base.retry_delay_ms,
                )
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

            memories = await asyncio.wait_for(
                provider.recall(
                    query=message[:500],  # Truncate for search
                    user_id=self.config.user_id,
                    limit=self.config.base.max_memories,
                ),
                timeout=self.config.base.query_timeout_ms / 1000,
            )

            if not memories:
                logger.debug("No memories found, returning original message")
                return message

            # Format memories as XML block
            memory_xml = self._format_memories(memories)
            logger.info(f"Enriched message with {len(memories)} memories")
            return f"{memory_xml}\\n\\n{message}"

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

    def _format_memories(self, memories: List[Memory]) -> str:
        """Format memories for LLM injection."""
        if not memories:
            return ""

        lines = ["<subconscious_context>"]
        lines.append("  <relevant_memories>")
        for mem in memories[: self.config.base.max_memories]:
            # Truncate long memories
            content = mem.content[:200]
            if len(mem.content) > 200:
                content += "..."
            lines.append(f"    - {content}")
        lines.append("  </relevant_memories>")
        lines.append("</subconscious_context>")

        return "\\n".join(lines)

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

        # Deep copy to prevent mutation
        self._conversation.append({"role": role, "content": copy.deepcopy(content)})

    async def reflect(self):
        """
        Trigger reflection on the conversation.

        Analyzes the conversation and stores extracted memories.
        """
        if self._closed:
            logger.warning("Attempted to reflect after state closed")
            return

        if not self._conversation:
            logger.debug("No conversation to reflect on")
            return

        provider = await self._ensure_provider()

        # Build conversation text for analysis
        conv_text = "\\n".join(
            f"{m['role']}: {m['content']}" for m in self._conversation
        )

        # Extract learnings using LLM
        learnings = await self._extract_learnings(conv_text)

        # Store new memories
        stored_count = 0
        for learning in learnings:
            try:
                await provider.store(
                    content=learning,
                    user_id=self.config.user_id,
                    metadata={"source": "reflection"},
                )
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store memory: {e}", exc_info=True)

        logger.info(
            f"Reflected on {len(self._conversation)} messages, "
            f"stored {stored_count}/{len(learnings)} memories"
        )

    async def _extract_learnings(self, conversation: str) -> List[str]:
        """Use LLM to extract learnings from conversation."""
        # Skip if no API key
        if not self.config.base.api_key:
            logger.warning("No API key configured, skipping reflection")
            return []

        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.config.base.api_key,
                base_url=self.config.base.base_url,
            )

            prompt = f"""Extract learnings from conversation.

Focus on:
- User preferences and patterns
- Project-specific knowledge
- Important decisions or context

Conversation:
{conversation[:3000]}

Respond with a JSON array of learnings. Each learning should be a single string.
Example: ["User prefers TypeScript over JavaScript", "Project uses pnpm not npm"]
"""

            response = await client.chat.completions.create(
                model=self.config.base.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )

            content = response.choices[0].message.content or "[]"

            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Extract array if wrapped in markdown code blocks or text
                # Use regex for more robust extraction
                match = re.search(r"\[[\s\S]*?\]", content)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"JSON parse failed: {match.group(0)[:50]}"
                        )
                return []

        except Exception as e:
            logger.error(f"LLM reflection failed: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Close the state, triggering reflection if configured.

        This is called automatically by reset_subconscious().
        """
        if self._closed:
            logger.debug("State already closed")
            return

        self._closed = True
        logger.debug(f"Closing subconscious state for user {self.config.user_id}")

        if self.config.base.reflect_on_close and self._conversation:
            await self.reflect()

        # Close provider connection if it has one
        if self._provider and hasattr(self._provider, "close"):
            await self._provider.close()

        # Clear conversation
        self._conversation.clear()


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
