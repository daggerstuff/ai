"""
SubconsciousClient - Explicit API for memory-aware LLM calls.

Use this when you want direct control. For transparent injection,
use the contextvars API instead.

Usage:
    from ai.memory.v3 import SubconsciousClient, SubconsciousConfig

    config = SubconsciousConfig.from_env()
    client = await SubconsciousClient.create(config, user_id="alice")

    response = await client.chat([
        {"role": "user", "content": "How do I run tests?"}
    ])

    await client.close()  # Triggers reflection
"""

import asyncio
import copy
import json
import logging
import re
from typing import Any, List, Optional

from .config import SubconsciousConfig, UserConfig
from .provider import LocalHindsightProvider, MemoryProvider

__all__ = ["SubconsciousClient"]

logger = logging.getLogger(__name__)

# Constants for magic numbers
MAX_CONVERSATION_LENGTH = 3000
MAX_QUERY_LENGTH = 500
MAX_TOKENS = 500


class SubconsciousClient:
    """
    Memory-aware wrapper for any LLM client.

    This is the explicit API. Set it up, use it, close it.

    For transparent injection (Claude Code sessions), use the
    contextvars API instead: set_subconscious(), get_subconscious().
    """

    def __init__(self, config: UserConfig, llm_client: Any):
        """
        Create a client. Use create() factory instead.

        Args:
            config: User-bound config
            llm_client: Any OpenAI-compatible async client
        """
        self.config = config
        self.llm_client = llm_client
        self._provider: Optional[MemoryProvider] = None
        self._conversation: List[dict] = []
        self._closed = False

    @classmethod
    async def create(
        cls,
        config: SubconsciousConfig,
        user_id: str,
        llm_client: Optional[Any] = None,
    ) -> "SubconsciousClient":
        """
        Create and initialize a SubconsciousClient.

        Args:
            config: Base configuration
            user_id: User identifier
            llm_client: Optional LLM client (creates default if None)

        Returns:
            Initialized client ready to use

        Raises:
            ValueError: If user_id is empty
        """
        user_config = config.with_user(user_id)

        # Create default client if none provided
        if llm_client is None:
            import openai

            llm_client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            logger.debug("Created default OpenAI client")

        client = cls(user_config, llm_client)
        await client._init_provider()
        return client

    async def _init_provider(self):
        """Initialize memory provider."""
        if self.config.base.memory_provider == "mock":
            from .provider import MockProvider

            self._provider = MockProvider()
        else:
            self._provider = LocalHindsightProvider(
                self.config.base.bank_id,
                max_retries=self.config.base.max_retries,
                retry_delay_ms=self.config.base.retry_delay_ms,
            )
        logger.debug(f"Initialized {self.config.base.memory_provider} provider")

    async def chat(
        self,
        messages: List[dict],
        enrich: bool = True,
        **kwargs,
    ) -> Any:
        """
        Send a chat request with memory enrichment.

        Args:
            messages: OpenAI-format message list
            enrich: If True, enrich the last user message with memories
            **kwargs: Passed to underlying client

        Returns:
            Response from underlying client
        """
        if self._closed:
            raise RuntimeError("Client is closed")

        if not messages:
            raise ValueError("Messages cannot be empty")

        if not self.config.base.enabled:
            logger.debug("Subconscious disabled, passing through to LLM")
            return await self._call_llm(messages, **kwargs)

        # Deep copy to prevent mutation of original
        enriched_messages = copy.deepcopy(messages)

        # Enrich last user message
        if enrich:
            last_user_idx = self._find_last_user_message(enriched_messages)

            if last_user_idx is not None and self._provider:
                original = enriched_messages[last_user_idx].get("content", "")
                memories = await self._safe_recall(original)

                if memories:
                    memory_block = self._format_memories(memories)
                    enriched_messages[last_user_idx]["content"] = (
                        f"{memory_block}\\n\\n{original}"
                    )
                    logger.info(f"Enriched message with {len(memories)} memories")

        # Record for reflection (deep copy)
        self._conversation.extend(copy.deepcopy(enriched_messages))

        # Call LLM
        response = await self._call_llm(enriched_messages, **kwargs)

        # Record assistant response
        assistant_content = self._extract_content(response)
        if assistant_content:
            self._conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )

        return response

    def _find_last_user_message(self, messages: List[dict]) -> Optional[int]:
        """Find the index of the last user message."""
        for i in reversed(range(len(messages))):
            if messages[i].get("role") == "user":
                return i
        return None

    async def _safe_recall(self, query: str) -> List:
        """Safely recall memories with error handling."""
        try:
            return await self._provider.recall(
                query=query[:MAX_QUERY_LENGTH],
                user_id=self.config.user_id,
                limit=self.config.base.max_memories,
            )
        except Exception as e:
            logger.error(
                f"Failed to recall memories for user={self.config.user_id} "
                f"query='{query[:50]}...': {e}",
                exc_info=True,
            )
            if self.config.base.fail_open:
                return []
            raise

    async def _call_llm(self, messages: List[dict], **kwargs) -> Any:
        """Call the underlying LLM client."""
        client = self.llm_client

        # OpenAI-style
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return await client.chat.completions.create(messages=messages, **kwargs)

        # Anthropic-style
        if hasattr(client, "messages"):
            return await client.messages.create(messages=messages, **kwargs)

        # Fallback: try calling directly
        if callable(client):
            result = client(messages=messages, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        raise ValueError(f"Unsupported client type: {type(client)}")

    def _format_memories(self, memories: List[Any]) -> str:
        """Format memories for injection."""
        lines = ["<subconscious_context>"]
        lines.append("  <relevant_memories>")
        for mem in memories[: self.config.base.max_memories]:
            content = mem.content[:200]
            if len(mem.content) > 200:
                content += "..."
            lines.append(f"    - {content}")
        lines.append("  </relevant_memories>")
        lines.append("</subconscious_context>")
        return "\\n".join(lines)

    def _extract_content(self, response: Any) -> Optional[str]:
        """
        Extract content from various response formats.

        Args:
            response: LLM response object

        Returns:
            Extracted content or None
        """
        try:
            # OpenAI
            if hasattr(response, "choices"):
                return response.choices[0].message.content

            # Anthropic
            if hasattr(response, "content"):
                return response.content[0].text

            return str(response)
        except Exception as e:
            logger.debug(f"Could not extract content from response: {e}")
            return None

    async def reflect(self):
        """
        Manually trigger reflection on conversation history.

        Returns:
            Number of memories stored
        """
        if self._closed:
            logger.warning("Attempted to reflect on closed client")
            return 0

        if not self._conversation:
            logger.debug("No conversation to reflect on")
            return 0

        if not self._provider:
            logger.warning("No provider available for reflection")
            return 0

        # Build conversation text
        conv_text = "\\n".join(
            f"{m['role']}: {m['content']}" for m in self._conversation
        )

        # Extract learnings
        learnings = await self._extract_learnings(conv_text)

        # Store
        stored_count = 0
        for learning in learnings:
            try:
                await self._provider.store(
                    content=learning,
                    user_id=self.config.user_id,
                    metadata={"source": "reflection"},
                )
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store memory: {e}", exc_info=True)

        logger.info(
            f"Reflection complete: {stored_count}/{len(learnings)} memories stored"
        )
        return stored_count

    async def _extract_learnings(self, conversation: str) -> List[str]:
        """
        Use LLM to extract learnings from conversation.

        Args:
            conversation: Conversation text

        Returns:
            List of extracted learnings
        """
        if not self.config.base.api_key:
            logger.warning("No API key configured, skipping reflection")
            return []

        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.config.base.api_key,
                base_url=self.config.base.base_url,
            )

            prompt = f"""Extract key learnings from this conversation.

Focus on: user preferences, project context, important decisions.

Conversation:
{conversation[:MAX_CONVERSATION_LENGTH]}

Return a JSON array of strings. Example:
["User prefers TypeScript", "Project uses pnpm"]
"""

            response = await client.chat.completions.create(
                model=self.config.base.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
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
        Close the client, triggering reflection if configured.
        """
        if self._closed:
            logger.debug("Client already closed")
            return

        self._closed = True
        logger.debug(f"Closing client for user {self.config.user_id}")

        if self.config.base.reflect_on_close and self._conversation:
            await self.reflect()

        # Close provider if it has a close method
        if self._provider and hasattr(self._provider, "close"):
            await self._provider.close()

        self._conversation.clear()

    async def health_check(self) -> bool:
        """
        Check if the client and provider are healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self._closed:
            return False

        if not self._provider:
            return False

        return await self._provider.health_check()
