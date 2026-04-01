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
import logging
from typing import Any, List, Optional

from .config import SubconsciousConfig, UserConfig
from .provider import MemoryProvider, LocalHindsightProvider

logger = logging.getLogger(__name__)


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
        """
        user_config = config.with_user(user_id)

        # Create default client if none provided
        if llm_client is None:
            import openai
            llm_client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )

        client = cls(user_config, llm_client)
        await client._init_provider()
        return client

    async def _init_provider(self):
        """Initialize memory provider."""
        if self.config.base.memory_provider == "mock":
            from .provider import MockProvider
            self._provider = MockProvider()
        else:
            self._provider = LocalHindsightProvider(self.config.base.bank_id)

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
        if not self.config.base.enabled:
            return await self._call_llm(messages, **kwargs)

        # Work on a copy
        enriched_messages = [m.copy() for m in messages]

        # Enrich last user message
        if enrich:
            last_user_idx = None
            for i in reversed(range(len(enriched_messages))):
                if enriched_messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None and self._provider:
                original = enriched_messages[last_user_idx].get("content", "")
                memories = await self._provider.recall(
                    query=original[:500],
                    user_id=self.config.user_id,
                    limit=self.config.base.max_memories,
                )

                if memories:
                    memory_block = self._format_memories(memories)
                    enriched_messages[last_user_idx]["content"] = f"{memory_block}\\n\\n{original}"

        # Record for reflection
        self._conversation.extend(enriched_messages)

        # Call LLM
        response = await self._call_llm(enriched_messages, **kwargs)

        # Record assistant response
        assistant_content = self._extract_content(response)
        if assistant_content:
            self._conversation.append({
                "role": "assistant",
                "content": assistant_content,
            })

        return response

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
            import asyncio
            result = client(messages=messages, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        raise ValueError(f"Unsupported client type: {type(client)}")

    def _format_memories(self, memories) -> str:
        """Format memories for injection."""
        lines = ["<subconscious_context>"]
        lines.append("  <relevant_memories>")
        for mem in memories[:self.config.base.max_memories]:
            content = mem.content[:200]
            if len(mem.content) > 200:
                content += "..."
            lines.append(f"    - {content}")
        lines.append("  </relevant_memories>")
        lines.append("</subconscious_context>")
        return "\\n".join(lines)

    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract content from various response formats."""
        try:
            # OpenAI
            if hasattr(response, "choices"):
                return response.choices[0].message.content

            # Anthropic
            if hasattr(response, "content"):
                return response.content[0].text

            return str(response)
        except Exception:
            return None

    async def reflect(self):
        """Manually trigger reflection on conversation history."""
        if not self._conversation or not self._provider:
            return

        # Build conversation text
        conv_text = "\\n".join(
            f"{m['role']}: {m['content']}"
            for m in self._conversation
        )

        # Extract learnings
        learnings = await self._extract_learnings(conv_text)

        # Store
        for learning in learnings:
            try:
                await self._provider.store(
                    content=learning,
                    user_id=self.config.user_id,
                    metadata={"source": "reflection"},
                )
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")

        logger.info(f"Reflection complete: {len(learnings)} memories stored")

    async def _extract_learnings(self, conversation: str) -> List[str]:
        """Use LLM to extract learnings."""
        if not self.config.base.api_key:
            return []

        try:
            import openai
            import json

            client = openai.AsyncOpenAI(
                api_key=self.config.base.api_key,
                base_url=self.config.base.base_url,
            )

            prompt = f"""Extract key learnings from this conversation.

Focus on: user preferences, project context, important decisions.

Conversation:
{conversation[:3000]}

Return a JSON array of strings. Example:
["User prefers TypeScript", "Project uses pnpm"]
"""

            response = await client.chat.completions.create(
                model=self.config.base.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )

            content = response.choices[0].message.content or "[]"

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if "[" in content and "]" in content:
                    start = content.index("[")
                    end = content.rindex("]") + 1
                    return json.loads(content[start:end])
                return []

        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            return []

    async def close(self):
        """
        Close the client, triggering reflection if configured.
        """
        if self.config.base.reflect_on_close and self._conversation:
            await self.reflect()
        self._conversation.clear()
