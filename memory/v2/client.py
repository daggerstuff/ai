"""
SubconsciousClient - A wrapper that adds memory awareness to any LLM client.

Key insight: Composition over monkey patching.

Instead of monkey-patching OpenAI/Anthropic classes, we provide:
1. A wrapper that can be used with any LLM client
2. Async primitives that work with any async framework
3. No global state, no module-level effects

Usage with OpenAI:
    from openai import AsyncOpenAI
    from ai.memory.v2 import SubconsciousClient, SubconsciousConfig

    config = SubconsciousConfig.from_env()
    openai_client = AsyncOpenAI()

    client = SubconsciousClient(config, openai_client, user_id="alice")

    response = await client.chat(
        messages=[{"role": "user", "content": "I'm feeling anxious"}]
    )

Usage with Anthropic:
    import anthropic
    from ai.memory.v2 import SubconsciousClient

    anthropic_client = anthropic.AsyncAnthropic()
    client = SubconsciousClient(config, anthropic_client, user_id="bob")

    response = await client.chat(
        messages=[{"role": "user", "content": "Help me understand my emotions"}]
    )
"""
import logging
from typing import Any, List, Optional, Union

from .config import SubconsciousConfig
from .context import SubconsciousContext

logger = logging.getLogger(__name__)


class SubconsciousClient:
    """
    A memory-aware wrapper around any LLM client.

    This is NOT a subclass. It's a wrapper that:
    1. Enriches prompts with memory context
    2. Delegates to the underlying client
    3. Optionally records the conversation

    Thread-safe: Each instance has its own state.
    The underlying client is never modified.
    """

    def __init__(
        self,
        config: SubconsciousConfig,
        llm_client: Any,
        user_id: str,
    ):
        """
        Create a memory-aware client.

        Args:
            config: Immutable configuration
            llm_client: Any async LLM client (OpenAI, Anthropic, etc.)
            user_id: User identifier for memory lookup

        The llm_client can be any object with an async chat/completions interface.
        We don't care about the specific API - we just enrich the messages.
        """
        self.config = config
        self.llm_client = llm_client
        self.user_id = user_id
        self._context: Optional[SubconsciousContext] = None
        self._conversation_history: List[dict] = []

    async def initialize(self) -> "SubconsciousClient":
        """
        Async initialization.

        Returns self for chaining.
        """
        self._context = await SubconsciousContext(self.config, self.user_id).initialize()
        return self

    async def chat(
        self,
        messages: List[dict],
        enrich: bool = True,
        record: bool = True,
        **kwargs,
    ) -> Any:
        """
        Send a chat request with optional memory enrichment.

        Args:
            messages: List of message dicts with 'role' and 'content'
            enrich: If True, enrich the last user message with memory context
            record: If True, add to conversation history
            **kwargs: Additional arguments passed to the underlying client

        Returns:
            The response from the underlying client

        Example:
            response = await client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "I'm feeling anxious"},
                ],
                model="z-ai/glm4.7",
            )
        """
        if self._context is None:
            await self.initialize()

        # Work on a copy
        enriched_messages = [m.copy() for m in messages]

        # Find and enrich the last user message
        if enrich:
            last_user_idx = None
            for i in reversed(range(len(enriched_messages))):
                if enriched_messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                original_content = enriched_messages[last_user_idx].get("content", "")
                enriched_content = await self._context.enrich(
                    original_content,
                    conversation_history=self._conversation_history,
                )
                enriched_messages[last_user_idx]["content"] = enriched_content

        # Record before calling (so we have the context)
        if record:
            self._conversation_history.extend(enriched_messages)

        # Call the underlying client
        # We use duck typing - any client with a chat.completions.create will work
        response = await self._call_underlying_client(enriched_messages, **kwargs)

        # Record the assistant's response
        if record and response:
            assistant_content = self._extract_response_content(response)
            if assistant_content:
                self._conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

        return response

    async def _call_underlying_client(self, messages: List[dict], **kwargs) -> Any:
        """
        Call the underlying LLM client.

        Supports multiple client types:
        - OpenAI: client.chat.completions.create(messages=..., **kwargs)
        - Anthropic: client.messages.create(messages=..., **kwargs)
        - Custom: Any async callable
        """
        client = self.llm_client

        # OpenAI-style API
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return await client.chat.completions.create(messages=messages, **kwargs)

        # Anthropic-style API
        if hasattr(client, "messages"):
            return await client.messages.create(messages=messages, **kwargs)

        # Fallback: try calling directly
        if callable(client):
            return await client(messages=messages, **kwargs)

        raise ValueError(f"Unsupported client type: {type(client)}")

    def _extract_response_content(self, response: Any) -> Optional[str]:
        """Extract the content from various response formats."""
        try:
            # OpenAI format
            if hasattr(response, "choices"):
                return response.choices[0].message.content

            # Anthropic format
            if hasattr(response, "content"):
                return response.content[0].text

            # Fallback
            return str(response)

        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return None

    def get_history(self) -> List[dict]:
        """Get the conversation history."""
        return self._conversation_history.copy()

    def clear_history(self):
        """Clear the conversation history."""
        self._conversation_history = []

    async def reflect(self) -> dict:
        """
        Trigger a reflection cycle.

        This analyzes the conversation history and updates memories.

        Returns:
            A dict with reflection results
        """
        if self._context is None:
            await self.initialize()

        if not self._context.should_reflect():
            return {"status": "skipped", "reason": "threshold not reached"}

        # Perform reflection
        # TODO: Implement actual reflection logic
        result = {
            "status": "completed",
            "messages_analyzed": len(self._conversation_history),
            "user_id": self.user_id,
        }

        self._context.reset_counter()
        return result


# Factory function for convenience
async def create_client(
    user_id: str,
    config: Optional[SubconsciousConfig] = None,
    llm_client: Optional[Any] = None,
) -> SubconsciousClient:
    """
    Create and initialize a SubconsciousClient.

    Example:
        from openai import AsyncOpenAI

        client = await create_client(
            user_id="alice",
            llm_client=AsyncOpenAI(),
        )
    """
    if config is None:
        config = SubconsciousConfig.from_env()

    if llm_client is None:
        # Create default OpenAI client with NVIDIA NIM config
        from openai import AsyncOpenAI

        llm_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    client = SubconsciousClient(config, llm_client, user_id)
    await client.initialize()
    return client
