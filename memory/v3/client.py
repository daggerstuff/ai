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
from .client_runtime import SubconsciousClientRuntime
from .llm_client_factory import create_default_llm_client
from .llm_transport import call_llm

__all__ = ["SubconsciousClient"]

logger = logging.getLogger(__name__)

# Constants for magic numbers
MAX_QUERY_LENGTH = 500


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
        self._owns_llm_client = False
        self._runtime = SubconsciousClientRuntime(config)
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
        created_default_llm_client = llm_client is None
        if created_default_llm_client:
            llm_client = create_default_llm_client(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            logger.debug("Created default OpenAI client")

        client = cls(user_config, llm_client)
        client._owns_llm_client = created_default_llm_client
        await client._init_provider()
        return client

    async def _init_provider(self):
        """Initialize memory provider."""
        await self._runtime.init_provider()

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

        enriched_messages = await self._runtime.prepare_messages(
            messages,
            enrich=enrich,
            query_length_limit=MAX_QUERY_LENGTH,
        )
        response: Optional[Any] = None
        llm_error: Optional[Exception] = None
        try:
            response = await self._call_llm(enriched_messages, **kwargs)
        except Exception as exc:
            llm_error = exc
        self._runtime.record_messages(messages)
        if llm_error is not None:
            raise llm_error
        self._runtime.record_assistant_response(response)
        return response

    async def _call_llm(self, messages: List[dict], **kwargs) -> Any:
        """Call the underlying LLM client."""
        return await call_llm(self.llm_client, messages, **kwargs)

    async def reflect(self):
        """
        Manually trigger reflection on conversation history.

        Returns:
            Number of memories stored
        """
        if self._closed:
            logger.warning("Attempted to reflect on closed client")
            return 0

        if not self._runtime.has_conversation():
            logger.debug("No conversation to reflect on")
            return 0

        stored_count = await self._runtime.reflect(llm_client=self.llm_client)

        logger.info("Reflection complete: %s memories stored", stored_count)
        return stored_count

    async def close(self) -> None:
        """
        Close the client, triggering reflection if configured.
        """
        if self._closed:
            logger.debug("Client already closed")
            return

        logger.debug(f"Closing client for user {self.config.user_id}")
        if self.config.base.reflect_on_close and self._runtime.has_conversation():
            await self.reflect()
        await self._runtime.close()
        if (
            self._owns_llm_client
            and self.llm_client is not None
            and hasattr(self.llm_client, "aclose")
        ):
            await self.llm_client.aclose()
        self._closed = True

    async def health_check(self) -> bool:
        """
        Check if the client and provider are healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self._closed:
            return False

        return await self._runtime.health_check()
