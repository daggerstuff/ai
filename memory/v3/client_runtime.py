"""Runtime coordinator for subconscious v3 client sessions."""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional

from .config import UserConfig
from .conversation_manager import ConversationManager
from .llm_transport import extract_response_content
from .memory_enrichment import enrich_user_message
from .provider import (
    MemoryProvider,
    close_memory_provider,
    create_memory_provider,
    flush_memory_provider,
)
from .reflection import reflect_conversation

logger = logging.getLogger(__name__)


class SubconsciousClientRuntime:
    """Own provider lifecycle, enrichment, conversation history, and reflection."""

    def __init__(self, config: UserConfig) -> None:
        self.config = config
        self.provider: Optional[MemoryProvider] = None
        self.conversation_manager = ConversationManager()

    async def init_provider(self) -> None:
        """Initialize the configured memory provider."""
        self.provider = create_memory_provider(self.config.base)
        logger.debug("Initialized %s provider", self.config.base.memory_provider)

    async def prepare_messages(
        self,
        messages: List[dict],
        *,
        enrich: bool,
        query_length_limit: int,
    ) -> List[dict]:
        """Copy and optionally enrich the outbound message list."""
        enriched_messages = copy.deepcopy(messages)
        if not enrich or self.provider is None:
            return enriched_messages

        last_user_idx = self.find_last_user_message(enriched_messages)
        if last_user_idx is None:
            return enriched_messages

        original = enriched_messages[last_user_idx].get("content", "")
        memories = await self.safe_recall(
            query=original[:query_length_limit],
            user_id=self.config.user_id,
            limit=self.config.base.max_memories,
        )
        if not memories:
            return enriched_messages

        enriched_messages[last_user_idx]["content"] = enrich_user_message(
            conversation_manager=self.conversation_manager,
            message=original,
            memories=memories,
            max_memories=self.config.base.max_memories,
        )
        logger.info("Enriched message with %s memories", len(memories))
        return enriched_messages

    async def safe_recall(self, *, query: str, user_id: str, limit: int) -> List[Any]:
        """Recall memories with consistent error handling."""
        if self.provider is None:
            return []
        try:
            return await self.provider.recall(
                query=query,
                user_id=user_id,
                limit=limit,
            )
        except Exception as exc:
            logger.error(
                "Failed to recall memories for user=%s query='%s...': %s",
                user_id,
                query[:50],
                exc,
                exc_info=True,
            )
            if self.config.base.fail_open:
                return []
            raise

    def record_messages(self, messages: List[dict]) -> None:
        """Append the unseen suffix of outbound messages to conversation history."""
        self.conversation_manager.record_messages(messages)

    def record_assistant_response(self, response: Any) -> None:
        """Append the assistant response content to the conversation history."""
        assistant_content = extract_response_content(response)
        if assistant_content:
            self.conversation_manager.record_message("assistant", assistant_content)

    def has_conversation(self) -> bool:
        """Return whether the runtime has recorded any conversation state."""
        return bool(self.conversation_manager.conversation)

    async def reflect(self, *, llm_client: Any) -> int:
        """Reflect the recorded conversation into durable learnings."""
        if not self.has_conversation() or self.provider is None:
            return 0
        return await reflect_conversation(
            provider=self.provider,
            user_id=self.config.user_id,
            conversation=self.conversation_manager.conversation,
            api_key=self.config.base.api_key,
            base_url=self.config.base.base_url,
            model=self.config.base.model,
            focus_prompt=(
                "Extract key learnings from this conversation.\n\n"
                "Focus on: user preferences, project context, important decisions."
            ),
            llm_client=llm_client,
        )

    async def close(self) -> None:
        """Flush and close provider resources."""
        await flush_memory_provider(self.provider)
        await close_memory_provider(self.provider)
        self.conversation_manager.clear()
        self.provider = None

    async def health_check(self) -> bool:
        """Check whether the provider is healthy."""
        if self.provider is None:
            return False
        return await self.provider.health_check()

    @staticmethod
    def find_last_user_message(messages: List[dict]) -> Optional[int]:
        """Find the index of the last user message."""
        for index in reversed(range(len(messages))):
            if messages[index].get("role") == "user":
                return index
        return None
