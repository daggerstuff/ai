from __future__ import annotations

"""
NVIDIA NIM integration backed by the shared local memory service.

This replaces the old cloud/local split for Foresight memory. NVIDIA-generated
responses still use the configured model endpoint, but durable memory is stored
only in the repository's local shared memory backend.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from ai.research.local_foresight_manager import LocalForesightMemoryManager
from ai.research.therapeutic_processor import TherapeuticProcessor

from .interaction_service import NvidiaTherapeuticInteractionService
from .memory_ingestion_config import TherapeuticMemoryConfig
from .rate_limiter import NvidiaRateLimiter

logger = logging.getLogger("foresight_nvidia")


class NvidiaForesightConfig(BaseModel):
    """Configuration for NVIDIA NIM with shared local memory."""

    nvidia_api_key: str = Field(..., description="NVIDIA API key")
    model_name: str = Field("meta/llama-3.1-405b-instruct", description="NVIDIA NIM model to use")
    base_url: str = Field("https://integrate.api.nvidia.com/v1", description="NVIDIA NIM Base URL")
    user_id: str = Field("default_user", description="Default user ID for memory")
    db_path: str = Field(..., description="Path to the shared local memory database")
    bank_id: str = Field("pixelated", description="Shared memory bank identifier")
    foresight_api_key: str | None = Field(
        default=None,
        description="Deprecated; local shared memory is the only supported backend",
    )
    memory_config: dict[str, Any] | None = Field(
        default=None,
        description="Deprecated; local shared memory is the only supported backend",
    )
    therapeutic_config: TherapeuticMemoryConfig | None = Field(
        default=None, description="Therapeutic memory ingestion configuration"
    )


class NvidiaForesightManager:
    """
    NVIDIA NIM manager with local-only shared memory.

    Response generation remains NVIDIA-backed. Memory retention, recall, and
    update operations all route through LocalForesightMemoryManager.
    """

    def __init__(self, config: NvidiaForesightConfig):
        self.config = config
        self.therapeutic_config = config.therapeutic_config or TherapeuticMemoryConfig()

        self.processor = TherapeuticProcessor(self.therapeutic_config)
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.nvidia_api_key)
        self.async_client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.nvidia_api_key,
        )
        self.memory = LocalForesightMemoryManager(
            db_path=self.config.db_path,
            bank_id=self.config.bank_id,
        )
        self.interactions = NvidiaTherapeuticInteractionService(
            memory=self.memory,
            processor=self.processor,
        )
        self.rate_limiter = NvidiaRateLimiter()
        logger.info(
            "Initialized NvidiaForesightManager with model %s using local shared memory",
            self.config.model_name,
        )

    def _memory_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(metadata or {})
        merged.setdefault("provider", "nvidia")
        merged.setdefault("model_name", self.config.model_name)
        return merged

    def _filter_for_storage(self, content: str) -> str | None:
        return self.interactions.filter_for_storage(content)

    async def generate_content(self, prompt: str, system_instruction: str | None = None) -> str:
        await self.rate_limiter.wait("generation")
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        response = await self.async_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        return content or ""

    def _get_base_instructions(self) -> str:
        return """You are Antigravity, a therapeutic companion AI.
Your goal is to provide empathetic, validating, and safe support.
Maintain professional boundaries and safety protocols at all times."""

    async def get_response(self, user_id: str, message: str, session_id: str | None = None) -> str:
        memories = self.search_memories(message, user_id)
        facts = [m.get("memory", "") or m.get("content", "") for m in memories]
        system_prompt = self.processor.build_system_prompt(self._get_base_instructions(), facts)
        response_text = await self.generate_content(message, system_prompt)
        crisis_severity = self.processor.detect_crisis(message)
        self.interactions.store_interaction(
            user_id=user_id,
            query=message,
            response=response_text,
            session_id=session_id,
            provider_metadata=self._memory_metadata(),
            crisis_severity=crisis_severity,
        )
        return response_text

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        filtered_content = self._filter_for_storage(new_content)
        if not filtered_content:
            logger.warning("Update rejected: content failed filtering")
            return False
        return self.memory.update_memory(
            memory_id=memory_id,
            new_content=filtered_content,
            metadata=self._memory_metadata(metadata),
            user_id=user_id,
        )

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        return self.memory.delete_memory(memory_id, user_id=user_id)

    def clear_memory(self, user_id: str) -> None:
        self.memory.clear_memory(user_id)

    def list_entities(self, limit: int = 20, page: int = 1) -> list[dict[str, Any]]:
        entities = [{"id": self.config.user_id, "type": "user"}]
        start = max(page - 1, 0) * limit
        end = start + limit
        return entities[start:end]

    def get_all_memories(self, user_id: str, limit: int = 100, page: int = 1) -> list[dict[str, Any]]:
        offset = max(page - 1, 0) * limit
        if hasattr(self.memory, "get_all_memories_scoped"):
            return self.memory.get_all_memories_scoped(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )
        memories = self.memory.get_all_memories(user_id=user_id, limit=limit * max(page, 1))
        return self._paginate(memories, limit, page)

    def search_memories(self, query: str, user_id: str, limit: int = 10, page: int = 1) -> list[dict[str, Any]]:
        offset = max(page - 1, 0) * limit
        if hasattr(self.memory, "search_memories_scoped"):
            return self.memory.search_memories_scoped(
                query=query,
                user_id=user_id,
                limit=limit,
                offset=offset,
            )
        memories = self.memory.search_memories(query=query, user_id=user_id, limit=limit * max(page, 1))
        return self._paginate(memories, limit, page)

    def _paginate(self, items: list[Any], limit: int, page: int) -> list[Any]:
        if not items:
            return []
        start = max(page - 1, 0) * limit
        end = start + limit
        return items[start:end]

    def get_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.memory.get_memory(memory_id, user_id=user_id)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> str | None:
        filtered_content = self._filter_for_storage(content)
        if not filtered_content:
            logger.warning("Memory addition rejected: content failed filtering")
            return None
        full_metadata = self._memory_metadata(metadata)
        if "timestamp" not in full_metadata:
            full_metadata["timestamp"] = datetime.now(UTC).isoformat()
        return self.memory.add_memory(
            filtered_content,
            user_id=user_id,
            metadata=full_metadata,
            category=category,
        )
