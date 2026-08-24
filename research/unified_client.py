"""
Compatibility unified-memory client used by legacy tests.
"""

from __future__ import annotations

from typing import Any

from .dual_storage_provider import DualStorageProvider
from .foresight_provider import ForesightMemoryProvider
from .letta_provider import LettaMemoryProvider
from .reflection_types import Memory, MemoryCategory, MemoryMetadata
from .unified_memory import MemoryProvider


class UnifiedMemoryClient:
    """Legacy compatibility façade for memory operations."""

    def __init__(self, mode: str = "dual", config: dict[str, Any] | None = None):
        self.mode = mode
        self.config = config or {}
        self._initialized = False
        self._foresight_provider: MemoryProvider | None = None
        self._letta_provider: MemoryProvider | None = None

        if mode == "foresight":
            self.provider = ForesightMemoryProvider(
                foresight=self.config.get("foresight") or object(),
                config={"bank_id": self.config.get("bank_id", "pixelated")},
            )
        elif mode == "letta":
            self.provider = LettaMemoryProvider(self.config.get("letta") or object())
        else:
            self.provider = DualStorageProvider(
                ForesightMemoryProvider(
                    foresight=self.config.get("foresight") or object(),
                    config={"bank_id": self.config.get("bank_id", "pixelated")},
                ),
                LettaMemoryProvider(self.config.get("letta") or object()),
            )

    async def retain(
        self,
        content: str,
        user_id: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        metadata: MemoryMetadata | None = None,
    ) -> str:
        memory_metadata = metadata or MemoryMetadata(
            user_id=user_id,
            category=category,
            tags=[],
        )
        memory_metadata.user_id = user_id
        memory_metadata.category = category
        return await self.provider.add_memory(content, memory_metadata)

    async def recall(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[Memory]:
        return await self.provider.search_memories(query, user_id=user_id, limit=limit)

    async def delete(self, memory_id: str) -> None:
        await self.provider.delete_memory(memory_id)


def create_client(mode: str = "dual", config: dict[str, Any] | None = None) -> UnifiedMemoryClient:
    return UnifiedMemoryClient(mode=mode, config=config)
