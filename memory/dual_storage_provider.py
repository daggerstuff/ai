"""
Legacy dual-storage provider combining Foresight and Letta backends.
"""

from __future__ import annotations

from .reflection_types import Memory, MemoryCategory, MemoryMetadata
from .unified_memory import MemoryProvider


class DualStorageProvider:
    """Route memory writes between two providers with crisis-safe policy."""

    def __init__(self, foresight: MemoryProvider, letta: MemoryProvider):
        self.foresight = foresight
        self.letta = letta

    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str:
        if metadata.category in {
            MemoryCategory.CRISIS_CONTEXT,
            MemoryCategory.EMOTIONAL_STATE,
            MemoryCategory.THERAPEUTIC_INSIGHT,
        }:
            # Never write crisis-signal content to Letta; keep in Foresight only.
            return await self.foresight.add_memory(content, metadata)
        memory_id = await self.foresight.add_memory(content, metadata)
        await self.letta.add_memory(content, metadata)
        return memory_id

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self.foresight.get_memory(memory_id)

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> None:
        await self.foresight.update_memory(memory_id, content=content, metadata=metadata)

    async def delete_memory(self, memory_id: str) -> None:
        await self.foresight.delete_memory(memory_id)

    async def search_memories(self, query: str, user_id: str, limit: int = 10) -> list[Memory]:
        return await self.foresight.search_memories(query, user_id, limit=limit)

    async def get_memories_by_user(self, user_id: str, limit: int = 100) -> list[Memory]:
        return await self.foresight.get_memories_by_user(user_id, limit=limit)

    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        return await self.foresight.get_memories_by_category(category, user_id=user_id, limit=limit)
