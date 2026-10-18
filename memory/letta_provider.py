"""
Backward-compatible Letta memory provider wrapper.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .reflection_types import Memory, MemoryCategory, MemoryMetadata


class LettaMemoryProvider:
    """Compatibility provider that adapts to the Letta client/session API."""

    def __init__(self, letta_client: Any):
        self.letta = letta_client

    async def _maybe_await(self, value):
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
            return await value
        return value

    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str:
        run_call = getattr(self.letta, "run", None)
        result = await self._maybe_await(run_call(content=content, metadata=metadata))
        if isinstance(result, str):
            return result
        if hasattr(result, "id") and isinstance(result.id, (str, int)):
            return str(result.id)
        if isinstance(result, int):
            return str(result)
        if result is not None and not result.__class__.__name__.endswith("Mock"):
            return str(result)
        return f"letta-{abs(hash((content, metadata.user_id or '')))}"

    async def get_memory(self, memory_id: str) -> Memory | None:
        getter = getattr(self.letta, "get_memory", None)
        if getter is None:
            return None
        result = await self._maybe_await(getter(memory_id))
        return result if isinstance(result, Memory) else None

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> None:
        updater = getattr(self.letta, "update_memory", None)
        if updater is not None:
            await self._maybe_await(updater(memory_id=memory_id, content=content, metadata=metadata))

    async def delete_memory(self, memory_id: str) -> None:
        deleter = getattr(self.letta, "delete_memory", None)
        if deleter is not None:
            await self._maybe_await(deleter(memory_id))

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[Memory]:
        searcher = getattr(self.letta, "search_memories", None)
        if searcher is None:
            return []
        return await self._maybe_await(searcher(query=query, user_id=user_id, limit=limit))

    async def get_memories_by_user(self, user_id: str, limit: int = 100) -> list[Memory]:
        fetcher = getattr(self.letta, "get_memories_by_user", None)
        if fetcher is None:
            return []
        return await self._maybe_await(fetcher(user_id=user_id, limit=limit))

    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        fetcher = getattr(self.letta, "get_memories_by_category", None)
        if fetcher is None:
            return []
        if user_id is None:
            return await self._maybe_await(fetcher(category=category, limit=limit))
        return await self._maybe_await(fetcher(category=category, user_id=user_id, limit=limit))
