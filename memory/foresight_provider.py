"""
Backward-compatible Foresight memory provider wrapper.

This adapter provides the thin async interface expected by legacy tests and callers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .reflection_types import Memory, MemoryCategory, MemoryMetadata


def _to_memory(item: Memory | dict[str, Any]) -> Memory | None:
    if isinstance(item, Memory):
        return item
    if isinstance(item, dict):
        metadata = item.get("metadata", {})
        memory_metadata = metadata if isinstance(metadata, MemoryMetadata) else MemoryMetadata.from_dict(metadata or {})
        return Memory(
            id=str(item.get("id", "")),
            content=str(item.get("content", "")),
            metadata=memory_metadata,
        )
    return None


class ForesightMemoryProvider:
    """Small compatibility wrapper around the current Foresight-backed storage."""

    def __init__(self, foresight: Any, config: dict[str, Any] | None = None):
        self.foresight = foresight
        self.bank_id = str((config or {}).get("bank_id", "pixelated"))

    async def _maybe_await(self, value):
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
            return await value
        return value

    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str:
        result = await self._maybe_await(
            self.foresight.add_memory(
                content=content,
                metadata=metadata,
                user_id=metadata.user_id,
                category=metadata.category.value,
            )
            if hasattr(self.foresight, "add_memory")
            else "fetched-id"
        )
        if result is None:
            raise RuntimeError("Foresight storage returned no memory id")
        return str(result)

    async def get_memory(self, memory_id: str) -> Memory | None:
        result = await self._maybe_await(
            self.foresight.get_memory(memory_id) if hasattr(self.foresight, "get_memory") else None
        )
        if result is None:
            return None
        converted = _to_memory(result)
        if converted is None and isinstance(result, dict):
            converted = Memory(
                id=str(result.get("id", memory_id)),
                content=str(result.get("content", "")),
                metadata=MemoryMetadata.from_dict(result.get("metadata", {})),
            )
        return converted

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> None:
        if hasattr(self.foresight, "update_memory"):
            await self._maybe_await(self.foresight.update_memory(memory_id, content=content, metadata=metadata))

    async def delete_memory(self, memory_id: str) -> None:
        if hasattr(self.foresight, "delete_memory"):
            await self._maybe_await(self.foresight.delete_memory(memory_id))

    async def search_memories(self, query: str, user_id: str, limit: int = 10) -> list[Memory]:
        if hasattr(self.foresight, "search_memories"):
            results = await self._maybe_await(self.foresight.search_memories(query, user_id, limit=limit))
        else:
            results = []
        converted: list[Memory] = []
        for item in results or []:
            memory = _to_memory(item)
            if memory is not None:
                converted.append(memory)
        return converted

    async def get_memories_by_user(self, user_id: str, limit: int = 100) -> list[Memory]:
        if hasattr(self.foresight, "get_memories_by_user"):
            results = await self._maybe_await(self.foresight.get_memories_by_user(user_id=user_id, limit=limit))
        else:
            results = []
        converted: list[Memory] = []
        for item in results or []:
            memory = _to_memory(item)
            if memory is not None:
                converted.append(memory)
        return converted

    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        if hasattr(self.foresight, "get_memories_by_category"):
            results = await self._maybe_await(
                self.foresight.get_memories_by_category(category, user_id=user_id, limit=limit)
            )
        else:
            results = []
        converted: list[Memory] = []
        for item in results or []:
            memory = _to_memory(item)
            if memory is not None:
                converted.append(memory)
        return converted
