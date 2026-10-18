from __future__ import annotations

"""
Local-only reflection memory models and client wrapper.

This replaces the old unified/dual/cloud reflection memory layer with a
single async-friendly adapter over the shared local memory service.

Gate evaluation is handled by LocalForesightMemoryManager.add_memory()
which routes through gated_add_memory() — no duplicate gating needed here.
"""


import asyncio
import os
from typing import Any

from .local_foresight_manager import LocalForesightMemoryManager
from .reflection_memory_mapper import record_to_memory
from .reflection_types import Memory, MemoryCategory, MemoryMetadata


class LocalReflectionMemoryClient:
    """Async-friendly adapter over LocalForesightMemoryManager for reflection."""

    def __init__(self, manager: LocalForesightMemoryManager | None = None) -> None:
        self.manager = manager or LocalForesightMemoryManager(
            db_path=os.environ.get("FORESIGHT_DB_PATH", "foresight.db")
        )

    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str | None:
        # Gating is handled by the manager's gated_add_memory path.
        # Returns None if content is blocked by any gate.
        return await asyncio.to_thread(
            self.manager.add_memory,
            content=content,
            user_id=metadata.user_id or "system",
            metadata=metadata,
            category=metadata.category.value,
        )

    async def get_memory(self, memory_id: str, user_id: str | None = None) -> Memory:
        record = await asyncio.to_thread(self.manager.get_memory, memory_id, user_id)
        if record is None:
            raise ValueError(f"Memory not found: {memory_id}")
        return record_to_memory(record)

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> None:
        user_id = metadata.user_id if metadata is not None else None
        existing = await asyncio.to_thread(self.manager.get_memory, memory_id, user_id)
        if existing is None:
            raise ValueError(f"Memory not found: {memory_id}")
        next_content = content if content is not None else (existing.get("content") or "")
        next_metadata = metadata if metadata is not None else MemoryMetadata.from_dict(existing.get("metadata", {}))
        updated = await asyncio.to_thread(
            self.manager.update_memory,
            memory_id=memory_id,
            new_content=next_content,
            metadata=next_metadata,
            user_id=next_metadata.user_id,
        )
        if not updated:
            raise ValueError(f"Failed to update memory: {memory_id}")

    async def delete_memory(self, memory_id: str) -> None:
        deleted = await asyncio.to_thread(self.manager.delete_memory, memory_id)
        if not deleted:
            raise ValueError(f"Memory not found: {memory_id}")

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[Memory]:
        records = await asyncio.to_thread(
            self.manager.search_memories,
            query,
            user_id,
            limit,
        )
        return [record_to_memory(record) for record in records]

    async def get_memories_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        records = await asyncio.to_thread(self.manager.get_all_memories, user_id, limit)
        return [record_to_memory(record) for record in records]

    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        if user_id is None:
            raise ValueError("user_id is required for category-scoped reflection memory access")
        records = await asyncio.to_thread(
            self.manager.get_memories_by_category,
            user_id,
            category.value,
            limit,
        )
        return [record_to_memory(record) for record in records]

    async def execute_consolidation(
        self,
        result: Any,
        *,
        user_id: str,
        allow_crisis_deletions: bool = False,
    ) -> dict[str, int]:
        """
        Execute memory cleanup actions from a reflection result.

        This is a low-level execution primitive. The caller is responsible for
        deciding which deletions are safe before invoking it.
        """
        delete_ids = list(getattr(result, "memories_deleted", []))
        if getattr(result, "crisis_detected", False) and not allow_crisis_deletions:
            delete_ids = []
        stats = {
            "preserved": len(getattr(result, "memories_preserved", [])),
            "consolidated": len(getattr(result, "memories_consolidated", [])),
            "deleted": 0,
            "errors": 0,
        }
        if delete_ids:
            deleted = await asyncio.to_thread(
                self.manager.delete_memories,
                delete_ids,
                user_id,
            )
            stats["deleted"] = deleted
            stats["errors"] = max(len(delete_ids) - deleted, 0)
        return stats

    async def close(self) -> None:
        await asyncio.to_thread(self.manager.close)
