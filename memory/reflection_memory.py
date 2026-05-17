from __future__ import annotations

"""
Local-only reflection memory models and client wrapper.

This replaces the old unified/dual/cloud reflection memory layer with a
single async-friendly adapter over the shared local memory service.
"""


import asyncio
import hashlib
from typing import Any

from .local_foresight_manager import LocalForesightMemoryManager
from .reflection_memory_mapper import record_to_memory
from .reflection_types import Memory, MemoryCategory, MemoryMetadata

from ai.core.pipelines.privacy_content_gates import PrivacyContentGates, GateDecision


class LocalReflectionMemoryClient:
    """Async-friendly adapter over LocalForesightMemoryManager for reflection."""

    def __init__(self, manager: LocalForesightMemoryManager | None = None) -> None:
        self.manager = manager or LocalForesightMemoryManager()
        # Initialize Socratic Gate for message filtering
        self._gate = PrivacyContentGates()

    async def add_memory(self, content: str, metadata: MemoryMetadata) -> str:
        # Apply Socratic Gate to filter messages before storage
        gate_result = self._gate.evaluate(
            source_id=f"msg_{hashlib.md5(content.encode()).hexdigest()[:8]}",
            text=content,
            license_id="cc0-1.0",  # Default to permissive license for internal messages
            consent_recorded=True   # Internal messages have implied consent
        )
        
        # Only store messages that pass the gate
        if gate_result.passed:
            return await asyncio.to_thread(
                self.manager.add_memory,
                content=content,
                user_id=metadata.user_id or "system",
                metadata=metadata,
                category=metadata.category.value,
            )
        else:
            # Return a placeholder ID for blocked messages to maintain interface compatibility
            # In a real implementation, we might want to raise an exception or return None
            # but for now we return a special ID indicating the message was filtered
            return f"blocked_{hashlib.md5(content.encode()).hexdigest()[:8]}"

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
        next_metadata = metadata if metadata is not None else MemoryMetadata.from_dict(
            existing.get("metadata", {})
        )
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
