"""
In-memory fallback memory backend for local tests and degraded service paths.
"""

from __future__ import annotations

from typing import Any

from ai.inference.api.memory.base import BaseMemoryManager

from .null_memory_health_service import NullMemoryHealthService
from .null_memory_legacy_adapter import NullMemoryLegacyAdapter
from .null_memory_manager_factory import build_null_memory_manager_runtime


class NullMemoryManager(BaseMemoryManager):
    """High-level memory manager facade backed by an in-memory store."""

    def __init__(self, runtime=None, *_args, **_kwargs) -> None:
        runtime = runtime or build_null_memory_manager_runtime()
        self.coordination = runtime.coordination
        self.store = runtime.store
        self.queries = runtime.queries
        self.protocol = runtime.protocol
        self.health = NullMemoryHealthService(self.store)

        legacy = NullMemoryLegacyAdapter(self)
        self.search_memories_scoped = self.queries.search_memories_scoped
        self.get_all_memories_scoped = self.queries.get_all_memories_scoped
        self.count_memories_by_category_scoped = self.queries.count_memories_by_category_scoped
        self.add = legacy.add
        self.search = legacy.search
        self.get_all = legacy.get_all
        self.get = legacy.get
        self.update = legacy.update
        self.delete = legacy.delete
        self.delete_all = legacy.delete_all

    def _query(self, method: str, *args: Any, **kwargs: Any):
        return getattr(self.queries, method)(*args, **kwargs)

    def _protocol(self, method: str, *args: Any, **kwargs: Any):
        return getattr(self.protocol, method)(*args, **kwargs)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> str:
        return self._protocol(
            "add_memory",
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._query("search_memories", query=query, user_id=user_id, limit=limit)

    def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._query("get_all_memories", user_id=user_id, limit=limit)

    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id is None:
            return None
        memory = self.store.get_record(memory_id=memory_id, user_id=user_id)
        if memory is None:
            return None
        owner_user_id = memory.get("user_id")
        if owner_user_id != user_id:
            return None
        return memory

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        if user_id is not None:
            memory = self.get_memory(memory_id, user_id=user_id)
            if memory is None:
                return False
        if not user_id:
            return False
        return self.store.update_record(
            memory_id=memory_id,
            user_id=user_id,
            new_content=new_content,
            metadata=metadata,
        )

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        if user_id is not None:
            memory = self.get_memory(memory_id, user_id=user_id)
            if memory is None:
                return False
        if not user_id:
            return False
        return self.store.delete_record(memory_id=memory_id, user_id=user_id)

    def clear_memory(self, user_id: str) -> bool:
        return self.store.clear_user(user_id=user_id)

    def get_health_status(self) -> dict[str, Any]:
        return self.health.status()

    @property
    def project(self):
        class NullProject:
            def update(self, **_kwargs: Any) -> None:
                return None

        return NullProject()
