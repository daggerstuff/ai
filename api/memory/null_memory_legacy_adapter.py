from __future__ import annotations

from typing import Any, Protocol


class NullMemoryLegacySurface(Protocol):
    def add_memory(
        self, content: str, user_id: str, metadata: dict | None = None, category: str | None = None
    ) -> str: ...
    def search_memories(self, query: str, user_id: str, limit: int = 10) -> list[dict]: ...
    def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict]: ...
    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict | None: ...
    def update_memory(
        self, memory_id: str, new_content: str, metadata: dict | None = None, user_id: str | None = None
    ) -> bool: ...
    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool: ...
    def clear_memory(self, user_id: str) -> bool: ...


class NullMemoryLegacyAdapter:
    """Legacy dict-shaped API wrapper over the null memory manager."""

    def __init__(self, manager: NullMemoryLegacySurface) -> None:
        self.manager = manager

    def add(self, content: str, user_id: str, metadata: dict | None = None, **kwargs: Any) -> dict:
        record_id = self.manager.add_memory(
            content,
            user_id,
            metadata=metadata,
            category=kwargs.get("category"),
        )
        return {"results": [{"id": record_id}]}

    def search(self, query: str, user_id: str, **kwargs: Any) -> dict:
        return {
            "results": self.manager.search_memories(
                query=query,
                user_id=user_id,
                limit=kwargs.get("limit", 1000),
            )
        }

    def get_all(self, user_id: str, **kwargs: Any) -> dict:
        return {
            "results": self.manager.get_all_memories(
                user_id=user_id,
                limit=kwargs.get("limit", 1000),
            )
        }

    def get(self, memory_id: str, **kwargs: Any) -> dict | None:
        user_id = kwargs.get("user_id")
        if not user_id:
            return None
        return self.manager.get_memory(memory_id, user_id=user_id)

    def update(self, memory_id: str, new_content: str, **kwargs: Any) -> bool:
        user_id = kwargs.get("user_id", "")
        if not user_id:
            return False
        return self.manager.update_memory(
            memory_id,
            new_content,
            metadata=kwargs.get("metadata"),
            user_id=user_id,
        )

    def delete(self, memory_id: str, **kwargs: Any) -> bool:
        user_id = kwargs.get("user_id", "")
        if not user_id:
            return False
        return self.manager.delete_memory(memory_id, user_id=user_id)

    def delete_all(self, user_id: str, **_kwargs: Any) -> bool:
        return self.manager.clear_memory(user_id)
