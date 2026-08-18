from __future__ import annotations

from typing import Any

from .null_memory_repository import NullMemoryRepository


class NullMemoryCommandService:
    """Write and direct-record operations for the null memory backend."""

    def __init__(self, store: NullMemoryRepository) -> None:
        self.store = store

    def add(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        return self.store.add_record(
            content=content,
            user_id=user_id,
            metadata=metadata,
            memory_id=memory_id,
        )

    def get(self, *, memory_id: str, user_id: str) -> dict[str, Any] | None:
        return self.store.get_record(memory_id=memory_id, user_id=user_id)

    def update(
        self,
        *,
        memory_id: str,
        user_id: str,
        new_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return self.store.update_record(
            memory_id=memory_id,
            user_id=user_id,
            new_content=new_content,
            metadata=metadata,
        )

    def delete(self, *, memory_id: str, user_id: str) -> bool:
        return self.store.delete_record(memory_id=memory_id, user_id=user_id)

    def clear(self, *, user_id: str) -> bool:
        return self.store.clear_user(user_id=user_id)
