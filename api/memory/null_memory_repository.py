from __future__ import annotations

from typing import Protocol


class NullMemoryRepository(Protocol):
    def add_record(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> dict: ...

    def search_records(
        self,
        *,
        query: str,
        user_id: str,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> list[dict]: ...

    def list_records(self, *, user_id: str) -> list[dict]: ...

    def get_record(self, *, memory_id: str, user_id: str) -> dict | None: ...

    def update_record(
        self,
        *,
        memory_id: str,
        user_id: str,
        new_content: str,
        metadata: dict | None = None,
    ) -> bool: ...

    def delete_record(self, *, memory_id: str, user_id: str) -> bool: ...

    def clear_user(self, *, user_id: str) -> bool: ...

    def get_category_counts(self, *, user_id: str) -> dict[str, int]: ...

    def user_revision(self, user_id: str) -> int: ...

    def has_capacity_pressure(self) -> bool: ...

    def has_capacity_pressure_for_user(self, *, user_id: str) -> bool: ...
