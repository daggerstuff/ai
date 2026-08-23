from __future__ import annotations

from datetime import UTC, datetime

from ai.memory.local_memory_adapter import normalize_tags

from .null_memory_coordination import NullMemoryCoordination
from .null_memory_lifecycle import NullMemoryLifecycle
from .null_memory_mutation_service import InMemoryMutationService
from .null_memory_record_store import NullMemoryRecordStore


class InMemoryStore:
    """Repository facade over record persistence plus lifecycle policy."""

    def __init__(
        self,
        *,
        coordination: NullMemoryCoordination,
        records: NullMemoryRecordStore,
        lifecycle: NullMemoryLifecycle,
        mutation_service: InMemoryMutationService,
    ) -> None:
        self._coordination = coordination
        self._records = records
        self._lifecycle = lifecycle
        self._mutations = mutation_service

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def user_revision(self, user_id: str) -> int:
        return self._coordination.user_revision(user_id)

    def has_capacity_pressure(self) -> bool:
        return self._lifecycle.has_capacity_pressure()

    def has_capacity_pressure_for_user(self, *, user_id: str) -> bool:
        return self._lifecycle.has_capacity_pressure_for_user(user_id=user_id)

    def get_category_counts(self, *, user_id: str) -> dict[str, int]:
        return self._lifecycle.get_category_counts(user_id=user_id)

    def add_record(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> dict:
        return self._mutations.add_record(
            content=content,
            user_id=user_id,
            metadata=metadata,
            memory_id=memory_id,
        )

    def search_records(
        self,
        *,
        query: str,
        user_id: str,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> list[dict]:
        query_lower = query.lower()
        normalized_tags = tuple(normalize_tags(tags))
        matches = self._records.search_for_user(
            query_lower=query_lower,
            user_id=user_id,
            tags=normalized_tags,
            tags_match=tags_match,
        )
        return [record.to_dict() for record in matches]

    def list_records(self, *, user_id: str) -> list[dict]:
        snapshots = self._records.snapshot_for_user(user_id=user_id)
        return [record.to_dict() for record in snapshots]

    def get_record(self, *, memory_id: str, user_id: str) -> dict | None:
        with self._coordination.user_lock(user_id):
            memory = self._records.get(memory_id=memory_id, user_id=user_id)
        return memory.to_dict() if memory is not None else None

    def update_record(
        self,
        *,
        memory_id: str,
        user_id: str,
        new_content: str,
        metadata: dict | None = None,
    ) -> bool:
        return self._mutations.update_record(
            memory_id=memory_id,
            user_id=user_id,
            new_content=new_content,
            metadata=metadata,
        )

    def delete_record(self, *, memory_id: str, user_id: str) -> bool:
        return self._mutations.delete_record(memory_id=memory_id, user_id=user_id)

    def clear_user(self, *, user_id: str) -> bool:
        return self._mutations.clear_user(user_id=user_id)


NullMemoryStore = InMemoryStore
