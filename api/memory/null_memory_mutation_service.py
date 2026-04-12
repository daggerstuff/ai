from __future__ import annotations

from .null_memory_coordination import NullMemoryCoordination
from .null_memory_lifecycle import NullMemoryLifecycle
from .null_memory_record import NullMemoryRecord
from .null_memory_record_store import NullMemoryRecordStore


class InMemoryMutationService:
    """Coordinates write operations across record storage and lifecycle policy."""

    def __init__(
        self,
        *,
        coordination: NullMemoryCoordination,
        records: NullMemoryRecordStore,
        lifecycle: NullMemoryLifecycle,
    ) -> None:
        self._coordination = coordination
        self._records = records
        self._lifecycle = lifecycle

    def add_record(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> dict:
        with self._coordination.user_lock(user_id):
            record, existing_record = self._records.upsert_content(
                content=content,
                user_id=user_id,
                metadata=metadata,
                memory_id=memory_id,
            )
            if existing_record is not None:
                return self._replace_record(
                    user_id=user_id,
                    existing_record=existing_record,
                    record=record,
                )
            return self._insert_record(user_id=user_id, record=record)

    def update_record(
        self,
        *,
        memory_id: str,
        user_id: str,
        new_content: str,
        metadata: dict | None = None,
    ) -> bool:
        with self._coordination.user_lock(user_id):
            memory = self._records.get(memory_id=memory_id, user_id=user_id)
            if memory is None:
                return False
            updated_memory = memory.with_updates(
                new_content=new_content,
                metadata=metadata,
                updated_at=self._records.now(),
            )
            self._records.replace(
                memory_id=memory_id,
                user_id=user_id,
                record=updated_memory,
            )
            self._lifecycle.apply_category_delta(
                user_id=user_id,
                previous=memory,
                current=updated_memory,
            )
        self._coordination.touch(user_id=user_id)
        return True

    def delete_record(self, *, memory_id: str, user_id: str) -> bool:
        with self._coordination.user_lock(user_id):
            memory = self._records.delete(memory_id=memory_id, user_id=user_id)
            if memory is None:
                return False
            self._lifecycle.apply_category_delta(user_id=user_id, previous=memory, current=None)
            self._lifecycle.sync_capacity_state(user_id=user_id)
        self._coordination.touch(user_id=user_id)
        return True

    def clear_user(self, *, user_id: str) -> bool:
        with self._coordination.user_lock(user_id):
            removed_ids = self._records.clear_user(user_id=user_id)
            if not removed_ids:
                return False
        self._lifecycle.clear_user_state(user_id=user_id)
        self._coordination.touch(user_id=user_id)
        return True

    def _replace_record(
        self,
        *,
        user_id: str,
        existing_record: NullMemoryRecord,
        record: NullMemoryRecord,
    ) -> dict:
        self._records.upsert(user_id=user_id, record=record)
        self._lifecycle.apply_category_delta(
            user_id=user_id,
            previous=existing_record,
            current=record,
        )
        return self._finalize_mutation(user_id=user_id, record=record, sync_capacity=False)

    def _insert_record(self, *, user_id: str, record: NullMemoryRecord) -> dict:
        self._lifecycle.ensure_capacity(user_id=user_id)
        self._records.upsert(user_id=user_id, record=record)
        self._lifecycle.apply_category_delta(user_id=user_id, previous=None, current=record)
        return self._finalize_mutation(user_id=user_id, record=record, sync_capacity=True)

    def _finalize_mutation(
        self,
        *,
        user_id: str,
        record: NullMemoryRecord,
        sync_capacity: bool,
    ) -> dict:
        self._coordination.touch(user_id=user_id)
        if sync_capacity:
            self._lifecycle.sync_capacity_state(user_id=user_id)
        return record.to_dict()


NullMemoryMutationService = InMemoryMutationService
