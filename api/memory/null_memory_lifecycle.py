from __future__ import annotations

from typing import Protocol

from .null_memory_capacity_tracker import NullMemoryCapacityTracker
from .null_memory_category_tracker import NullMemoryCategoryTracker
from .null_memory_record import NullMemoryRecord


class NullMemoryLifecycleRecords(Protocol):
    def user_count(self, *, user_id: str) -> int: ...
    def evict_oldest(self, *, user_id: str) -> NullMemoryRecord | None: ...


class NullMemoryLifecycleCoordination(Protocol):
    def touch(self, *, user_id: str) -> None: ...
    def drop_user_lock(self, user_id: str) -> None: ...


class NullMemoryLifecycle:
    """Coordinates policy side-effects around low-level record persistence."""

    def __init__(
        self,
        *,
        records: NullMemoryLifecycleRecords,
        categories: NullMemoryCategoryTracker,
        capacity: NullMemoryCapacityTracker,
        coordination: NullMemoryLifecycleCoordination,
        max_memories_per_user: int,
    ) -> None:
        self.records = records
        self.categories = categories
        self.capacity = capacity
        self.coordination = coordination
        self.max_memories_per_user = max_memories_per_user

    @staticmethod
    def category_for(record: NullMemoryRecord | None) -> str | None:
        if record is None:
            return None
        return record.metadata.get("category", "general")

    def apply_category_delta(
        self,
        *,
        user_id: str,
        previous: NullMemoryRecord | None,
        current: NullMemoryRecord | None,
    ) -> None:
        self.categories.apply(
            user_id=user_id,
            previous_category=self.category_for(previous),
            current_category=self.category_for(current),
        )

    def ensure_capacity(self, *, user_id: str) -> None:
        if self.records.user_count(user_id=user_id) < self.max_memories_per_user:
            return
        evicted = self.records.evict_oldest(user_id=user_id)
        if evicted is None:
            return
        self.apply_category_delta(user_id=user_id, previous=evicted, current=None)
        self.coordination.touch(user_id=user_id)
        self.capacity.set_user_count(
            user_id=user_id,
            count=self.records.user_count(user_id=user_id),
            max_memories=self.max_memories_per_user,
        )

    def sync_capacity_state(self, *, user_id: str) -> None:
        self.capacity.set_user_count(
            user_id=user_id,
            count=self.records.user_count(user_id=user_id),
            max_memories=self.max_memories_per_user,
        )

    def has_capacity_pressure(self) -> bool:
        return self.capacity.has_pressure()

    def has_capacity_pressure_for_user(self, *, user_id: str) -> bool:
        return self.capacity.user_has_pressure(user_id=user_id)

    def get_category_counts(self, *, user_id: str) -> dict[str, int]:
        return self.categories.get_counts(user_id=user_id)

    def clear_user_state(self, *, user_id: str) -> None:
        self.categories.clear_user(user_id=user_id)
        self.capacity.clear_user(user_id=user_id)
        self.coordination.drop_user_lock(user_id)
