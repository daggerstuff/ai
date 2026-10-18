from __future__ import annotations

from datetime import UTC, datetime

from .null_memory_capacity_tracker import NullMemoryCapacityTracker
from .null_memory_category_tracker import NullMemoryCategoryTracker
from .null_memory_coordination import NullMemoryCoordination
from .null_memory_index import NullMemoryIndex
from .null_memory_lifecycle import NullMemoryLifecycle
from .null_memory_mutation_service import InMemoryMutationService
from .null_memory_record_factory import NullMemoryRecordFactory
from .null_memory_record_store import NullMemoryRecordStore
from .null_memory_store import NullMemoryStore


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_null_memory_store(
    *,
    coordination: NullMemoryCoordination,
    max_memories_per_user: int = 1000,
) -> NullMemoryStore:
    index = NullMemoryIndex()
    record_factory = NullMemoryRecordFactory(
        generate_id=index.generate_id,
        now=_utc_now_iso,
    )
    records = NullMemoryRecordStore(index=index, record_factory=record_factory)
    categories = NullMemoryCategoryTracker()
    capacity = NullMemoryCapacityTracker()
    lifecycle = NullMemoryLifecycle(
        records=records,
        categories=categories,
        capacity=capacity,
        coordination=coordination,
        max_memories_per_user=max_memories_per_user,
    )
    mutation_service = InMemoryMutationService(
        coordination=coordination,
        records=records,
        lifecycle=lifecycle,
    )
    return NullMemoryStore(
        coordination=coordination,
        records=records,
        lifecycle=lifecycle,
        mutation_service=mutation_service,
    )
