from __future__ import annotations

from dataclasses import dataclass

from .null_memory_coordination import NullMemoryCoordination
from .null_memory_protocol_adapter import NullMemoryProtocolAdapter
from .null_memory_query_service import NullMemoryQueryService
from .null_memory_store import NullMemoryStore
from .null_memory_store_factory import build_null_memory_store


@dataclass(frozen=True)
class NullMemoryManagerRuntime:
    coordination: NullMemoryCoordination
    store: NullMemoryStore
    queries: NullMemoryQueryService
    protocol: NullMemoryProtocolAdapter


def build_null_memory_manager_runtime() -> NullMemoryManagerRuntime:
    coordination = NullMemoryCoordination()
    store = build_null_memory_store(coordination=coordination)
    queries = NullMemoryQueryService(store)
    protocol = NullMemoryProtocolAdapter(store, queries=queries)
    return NullMemoryManagerRuntime(
        coordination=coordination,
        store=store,
        queries=queries,
        protocol=protocol,
    )
