from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryCategoryCounter(Protocol):
    def count_memories_by_category_scoped(
        self,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, int]: ...


@runtime_checkable
class MemoryCategoryCounterProvider(Protocol):
    queries: MemoryCategoryCounter


def resolve_memory_category_counter(manager) -> MemoryCategoryCounter | None:
    if isinstance(manager, MemoryCategoryCounter):
        return manager
    if isinstance(manager, MemoryCategoryCounterProvider):
        return manager.queries
    return None
