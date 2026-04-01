from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai.api.mcp_server.fastmcp_protocols import (
    MemoryQueryServiceProvider,
    ScopedMemoryCategoryCounter,
)
from ai.api.mcp_server.fastmcp_search import get_scoped_recent_memories

from .memory_category_counts import count_memory_categories


@dataclass(frozen=True)
class MemoryStatusSummary:
    total_anchors: int
    health: str
    categories: dict[str, int]


def summarize_memory_status(*, manager: Any, scope: Any, user_id: str) -> MemoryStatusSummary:
    memories = get_scoped_recent_memories(
        manager=manager,
        scope=scope,
        user_id=user_id,
        limit=100,
    )
    categories = _count_categories(manager=manager, scope=scope, user_id=user_id, memories=memories)
    health = "Stable" if len(memories) > 10 else "Developing"
    return MemoryStatusSummary(
        total_anchors=len(memories),
        health=health,
        categories=categories,
    )


def _count_categories(
    *,
    manager: Any,
    scope: Any,
    user_id: str,
    memories: list[dict[str, Any]],
) -> dict[str, int]:
    counter = manager if isinstance(manager, ScopedMemoryCategoryCounter) else None
    if counter is None and isinstance(manager, MemoryQueryServiceProvider):
        counter = manager.queries
    if isinstance(counter, ScopedMemoryCategoryCounter):
        return counter.count_memories_by_category_scoped(
            user_id=user_id,
            org_id=scope.org_id,
            project_id=scope.project_id,
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            include_shared=scope.include_shared,
        )
    return count_memory_categories(memories)
