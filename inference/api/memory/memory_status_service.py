from __future__ import annotations

from ai.api.mcp_server.fastmcp_search import get_scoped_recent_memories
from ai.api.mcp_server.memory_scope import MemoryScope

from .memory_category_counter import resolve_memory_category_counter
from .memory_category_counts import count_memory_categories
from .memory_health import resolve_memory_readiness
from .memory_status_summary import MemoryStatusSummary, summarize_memory_status


def build_memory_status_summary(
    *,
    manager,
    scope: MemoryScope,
    user_id: str,
    fallback_limit: int = 100,
) -> MemoryStatusSummary:
    category_counter = resolve_memory_category_counter(manager)
    is_sampled = False
    if category_counter is not None:
        category_counts = category_counter.count_memories_by_category_scoped(
            user_id=user_id,
            org_id=scope.org_id,
            project_id=scope.project_id,
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            include_shared=scope.include_shared,
        )
    else:
        is_sampled = True
        category_counts = count_memory_categories(
            get_scoped_recent_memories(
                manager=manager,
                scope=scope,
                user_id=user_id,
                limit=fallback_limit,
            )
        )
    return summarize_memory_status(
        category_counts=category_counts,
        backend_readiness=resolve_memory_readiness(manager),
        is_sampled=is_sampled,
    )
