from __future__ import annotations

import os

from ai.api.mcp_server.memory_scope import (
    filter_memories_by_scope,
    search_with_overfetch,
)

from .fastmcp_protocols import ScopedMemorySearcher


def _fallback_candidate_limit(*, requested_limit: int, aggressive: bool) -> int:
    env_multiplier = os.getenv("PIXELATED_MEMORY_SCOPE_OVERFETCH_MULTIPLIER")
    if env_multiplier:
        try:
            configured_multiplier = max(1, int(env_multiplier))
            return max(requested_limit * configured_multiplier, requested_limit + 8)
        except ValueError:
            pass
    if aggressive:
        return max(requested_limit * 4, requested_limit + 8)
    return max(requested_limit * 2, requested_limit + 4)


def _filter_scoped_candidates(*, scope, candidates):
    return filter_memories_by_scope(
        scope=scope,
        memories=candidates,
        limit=None,
    )


def _fallback_scoped_search_candidates(
    *,
    manager,
    query: str,
    user_id: str,
    scope,
    requested_limit: int,
) -> list[dict]:
    candidate_limit = _fallback_candidate_limit(
        requested_limit=requested_limit,
        aggressive=True,
    )
    candidates = search_with_overfetch(
        manager=manager,
        query=query,
        user_id=user_id,
        requested_limit=candidate_limit,
    )
    if not candidates:
        return []
    return _filter_scoped_candidates(
        scope=scope,
        candidates=candidates,
    )


def _search_with_scope_backend(
    *,
    manager: ScopedMemorySearcher,
    query: str,
    user_id: str,
    scope,
    requested_limit: int,
):
    return manager.search_memories_scoped(
        query=query,
        user_id=user_id,
        org_id=scope.org_id,
        project_id=scope.project_id,
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        include_shared=scope.include_shared,
        limit=requested_limit,
    )


def search_scoped_memories(
    *,
    manager,
    query: str,
    user_id: str,
    scope,
    limit: int,
):
    requested_limit = max(limit, 1)
    if isinstance(manager, ScopedMemorySearcher):
        return _search_with_scope_backend(
            manager=manager,
            query=query,
            user_id=user_id,
            scope=scope,
            requested_limit=requested_limit,
        )

    filtered_results = _fallback_scoped_search_candidates(
        manager=manager,
        query=query,
        user_id=user_id,
        scope=scope,
        requested_limit=requested_limit,
    )
    return filtered_results[:requested_limit]
