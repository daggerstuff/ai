from __future__ import annotations

from ai.api.mcp_server.memory_scope import (
    filter_memories_by_scope,
    search_with_overfetch,
)

from .fastmcp_protocols import ScopedMemoryLister, ScopedMemorySearcher
from .fastmcp_shared import get_recent_memories

_MAX_FALLBACK_CANDIDATES = 100
_MAX_FALLBACK_RECENT_MEMORIES = 100


def _fallback_candidate_limit(*, requested_limit: int, aggressive: bool) -> int:
    if aggressive:
        computed = max(requested_limit * 4, requested_limit + 8)
    else:
        computed = max(requested_limit * 2, requested_limit + 4)
    return min(computed, _MAX_FALLBACK_CANDIDATES)


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
    return search_with_overfetch(
        manager=manager,
        query=query,
        user_id=user_id,
        requested_limit=candidate_limit,
        scope=scope,
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


def get_scoped_recent_memories(
    *,
    manager,
    user_id: str,
    scope,
    limit: int,
):
    requested_limit = min(max(limit, 1), _MAX_FALLBACK_RECENT_MEMORIES)
    if isinstance(manager, ScopedMemoryLister):
        return manager.get_all_memories_scoped(
            user_id=user_id,
            org_id=scope.org_id,
            project_id=scope.project_id,
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            include_shared=scope.include_shared,
            limit=requested_limit,
        )

    memories = get_recent_memories(manager, user_id, limit=requested_limit)
    return filter_memories_by_scope(
        scope=scope,
        memories=memories,
        limit=requested_limit,
    )
