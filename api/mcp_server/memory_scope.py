"""
Scope utilities for memory operations.

Centralizes how scope metadata is written and filtered across MCP surfaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .fastmcp_protocols import (
    BasicMemorySearcher,
    LegacyMemorySearcher,
    MemoryReader,
)


@dataclass(frozen=True)
class MemoryScope:
    """Canonical scope used by MCP memory APIs."""

    user_id: str
    org_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    visibility: str = "private"
    include_shared: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Convert scope to normalized metadata."""
        return scope_metadata_dict(self)


def scope_metadata_dict(scope: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "visibility": getattr(scope, "visibility", "private"),
    }
    for key in ("org_id", "project_id", "agent_id", "run_id", "session_id"):
        value = getattr(scope, key, None)
        if value:
            metadata[key] = value
    return metadata


def build_scope_metadata(
    *,
    scope: Any,
    incoming_metadata: dict[str, Any] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Merge normalized scope metadata with caller-provided metadata."""
    metadata = dict(incoming_metadata or {})
    metadata.update(scope_metadata_dict(scope))
    if category:
        metadata.setdefault("category", category)
    return metadata


def scope_from_kwargs(
    *,
    user_id: str,
    org_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    visibility: str = "private",
    include_shared: bool = True,
) -> MemoryScope:
    """Build canonical scope from common keyword args."""
    return MemoryScope(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        agent_id=agent_id,
        run_id=run_id,
        session_id=session_id,
        visibility=visibility,
        include_shared=include_shared,
    )


def scope_input_schema_properties(
    *,
    include_visibility: bool = False,
) -> dict[str, dict[str, Any]]:
    """Shared JSON-schema properties for scope arguments."""
    props: dict[str, dict[str, Any]] = {
        "org_id": {"type": "string", "description": "Organization scope (optional)"},
        "project_id": {"type": "string", "description": "Project scope (optional)"},
        "session_id": {"type": "string", "description": "Session scope (optional)"},
        "agent_id": {"type": "string", "description": "Agent scope (optional)"},
        "run_id": {"type": "string", "description": "Run scope (optional)"},
        "include_shared": {
            "type": "boolean",
            "description": "Whether shared memories are included",
            "default": True,
        },
    }
    if include_visibility:
        props["visibility"] = {
            "type": "string",
            "description": "private/shared scope visibility",
            "default": "private",
        }
    return props


def _matches_value(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return True
    return actual == expected


def _is_shared(visibility: str | None) -> bool:
    return (visibility or "").lower() in {"shared", "org", "project", "system"}


def _matches_shared_scope(scope: MemoryScope, metadata: dict[str, Any]) -> bool:
    if not _matches_value(scope.org_id, metadata.get("org_id")):
        return False
    return _matches_value(scope.project_id, metadata.get("project_id"))


def _matches_private_scope(scope: MemoryScope, metadata: dict[str, Any]) -> bool:
    if not _matches_shared_scope(scope, metadata):
        return False
    if not _matches_value(scope.agent_id, metadata.get("agent_id")):
        return False
    if not _matches_value(scope.run_id, metadata.get("run_id")):
        return False
    return _matches_value(scope.session_id, metadata.get("session_id"))


def _scope_matches(scope: MemoryScope, metadata: dict[str, Any]) -> bool:
    visibility = (metadata.get("visibility") or "private").lower()
    is_shared = _is_shared(visibility)

    if not scope.include_shared and is_shared:
        return False
    if is_shared:
        return _matches_shared_scope(scope, metadata)

    return _matches_private_scope(scope, metadata)


def filter_memories_by_scope(
    *,
    scope: MemoryScope,
    memories: Iterable[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter memory records using normalized scope metadata."""
    filtered: list[dict[str, Any]] = []
    for memory in memories:
        metadata = memory.get("metadata") or {}
        if _scope_matches(scope, metadata):
            filtered.append(memory)

    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


@dataclass(frozen=True)
class SearchCandidateFetchPolicy:
    initial_multiplier: int = 4
    max_candidates: int = 100

    def initial_limit(self, requested_limit: int) -> int:
        minimum = max(requested_limit, 1)
        return min(
            max(minimum * self.initial_multiplier, minimum + 8),
            self.max_candidates,
        )


_DEFAULT_SEARCH_FETCH_POLICY = SearchCandidateFetchPolicy()


@dataclass(frozen=True)
class _MemorySearchAdapter:
    manager: Any

    @classmethod
    def from_manager(cls, manager: Any) -> _MemorySearchAdapter:
        if isinstance(manager, BasicMemorySearcher):
            return _BasicMemorySearchAdapter(manager)
        if isinstance(manager, LegacyMemorySearcher):
            return _LegacyMemorySearchAdapter(manager)
        raise TypeError("Memory manager does not support search operations.")

    def run_search(self, query: str, user_id: str, limit: int) -> Any:
        raise NotImplementedError  # subclasses must override


@dataclass(frozen=True)
class _BasicMemorySearchAdapter(_MemorySearchAdapter):
    manager: BasicMemorySearcher

    def run_search(self, query: str, user_id: str, limit: int):
        return self.manager.search_memories(query, user_id, limit)


@dataclass(frozen=True)
class _LegacyMemorySearchAdapter(_MemorySearchAdapter):
    manager: LegacyMemorySearcher

    def run_search(self, query: str, user_id: str, limit: int):
        return self.manager.search(query, user_id=user_id, limit=limit)


def _progressive_fetch_limits(
    *,
    requested_limit: int,
    fetch_policy: SearchCandidateFetchPolicy,
):
    fetch_limit = fetch_policy.initial_limit(requested_limit)
    yield fetch_limit
    if fetch_limit < fetch_policy.max_candidates:
        yield fetch_policy.max_candidates


def search_with_overfetch(
    *,
    manager: Any,
    query: str,
    user_id: str,
    requested_limit: int,
    scope: MemoryScope | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch candidates using a bounded progressive window so restrictive scopes
    can still fill the requested result count without a fixed 5x overfetch.
    """
    adapter = _MemorySearchAdapter.from_manager(manager)
    fetch_policy = _DEFAULT_SEARCH_FETCH_POLICY
    best_match: list[dict[str, Any]] = []

    for fetch_limit in _progressive_fetch_limits(
        requested_limit=requested_limit,
        fetch_policy=fetch_policy,
    ):
        result = adapter.run_search(query, user_id, fetch_limit)
        candidates: list[dict[str, Any]]
        if isinstance(result, dict):
            candidates = result.get("results", [])
        elif isinstance(result, list):
            candidates = result
        else:
            candidates = []
        if not candidates:
            return []

        if scope is None:
            return candidates

        best_match = filter_memories_by_scope(
            scope=scope,
            memories=candidates,
            limit=requested_limit,
        )
        if len(best_match) >= requested_limit:
            return best_match
    return best_match


def memory_in_scope(
    *,
    manager: Any,
    scope: MemoryScope,
    memory_id: str,
) -> bool:
    """
    Verify memory ownership/scope using direct lookup metadata.
    """
    if not isinstance(manager, MemoryReader):
        return False
    try:
        record = manager.get_memory(memory_id)
    except Exception:
        return False

    if not record:
        return False

    scoped = filter_memories_by_scope(scope=scope, memories=[record], limit=1)
    return bool(scoped)
