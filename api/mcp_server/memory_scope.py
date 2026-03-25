"""
Scope utilities for memory operations.

Centralizes how scope metadata is written and filtered across MCP surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MemoryScope:
    """Canonical scope used by MCP memory APIs."""

    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    visibility: str = "private"
    include_shared: bool = True

    def to_metadata(self) -> Dict[str, Any]:
        """Convert scope to normalized metadata."""
        metadata: Dict[str, Any] = {
            "visibility": self.visibility,
        }
        if self.org_id:
            metadata["org_id"] = self.org_id
        if self.project_id:
            metadata["project_id"] = self.project_id
        if self.agent_id:
            metadata["agent_id"] = self.agent_id
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.session_id:
            metadata["session_id"] = self.session_id
        return metadata


def build_scope_metadata(
    *,
    scope: MemoryScope,
    incoming_metadata: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge normalized scope metadata with caller-provided metadata."""
    metadata = dict(incoming_metadata or {})
    metadata.update(scope.to_metadata())
    if category:
        metadata.setdefault("category", category)
    return metadata


def scope_from_kwargs(
    *,
    user_id: str,
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
) -> Dict[str, Dict[str, Any]]:
    """Shared JSON-schema properties for scope arguments."""
    props: Dict[str, Dict[str, Any]] = {
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


def _matches_value(expected: Optional[str], actual: Optional[str]) -> bool:
    if not expected:
        return True
    return actual == expected


def _is_shared(visibility: Optional[str]) -> bool:
    return (visibility or "").lower() in {"shared", "org", "project", "system"}


def _scope_matches(scope: MemoryScope, metadata: Dict[str, Any]) -> bool:
    visibility = (metadata.get("visibility") or "private").lower()
    is_shared = _is_shared(visibility)

    if not scope.include_shared and is_shared:
        return False

    if not _matches_value(scope.org_id, metadata.get("org_id")):
        return False
    if not _matches_value(scope.project_id, metadata.get("project_id")):
        return False

    # Shared records ignore agent/run/session constraints unless explicitly set.
    if not is_shared:
        if not _matches_value(scope.agent_id, metadata.get("agent_id")):
            return False
        if not _matches_value(scope.run_id, metadata.get("run_id")):
            return False
        if not _matches_value(scope.session_id, metadata.get("session_id")):
            return False

    return True


def filter_memories_by_scope(
    *,
    scope: MemoryScope,
    memories: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter memory records using normalized scope metadata."""
    filtered: List[Dict[str, Any]] = []
    for memory in memories:
        metadata = memory.get("metadata") or {}
        if _scope_matches(scope, metadata):
            filtered.append(memory)

    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


def search_with_overfetch(
    *,
    manager: Any,
    query: str,
    user_id: str,
    requested_limit: int,
) -> List[Dict[str, Any]]:
    """
    Fetch extra search candidates before in-process scope filtering.
    """
    fetch_limit = max(requested_limit * 5, 50)
    try:
        result = manager.search_memories(query, user_id=user_id, limit=fetch_limit)
    except (TypeError, AttributeError):
        if hasattr(manager, "search_memories"):
            result = manager.search_memories(query, user_id)
        else:
            result = manager.search(query, user_id=user_id, limit=fetch_limit)

    if isinstance(result, dict):
        return result.get("results", [])
    if isinstance(result, list):
        return result
    return []


def memory_in_scope(
    *,
    manager: Any,
    scope: MemoryScope,
    memory_id: str,
) -> bool:
    """
    Verify memory ownership/scope using direct lookup metadata.
    """
    try:
        record = manager.get_memory(memory_id)
    except Exception:
        return False

    if not record:
        return False

    scoped = filter_memories_by_scope(scope=scope, memories=[record], limit=1)
    return bool(scoped)
