from __future__ import annotations

from typing import Any

from ai.inference.api.mcp_server.memory_scope import _scope_matches, filter_memories_by_scope, scope_from_kwargs


def scoped_memories_from_records(
    *,
    records: list[dict[str, Any]],
    user_id: str,
    org_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    include_shared: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    scope = scope_from_kwargs(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    return filter_memories_by_scope(
        scope=scope,
        memories=records,
        limit=limit,
    )


def scoped_category_counts_from_records(
    *,
    records: list[dict[str, Any]],
    user_id: str,
    org_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    include_shared: bool = True,
) -> dict[str, int]:
    scope = scope_from_kwargs(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    categories: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        if not _scope_matches(scope, metadata):
            continue
        category = metadata.get("category", "general")
        categories[category] = categories.get(category, 0) + 1
    return categories
