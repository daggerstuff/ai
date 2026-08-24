from __future__ import annotations

from typing import Any, cast

from ai.research.base import (
    CategoryScopedMemoryManager,
    ForesightCompatibleMemoryManager,
    ScopedMemoryManager,
)


def get_scoped_memories(
    manager: ScopedMemoryManager,
    *,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
    limit: int,
    offset: int = 0,
    category: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    result = manager.get_all_memories_scoped(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
        limit=limit,
        offset=offset,
        category=category,
        tags=tags,
    )
    if isinstance(result, dict) and "results" in result:
        return cast(dict[str, Any], result)["results"]
    return result or []


def recall_memories_for_user(
    manager: ForesightCompatibleMemoryManager,
    *,
    bank_id: str,
    user_id: str,
    query: str,
    limit: int,
    tags: list[str] | None,
    tags_match: str | None,
) -> dict[str, Any]:
    return manager.recall_for_user(
        bank_id,
        user_id=user_id,
        query=query,
        limit=limit,
        tags=tags,
        tags_match=tags_match or "any",
    )


def get_scoped_memory_stats(
    manager: CategoryScopedMemoryManager,
    *,
    user_id: str,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    include_shared: bool,
) -> dict[str, int]:
    category_counts = manager.count_memories_by_category_scoped(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    return {str(category): int(count) for category, count in dict(category_counts or {}).items()}
