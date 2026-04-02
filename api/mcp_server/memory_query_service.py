from __future__ import annotations

from typing import Dict, List, Optional

from ai.memory.base import (
    CategoryScopedMemoryManager,
    HindsightCompatibleMemoryManager,
    ScopedMemoryManager,
)


def get_scoped_memories(
    manager: ScopedMemoryManager,
    *,
    user_id: str,
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    include_shared: bool,
    limit: int,
    offset: int = 0,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
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
        return result["results"]
    return result or []


def recall_memories_for_user(
    manager: HindsightCompatibleMemoryManager,
    *,
    bank_id: str,
    user_id: str,
    query: str,
    limit: int,
    tags: Optional[List[str]],
    tags_match: Optional[str],
) -> Dict[str, Any]:
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
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    include_shared: bool,
) -> Dict[str, int]:
    category_counts = manager.count_memories_by_category_scoped(
        user_id=user_id,
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        include_shared=include_shared,
    )
    return {
        str(category): int(count)
        for category, count in dict(category_counts or {}).items()
    }
