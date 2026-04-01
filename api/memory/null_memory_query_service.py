from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.memory.hindsight_local_adapter import normalize_tags

from .null_memory_cache import NullMemoryCategoryCountCache
from .null_memory_repository import NullMemoryRepository
from .scoped_memory_records import (
    scoped_category_counts_from_records,
    scoped_memories_from_records,
)


class NullMemoryQueryService:
    """Scoped query helpers layered on top of NullMemoryStore."""

    def __init__(self, store: NullMemoryRepository) -> None:
        self.store = store
        self._category_count_cache = NullMemoryCategoryCountCache(max_entries=64)

    def scoped_memories(
        self,
        *,
        records: List[Dict[str, Any]],
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        return scoped_memories_from_records(
            records=records,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
        )

    def recall_records(
        self,
        *,
        query: str,
        user_id: str,
        tags: Optional[List[str]],
        tags_match: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        requested_tags = normalize_tags(tags)
        normalized_tags_match = (tags_match or "any").lower()
        if normalized_tags_match not in {"any", "all"}:
            normalized_tags_match = "any"
        matches: List[Dict[str, Any]] = []
        for memory in self.search_memories(
            query=query,
            user_id=user_id,
            tags=requested_tags,
            tags_match=normalized_tags_match,
            limit=limit,
        ):
            memory_tags = normalize_tags((memory.get("metadata") or {}).get("tags", []))
            matches.append({
                "document_id": memory["id"],
                "text": memory["content"],
                "tags": memory_tags,
            })
            if len(matches) >= limit:
                break
        return matches

    def get_all_memories(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.store.list_records(user_id=user_id)[:limit]

    def search_memories(
        self,
        *,
        query: str,
        user_id: str,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.store.search_records(
            query=query,
            user_id=user_id,
            tags=tags,
            tags_match=tags_match,
        )[:limit]

    def get_all_memories_scoped(
        self,
        *,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.scoped_memories(
            records=self.store.list_records(user_id=user_id),
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
        )

    def search_memories_scoped(
        self,
        *,
        query: str,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.scoped_memories(
            records=self.store.search_records(query=query, user_id=user_id),
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
        )

    def count_records_by_category(
        self,
        *,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> Dict[str, int]:
        cache_key = (
            self.store.user_revision(user_id),
            user_id,
            org_id,
            project_id,
            session_id,
            agent_id,
            run_id,
            include_shared,
        )
        cached = self._category_count_cache.get(cache_key)
        if cached is not None:
            return cached
        if (
            org_id is None
            and project_id is None
            and session_id is None
            and agent_id is None
            and run_id is None
            and include_shared
        ):
            categories = self.store.get_category_counts(user_id=user_id)
        else:
            categories = scoped_category_counts_from_records(
                records=self.store.list_records(user_id=user_id),
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
            )
        self._category_count_cache.put(cache_key, value=categories)
        return categories

    def count_memories_by_category_scoped(
        self,
        *,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> Dict[str, int]:
        return self.count_records_by_category(
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
        )
