from __future__ import annotations

from typing import Any, Dict, List, Optional

from .null_memory_cache import NullMemoryCategoryCountCache
from .scoped_memory_records import (
    scoped_category_counts_from_records,
    scoped_memories_from_records,
)
from .null_memory_store import NullMemoryStore


class NullMemoryProtocolAdapter:
    """Scoped and Hindsight-compatible helpers backed by NullMemoryStore."""

    def __init__(self, store: NullMemoryStore) -> None:
        self.store = store
        self._category_count_cache = NullMemoryCategoryCountCache(max_entries=64)

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
        return scoped_memories_from_records(
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

    def recall(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
        tags: Optional[List[str]],
        tags_match: str,
    ) -> Dict[str, Any]:
        return {
            "results": self.store.recall_records(
                query=query,
                user_id=user_id,
                tags=tags,
                tags_match=tags_match,
                limit=limit,
            )
        }

    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]],
        category: Optional[str],
    ) -> str:
        merged_metadata = dict(metadata or {})
        if category:
            merged_metadata["category"] = category
        record = self.store.add_record(
            content=content,
            user_id=user_id,
            metadata=merged_metadata,
        )
        return record["id"]

    def search_memories(self, *, query: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        return self.store.search_records(query=query, user_id=user_id)[:limit]

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
        return scoped_memories_from_records(
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

    def get_all_memories(self, *, user_id: str, limit: int) -> List[Dict[str, Any]]:
        return self.store.list_records(user_id=user_id)[:limit]

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
        self._category_count_cache.put(cache_key, categories)
        return categories
