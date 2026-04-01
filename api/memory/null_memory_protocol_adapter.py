from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.api.mcp_server.memory_scope import filter_memories_by_scope, scope_from_kwargs

from .memory_category_counts import count_memory_categories
from .null_memory_cache import NullMemoryCategoryCountCache
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
            memories=self.store.list_records(user_id=user_id),
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
            memories=self.store.search_records(query=query, user_id=user_id),
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

        categories = count_memory_categories(
            self.get_all_memories_scoped(
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
                limit=100,
            )
        )
        self._category_count_cache.put(cache_key, categories)
        return categories
