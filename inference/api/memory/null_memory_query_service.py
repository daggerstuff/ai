from __future__ import annotations

from typing import Any

from ai.research.local_memory_adapter import normalize_tags

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
        tags: list[str] | None,
        tags_match: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        requested_tags = normalize_tags(tags)
        normalized_tags_match = (tags_match or "any").lower()
        if normalized_tags_match not in {"any", "all"}:
            normalized_tags_match = "any"
        matches: list[dict[str, Any]] = []
        for memory in self.search_memories(
            query=query,
            user_id=user_id,
            tags=requested_tags,
            tags_match=normalized_tags_match,
            limit=limit,
        ):
            memory_tags = normalize_tags((memory.get("metadata") or {}).get("tags", []))
            matches.append(
                {
                    "document_id": memory["id"],
                    "text": memory["content"],
                    "tags": memory_tags,
                }
            )
            if len(matches) >= limit:
                break
        return matches

    def get_all_memories(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.store.list_records(user_id=user_id)[:limit]

    def search_memories(
        self,
        *,
        query: str,
        user_id: str,
        tags: list[str] | None = None,
        tags_match: str = "any",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
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
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int = 100,
        offset: int = 0,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records = self.store.list_records(user_id=user_id)
        if category or tags:
            filtered: list[dict[str, Any]] = []
            for record in records:
                metadata = record.get("metadata") or {}
                if category and metadata.get("category") != category:
                    continue
                if tags:
                    record_tags = normalize_tags(metadata.get("tags") or [])
                    if not set(tags).issubset(set(record_tags)):
                        continue
                filtered.append(record)
            records = filtered
        scoped = self.scoped_memories(
            records=records,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=None if limit is None else (offset + limit),
        )
        return scoped[offset : offset + limit]

    def search_memories_scoped(
        self,
        *,
        query: str,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        limit: int = 10,
        offset: int = 0,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records = self.store.search_records(query=query, user_id=user_id)
        if category or tags:
            filtered: list[dict[str, Any]] = []
            for record in records:
                metadata = record.get("metadata") or {}
                if category and metadata.get("category") != category:
                    continue
                if tags:
                    record_tags = normalize_tags(metadata.get("tags") or [])
                    if not set(tags).issubset(set(record_tags)):
                        continue
                filtered.append(record)
            records = filtered
        scoped = self.scoped_memories(
            records=records,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=None if limit is None else (offset + limit),
        )
        return scoped[offset : offset + limit]

    def count_records_by_category(
        self,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, int]:
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
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, int]:
        return self.count_records_by_category(
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
        )
