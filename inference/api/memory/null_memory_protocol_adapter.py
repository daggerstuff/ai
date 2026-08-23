from __future__ import annotations

from typing import Any

from .null_memory_query_service import NullMemoryQueryService
from .null_memory_repository import NullMemoryRepository


class NullMemoryProtocolAdapter:
    """Scoped and Foresight-compatible helpers backed by NullMemoryStore."""

    def __init__(
        self,
        store: NullMemoryRepository,
        *,
        queries: NullMemoryQueryService | None = None,
    ) -> None:
        self.store = store
        self.queries = queries or NullMemoryQueryService(store)

    def recall(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
        tags: list[str] | None,
        tags_match: str,
    ) -> dict[str, Any]:
        return {
            "results": self.queries.recall_records(
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
        metadata: dict[str, Any] | None,
        category: str | None,
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
