from __future__ import annotations

from typing import Dict, List, Optional

from ai.api.mcp_server.memory_scope import filter_memories_by_scope, scope_from_kwargs
from ai.memory.hindsight_local_adapter import normalize_tags

from .null_memory_store import NullMemoryStore


class NullMemoryProtocolAdapter:
    """Scoped and Hindsight-compatible helpers backed by NullMemoryStore."""

    def __init__(self, store: NullMemoryStore) -> None:
        self.store = store

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
        requested_tags = normalize_tags(tags)
        matches: List[Dict[str, Any]] = []
        for memory in self.store.search_records(query=query, user_id=user_id):
            memory_tags = normalize_tags(memory.get("metadata", {}).get("tags", []))
            if requested_tags:
                if tags_match == "all":
                    if not all(tag in memory_tags for tag in requested_tags):
                        continue
                elif not any(tag in memory_tags for tag in requested_tags):
                    continue
            matches.append(
                {
                    "document_id": memory["id"],
                    "text": memory["content"],
                    "tags": memory_tags,
                }
            )
            if len(matches) >= limit:
                break
        return {"results": matches}

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

    def get_all_memories(self, *, user_id: str, limit: int) -> List[Dict[str, Any]]:
        return self.store.list_records(user_id=user_id)[:limit]
