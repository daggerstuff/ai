from __future__ import annotations

from typing import Any

from .foresight_local_adapter import memory_record_from_storage
from .local_foresight_repository import LocalForesightRepository


class LocalForesightMemoryQueryService:
    """Read/query operations for user-scoped local memories."""

    def __init__(
        self,
        *,
        default_bank_id: str,
        repository: LocalForesightRepository,
    ) -> None:
        self.default_bank_id = default_bank_id
        self.repository = repository

    def get_all_memories(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        records = self.repository.list_documents_for_user(
            self.default_bank_id,
            user_id=user_id,
            limit=limit,
            offset=0,
        )
        return [memory_record_from_storage(record) for record in records]

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
        records = self.repository.list_documents_for_scope(
            self.default_bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            category=category,
            tags=tags,
            limit=limit,
            offset=offset,
        )
        return [memory_record_from_storage(record) for record in records]

    def get_memories_by_category(
        self,
        *,
        user_id: str,
        category: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        records = self.repository.list_documents_for_user_by_category(
            self.default_bank_id,
            user_id=user_id,
            category=category,
            limit=limit,
            offset=0,
        )
        return [memory_record_from_storage(record) for record in records]

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
    ) -> list[dict[str, Any]]:
        records = self.repository.search_documents_for_scope(
            self.default_bank_id,
            user_id=user_id,
            query=query,
            limit=limit,
            offset=offset,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
        )
        return [memory_record_from_storage(record) for record in records]

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
        return self.repository.count_documents_by_category_for_scope(
            self.default_bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
        )
