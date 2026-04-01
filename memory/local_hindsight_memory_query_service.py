from __future__ import annotations

from typing import Any, Dict, List, Optional

from .hindsight_local_adapter import memory_record_from_storage
from .local_hindsight_repository import LocalHindsightRepository


class LocalHindsightMemoryQueryService:
    """Read/query operations for user-scoped local memories."""

    def __init__(
        self,
        *,
        default_bank_id: str,
        repository: LocalHindsightRepository,
    ) -> None:
        self.default_bank_id = default_bank_id
        self.repository = repository

    def get_all_memories(self, *, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
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
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        records = self.repository.list_documents_for_scope(
            self.default_bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
            offset=0,
        )
        return [memory_record_from_storage(record) for record in records]

    def get_memories_by_category(
        self,
        *,
        user_id: str,
        category: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        records = self.repository.list_documents_for_user_by_category(
            self.default_bank_id,
            user_id=user_id,
            category=category,
            limit=limit,
            offset=0,
        )
        return [memory_record_from_storage(record) for record in records]
