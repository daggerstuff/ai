from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseMemoryManager
from .local_hindsight_document_service import LocalHindsightDocumentService
from .local_hindsight_memory_query_service import LocalHindsightMemoryQueryService
from .local_hindsight_memory_record_service import LocalHindsightMemoryRecordService
from .local_hindsight_memory_write_service import LocalHindsightMemoryWriteService
from .local_hindsight_protocol_adapter import LocalHindsightProtocolAdapter
from .local_hindsight_repository import LocalHindsightRepository


class LocalHindsightMemoryManager(BaseMemoryManager):
    """Persistent local Hindsight-compatible store backed by SQLite."""

    def __init__(
        self,
        db_path: str,
        bank_id: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.default_bank_id = bank_id or "pixelated"
        self.bank_id = self.default_bank_id
        self.repository = LocalHindsightRepository(self.db_path)
        self.documents = LocalHindsightDocumentService(self.repository)
        self.protocol = LocalHindsightProtocolAdapter(
            default_bank_id=self.default_bank_id,
            documents=self.documents,
        )
        self.writes = LocalHindsightMemoryWriteService(
            protocol=self.protocol,
            default_bank_id=self.default_bank_id,
        )
        self.records = LocalHindsightMemoryRecordService(
            default_bank_id=self.default_bank_id,
            repository=self.repository,
            writes=self.writes,
            provider_name=self.get_provider_name(),
        )
        self.queries = LocalHindsightMemoryQueryService(
            default_bank_id=self.default_bank_id,
            repository=self.repository,
        )
        self.retain_items = self.protocol.retain_items
        self.recall = self.protocol.recall
        self.recall_for_user = self.protocol.recall_for_user
        self.can_write_document = self.protocol.can_write_document
        self.prepare_retained_items = self.protocol.prepare_retained_items
        self.add_memory_scoped = self.writes.add_memory
        self.search_memories_scoped = self.queries.search_memories_scoped
        self.get_all_memories_scoped = self.queries.get_all_memories_scoped
        self.count_memories_by_category_scoped = self.queries.count_memories_by_category_scoped
        self.get_memories_by_category = self.queries.get_memories_by_category
        self.delete_memories = self._delete_memories

    def list_documents(
        self,
        bank_id: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required when listing documents")
        return self.protocol.list_documents(
            bank_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not user_id:
            raise ValueError("user_id is required when fetching documents")
        return self.protocol.get_document(bank_id, document_id, user_id=user_id)

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> bool:
        if not user_id:
            raise ValueError("user_id is required when deleting documents")
        return self.protocol.delete_document(bank_id, document_id, user_id=user_id)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Any] = None,
        category: Optional[str] = None,
    ) -> str:
        return self.writes.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )

    def search_memories(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.recall(
            self.default_bank_id,
            query=query,
            limit=limit,
            tags=[f"user:{user_id}"],
            tags_match="any",
        )["results"]

    def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.queries.get_all_memories(user_id=user_id, limit=limit)

    def get_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.records.get_memory(memory_id, user_id=user_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        return self.records.update_memory(
            memory_id=memory_id,
            new_content=new_content,
            metadata=metadata,
            user_id=user_id,
        )

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        return self.records.delete_memory(
            memory_id=memory_id,
            user_id=user_id,
        )

    def _delete_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> int:
        return self.records.delete_memories(
            memory_ids,
            user_id=user_id,
        )

    def clear_memory(self, user_id: str) -> bool:
        return self.records.clear_memory(user_id)

    def get_health_status(self) -> Dict[str, Any]:
        return self.records.get_health_status()

    def close(self) -> None:
        self.repository.close()
