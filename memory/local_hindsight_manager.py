from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .base import BaseMemoryManager
from .hindsight_local_adapter import (
    memory_record_from_storage,
    metadata_to_tags,
    serialize_context,
)
from .local_hindsight_document_service import LocalHindsightDocumentService
from .local_hindsight_memory_query_service import LocalHindsightMemoryQueryService
from .local_hindsight_memory_update import build_updated_document_payload
from .local_hindsight_protocol_adapter import LocalHindsightProtocolAdapter
from .local_hindsight_repository import LocalHindsightRepository


class LocalHindsightMemoryManager(BaseMemoryManager):
    """Persistent local Hindsight-compatible store backed by SQLite."""

    @staticmethod
    def _metadata_dict(metadata: Optional[Any]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if hasattr(metadata, "to_dict") and callable(metadata.to_dict):
            data = metadata.to_dict()
            return dict(data) if isinstance(data, dict) else {}
        raise TypeError("metadata must be a mapping or expose to_dict()")

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
        self.queries = LocalHindsightMemoryQueryService(
            default_bank_id=self.default_bank_id,
            repository=self.repository,
        )

    def retain_items(self, bank_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.protocol.retain_items(bank_id, items)

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> Dict[str, Any]:
        return self.protocol.recall(
            bank_id,
            query=query,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
        )

    def recall_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> Dict[str, Any]:
        return self.protocol.recall_for_user(
            bank_id,
            user_id=user_id,
            query=query,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
        )

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

    def can_write_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        return self.protocol.can_write_document(bank_id, document_id, user_id=user_id)

    def prepare_retained_items(
        self,
        *,
        bank_id: str,
        items: List[Dict[str, Any]],
        user_id: str,
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return self.protocol.prepare_retained_items(
            bank_id=bank_id,
            items=items,
            user_id=user_id,
            base_metadata=base_metadata,
        )

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Any] = None,
        category: Optional[str] = None,
    ) -> str:
        merged = self._metadata_dict(metadata)
        if category:
            merged["category"] = category
        retained = self.retain_items(
            self.default_bank_id,
            [self.protocol.build_add_memory_item(user_id=user_id, content=content, metadata=merged)],
        )
        results = retained.get("results")
        if not isinstance(results, list) or not results:
            raise RuntimeError("Retain operation returned no document identifiers")
        first = results[0]
        if not isinstance(first, dict):
            raise RuntimeError("Retain operation returned an invalid document payload")
        document_id = first.get("id")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("Retain operation did not provide a valid document identifier")
        return document_id

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
        return self.queries.get_all_memories_scoped(
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            limit=limit,
        )

    def get_memories_by_category(
        self,
        user_id: str,
        category: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.queries.get_memories_by_category(
            user_id=user_id,
            category=category,
            limit=limit,
        )

    def get_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        record = self.repository.get_document(
            self.default_bank_id,
            memory_id,
            user_id=user_id,
        )
        if not record:
            return None
        return memory_record_from_storage(record)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        existing_record = self.repository.get_document(
            self.default_bank_id,
            memory_id,
            user_id=user_id,
        )
        if existing_record is None:
            return False
        owner_user_id, merged_metadata, category = build_updated_document_payload(
            repository=self.repository,
            existing_record=existing_record,
            metadata=self._metadata_dict(metadata),
        )
        self.repository.upsert_document(
            bank_id=self.default_bank_id,
            document_id=memory_id,
            content=new_content,
            context=serialize_context(
                user_id=owner_user_id,
                metadata=merged_metadata,
                category=category,
            ),
            tags=metadata_to_tags(
                user_id=owner_user_id,
                metadata=merged_metadata,
                category=category,
            ),
        )
        return True

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        deleted_count = self.repository.delete_documents(
            self.default_bank_id,
            [memory_id],
            user_id=user_id,
        )
        return deleted_count > 0

    def delete_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> int:
        """Delete multiple memories and return the number of removed records."""
        return self.repository.delete_documents(
            self.default_bank_id,
            memory_ids,
            user_id=user_id,
        )

    def clear_memory(self, user_id: str) -> bool:
        return self.repository.delete_documents_for_user(self.default_bank_id, user_id=user_id)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "provider": self.get_provider_name(),
        }
