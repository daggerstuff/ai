from __future__ import annotations

from typing import Any, Dict, List, Optional

from .hindsight_local_adapter import memory_record_from_storage, metadata_to_tags, serialize_context
from .local_hindsight_document_store import LocalHindsightDocumentStore
from .local_hindsight_memory_update import build_updated_document_payload
from .local_hindsight_memory_write_service import LocalHindsightMemoryWriteService


class LocalHindsightMemoryRecordService:
    """Mutation and record-level retrieval for local memories."""

    def __init__(
        self,
        *,
        default_bank_id: str,
        documents: LocalHindsightDocumentStore,
        writes: LocalHindsightMemoryWriteService,
        provider_name: str,
    ) -> None:
        self.default_bank_id = default_bank_id
        self.documents = documents
        self.writes = writes
        self.provider_name = provider_name

    def get_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        record = self.documents.get_document(
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
        existing_record = self.documents.get_document(
            self.default_bank_id,
            memory_id,
            user_id=user_id,
        )
        if existing_record is None:
            return False
        owner_user_id, merged_metadata, category = build_updated_document_payload(
            existing_record=existing_record,
            metadata=self.writes.coerce_metadata(metadata),
        )
        self.documents.upsert_document(
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
        deleted_count = self.documents.delete_documents(
            self.default_bank_id,
            [memory_id],
            user_id=user_id,
        )
        return deleted_count > 0

    def delete_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> int:
        return self.documents.delete_documents(
            self.default_bank_id,
            memory_ids,
            user_id=user_id,
        )

    def clear_memory(self, user_id: str) -> bool:
        return self.documents.delete_documents_for_user(self.default_bank_id, user_id=user_id)

    def get_health_status(self) -> Dict[str, Any]:
        db_health = self.documents.db.health_details()
        return {
            "status": "healthy" if db_health.get("db_ready") else "degraded",
            "provider": self.provider_name,
            "readiness": db_health,
        }
