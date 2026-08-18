from __future__ import annotations

from typing import Any

from .foresight_local_adapter import memory_record_from_storage, metadata_to_tags, serialize_context
from .local_foresight_document_store import LocalForesightDocumentStore
from .local_foresight_memory_update import build_updated_document_payload
from .local_foresight_memory_write_service import LocalForesightMemoryWriteService


class LocalForesightMemoryRecordService:
    """Mutation and record-level retrieval for local memories."""

    def __init__(
        self,
        *,
        default_bank_id: str,
        document_store: LocalForesightDocumentStore,
        writes: LocalForesightMemoryWriteService,
        provider_name: str,
    ) -> None:
        self.default_bank_id = default_bank_id
        self.document_store = document_store
        self.writes = writes
        self.provider_name = provider_name

    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        record = self.document_store.get_document(
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
        metadata: Any | None = None,
        user_id: str | None = None,
    ) -> bool:
        existing_record = self.document_store.get_document(
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
        self.document_store.upsert_document(
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

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        deleted_count = self.document_store.delete_documents(
            self.default_bank_id,
            [memory_id],
            user_id=user_id,
        )
        return deleted_count > 0

    def delete_memories(self, memory_ids: list[str], user_id: str | None = None) -> int:
        return self.document_store.delete_documents(
            self.default_bank_id,
            memory_ids,
            user_id=user_id,
        )

    def clear_memory(self, user_id: str) -> bool:
        return self.document_store.delete_documents_for_user(self.default_bank_id, user_id=user_id)

    def get_health_status(self) -> dict[str, Any]:
        db_health = self.document_store.db.health_details()
        return {
            "status": "healthy" if db_health.get("db_ready") else "degraded",
            "provider": self.provider_name,
            "readiness": db_health,
        }
