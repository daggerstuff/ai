from __future__ import annotations

from typing import Any

from .base import BaseMemoryManager
from .local_hindsight_compat_mixin import LocalHindsightCompatibilityMixin
from .local_hindsight_document_service import LocalHindsightDocumentService
from .local_hindsight_memory_query_service import LocalHindsightMemoryQueryService
from .local_hindsight_memory_record_service import LocalHindsightMemoryRecordService
from .local_hindsight_memory_write_service import LocalHindsightMemoryWriteService
from .local_hindsight_protocol_adapter import LocalHindsightProtocolAdapter
from .local_hindsight_repository import LocalHindsightRepository


class LocalHindsightMemoryManager(LocalHindsightCompatibilityMixin, BaseMemoryManager):
    """Persistent local Hindsight-compatible store backed by SQLite."""

    def __init__(
        self,
        db_path: str,
        bank_id: str | None = None,
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
            document_store=self.repository.documents,
            writes=self.writes,
            provider_name=self.get_provider_name(),
        )
        self.queries = LocalHindsightMemoryQueryService(
            default_bank_id=self.default_bank_id,
            repository=self.repository,
        )

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str:
        return self.writes.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )

    def add_memory_scoped(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
        scope_metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.writes.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
            scope_metadata=scope_metadata,
        )

    def get_all_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.queries.get_all_memories(user_id=user_id, limit=limit)

    def get_memory(self, memory_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        return self.records.get_memory(memory_id, user_id=user_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Any | None = None,
        user_id: str | None = None,
    ) -> bool:
        return self.records.update_memory(
            memory_id=memory_id,
            new_content=new_content,
            metadata=metadata,
            user_id=user_id,
        )

    def delete_memory(self, memory_id: str, user_id: str | None = None) -> bool:
        return self.records.delete_memory(
            memory_id=memory_id,
            user_id=user_id,
        )

    def _delete_memories(self, memory_ids: list[str], user_id: str | None = None) -> int:
        return self.records.delete_memories(
            memory_ids,
            user_id=user_id,
        )

    def clear_memory(self, user_id: str) -> bool:
        return self.records.clear_memory(user_id)

    def get_health_status(self) -> dict[str, Any]:
        return self.records.get_health_status()

    def close(self) -> None:
        self.repository.close()
