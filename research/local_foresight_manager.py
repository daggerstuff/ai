from __future__ import annotations

from typing import Any

from .base import BaseMemoryManager
from .local_foresight_document_service import LocalForesightDocumentService
from .local_foresight_memory_query_service import LocalForesightMemoryQueryService
from .local_foresight_memory_record_service import LocalForesightMemoryRecordService
from .local_foresight_memory_write_service import LocalForesightMemoryWriteService
from .local_foresight_protocol_adapter import LocalForesightProtocolAdapter
from .local_foresight_repository import LocalForesightRepository
from .local_memory_compat_mixin import LocalMemoryCompatibilityMixin


class LocalForesightMemoryManager(LocalMemoryCompatibilityMixin, BaseMemoryManager):
    """Persistent local Foresight-compatible store backed by SQLite.
    Successor to the legacy Foresight implementation.
    """

    def __init__(
        self,
        db_path: str,
        bank_id: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.default_bank_id = bank_id or "pixelated"
        self.bank_id = self.default_bank_id
        self.repository = LocalForesightRepository(self.db_path)
        self.documents = LocalForesightDocumentService(self.repository)
        self.protocol = LocalForesightProtocolAdapter(
            default_bank_id=self.default_bank_id,
            documents=self.documents,
        )
        self.writes = LocalForesightMemoryWriteService(
            protocol=self.protocol,
            default_bank_id=self.default_bank_id,
        )
        self.records = LocalForesightMemoryRecordService(
            default_bank_id=self.default_bank_id,
            document_store=self.repository.documents,
            writes=self.writes,
            provider_name=self.get_provider_name(),
        )
        self.queries = LocalForesightMemoryQueryService(
            default_bank_id=self.default_bank_id,
            repository=self.repository,
        )

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str | None:
        doc_id, _report = self.writes.gated_add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
        )
        return doc_id

    def add_memory_scoped(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
        scope_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        doc_id, _report = self.writes.gated_add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            category=category,
            scope_metadata=scope_metadata,
        )
        return doc_id

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
