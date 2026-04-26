from __future__ import annotations

import uuid
from typing import Any

from .foresight_local_adapter import (
    foresight_document_from_storage,
    foresight_document_summary_from_storage,
    metadata_to_tags,
    normalize_tags,
    serialize_context,
)
from .foresight_local_retention import build_scoped_retain_items
from .local_foresight_repository import LocalForesightRepository
from .local_foresight_search import build_recall_results


class DocumentAccessError(Exception):
    """Raised when a caller attempts to operate on a document outside its scope."""


class LocalForesightDocumentService:
    """Foresight-compatible document operations backed by the local repository."""

    def __init__(self, repository: LocalForesightRepository) -> None:
        self.repository = repository

    def retain_items(self, bank_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        chunk_size = 250
        for start in range(0, len(items), chunk_size):
            prepared = []
            for item in items[start : start + chunk_size]:
                document_id = item.get("document_id") or f"local-{uuid.uuid4().hex}"
                prepared.append(
                    {
                        "document_id": document_id,
                        "content": item["content"],
                        "context": item.get("context"),
                        "tags": item.get("tags"),
                    }
                )
                results.append({"id": document_id})
            self.repository.upsert_documents(bank_id, prepared)
        return {"results": results}

    def prepare_retained_items(
        self,
        *,
        bank_id: str,
        items: list[dict[str, Any]],
        user_id: str,
        base_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return build_scoped_retain_items(
            items=items,
            user_id=user_id,
            base_metadata=base_metadata,
            ownership_validator=lambda document_id: self._ensure_document_write_access(
                bank_id=bank_id,
                document_id=document_id,
                user_id=user_id,
            ),
        )

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        documents = self.repository.recall_documents(
            bank_id,
            query=query,
            fetch_limit=limit,
            tags=normalize_tags(tags),
            tags_match=tags_match,
        )
        return {"results": build_recall_results(documents, limit=limit)}

    def recall_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        documents = self.repository.recall_documents(
            bank_id,
            query=query,
            fetch_limit=limit,
            tags=normalize_tags(tags),
            required_tags=[f"user:{user_id}"],
            tags_match=tags_match,
        )
        return {"results": build_recall_results(documents, limit=limit)}

    def list_documents(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return {
            "items": [
                foresight_document_summary_from_storage(record)
                for record in self.repository.list_documents_for_user(
                    bank_id,
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                )
            ]
        }

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        record = self.repository.get_document(bank_id, document_id, user_id=user_id)
        if not record:
            return None
        return foresight_document_from_storage(record)

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        return self.repository.delete_document(bank_id, document_id, user_id=user_id)

    def can_write_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        record = self.repository.get_document(bank_id, document_id)
        if record is None:
            return True
        return record.get("user_id") == user_id

    def _ensure_document_write_access(
        self,
        *,
        bank_id: str,
        document_id: str,
        user_id: str,
    ) -> None:
        if not self.can_write_document(bank_id, document_id, user_id=user_id):
            raise DocumentAccessError("Document not found")

    def build_add_memory_item(
        self,
        *,
        user_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        document_id = f"foresight-{user_id}-{uuid.uuid4().hex[:12]}"
        category = metadata.get("category")
        return {
            "content": content,
            "document_id": document_id,
            "context": serialize_context(
                user_id=user_id,
                metadata=metadata,
                category=category if isinstance(category, str) else None,
            ),
            "tags": metadata_to_tags(
                user_id=user_id,
                metadata=metadata,
                category=category if isinstance(category, str) else None,
            ),
        }
