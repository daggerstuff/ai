from __future__ import annotations

from typing import Any

from .local_foresight_document_service import LocalForesightDocumentService


class LocalForesightProtocolAdapter:
    """Protocol-facing adapter for Foresight-compatible document operations."""

    def __init__(
        self,
        *,
        default_bank_id: str,
        documents: LocalForesightDocumentService,
    ) -> None:
        self.default_bank_id = default_bank_id
        self.documents = documents

    def retain_items(self, bank_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.documents.retain_items(bank_id, items)

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        return self.documents.recall(
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
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        return self.documents.recall_for_user(
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
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self.documents.list_documents(
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
        user_id: str,
    ) -> dict[str, Any] | None:
        return self.documents.get_document(bank_id, document_id, user_id=user_id)

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        return self.documents.delete_document(bank_id, document_id, user_id=user_id)

    def can_write_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str,
    ) -> bool:
        return self.documents.can_write_document(bank_id, document_id, user_id=user_id)

    def prepare_retained_items(
        self,
        *,
        bank_id: str,
        items: list[dict[str, Any]],
        user_id: str,
        base_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self.documents.prepare_retained_items(
            bank_id=bank_id,
            items=items,
            user_id=user_id,
            base_metadata=base_metadata,
        )

    def build_add_memory_item(
        self,
        *,
        user_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self.documents.build_add_memory_item(
            user_id=user_id,
            content=content,
            metadata=metadata,
        )
