from __future__ import annotations

from typing import Any


class LocalMemoryCompatibilityMixin:
    """Typed Foresight/document facade shared by the local memory manager."""

    default_bank_id: str
    protocol: Any

    def list_documents(
        self,
        bank_id: str,
        *,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required when listing documents")
        return self.protocol.list_documents(
            bank_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def retain_items(self, bank_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.protocol.retain_items(bank_id, items)

    def recall(
        self,
        bank_id: str,
        *,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
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
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> dict[str, Any]:
        return self.protocol.recall_for_user(
            bank_id,
            user_id=user_id,
            query=query,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
        )

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not user_id:
            raise ValueError("user_id is required when fetching documents")
        return self.protocol.get_document(bank_id, document_id, user_id=user_id)

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
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
        return self.protocol.can_write_document(
            bank_id,
            document_id,
            user_id=user_id,
        )

    def prepare_retained_items(
        self,
        *,
        bank_id: str,
        items: list[dict[str, Any]],
        user_id: str,
        base_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self.protocol.prepare_retained_items(
            bank_id=bank_id,
            items=items,
            user_id=user_id,
            base_metadata=base_metadata,
        )

    def search_memories(self, query: str, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        response = self.recall_for_user(
            self.default_bank_id,
            user_id=user_id,
            query=query,
            limit=limit,
            tags=[f"user:{user_id}"],
            tags_match="any",
        )
        results = response.get("results", [])
        return results if isinstance(results, list) else []
