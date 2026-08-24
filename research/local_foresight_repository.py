from __future__ import annotations

import sqlite3
from typing import Any

from .foresight_local_adapter import normalize_tags
from .foresight_local_domain import NON_PRIVATE_VISIBILITY_TAGS, resolve_user_id_from_record
from .local_foresight_db import LocalForesightDatabase
from .local_foresight_document_store import LocalForesightDocumentStore
from .local_foresight_queries import build_fts_query
from .local_foresight_query_executor import LocalForesightQueryExecutor


class LocalForesightRepository:
    """Search orchestration and repository facade for local memory storage."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.db = LocalForesightDatabase(db_path)
        self.documents = LocalForesightDocumentStore(self.db)

    def upsert_document(self, **kwargs: Any) -> dict[str, Any]:
        return self.documents.upsert_document(**kwargs)

    def upsert_documents(self, bank_id: str, items: list[dict[str, Any]]) -> None:
        self.documents.upsert_documents(bank_id, items)

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.documents.get_document(bank_id, document_id, user_id=user_id)

    def list_documents(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.documents.list_documents(bank_id, limit=limit, offset=offset)

    def list_documents_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.documents.list_documents_for_user(
            bank_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def list_documents_for_scope(
        self,
        bank_id: str,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.documents.list_documents_for_scope(
            bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
            category=category,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    def list_documents_for_user_by_category(
        self,
        bank_id: str,
        *,
        user_id: str,
        category: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.documents.list_documents_for_user_by_category(
            bank_id,
            user_id=user_id,
            category=category,
            limit=limit,
            offset=offset,
        )

    def count_documents_by_category_for_scope(
        self,
        bank_id: str,
        *,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> dict[str, int]:
        return self.documents.count_documents_by_category_for_scope(
            bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
        )

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: str | None = None,
    ) -> bool:
        return self.delete_documents(bank_id, [document_id], user_id=user_id) > 0

    def delete_documents(
        self,
        bank_id: str,
        document_ids: list[str],
        *,
        user_id: str | None = None,
    ) -> int:
        return self.documents.delete_documents(bank_id, document_ids, user_id=user_id)

    def _recall_rows(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        query: str,
        fetch_limit: int,
        normalized_tags: list[str],
        required_tags: list[str],
        tags_match: str,
    ) -> list[sqlite3.Row]:
        fts_query = build_fts_query(query)
        if fts_query:
            rows = self._recall_rows_via_fts(
                conn,
                bank_id=bank_id,
                fts_query=fts_query,
                fetch_limit=fetch_limit,
                normalized_tags=normalized_tags,
                required_tags=required_tags,
                tags_match=tags_match,
            )
            if rows:
                return rows

        return self._recall_rows_via_like(
            conn,
            bank_id=bank_id,
            query=query,
            fetch_limit=fetch_limit,
            normalized_tags=normalized_tags,
            required_tags=required_tags,
            tags_match=tags_match,
        )

    def _recall_rows_via_fts(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        fts_query: str,
        fetch_limit: int,
        normalized_tags: list[str],
        required_tags: list[str],
        tags_match: str,
    ) -> list[sqlite3.Row]:
        try:
            return LocalForesightQueryExecutor.execute_fts_query(
                conn,
                bank_id=bank_id,
                fts_query=fts_query,
                fetch_limit=fetch_limit,
                tags=normalized_tags,
                required_tags=required_tags,
                tags_match=tags_match,
            )
        except sqlite3.OperationalError:
            return []

    def _recall_rows_via_like(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        query: str,
        fetch_limit: int,
        normalized_tags: list[str],
        required_tags: list[str],
        tags_match: str,
    ) -> list[sqlite3.Row]:
        return LocalForesightQueryExecutor.execute_like_query(
            conn,
            bank_id=bank_id,
            query=query,
            fetch_limit=fetch_limit,
            tags=normalized_tags,
            required_tags=required_tags,
            tags_match=tags_match,
        )

    def recall_documents(
        self,
        bank_id: str,
        *,
        query: str,
        fetch_limit: int,
        tags: list[str] | None = None,
        required_tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> list[dict[str, Any]]:
        normalized_tags = normalize_tags(tags)
        normalized_required_tags = normalize_tags(required_tags)
        with self.db.lease() as conn:
            rows = self._recall_rows(
                conn,
                bank_id=bank_id,
                query=query,
                fetch_limit=fetch_limit,
                normalized_tags=normalized_tags,
                required_tags=normalized_required_tags,
                tags_match=tags_match,
            )

        documents = []
        for row in rows:
            parsed = self.documents.parse_row(row)
            parsed["rank"] = float(row["rank"]) if row["rank"] is not None else 999.0
            documents.append(parsed)
        return documents

    def search_documents_for_scope(
        self,
        bank_id: str,
        *,
        user_id: str,
        query: str,
        limit: int,
        offset: int = 0,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
    ) -> list[dict[str, Any]]:
        with self.db.lease() as conn:
            rows = LocalForesightQueryExecutor.execute_scoped_search_query(
                conn,
                bank_id=bank_id,
                user_id=user_id,
                query=query,
                fetch_limit=limit,
                offset=offset,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
                non_private_visibility_tags=list(NON_PRIVATE_VISIBILITY_TAGS),
            )

        documents = []
        for row in rows:
            parsed = self.documents.parse_row(row)
            parsed["rank"] = float(row["rank"]) if row["rank"] is not None else 999.0
            documents.append(parsed)
        return documents

    def delete_documents_for_user(self, bank_id: str, *, user_id: str) -> bool:
        return self.documents.delete_documents_for_user(bank_id, user_id=user_id)

    def resolve_user_id(self, record: dict[str, Any]) -> str | None:
        return resolve_user_id_from_record(record)

    def close(self) -> None:
        self.db.close()
