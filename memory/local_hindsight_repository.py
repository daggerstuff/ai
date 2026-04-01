from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from .hindsight_local_domain import resolve_user_id_from_record
from .local_hindsight_db import LocalHindsightDatabase
from .local_hindsight_document_store import LocalHindsightDocumentStore
from .local_hindsight_queries import build_fts_query
from .local_hindsight_query_executor import LocalHindsightQueryExecutor
from .hindsight_local_adapter import normalize_tags


class LocalHindsightRepository:
    """Search orchestration and repository facade for local memory storage."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.db = LocalHindsightDatabase(db_path)
        self.documents = LocalHindsightDocumentStore(self.db)

    def upsert_document(self, **kwargs: Any) -> Dict[str, Any]:
        return self.documents.upsert_document(**kwargs)

    def upsert_documents(self, bank_id: str, items: List[Dict[str, Any]]) -> None:
        self.documents.upsert_documents(bank_id, items)

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.documents.get_document(bank_id, document_id, user_id=user_id)

    def list_documents(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.documents.list_documents(bank_id, limit=limit, offset=offset)

    def list_documents_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
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
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.documents.list_documents_for_scope(
            bank_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            include_shared=include_shared,
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
    ) -> List[Dict[str, Any]]:
        return self.documents.list_documents_for_user_by_category(
            bank_id,
            user_id=user_id,
            category=category,
            limit=limit,
            offset=offset,
        )

    def delete_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> bool:
        return self.delete_documents(bank_id, [document_id], user_id=user_id) > 0

    def delete_documents(
        self,
        bank_id: str,
        document_ids: List[str],
        *,
        user_id: Optional[str] = None,
    ) -> int:
        return self.documents.delete_documents(bank_id, document_ids, user_id=user_id)

    def _recall_rows(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        query: str,
        fetch_limit: int,
        normalized_tags: List[str],
        required_tags: List[str],
        tags_match: str,
    ) -> List[sqlite3.Row]:
        rows: List[sqlite3.Row] = []
        fts_query = build_fts_query(query)
        if fts_query:
            try:
                rows = LocalHindsightQueryExecutor.execute_fts_query(
                conn,
                bank_id=bank_id,
                fts_query=fts_query,
                fetch_limit=fetch_limit,
                tags=normalized_tags,
                required_tags=required_tags,
                tags_match=tags_match,
            )
            except sqlite3.OperationalError:
                rows = []
        if rows:
            return rows

        return LocalHindsightQueryExecutor.execute_like_query(
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
        tags: Optional[List[str]] = None,
        required_tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> List[Dict[str, Any]]:
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

    def delete_documents_for_user(self, bank_id: str, *, user_id: str) -> bool:
        return self.documents.delete_documents_for_user(bank_id, user_id=user_id)

    def resolve_user_id(self, record: Dict[str, Any]) -> Optional[str]:
        return resolve_user_id_from_record(record)

    def close(self) -> None:
        self.db.close()
