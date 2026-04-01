from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .hindsight_local_adapter import encode_tags_json, normalize_tags
from .hindsight_local_domain import NON_PRIVATE_VISIBILITY_TAGS, resolve_user_id_from_context
from .local_hindsight_db import LocalHindsightDatabase
from .local_hindsight_query_executor import LocalHindsightQueryExecutor


class LocalHindsightDocumentStore:
    """CRUD and batch document operations for the local memory store."""

    def __init__(self, db: LocalHindsightDatabase) -> None:
        self.db = db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def parse_row(row: sqlite3.Row) -> Dict[str, Any]:
        raw_tags = row["tags_json"] or "[]"
        try:
            tags = json.loads(raw_tags)
        except json.JSONDecodeError:
            tags = []
        if not isinstance(tags, list):
            tags = []
        return {
            "bank_id": row["bank_id"],
            "id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "context": row["context"] or "",
            "tags": [str(tag) for tag in tags if tag],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _replace_tags_tx(
        self,
        conn: sqlite3.Connection,
        bank_id: str,
        document_id: str,
        tags: List[str],
    ) -> None:
        conn.execute(
            "DELETE FROM document_tags WHERE bank_id = ? AND document_id = ?",
            (bank_id, document_id),
        )
        if tags:
            conn.executemany(
                "INSERT INTO document_tags(bank_id, document_id, tag) VALUES (?, ?, ?)",
                [(bank_id, document_id, tag) for tag in tags],
            )

    def _upsert_document_tx(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        document_id: str,
        content: str,
        context: Optional[str],
        tags: Optional[Iterable[str]],
    ) -> None:
        now = self._now()
        user_id = resolve_user_id_from_context(context)
        normalized = normalize_tags(tags)
        conn.execute(
            """
            INSERT INTO documents(bank_id, id, user_id, content, context, tags_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bank_id, id) DO UPDATE SET
                user_id = excluded.user_id,
                content = excluded.content,
                context = excluded.context,
                tags_json = excluded.tags_json,
                updated_at = excluded.updated_at
            """,
            (
                bank_id,
                document_id,
                user_id,
                content,
                context or "",
                encode_tags_json(normalized),
                now,
                now,
            ),
        )
        self._replace_tags_tx(conn, bank_id, document_id, normalized)

    def upsert_document(
        self,
        *,
        bank_id: str,
        document_id: str,
        content: str,
        context: Optional[str],
        tags: Optional[Iterable[str]],
    ) -> Dict[str, Any]:
        normalized = normalize_tags(tags)
        with self.db.lease() as conn:
            self._upsert_document_tx(
                conn,
                bank_id=bank_id,
                document_id=document_id,
                content=content,
                context=context,
                tags=normalized,
            )
        return {
            "bank_id": bank_id,
            "id": document_id,
            "user_id": resolve_user_id_from_context(context),
            "content": content,
            "context": context or "",
            "tags": normalized,
        }

    def upsert_documents(self, bank_id: str, items: List[Dict[str, Any]]) -> None:
        with self.db.lease() as conn:
            for item in items:
                self._upsert_document_tx(
                    conn,
                    bank_id=bank_id,
                    document_id=item["document_id"],
                    content=item["content"],
                    context=item.get("context"),
                    tags=item.get("tags"),
                )

    def get_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self.db.lease() as conn:
            cursor = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE bank_id = ? AND id = ?
                  AND (? IS NULL OR user_id = ?)
                LIMIT 1
                """,
                (bank_id, document_id, user_id, user_id),
            )
            row = cursor.fetchone()
        return self.parse_row(row) if row else None

    def list_documents(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.db.lease() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE bank_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (bank_id, limit, offset),
            ).fetchall()
        return [self.parse_row(row) for row in rows]

    def list_documents_for_user(
        self,
        bank_id: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.db.lease() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE bank_id = ?
                  AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (bank_id, user_id, limit, offset),
            ).fetchall()
        return [self.parse_row(row) for row in rows]

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
        with self.db.lease() as conn:
            rows = LocalHindsightQueryExecutor.execute_scope_query(
                conn,
                bank_id=bank_id,
                user_id=user_id,
                org_id=org_id,
                project_id=project_id,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                include_shared=include_shared,
                non_private_visibility_tags=list(NON_PRIVATE_VISIBILITY_TAGS),
                limit=limit,
                offset=offset,
            )
        return [self.parse_row(row) for row in rows]

    def list_documents_for_user_by_category(
        self,
        bank_id: str,
        *,
        user_id: str,
        category: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.db.lease() as conn:
            rows = conn.execute(
                """
                SELECT d.*
                FROM documents d
                INNER JOIN document_tags t
                    ON t.bank_id = d.bank_id
                   AND t.document_id = d.id
                WHERE d.bank_id = ?
                  AND d.user_id = ?
                  AND t.tag = ?
                ORDER BY d.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (bank_id, user_id, f"category:{category}", limit, offset),
            ).fetchall()
        return [self.parse_row(row) for row in rows]

    def delete_documents(
        self,
        bank_id: str,
        document_ids: List[str],
        *,
        user_id: Optional[str] = None,
    ) -> int:
        if not document_ids:
            return 0
        payload = json.dumps(document_ids, separators=(",", ":"))
        with self.db.lease() as conn:
            conn.execute(
                """
                DELETE FROM document_tags
                WHERE bank_id = ?
                  AND document_id IN (
                    SELECT id
                    FROM documents
                    WHERE bank_id = ?
                      AND id IN (SELECT value FROM json_each(?))
                      AND (? IS NULL OR user_id = ?)
                  )
                """,
                (bank_id, bank_id, payload, user_id, user_id),
            )
            cursor = conn.execute(
                """
                DELETE FROM documents
                WHERE bank_id = ?
                  AND id IN (SELECT value FROM json_each(?))
                  AND (? IS NULL OR user_id = ?)
                """,
                (bank_id, payload, user_id, user_id),
            )
            return cursor.rowcount

    def delete_documents_for_user(self, bank_id: str, *, user_id: str) -> bool:
        user_tag = f"user:{user_id}"
        with self.db.lease() as conn:
            conn.execute(
                """
                DELETE FROM document_tags
                WHERE bank_id = ?
                  AND document_id IN (
                    SELECT id
                    FROM documents
                    WHERE bank_id = ?
                      AND (user_id = ? OR id IN (
                        SELECT document_id FROM document_tags WHERE bank_id = ? AND tag = ?
                      ))
                  )
                """,
                (bank_id, bank_id, user_id, bank_id, user_tag),
            )
            deleted = conn.execute(
                """
                DELETE FROM documents
                WHERE bank_id = ?
                  AND (user_id = ? OR id IN (
                    SELECT document_id FROM document_tags WHERE bank_id = ? AND tag = ?
                  ))
                """,
                (bank_id, user_id, bank_id, user_tag),
            )
            return deleted.rowcount > 0
