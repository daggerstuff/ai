import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from .foresight_local_adapter import encode_tags_json, normalize_tags
from .foresight_local_domain import NON_PRIVATE_VISIBILITY_TAGS, resolve_user_id_from_context
from .local_foresight_db import LocalForesightDatabase
from .local_foresight_query_builders import build_scope_category_count_query, scope_tags
from .local_foresight_query_executor import LocalForesightQueryExecutor


class LocalForesightDocumentStore:
    """CRUD and batch document operations for the local memory store."""

    def __init__(self, db: LocalForesightDatabase) -> None:
        self.db = db

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def parse_row(row: sqlite3.Row) -> dict[str, Any]:
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
        tags: list[str],
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

    @staticmethod
    def _json_payload(values: list[str]) -> str:
        return json.dumps(values, separators=(",", ":"))

    def _document_ids_matching_payload(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        payload: str,
        user_id: str | None = None,
    ) -> list[str]:
        rows = conn.execute(
            """
            SELECT id
            FROM documents
            WHERE bank_id = ?
              AND id IN (SELECT value FROM json_each(?))
              AND (? IS NULL OR user_id = ?)
            """,
            (bank_id, payload, user_id, user_id),
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def _delete_documents_by_ids_tx(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        document_ids: list[str],
    ) -> int:
        if not document_ids:
            return 0
        payload = self._json_payload(document_ids)
        conn.execute(
            """
            DELETE FROM document_tags
            WHERE bank_id = ?
              AND document_id IN (SELECT value FROM json_each(?))
            """,
            (bank_id, payload),
        )
        cursor = conn.execute(
            """
            DELETE FROM documents
            WHERE bank_id = ?
              AND id IN (SELECT value FROM json_each(?))
            """,
            (bank_id, payload),
        )
        return cursor.rowcount

    def _upsert_document_tx(
        self,
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        document_id: str,
        content: str,
        context: str | None,
        tags: Iterable[str] | None,
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
        context: str | None,
        tags: Iterable[str] | None,
    ) -> dict[str, Any]:
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

    def upsert_documents(self, bank_id: str, items: list[dict[str, Any]]) -> None:
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
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
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
    ) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
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
        required_tags = [*normalize_tags(tags)]
        if category:
            required_tags.append(f"category:{category}")
        with self.db.lease() as conn:
            rows = LocalForesightQueryExecutor.execute_scope_query(
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
                required_tags=required_tags,
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
    ) -> list[dict[str, Any]]:
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
        required_scope_tags = scope_tags(
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        built_query = build_scope_category_count_query(
            bank_id=bank_id,
            user_id=user_id,
            required_scope_tags=required_scope_tags,
            include_shared=include_shared,
            non_private_visibility_tags=list(NON_PRIVATE_VISIBILITY_TAGS),
        )
        with self.db.lease() as conn:
            rows = conn.execute(built_query.sql, built_query.params).fetchall()
        return {str(row["category"]): int(row["total"]) for row in rows}

    def delete_documents(
        self,
        bank_id: str,
        document_ids: list[str],
        *,
        user_id: str | None = None,
    ) -> int:
        if not document_ids:
            return 0
        with self.db.lease() as conn:
            payload = self._json_payload(document_ids)
            matching_ids = self._document_ids_matching_payload(
                conn,
                bank_id=bank_id,
                payload=payload,
                user_id=user_id,
            )
            return self._delete_documents_by_ids_tx(
                conn,
                bank_id=bank_id,
                document_ids=matching_ids,
            )

    def delete_documents_for_user(self, bank_id: str, *, user_id: str) -> bool:
        user_tag = f"user:{user_id}"
        with self.db.lease() as conn:
            conn.execute(
                """
                DELETE FROM document_tags
                WHERE bank_id = ?
                  AND EXISTS (
                    SELECT 1
                    FROM documents d
                    LEFT JOIN document_tags owner_tag
                      ON owner_tag.bank_id = d.bank_id
                     AND owner_tag.document_id = d.id
                     AND owner_tag.tag = ?
                    WHERE d.bank_id = document_tags.bank_id
                      AND d.id = document_tags.document_id
                      AND (d.user_id = ? OR owner_tag.document_id IS NOT NULL)
                  )
                """,
                (bank_id, user_tag, user_id),
            )
            deleted = conn.execute(
                """
                DELETE FROM documents
                WHERE bank_id = ?
                  AND (
                    user_id = ?
                    OR EXISTS (
                        SELECT 1
                        FROM document_tags t
                        WHERE t.bank_id = documents.bank_id
                          AND t.document_id = documents.id
                          AND t.tag = ?
                    )
                  )
                """,
                (bank_id, user_id, user_tag),
            )
            return deleted.rowcount > 0
