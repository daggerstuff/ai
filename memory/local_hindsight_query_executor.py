from __future__ import annotations

import sqlite3
from typing import Any, List, Optional

from .hindsight_local_adapter import encode_tags_json, normalize_tags
from .local_hindsight_queries import query_tokens


class LocalHindsightQueryExecutor:
    """Execute prepared local memory queries against SQLite."""

    MAX_SCOPE_TAGS = 5

    @staticmethod
    def _rank_column(fts: bool) -> str:
        return "bm25(documents_fts)" if fts else "999.0"

    @staticmethod
    def _documents_source(*, fts: bool) -> str:
        if fts:
            return """
                FROM documents_fts
                JOIN documents d
                  ON d.bank_id = documents_fts.bank_id
                 AND d.id = documents_fts.document_id
                WHERE documents_fts.bank_id = ?
                  AND documents_fts MATCH ?
            """
        return """
            FROM documents
            WHERE bank_id = ?
              AND lower(content) LIKE ?
        """

    @staticmethod
    def _base_query(*, fts: bool) -> str:
        table_alias = "d" if fts else "documents"
        rank_column = LocalHindsightQueryExecutor._rank_column(fts)
        order_column = "rank ASC" if fts else "updated_at DESC"
        return f"""
            SELECT {table_alias}.*, {rank_column} AS rank
            {{source}}
            ORDER BY {order_column}
            LIMIT ?
        """

    @staticmethod
    def _tag_requirements_clause(*, table_alias: str, tags_match: str) -> str:
        if tags_match == "all":
            return f"""
                {table_alias}.id IN (
                    SELECT t.document_id
                    FROM document_tags t
                    JOIN json_each(?) jt ON jt.value = t.tag
                    WHERE t.bank_id = {table_alias}.bank_id
                    GROUP BY t.document_id
                    HAVING COUNT(DISTINCT t.tag) = ?
                )
            """
        return f"""
            EXISTS (
                SELECT 1
                FROM document_tags t
                JOIN json_each(?) jt ON jt.value = t.tag
                WHERE t.bank_id = {table_alias}.bank_id
                  AND t.document_id = {table_alias}.id
            )
        """

    @staticmethod
    def _filtered_query(
        *,
        fts: bool,
        required_tags: List[str],
        optional_tags: List[str],
        optional_match: str,
    ) -> str:
        table_alias = "d" if fts else "documents"
        rank_column = LocalHindsightQueryExecutor._rank_column(fts)
        order_column = "rank ASC" if fts else "updated_at DESC"
        filters: List[str] = []
        if required_tags:
            filters.append(
                LocalHindsightQueryExecutor._tag_requirements_clause(
                    table_alias=table_alias,
                    tags_match="all",
                ).strip()
            )
        if optional_tags:
            filters.append(
                LocalHindsightQueryExecutor._tag_requirements_clause(
                    table_alias=table_alias,
                    tags_match=optional_match,
                ).strip()
            )
        filter_sql = ""
        if filters:
            filter_sql = "\n  AND " + "\n  AND ".join(filters)
        return f"""
            SELECT {table_alias}.*, {rank_column} AS rank
            {{source}}{filter_sql}
            ORDER BY {order_column}
            LIMIT ?
        """

    @staticmethod
    def _filtered_params(
        *,
        required_tags: List[str],
        optional_tags: List[str],
        optional_match: str,
    ) -> List[Any]:
        params: List[Any] = []
        if required_tags:
            params.append(encode_tags_json(required_tags))
            params.append(len(required_tags))
        if optional_tags:
            params.append(encode_tags_json(optional_tags))
            if optional_match == "all":
                params.append(len(optional_tags))
        return params

    @staticmethod
    def execute_fts_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        fts_query: str,
        fetch_limit: int,
        tags: List[str],
        required_tags: Optional[List[str]] = None,
        tags_match: str,
    ) -> List[sqlite3.Row]:
        if not fts_query:
            return []
        source = LocalHindsightQueryExecutor._documents_source(fts=True)
        required = normalize_tags(required_tags)
        optional = normalize_tags(tags)
        if not required and not optional:
            query_sql = LocalHindsightQueryExecutor._base_query(fts=True).format(source=source)
            params = (bank_id, fts_query, fetch_limit)
            return conn.execute(query_sql, params).fetchall()

        query_sql = LocalHindsightQueryExecutor._filtered_query(
            fts=True,
            required_tags=required,
            optional_tags=optional,
            optional_match=tags_match,
        ).format(source=source)
        params = (
            bank_id,
            fts_query,
            *LocalHindsightQueryExecutor._filtered_params(
                required_tags=required,
                optional_tags=optional,
                optional_match=tags_match,
            ),
            fetch_limit,
        )
        return conn.execute(query_sql, params).fetchall()

    @staticmethod
    def execute_like_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        query: str,
        fetch_limit: int,
        tags: List[str],
        required_tags: Optional[List[str]] = None,
        tags_match: str,
    ) -> List[sqlite3.Row]:
        tokens = query_tokens(query)
        like_seed = (tokens[0] if tokens else query).lower()
        like = f"{like_seed}%"
        source = LocalHindsightQueryExecutor._documents_source(fts=False)
        required = normalize_tags(required_tags)
        optional = normalize_tags(tags)
        if not required and not optional:
            query_sql = LocalHindsightQueryExecutor._base_query(fts=False).format(source=source)
            params = (bank_id, like, fetch_limit)
            return conn.execute(query_sql, params).fetchall()

        query_sql = LocalHindsightQueryExecutor._filtered_query(
            fts=False,
            required_tags=required,
            optional_tags=optional,
            optional_match=tags_match,
        ).format(source=source)
        params = (
            bank_id,
            like,
            *LocalHindsightQueryExecutor._filtered_params(
                required_tags=required,
                optional_tags=optional,
                optional_match=tags_match,
            ),
            fetch_limit,
        )
        return conn.execute(query_sql, params).fetchall()

    @staticmethod
    def execute_scope_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        user_id: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_shared: bool = True,
        non_private_visibility_tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[sqlite3.Row]:
        conditions = ["d.bank_id = ?", "d.user_id = ?"]
        params: List[Any] = [bank_id, user_id]
        required_scope_tags = LocalHindsightQueryExecutor._scope_tags(
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        if required_scope_tags:
            conditions.append(
                LocalHindsightQueryExecutor._required_tags_clause(len(required_scope_tags))
            )
            params.extend(required_scope_tags)
        if not include_shared and non_private_visibility_tags:
            conditions.append(
                LocalHindsightQueryExecutor._non_shared_visibility_clause(
                    len(non_private_visibility_tags)
                )
            )
            params.extend(non_private_visibility_tags)

        where_clause = " AND ".join(f"({condition.strip()})" for condition in conditions)
        query = f"""
            SELECT d.*
            FROM documents d
            WHERE {where_clause}
            ORDER BY d.updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        return conn.execute(query, tuple(params)).fetchall()

    @staticmethod
    def _scope_tags(
        *,
        org_id: Optional[str],
        project_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
    ) -> List[str]:
        tags: List[str] = []
        for value, prefix in (
            (org_id, "org_id"),
            (project_id, "project_id"),
            (session_id, "session_id"),
            (agent_id, "agent_id"),
            (run_id, "run_id"),
        ):
            if not value:
                continue
            tags.append(f"{prefix}:{value}")
        return normalize_tags(tags)

    @staticmethod
    def _required_tags_clause(tag_count: int) -> str:
        if tag_count < 1 or tag_count > LocalHindsightQueryExecutor.MAX_SCOPE_TAGS:
            raise ValueError("Scope tag count is outside the supported range")
        selects = [
            """
            SELECT t.document_id
            FROM document_tags t
            WHERE t.bank_id = d.bank_id
              AND t.tag = ?
            """.strip()
            for _ in range(tag_count)
        ]
        joined = "\nINTERSECT\n".join(selects)
        return f"""
            d.id IN (
                {joined}
            )
        """

    @staticmethod
    def _non_shared_visibility_clause(tag_count: int) -> str:
        placeholders = ", ".join("?" for _ in range(tag_count))
        return """
            NOT EXISTS (
                SELECT 1
                FROM document_tags t
                WHERE t.bank_id = d.bank_id
                  AND t.document_id = d.id
                  AND t.tag IN ({placeholders})
            )
        """.format(placeholders=placeholders)
