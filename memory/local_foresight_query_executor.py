from __future__ import annotations

import sqlite3

from .foresight_local_adapter import normalize_tags
from .local_foresight_queries import build_fts_query
from .local_foresight_query_builders import (
    base_query,
    build_scope_listing_query,
    build_scoped_search_query,
    documents_source,
    filtered_params,
    filtered_query,
    like_query_value,
    scope_tags,
)


class LocalForesightQueryExecutor:
    """Execute prepared local memory queries against SQLite."""

    @staticmethod
    def execute_fts_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        fts_query: str,
        fetch_limit: int,
        tags: list[str],
        required_tags: list[str] | None = None,
        tags_match: str,
    ) -> list[sqlite3.Row]:
        if not fts_query:
            return []
        source = documents_source(fts=True)
        required = normalize_tags(required_tags)
        optional = normalize_tags(tags)
        if not required and not optional:
            query_sql = base_query(fts=True, source=source)
            params = (bank_id, fts_query, fetch_limit)
            return conn.execute(query_sql, params).fetchall()

        query_sql = filtered_query(
            fts=True,
            source=source,
            required_tags=required,
            optional_tags=optional,
            optional_match=tags_match,
        )
        params = (
            bank_id,
            fts_query,
            *filtered_params(
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
        tags: list[str],
        required_tags: list[str] | None = None,
        tags_match: str,
    ) -> list[sqlite3.Row]:
        like = like_query_value(query)
        source = documents_source(fts=False)
        required = normalize_tags(required_tags)
        optional = normalize_tags(tags)
        if not required and not optional:
            query_sql = base_query(fts=False, source=source)
            params = (bank_id, like, fetch_limit)
            return conn.execute(query_sql, params).fetchall()

        query_sql = filtered_query(
            fts=False,
            source=source,
            required_tags=required,
            optional_tags=optional,
            optional_match=tags_match,
        )
        params = (
            bank_id,
            like,
            *filtered_params(
                required_tags=required,
                optional_tags=optional,
                optional_match=tags_match,
            ),
            fetch_limit,
        )
        return conn.execute(query_sql, params).fetchall()

    @staticmethod
    def execute_scoped_search_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        user_id: str,
        query: str,
        fetch_limit: int,
        offset: int = 0,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        non_private_visibility_tags: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        fts_query = build_fts_query(query)
        fts = bool(fts_query)
        query_value = fts_query if fts else like_query_value(query)
        source = documents_source(fts=fts)
        required_scope_tags = scope_tags(
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        built_query = build_scoped_search_query(
            fts=fts,
            source=source,
            user_id=user_id,
            required_scope_tags=required_scope_tags,
            include_shared=include_shared,
            non_private_visibility_tags=non_private_visibility_tags,
            fetch_limit=fetch_limit,
            offset=offset,
            leading_params=[bank_id, query_value],
        )
        return conn.execute(built_query.sql, built_query.params).fetchall()

    @staticmethod
    def execute_scope_query(
        conn: sqlite3.Connection,
        *,
        bank_id: str,
        user_id: str,
        org_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_shared: bool = True,
        non_private_visibility_tags: list[str] | None = None,
        required_tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        required_scope_tags = scope_tags(
            org_id=org_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        built_query = build_scope_listing_query(
            bank_id=bank_id,
            user_id=user_id,
            required_scope_tags=required_scope_tags,
            required_filter_tags=normalize_tags(required_tags),
            include_shared=include_shared,
            non_private_visibility_tags=non_private_visibility_tags,
            limit=limit,
            offset=offset,
        )
        return conn.execute(built_query.sql, built_query.params).fetchall()
