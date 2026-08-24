from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .foresight_local_adapter import encode_tags_json, normalize_tags
from .local_foresight_queries import query_tokens


@dataclass(frozen=True)
class BuiltQuery:
    sql: str
    params: tuple[Any, ...]


@dataclass(frozen=True)
class QueryLayout:
    table_alias: str
    order_column: str
    rank_expression: str


def query_layout(*, fts: bool) -> QueryLayout:
    return QueryLayout(
        table_alias="d" if fts else "documents",
        order_column="rank ASC" if fts else "updated_at DESC",
        rank_expression=rank_column(fts=fts),
    )


def rank_column(*, fts: bool) -> str:
    return "bm25(documents_fts)" if fts else "999.0"


def documents_source(*, fts: bool) -> str:
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


def like_query_value(query: str) -> str:
    tokens = query_tokens(query)
    like_seed = (tokens[0] if tokens else query).lower()
    return f"{like_seed}%"


def base_query(*, fts: bool, source: str) -> str:
    layout = query_layout(fts=fts)
    return f"""
        SELECT {layout.table_alias}.*, {layout.rank_expression} AS rank
        {source}
        ORDER BY {layout.order_column}
        LIMIT ?
    """


def tag_requirements_clause(*, table_alias: str, tags_match: str) -> str:
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


def filtered_query(
    *,
    fts: bool,
    source: str,
    required_tags: list[str],
    optional_tags: list[str],
    optional_match: str,
) -> str:
    layout = query_layout(fts=fts)
    filters: list[str] = []
    if required_tags:
        filters.append(tag_requirements_clause(table_alias=layout.table_alias, tags_match="all").strip())
    if optional_tags:
        filters.append(
            tag_requirements_clause(
                table_alias=layout.table_alias,
                tags_match=optional_match,
            ).strip()
        )
    filter_sql = ""
    if filters:
        filter_sql = "\n  AND " + "\n  AND ".join(filters)
    return f"""
        SELECT {layout.table_alias}.*, {layout.rank_expression} AS rank
        {source}{filter_sql}
        ORDER BY {layout.order_column}
        LIMIT ?
    """


def filtered_params(
    *,
    required_tags: list[str],
    optional_tags: list[str],
    optional_match: str,
) -> list[Any]:
    params: list[Any] = []
    if required_tags:
        params.append(encode_tags_json(required_tags))
        params.append(len(required_tags))
    if optional_tags:
        params.append(encode_tags_json(optional_tags))
        if optional_match == "all":
            params.append(len(optional_tags))
    return params


def scope_tags(
    *,
    org_id: str | None,
    project_id: str | None,
    session_id: str | None,
    agent_id: str | None,
    run_id: str | None,
) -> list[str]:
    tags: list[str] = []
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


def required_tags_clause(tag_count: int) -> str:
    if tag_count < 1:
        raise ValueError("Scope tag count is outside the supported range")
    return f"""
        d.id IN (
            SELECT t.document_id
            FROM document_tags t
            JOIN json_each(?) jt ON jt.value = t.tag
            WHERE t.bank_id = d.bank_id
            GROUP BY t.document_id
            HAVING COUNT(DISTINCT t.tag) = {tag_count}
        )
    """


def non_shared_visibility_clause(tag_count: int) -> str:
    placeholders = ", ".join("?" for _ in range(tag_count))
    return f"""
        NOT EXISTS (
            SELECT 1
            FROM document_tags t
            WHERE t.bank_id = d.bank_id
              AND t.document_id = d.id
              AND t.tag IN ({placeholders})
        )
    """


def build_scoped_search_query(
    *,
    fts: bool,
    source: str,
    user_id: str,
    required_scope_tags: list[str],
    include_shared: bool,
    non_private_visibility_tags: list[str] | None,
    fetch_limit: int,
    offset: int,
    leading_params: list[Any],
) -> BuiltQuery:
    layout = query_layout(fts=fts)
    conditions = ["d.user_id = ?"]
    params: list[Any] = [*leading_params, user_id]
    _apply_scope_filters(
        conditions=conditions,
        params=params,
        required_scope_tags=required_scope_tags,
        required_filter_tags=[],
        include_shared=include_shared,
        non_private_visibility_tags=non_private_visibility_tags,
    )

    where_suffix = ""
    if conditions:
        where_suffix = "\n  AND " + "\n  AND ".join(f"({condition.strip()})" for condition in conditions)

    sql = f"""
        SELECT {layout.table_alias}.*, {layout.rank_expression} AS rank
        {source}{where_suffix}
        ORDER BY {layout.order_column}
        LIMIT ? OFFSET ?
    """
    params.extend([fetch_limit, offset])
    return BuiltQuery(sql=sql, params=tuple(params))


def build_scope_listing_query(
    *,
    bank_id: str,
    user_id: str,
    required_scope_tags: list[str],
    required_filter_tags: list[str] | None,
    include_shared: bool,
    non_private_visibility_tags: list[str] | None,
    limit: int,
    offset: int,
) -> BuiltQuery:
    conditions = ["d.bank_id = ?", "d.user_id = ?"]
    params: list[Any] = [bank_id, user_id]
    _apply_scope_filters(
        conditions=conditions,
        params=params,
        required_scope_tags=required_scope_tags,
        required_filter_tags=required_filter_tags or [],
        include_shared=include_shared,
        non_private_visibility_tags=non_private_visibility_tags,
    )
    where_clause = " AND ".join(f"({condition.strip()})" for condition in conditions)
    sql = f"""
        SELECT d.*
        FROM documents d
        WHERE {where_clause}
        ORDER BY d.updated_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    return BuiltQuery(sql=sql, params=tuple(params))


def build_scope_category_count_query(
    *,
    bank_id: str,
    user_id: str,
    required_scope_tags: list[str],
    include_shared: bool,
    non_private_visibility_tags: list[str] | None,
) -> BuiltQuery:
    conditions = ["d.bank_id = ?", "d.user_id = ?"]
    params: list[Any] = [bank_id, user_id]
    _apply_scope_filters(
        conditions=conditions,
        params=params,
        required_scope_tags=required_scope_tags,
        required_filter_tags=[],
        include_shared=include_shared,
        non_private_visibility_tags=non_private_visibility_tags,
    )
    where_clause = " AND ".join(f"({condition.strip()})" for condition in conditions)
    sql = f"""
        SELECT
            COALESCE(SUBSTR(category_tags.category_tag, 10), 'general') AS category,
            COUNT(DISTINCT d.id) AS total
        FROM documents d
        LEFT JOIN (
            SELECT
                bank_id,
                document_id,
                MIN(tag) AS category_tag
            FROM document_tags
            WHERE tag GLOB 'category:*'
            GROUP BY bank_id, document_id
        ) category_tags
          ON category_tags.bank_id = d.bank_id
         AND category_tags.document_id = d.id
        WHERE {where_clause}
        GROUP BY COALESCE(SUBSTR(category_tags.category_tag, 10), 'general')
        ORDER BY total DESC, category ASC
    """
    return BuiltQuery(sql=sql, params=tuple(params))


def _scoped_conditions_and_params(
    *,
    required_scope_tags: list[str],
    required_filter_tags: list[str],
    include_shared: bool,
    non_private_visibility_tags: list[str] | None,
) -> tuple[list[str], list[Any]]:
    all_required_tags = [*required_scope_tags, *required_filter_tags]
    conditions: list[str] = []
    params: list[Any] = []
    if all_required_tags:
        conditions.append(required_tags_clause(len(all_required_tags)))
        params.append(encode_tags_json(all_required_tags))
    if not include_shared and non_private_visibility_tags:
        conditions.append(non_shared_visibility_clause(len(non_private_visibility_tags)))
        params.extend(non_private_visibility_tags)
    return conditions, params


def _apply_scope_filters(
    *,
    conditions: list[str],
    params: list[Any],
    required_scope_tags: list[str],
    required_filter_tags: list[str],
    include_shared: bool,
    non_private_visibility_tags: list[str] | None,
) -> None:
    scoped_conditions, scoped_params = _scoped_conditions_and_params(
        required_scope_tags=required_scope_tags,
        required_filter_tags=required_filter_tags,
        include_shared=include_shared,
        non_private_visibility_tags=non_private_visibility_tags,
    )
    conditions.extend(scoped_conditions)
    params.extend(scoped_params)
