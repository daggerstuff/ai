from __future__ import annotations

from typing import List, Tuple

from .hindsight_local_adapter import encode_tags_json, normalize_tags


def query_tokens(query: str) -> List[str]:
    return [token for token in query.replace('"', " ").split() if token]


def build_fts_query(query: str) -> str:
    tokens = query_tokens(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def build_tag_filter(
    *,
    bank_id: str,
    column_alias: str,
    tags: List[str],
    tags_match: str,
) -> Tuple[str, List[str]]:
    normalized_tags = normalize_tags(tags)
    if not normalized_tags:
        return "", []

    tags_json = encode_tags_json(normalized_tags)
    params: List[str] = [bank_id, tags_json]
    if tags_match == "all":
        params.append(len(normalized_tags))
        return (
            f" AND {column_alias}.id IN ("
            f"SELECT document_id FROM document_tags "
            f"WHERE bank_id = ? AND tag IN (SELECT value FROM json_each(?)) "
            f"GROUP BY document_id HAVING COUNT(DISTINCT tag) = ?)",
            params,
        )

    return (
        f" AND {column_alias}.id IN ("
        f"SELECT document_id FROM document_tags "
        f"WHERE bank_id = ? AND tag IN (SELECT value FROM json_each(?)))",
        params,
    )
