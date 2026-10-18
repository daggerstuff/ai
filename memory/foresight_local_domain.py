from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

NON_PRIVATE_VISIBILITY_TAGS = (
    "visibility:shared",
    "visibility:org",
    "visibility:project",
    "visibility:system",
)


def resolve_user_id_from_record(record: dict[str, Any]) -> str | None:
    user_id = record.get("user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    return resolve_user_id_from_payload(
        context=record.get("context"),
        tags=record.get("tags") or (),
    )


def resolve_user_id_from_payload(
    *,
    context: str | None,
    tags: Iterable[Any] = (),
    fallback_user_id: str | None = None,
) -> str | None:
    return resolve_user_id_from_context(context) or resolve_user_id_from_tags(tags) or fallback_user_id


def resolve_user_id_from_context(context: str | None) -> str | None:
    payload = parse_context_payload(context)
    value = payload.get("user_id")
    return value if isinstance(value, str) and value else None


def resolve_user_id_from_tags(tags: Iterable[Any]) -> str | None:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("user:"):
            value = tag.split(":", 1)[1]
            if value:
                return value
    return None


def parse_context_payload(context: str | None) -> dict[str, Any]:
    if not context:
        return {}
    try:
        parsed = json.loads(context)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_category_for_update(
    *,
    existing_context: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> str | None:
    category = existing_context.get("category")
    if metadata and metadata.get("category") is not None:
        category = metadata.get("category")
    return category if isinstance(category, str) and category else None
