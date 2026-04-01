from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

NON_PRIVATE_VISIBILITY_TAGS = (
    "visibility:shared",
    "visibility:org",
    "visibility:project",
    "visibility:system",
)


def parse_context_payload(context: Optional[str]) -> Dict[str, Any]:
    if not context:
        return {}
    try:
        parsed = json.loads(context)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_user_id_from_context(context: Optional[str]) -> Optional[str]:
    payload = parse_context_payload(context)
    value = payload.get("user_id")
    return value if isinstance(value, str) and value else None


def resolve_user_id_from_tags(tags: Iterable[Any]) -> Optional[str]:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("user:"):
            value = tag.split(":", 1)[1]
            if value:
                return value
    return None


def resolve_user_id_from_payload(
    *,
    context: Optional[str],
    tags: Iterable[Any] = (),
    fallback_user_id: Optional[str] = None,
) -> Optional[str]:
    return (
        resolve_user_id_from_context(context)
        or resolve_user_id_from_tags(tags)
        or fallback_user_id
    )


def resolve_user_id_from_record(record: Dict[str, Any]) -> Optional[str]:
    user_id = record.get("user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    return resolve_user_id_from_payload(
        context=record.get("context"),
        tags=record.get("tags") or (),
    )


def resolve_category_for_update(
    *,
    existing_context: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
) -> Optional[str]:
    category = existing_context.get("category")
    if metadata and metadata.get("category") is not None:
        category = metadata.get("category")
    return category if isinstance(category, str) and category else None
