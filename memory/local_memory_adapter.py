from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .local_memory_domain import resolve_user_id_from_record


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    seen = set()
    normalized: list[str] = []
    for tag in tags or []:
        if not tag:
            continue
        value = str(tag)
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def encode_tags_json(tags: Iterable[str] | None) -> str:
    return json.dumps(normalize_tags(tags), separators=(",", ":"))


def parse_context_payload(context: str | None) -> dict[str, Any]:
    if not context:
        return {}
    try:
        parsed = json.loads(context)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def serialize_context(
    *,
    user_id: str | None,
    metadata: dict[str, Any] | None,
    category: str | None,
) -> str:
    payload = {
        "metadata": metadata or {},
        "category": category or (metadata or {}).get("category"),
    }
    if user_id:
        payload["user_id"] = user_id
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def metadata_to_tags(
    *,
    user_id: str | None,
    metadata: dict[str, Any] | None,
    category: str | None,
) -> list[str]:
    merged = dict(metadata or {})
    if category:
        merged["category"] = category
    tags: list[str] = []
    if user_id:
        tags.append(f"user:{user_id}")
    for key in (
        "visibility",
        "scope",
        "org_id",
        "project_id",
        "agent_id",
        "run_id",
        "session_id",
        "category",
    ):
        value = merged.get(key)
        if value not in (None, ""):
            tags.append(f"{key}:{value}")
    return normalize_tags(tags)


def context_to_retain_params(context: str | None) -> str:
    return json.dumps({"context": context or ""}, separators=(",", ":"))


def memory_record_from_storage(record: dict[str, Any]) -> dict[str, Any]:
    context_payload = parse_context_payload(record.get("context"))
    return {
        "id": record["id"],
        "memory": record["content"],
        "content": record["content"],
        "user_id": resolve_user_id_from_record(record),
        "metadata": context_payload.get("metadata") or {},
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


def document_from_storage(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "bank_id": record["bank_id"],
        "original_text": record["content"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "tags": list(record.get("tags") or []),
        "retain_params": context_to_retain_params(record.get("context")),
    }


def document_summary_from_storage(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "bank_id": record["bank_id"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "tags": list(record.get("tags") or []),
        "retain_params": context_to_retain_params(record.get("context")),
    }
