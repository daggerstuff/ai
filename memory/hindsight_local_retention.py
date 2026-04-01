from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from .hindsight_local_adapter import metadata_to_tags, normalize_tags, serialize_context
from .hindsight_local_domain import parse_context_payload

_SCOPE_METADATA_KEYS = {
    "org_id",
    "project_id",
    "agent_id",
    "run_id",
    "session_id",
    "scope",
    "visibility",
}
_RESERVED_TAG_PREFIXES = tuple(f"{key}:" for key in (*_SCOPE_METADATA_KEYS, "user", "category"))


def scope_metadata(
    *,
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    visibility: Optional[str],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if org_id:
        metadata["org_id"] = org_id
    if project_id:
        metadata["project_id"] = project_id
    if session_id:
        metadata["session_id"] = session_id
    if agent_id:
        metadata["agent_id"] = agent_id
    if run_id:
        metadata["run_id"] = run_id
    if visibility:
        metadata["visibility"] = visibility
    return metadata


def metadata_from_context(context: Optional[str]) -> Dict[str, Any]:
    payload = parse_context_payload(context)
    metadata = payload.get("metadata")
    merged: Dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
    category = payload.get("category")
    if isinstance(category, str) and category:
        merged["category"] = category
    return {
        key: value
        for key, value in merged.items()
        if key not in _SCOPE_METADATA_KEYS
    }


def custom_tags(tags: Iterable[str]) -> List[str]:
    filtered: List[str] = []
    for tag in tags:
        value = str(tag).strip()
        if not value or value.startswith(_RESERVED_TAG_PREFIXES):
            continue
        filtered.append(value)
    return normalize_tags(filtered)


def build_scoped_retain_items(
    *,
    items: Iterable[Dict[str, Any]],
    user_id: str,
    base_metadata: Dict[str, Any],
    ownership_validator: Callable[[str], None],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for item in items:
        document_id = item.get("document_id")
        if document_id:
            ownership_validator(str(document_id))
        merged_metadata = metadata_from_context(item.get("context"))
        merged_metadata.update(base_metadata)
        category = merged_metadata.get("category")
        scoped_tags = metadata_to_tags(
            user_id=user_id,
            metadata=merged_metadata,
            category=category if isinstance(category, str) else None,
        )
        prepared.append(
            {
                "content": item["content"],
                "document_id": document_id,
                "context": serialize_context(
                    user_id=user_id,
                    metadata=merged_metadata,
                    category=category if isinstance(category, str) else None,
                ),
                "tags": normalize_tags([*scoped_tags, *custom_tags(item.get("tags") or [])]),
            }
        )
    return prepared


def build_hindsight_retain_batch(
    *,
    items: Iterable[Dict[str, Any]],
    user_id: str,
    actor_metadata: Dict[str, Any],
    org_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    visibility: Optional[str],
    ownership_validator: Callable[[str], None],
) -> List[Dict[str, Any]]:
    base_metadata = scope_metadata(
        org_id=org_id,
        project_id=project_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        visibility=visibility,
    )
    base_metadata.update(actor_metadata)
    return build_scoped_retain_items(
        items=items,
        user_id=user_id,
        base_metadata=base_metadata,
        ownership_validator=ownership_validator,
    )
