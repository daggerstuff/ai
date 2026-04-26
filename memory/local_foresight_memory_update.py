from __future__ import annotations

from .foresight_local_adapter import memory_record_from_storage, parse_context_payload
from .foresight_local_domain import resolve_category_for_update, resolve_user_id_from_record


def build_updated_document_payload(
    *,
    existing_record: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], str | None]:
    owner_user_id = resolve_user_id_from_record(existing_record)
    existing = memory_record_from_storage(existing_record)
    merged_metadata = dict(existing.get("metadata") or {})
    next_metadata = dict(metadata or {})
    if next_metadata:
        merged_metadata.update(next_metadata)
    existing_context = parse_context_payload(existing_record.get("context"))
    category = resolve_category_for_update(
        existing_context=existing_context,
        metadata=next_metadata,
    )
    if category is not None:
        merged_metadata["category"] = category
    return owner_user_id or "system", merged_metadata, category
