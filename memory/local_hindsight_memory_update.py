from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .hindsight_local_adapter import memory_record_from_storage, parse_context_payload
from .hindsight_local_domain import resolve_category_for_update
from .local_hindsight_repository import LocalHindsightRepository


def build_updated_document_payload(
    *,
    repository: LocalHindsightRepository,
    existing_record: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    owner_user_id = repository.resolve_user_id(existing_record)
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
