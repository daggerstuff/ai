from __future__ import annotations

from .local_memory_adapter import (
    document_from_storage as foresight_document_from_storage,
    document_summary_from_storage as foresight_document_summary_from_storage,
    memory_record_from_storage,
    metadata_to_tags,
    normalize_tags,
    serialize_context,
    encode_tags_json,
    parse_context_payload,
)
