"""
Adapter utilities for the local Foresight-compatible memory layer.

Re-exports shared conversion helpers from local_memory_adapter and provides
foresight-prefixed aliases used by foresight-* service modules.
"""

from .local_memory_adapter import (
    document_from_storage,
    document_summary_from_storage,
    encode_tags_json,
    memory_record_from_storage,
    metadata_to_tags,
    normalize_tags,
    parse_context_payload,
    serialize_context,
)

# Foresight-prefixed aliases expected by foresight-* service modules
foresight_document_from_storage = document_from_storage
foresight_document_summary_from_storage = document_summary_from_storage

__all__ = [
    "encode_tags_json",
    "foresight_document_from_storage",
    "foresight_document_summary_from_storage",
    "memory_record_from_storage",
    "metadata_to_tags",
    "normalize_tags",
    "parse_context_payload",
    "serialize_context",
]
