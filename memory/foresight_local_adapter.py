from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .local_memory_adapter import (
    encode_tags_json,
    memory_record_from_storage,
    metadata_to_tags,
    normalize_tags,
    parse_context_payload,
    serialize_context,
    document_from_storage as foresight_document_from_storage,
    document_summary_from_storage as foresight_document_summary_from_storage,
)
