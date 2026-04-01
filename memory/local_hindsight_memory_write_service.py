from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .local_hindsight_protocol_adapter import LocalHindsightProtocolAdapter


class LocalHindsightMemoryWriteService:
    """Create normalized memory payloads before they are retained."""

    def __init__(self, *, protocol: LocalHindsightProtocolAdapter, default_bank_id: str) -> None:
        self.protocol = protocol
        self.default_bank_id = default_bank_id

    @staticmethod
    def _metadata_dict(metadata: Optional[Any]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if hasattr(metadata, "to_dict") and callable(metadata.to_dict):
            data = metadata.to_dict()
            return dict(data) if isinstance(data, dict) else {}
        raise TypeError("metadata must be a mapping or expose to_dict()")

    def coerce_metadata(self, metadata: Optional[Any]) -> Dict[str, Any]:
        return self._metadata_dict(metadata)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def prepare_metadata(
        self,
        *,
        metadata: Optional[Any],
        category: Optional[str],
        scope_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = self._metadata_dict(metadata)
        if scope_metadata:
            merged.update(scope_metadata)
        if category:
            merged["category"] = category
        merged.setdefault("timestamp", self._utc_now())
        return merged

    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Optional[Any] = None,
        category: Optional[str] = None,
        scope_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        merged = self.prepare_metadata(
            metadata=metadata,
            category=category,
            scope_metadata=scope_metadata,
        )
        retained = self.protocol.retain_items(
            self.default_bank_id,
            [self.protocol.build_add_memory_item(user_id=user_id, content=content, metadata=merged)],
        )
        results = retained.get("results")
        if not isinstance(results, list) or not results:
            raise RuntimeError("Retain operation returned no document identifiers")
        first = results[0]
        if not isinstance(first, dict):
            raise RuntimeError("Retain operation returned an invalid document payload")
        document_id = first.get("id")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("Retain operation did not provide a valid document identifier")
        return document_id
