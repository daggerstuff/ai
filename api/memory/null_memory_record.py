from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from ai.memory.local_memory_adapter import normalize_tags

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class NullMemoryRecord:
    id: str
    content: str
    content_lower: str
    content_tokens: tuple[str, ...]
    user_id: str
    metadata_items: tuple[tuple[str, Any], ...]
    normalized_tags: tuple[str, ...]
    created_at: str
    updated_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        memory_id: str,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None,
        created_at: str,
        updated_at: str | None = None,
    ) -> NullMemoryRecord:
        prepared = cls._prepare(content=content, metadata=metadata or {})
        return cls(
            id=memory_id,
            **prepared,
            user_id=user_id,
            created_at=created_at,
            updated_at=updated_at,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_items)

    def to_dict(self) -> dict[str, Any]:
        materialized = {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
        if self.updated_at is not None:
            materialized["updated_at"] = self.updated_at
        return materialized

    def with_updates(
        self,
        *,
        new_content: str | None = None,
        metadata: dict[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> NullMemoryRecord:
        merged_metadata = self.metadata
        if metadata is not None:
            merged_metadata.update(metadata)
        content = new_content if new_content is not None else self.content
        prepared = self._prepare(content=content, metadata=merged_metadata)
        return replace(
            self,
            **prepared,
            updated_at=updated_at or self.updated_at,
        )

    @staticmethod
    def _prepare(*, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        content_lower = content.lower()
        return {
            "content": content,
            "content_lower": content_lower,
            "content_tokens": tuple(_TOKEN_PATTERN.findall(content_lower)),
            "metadata_items": tuple(metadata.items()),
            "normalized_tags": tuple(normalize_tags(metadata.get("tags", []))),
        }
