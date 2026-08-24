from __future__ import annotations

"""Mapping helpers for reflection memory records."""


from typing import Any

from .reflection_types import Memory, MemoryCategory, MemoryMetadata


def record_to_memory(record: dict[str, Any]) -> Memory:
    return Memory(
        id=record["id"],
        content=record.get("content") or record.get("memory", ""),
        metadata=MemoryMetadata.from_dict(record.get("metadata") or {}),
    )


def summary_to_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "content": item.get("content", ""),
        "memory": item.get("content", ""),
        "metadata": {
            "category": item.get("category", MemoryCategory.GENERAL.value),
            "tags": item.get("tags", []),
        },
    }
