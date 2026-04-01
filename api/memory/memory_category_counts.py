from __future__ import annotations

from typing import Any


def count_memory_categories(memories: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for memory in memories:
        category = (memory.get("metadata") or {}).get("category", "general")
        counts[category] = counts.get(category, 0) + 1
    return counts
