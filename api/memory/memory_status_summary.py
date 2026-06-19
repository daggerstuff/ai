from __future__ import annotations

from dataclasses import dataclass

from .memory_health import resolve_memory_health


@dataclass(frozen=True)
class MemoryStatusSummary:
    total_memories: int
    health: str
    categories: dict[str, int]
    is_sampled: bool = False


def summarize_memory_status(
    *,
    category_counts: dict[str, int],
    backend_readiness: str | None = None,
    is_sampled: bool = False,
) -> MemoryStatusSummary:
    total_memories = sum(category_counts.values())
    health = resolve_memory_health(
        readiness=backend_readiness,
        _memory_count=total_memories,
    )
    return MemoryStatusSummary(
        total_memories=total_memories,
        health=health,
        categories=category_counts,
        is_sampled=is_sampled,
    )
