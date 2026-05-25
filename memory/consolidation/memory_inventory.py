"""Memory Inventory System — Sprint 3, Task 1.

Catalogs raw memories from sessions, sorts by importance, groups by
session/topic/emotional valence.  Builds in < 100 ms for 1000 memories.

Usage::

    from ai.memory.consolidation.memory_inventory import MemoryInventory
    inventory = MemoryInventory()
    inventory.add_memory(memory_block)
    catalog = inventory.build_catalog()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from ..schema import ConsolidationPhase, MemoryBlock

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InventoryGroup:
    """A group of memories sharing a common key (session, topic, or valence)."""

    key: str
    memories: list[MemoryBlock]
    total_importance: float
    avg_valence: float

    @property
    def count(self) -> int:
        return len(self.memories)


@dataclass
class MemoryCatalog:
    """Full inventory catalog with multiple groupings."""

    all_memories: list[MemoryBlock]
    by_importance: list[MemoryBlock]
    by_session: dict[str, InventoryGroup]
    by_topic: dict[str, InventoryGroup]
    by_valence: dict[str, InventoryGroup]
    build_time_ms: float
    total_count: int
    total_importance: float


class MemoryInventory:
    """Catalog memories for consolidation processing.

    Provides importance-sorted cataloging, session-based grouping,
    and emotional valence sorting.  Respects tenant boundaries.
    """

    def __init__(self) -> None:
        self._memories: list[MemoryBlock] = []
        self._tenant_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_memory(self, memory: MemoryBlock) -> None:
        """Add a memory block to the inventory."""
        self._memories.append(memory)
        self._tenant_ids.add(memory.tenantId)

    def add_memories(self, memories: list[MemoryBlock]) -> None:
        """Add multiple memory blocks at once."""
        for m in memories:
            self.add_memory(m)

    def clear(self) -> None:
        """Remove all memories from the inventory."""
        self._memories.clear()
        self._tenant_ids.clear()

    # ------------------------------------------------------------------
    # Catalog building
    # ------------------------------------------------------------------
    def build_catalog(self) -> MemoryCatalog:
        """Build a full catalog sorted and grouped multiple ways.

        Returns a :class:`MemoryCatalog` with all groupings.
        """
        t0 = time.perf_counter()

        by_importance = sorted(self._memories, key=lambda m: m.importance.raw, reverse=True)
        by_session = self._group_by_session()
        by_topic = self._group_by_topic()
        by_valence = self._group_by_valence()

        elapsed = (time.perf_counter() - t0) * 1000
        total_imp = sum(m.importance.raw for m in self._memories)

        catalog = MemoryCatalog(
            all_memories=list(self._memories),
            by_importance=by_importance,
            by_session=by_session,
            by_topic=by_topic,
            by_valence=by_valence,
            build_time_ms=round(elapsed, 2),
            total_count=len(self._memories),
            total_importance=round(total_imp, 6),
        )
        log.debug(
            "Catalog built: %d memories in %.1f ms",
            catalog.total_count,
            catalog.build_time_ms,
        )
        return catalog

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def get_tenant_memories(self, tenant_id: str) -> list[MemoryBlock]:
        """Return memories for a specific tenant (isolation guarantee)."""
        return [m for m in self._memories if m.tenantId == tenant_id]

    def get_session_memories(self, session_id: str) -> list[MemoryBlock]:
        """Return all memories from a specific session."""
        return [m for m in self._memories if m.sessionId == session_id]

    def get_crisis_memories(self) -> list[MemoryBlock]:
        """Return memories flagged as crisis content."""
        return [m for m in self._memories if m.gating.crisisFlag]

    def get_phase_memories(self, phase: ConsolidationPhase) -> list[MemoryBlock]:
        """Return memories in a specific consolidation phase."""
        return [m for m in self._memories if m.consolidation.phase == phase]

    # ------------------------------------------------------------------
    # Internal grouping
    # ------------------------------------------------------------------
    def _group_by_session(self) -> dict[str, InventoryGroup]:
        groups: dict[str, list[MemoryBlock]] = defaultdict(list)
        for m in self._memories:
            groups[m.sessionId].append(m)
        return {sid: self._make_group(sid, memories) for sid, memories in groups.items()}

    def _group_by_topic(self) -> dict[str, InventoryGroup]:
        """Group by emotional category labels as a proxy for topic."""
        groups: dict[str, list[MemoryBlock]] = defaultdict(list)
        for m in self._memories:
            categories = m.emotions.categories or ["uncategorized"]
            for cat in categories:
                groups[cat].append(m)
        return {cat: self._make_group(cat, memories) for cat, memories in groups.items()}

    def _group_by_valence(self) -> dict[str, InventoryGroup]:
        """Group into negative / neutral / positive valence buckets."""
        buckets: dict[str, list[MemoryBlock]] = {
            "negative": [],
            "neutral": [],
            "positive": [],
        }
        for m in self._memories:
            v = m.emotions.valence
            if v < -0.2:
                buckets["negative"].append(m)
            elif v > 0.2:
                buckets["positive"].append(m)
            else:
                buckets["neutral"].append(m)
        return {label: self._make_group(label, memories) for label, memories in buckets.items() if memories}

    @staticmethod
    def _make_group(key: str, memories: list[MemoryBlock]) -> InventoryGroup:
        total_imp = sum(m.importance.raw for m in memories)
        avg_val = sum(m.emotions.valence for m in memories) / len(memories) if memories else 0.0
        return InventoryGroup(
            key=key,
            memories=memories,
            total_importance=round(total_imp, 6),
            avg_valence=round(avg_val, 4),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self._memories)

    @property
    def tenant_ids(self) -> set[str]:
        return set(self._tenant_ids)

    def __repr__(self) -> str:
        return f"<MemoryInventory memories={self.count} tenants={len(self._tenant_ids)}>"
