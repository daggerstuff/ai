"""Lightweight evidence-based reprioritization engine for ai/memory integration.

Provides a simple ReprioritizationEngine that accepts evaluation evidence
and produces a reprioritized task order. This is distinct from the full-featured
engine in ai/core/pipelines/reprioritization_engine.py which handles the
modern dataset pipeline's Phase 4 evaluation-to-data feedback loop.

This module's lighter API is used by memory-layer services that need quick
priority adjustments without the overhead of the full pipeline integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EvidenceItem:
    """Simple representation of evaluation evidence impacting priority."""

    source_id: str
    evidence_type: str  # e.g., 'performance_gap', 'quality_issue'
    score: float  # magnitude of impact, >0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ReprioritizationEngine:
    """Core engine to adjust priorities based on evidence.

    The engine receives a list of EvidenceItem objects and produces a
    deterministic ordering for acquisition/curation tasks. It is deliberately
    lightweight — heavy logic (LLM scoring, database access) lives in the
    surrounding services.
    """

    def __init__(self, base_priority: list[str] | None = None) -> None:
        # base_priority is an ordered list of task identifiers (ids).
        self.base_priority = base_priority or []
        self.adjustments: dict[str, float] = {}

    def apply_evidence(self, evidence: list[EvidenceItem]) -> None:
        """Accumulate score adjustments for each task based on evidence."""
        for ev in evidence:
            task_id = ev.details.get("task_id")
            if not task_id:
                continue
            # Simple linear impact — can be replaced with more complex logic
            self.adjustments[task_id] = self.adjustments.get(task_id, 0.0) + ev.score

    def reprioritize(self) -> list[str]:
        """Return a new ordering of task ids.

        Tasks with higher accumulated adjustment move forward. Original order
        is preserved for ties.
        """
        # Combine base list with any new tasks discovered in adjustments
        all_tasks = list(dict.fromkeys(self.base_priority + list(self.adjustments)))

        # Sort by adjustment descending, then original index
        def sort_key(tid: str) -> tuple[float, float]:
            # negative because higher scores should come first
            base_idx = self.base_priority.index(tid) if tid in self.base_priority else float("inf")
            return (-self.adjustments.get(tid, 0.0), base_idx)

        return sorted(all_tasks, key=sort_key)

    def compute_new_order(self, evidence: list[EvidenceItem]) -> list[str]:
        """Convenience method: apply evidence and return new ordering."""
        self.apply_evidence(evidence)
        return self.reprioritize()


__all__ = ["EvidenceItem", "ReprioritizationEngine"]
