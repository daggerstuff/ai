"""State management for DeepRare multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConvergenceStatus(Enum):
    """How the diagnostic convergence process ended."""

    CONVERGED = "converged"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AGENT_DISAGREEMENT = "agent_disagreement"


@dataclass
class RareDiseaseState:
    """Mutable state tracked across orchestration iterations."""

    active_hypotheses: list[str] = field(default_factory=list)
    eliminated_conditions: list[str] = field(default_factory=list)
    pending_inquiries: list[str] = field(default_factory=list)
    evidence_strength: dict[str, float] = field(default_factory=dict)
    differential_history: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    convergence_status: ConvergenceStatus = ConvergenceStatus.CONVERGED

    def prune_below_threshold(self, floor: float = 0.01) -> list[str]:
        """Remove hypotheses whose posterior is below the floor.

        Returns the list of pruned disease IDs.
        """
        pruned: list[str] = []
        kept: list[str] = []
        for hid in self.active_hypotheses:
            score = self.evidence_strength.get(hid, 0.0)
            if score < floor:
                pruned.append(hid)
            else:
                kept.append(hid)
        self.active_hypotheses = kept
        for pid in pruned:
            if pid not in self.eliminated_conditions:
                self.eliminated_conditions.append(pid)
        return pruned

    def record_differential(self, entries: list[dict[str, Any]]) -> None:
        """Snapshot the current differential for convergence detection."""
        self.differential_history.append(
            {
                "iteration": self.iteration_count,
                "entries": entries,
            }
        )

    def has_converged(self, top_n: int = 3, stability_window: int = 3) -> bool:
        """Check whether the top-N differential has been stable for *stability_window* iterations."""
        if len(self.differential_history) < stability_window + 1:
            return False
        recent = self.differential_history[-stability_window:]
        top_keys = [tuple(e["disease_id"] for e in h["entries"][:top_n]) for h in recent]
        return len(set(top_keys)) == 1
