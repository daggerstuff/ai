"""Bayesian differential diagnosis manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .state import RareDiseaseState
from .types import (
    DifferentialEntry,
    Evidence,
    PatientCase,
)

logger = logging.getLogger(__name__)

DEFAULT_FLOOR = 0.01
PRUNE_WINDOW = 3


@dataclass
class DifferentialDiagnosisManager:
    """Maintains and updates a ranked differential using Bayesian evidence."""

    floor: float = DEFAULT_FLOOR
    prune_after: int = PRUNE_WINDOW

    def initialise(self, _case: PatientCase, candidates: list[DifferentialEntry]) -> RareDiseaseState:
        """Create an initial state from candidates (flat prior)."""
        n = len(candidates)
        prior = 1.0 / n if n else 0.0
        state = RareDiseaseState(active_hypotheses=[c.disease_id for c in candidates])
        for c in candidates:
            state.evidence_strength[c.disease_id] = prior
        return state

    def update(
        self,
        state: RareDiseaseState,
        evidence: list[Evidence],
        candidates: list[DifferentialEntry],
    ) -> list[DifferentialEntry]:
        """Apply Bayesian updating per candidate and return the updated ranked differential."""
        disease_map = {c.disease_id: c for c in candidates}
        for ev in evidence:
            if ev.disease_id and ev.disease_id not in disease_map:
                continue
            for did, entry in disease_map.items():
                lr = self._likelihood_ratio(ev, did)
                prior = state.evidence_strength.get(did, self.floor)
                posterior = prior * lr / (prior * lr + (1 - prior) * 1)
                state.evidence_strength[did] = posterior
                entry.posterior_probability = posterior

        ranked = sorted(candidates, key=lambda e: e.posterior_probability, reverse=True)
        state.record_differential(
            [
                {
                    "disease_id": e.disease_id,
                    "disease_name": e.disease_name,
                    "posterior": e.posterior_probability,
                }
                for e in ranked
            ]
        )
        state.iteration_count += 1
        return ranked

    def prune(self, state: RareDiseaseState) -> list[str]:
        return state.prune_below_threshold(self.floor)

    def has_converged(self, state: RareDiseaseState) -> bool:
        return state.has_converged()

    def _likelihood_ratio(self, evidence: Evidence, disease_id: str) -> float:
        base = max(evidence.weight, 0.01)
        if evidence.disease_id == disease_id:
            return max(base, 1.0)
        return 1.0 / max(base, 1.0)
