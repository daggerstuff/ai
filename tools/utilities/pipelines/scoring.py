"""Multi-criteria prioritization scoring model for PIX-4747.

Implements the mathematical prioritization model from the Workstream E spec:
priority scores combine Severity Weight, Frequency Score, System Performance
Gap, and User-Facing Impact using a NumPy/SciPy stack, then a rank-order
sorting routine produces the dynamically prioritized list of open tickets.

Design notes
------------
* All heavy lifting is vectorized over a 2D metric matrix (N tickets x 4
  criteria) so scoring + rank-order sorting runs in O(N log N), dominated by
  the final argsort.
* Each criterion is standardised (z-scored) across the ticket population so
  the four criteria are commensurable regardless of their native scales. A
  single-item population has zero variance; standardisation then yields 0.0
  and the composite score is well-defined.
* Weights are strictly positive and configurable; ``rank_order_tickets``
  breaks ties deterministically by item id so runs are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import zscore

#: Native severity weights (spec criterion 1).
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

#: Default composite-score weights (spec criteria 1-4).
DEFAULT_CRITERIA_WEIGHTS: dict[str, float] = {
    "severity_weight": 0.30,
    "frequency_score": 0.25,
    "performance_gap": 0.25,
    "user_facing_impact": 0.20,
}

#: Column order of the internal metric matrix; weights are read in this order.
CRITERIA_COLUMNS: tuple[str, ...] = (
    "severity_weight",
    "frequency_score",
    "performance_gap",
    "user_facing_impact",
)

#: Tier name constants matching reprioritization_engine.PriorityTier values.
TIER_URGENT = "urgent"
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_BACKLOG = "backlog"

#: Default quantile cut-points for tier assignment (applied to the composite
#: score; an item at or above the cut-point for tier k lands in tier k).
DEFAULT_TIER_QUANTILES: dict[str, float] = {
    TIER_URGENT: 0.95,
    TIER_HIGH: 0.80,
    TIER_MEDIUM: 0.50,
    TIER_LOW: 0.20,
}

_TIER_ORDER: tuple[str, ...] = (TIER_URGENT, TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_BACKLOG)


@dataclass
class TicketSignal:
    """Raw per-ticket evidence feeding the scoring model.

    Attributes mirror the spec's four criteria:

    * ``severity``: one of ``SEVERITY_WEIGHTS`` keys (clinical safety /
      performance gap severity).
    * ``frequency_score``: normalized interaction frequency in [0, 1].
    * ``performance_gap``: normalized system performance gap in [0, 1].
    * ``user_facing_impact``: normalized user-facing impact in [0, 1].
    """

    item_id: str
    title: str = ""
    severity: str = "medium"
    frequency_score: float = 0.0
    performance_gap: float = 0.0
    user_facing_impact: float = 0.0
    ticket_priority: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def severity_weight(self, weights: dict[str, float] | None = None) -> float:
        table = weights if weights is not None else SEVERITY_WEIGHTS
        return table.get(self.severity, table.get("low", 1.0))


@dataclass
class ScoredTicket:
    """A ticket with its composite score, rank, and assigned tier."""

    ticket: TicketSignal
    score: float
    rank: int
    tier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.ticket.item_id,
            "title": self.ticket.title,
            "severity": self.ticket.severity,
            "frequency_score": self.ticket.frequency_score,
            "performance_gap": self.ticket.performance_gap,
            "user_facing_impact": self.ticket.user_facing_impact,
            "score": round(self.score, 6),
            "rank": self.rank,
            "tier": self.tier,
        }


@dataclass
class PriorityWeights:
    """Composite-score weights for the four spec criteria.

    All weights must be positive. They are normalised to sum to 1.0 before
    use so callers can express relative emphasis without normalising by hand.
    """

    severity_weight: float = DEFAULT_CRITERIA_WEIGHTS["severity_weight"]
    frequency_score: float = DEFAULT_CRITERIA_WEIGHTS["frequency_score"]
    performance_gap: float = DEFAULT_CRITERIA_WEIGHTS["performance_gap"]
    user_facing_impact: float = DEFAULT_CRITERIA_WEIGHTS["user_facing_impact"]

    def as_vector(self) -> np.ndarray:
        raw = np.array(
            [
                self.severity_weight,
                self.frequency_score,
                self.performance_gap,
                self.user_facing_impact,
            ],
            dtype=np.float64,
        )
        if np.any(raw <= 0):
            raise ValueError("All criteria weights must be positive")
        return raw / raw.sum()


class MultiCriteriaScorer:
    """Vectorised multi-criteria scorer and rank-order sorter.

    The composite score for ticket *i* is::

        score_i = sum_k  w_k * z_k[i]

    where ``z_k`` is the z-score of criterion ``k`` across the ticket
    population. Composite scores are then min-max normalised to [0, 1] so
    tier assignment and cross-run comparisons are stable.
    """

    def __init__(
        self,
        weights: PriorityWeights | None = None,
        severity_weights: dict[str, float] | None = None,
        tier_quantiles: dict[str, float] | None = None,
    ) -> None:
        self._weights = weights or PriorityWeights()
        self._weights.as_vector()  # validate positivity eagerly
        self._severity_weights = severity_weights or dict(SEVERITY_WEIGHTS)
        self._tier_quantiles = tier_quantiles or dict(DEFAULT_TIER_QUANTILES)

    @property
    def weights(self) -> PriorityWeights:
        return self._weights

    def build_metric_matrix(self, tickets: Sequence[TicketSignal]) -> np.ndarray:
        """Build the (N, 4) criterion matrix in ``CRITERIA_COLUMNS`` order."""
        n = len(tickets)
        matrix = np.empty((n, len(CRITERIA_COLUMNS)), dtype=np.float64)
        matrix[:, 0] = [t.severity_weight(self._severity_weights) for t in tickets]
        matrix[:, 1] = [t.frequency_score for t in tickets]
        matrix[:, 2] = [t.performance_gap for t in tickets]
        matrix[:, 3] = [t.user_facing_impact for t in tickets]
        return matrix

    def standardise(self, matrix: np.ndarray) -> np.ndarray:
        """Z-score each criterion column (axis=0); zero-variance -> 0.0."""
        if matrix.shape[0] <= 1:
            return np.zeros_like(matrix)
        with np.errstate(invalid="ignore", divide="ignore"):
            standardized = zscore(matrix, axis=0, ddof=0)
        # Single distinct value in a column -> NaN; treat as zero variance.
        return np.nan_to_num(standardized, nan=0.0)

    def score(self, tickets: Sequence[TicketSignal]) -> np.ndarray:
        """Composite, min-max normalised [0, 1] score per ticket."""
        if not tickets:
            return np.empty(0, dtype=np.float64)
        if len(tickets) == 1:
            return np.full(1, 0.5, dtype=np.float64)
        standardized = self.standardise(self.build_metric_matrix(tickets))
        # Single distinct value per column yields NaN from zscore; treat as
        # zero variance so a homogeneous population scores a neutral 0.5.
        if np.unique(standardized, axis=0).shape[0] == 1:
            return np.full(len(tickets), 0.5, dtype=np.float64)
        composite = standardized @ self._weights.as_vector()
        lo, hi = float(composite.min()), float(composite.max())
        if hi - lo < 1e-12:
            return np.full(len(tickets), 0.5, dtype=np.float64)
        return (composite - lo) / (hi - lo)

    def assign_tiers(self, scores: np.ndarray) -> list[str]:
        """Map normalised scores onto the 5-level tier scale by quantile."""
        if len(scores) == 0:
            return []
        thresholds = {tier: float(np.quantile(scores, q)) for tier, q in self._tier_quantiles.items()}
        tiers: list[str] = []
        for s in scores:
            tier = TIER_BACKLOG
            for name in _TIER_ORDER[:-1]:
                if s >= thresholds[name]:
                    tier = name
                    break
            tiers.append(tier)
        return tiers

    def rank_order_tickets(
        self,
        tickets: Sequence[TicketSignal],
        weights: PriorityWeights | None = None,
    ) -> list[ScoredTicket]:
        """Score, tier-assign, and sort tickets by rank (best first).

        Ties in score are broken deterministically by ``item_id`` so that
        repeated runs over the same population produce identical rankings.
        Runs in O(N log N) -- vectorised scoring plus a single argsort.
        """
        effective_weights = weights or self._weights
        effective_weights.as_vector()  # validate positivity
        previous_weights = self._weights
        self._weights = effective_weights
        try:
            scores = self.score(tickets)
            tiers = self.assign_tiers(scores)
        finally:
            self._weights = previous_weights

        order = np.lexsort((np.array([t.item_id for t in tickets]), -scores))
        ranked: list[ScoredTicket] = [None] * len(tickets)
        for rank, idx in enumerate(order, start=1):
            ranked[idx] = ScoredTicket(
                ticket=tickets[idx],
                score=float(scores[idx]),
                rank=rank,
                tier=tiers[idx],
            )
        return [ranked[i] for i in order]

    def rank_shift_audit(
        self,
        baseline: list[ScoredTicket],
        perturbed: list[ScoredTicket],
    ) -> dict[str, Any]:
        """Compare two rank-orderings and report anomalous rank shifts.

        Returns a report with the items whose rank changed, the largest
        absolute shift, and a boolean flag when any non-backlog item fell
        out of the top quartile (the "higher-impact anomaly" check).
        """
        if len(baseline) != len(perturbed):
            raise ValueError("baseline and perturbed rank lists must have the same length")
        base_rank = {s.ticket.item_id: s.rank for s in baseline}
        shifts: list[dict[str, Any]] = []
        max_shift = 0
        for s in perturbed:
            previous = base_rank[s.ticket.item_id]
            delta = s.rank - previous
            if delta != 0:
                shifts.append(
                    {
                        "item_id": s.ticket.item_id,
                        "previous_rank": previous,
                        "new_rank": s.rank,
                        "delta": delta,
                    }
                )
                max_shift = max(max_shift, abs(delta))
        quartile_cut = max(1, len(baseline) // 4)
        anomalies = [
            shift for shift in shifts if shift["previous_rank"] <= quartile_cut and shift["new_rank"] > quartile_cut
        ]
        return {
            "total_items": len(perturbed),
            "items_with_rank_shift": len(shifts),
            "max_abs_shift": max_shift,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
        }


__all__ = [
    "CRITERIA_COLUMNS",
    "DEFAULT_CRITERIA_WEIGHTS",
    "DEFAULT_TIER_QUANTILES",
    "SEVERITY_WEIGHTS",
    "TIER_BACKLOG",
    "TIER_HIGH",
    "TIER_LOW",
    "TIER_MEDIUM",
    "TIER_URGENT",
    "MultiCriteriaScorer",
    "PriorityWeights",
    "ScoredTicket",
    "TicketSignal",
]
