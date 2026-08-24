"""DiagnosisArena evaluation adapter — metrics from the DeepRare paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import (
    DiseaseRarity,
    OrganSystem,
)


@dataclass(frozen=True)
class DiagnosisArenaMetrics:
    """Metrics reported by the DiagnosisArena adapter."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0
    accuracy_by_organ_system: dict[str, float] = field(default_factory=dict)
    accuracy_by_rarity: dict[str, float] = field(default_factory=dict)
    accuracy_by_complexity: dict[str, float] = field(default_factory=dict)


class DiagnosisArenaAdapter:
    """Reuses the DiagnosisArena evaluation methodology for end-to-end
    assessment of the multi-agent diagnostic pipeline."""

    def __init__(self) -> None:
        self._hits: list[list[str]] = []
        self._ground_truth: list[str] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def record(self, predicted_rank_list: list[str], ground_truth_disease_id: str) -> None:
        """Record one prediction / ground-truth pair."""
        self._hits.append(predicted_rank_list)
        self._ground_truth.append(ground_truth_disease_id)

    def compute(self) -> DiagnosisArenaMetrics:
        """Compute all DeepRare metrics across all recorded cases."""
        if not self._hits:
            return DiagnosisArenaMetrics()

        r1 = sum(1 for h, g in zip(self._hits, self._ground_truth) if h and h[0] == g)
        r5 = sum(
            1 for h, g in zip(self._hits, self._ground_truth) if g in h[:5]
        )
        r10 = sum(
            1 for h, g in zip(self._hits, self._ground_truth) if g in h[:10]
        )
        mrr = self._compute_mrr()

        return DiagnosisArenaMetrics(
            recall_at_1=r1 / len(self._hits),
            recall_at_5=r5 / len(self._hits),
            recall_at_10=r10 / len(self._hits),
            mean_reciprocal_rank=mrr,
        )

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    def _compute_mrr(self) -> float:
        recips = []
        for h, g in zip(self._hits, self._ground_truth):
            for rank, pred in enumerate(h, start=1):
                if pred == g:
                    recips.append(1.0 / rank)
                    break
            else:
                recips.append(0.0)
        return sum(recips) / len(recips) if recips else 0.0
