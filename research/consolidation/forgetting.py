"""Forgetting Mechanisms — Sprint 3, Task 4.

Ebbinghaus forgetting curve with therapeutic modifications,
archive vs. delete decision logic, crisis preservation guarantee,
and memory pruning scheduler.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import StrEnum

from ..schema import ConsolidationPhase, MemoryBlock

log = logging.getLogger(__name__)


class ForgetAction(StrEnum):
    PRESERVE = "preserve"
    ARCHIVE = "archive"
    DELETE = "delete"
    LATENT = "latent"


@dataclass(frozen=True)
class ForgetDecision:
    memory_id: str
    action: ForgetAction
    reason: str
    retention_score: float


@dataclass
class ForgettingConfig:
    half_life_days: float = 30.0
    archive_threshold: float = 0.15
    delete_threshold: float = 0.05
    crisis_preserve: bool = True
    min_rem_cycles: int = 1
    reverie_eligible_min_emotional_weight: float = 2.0


class ForgettingEngine:
    """Apply forgetting curve to memories with safety guarantees."""

    def __init__(self, config: ForgettingConfig | None = None) -> None:
        self._config = config or ForgettingConfig()

    def evaluate(self, memory: MemoryBlock, now_ms: int | None = None) -> ForgetDecision:
        """Evaluate a single memory for forgetting action."""
        if memory.gating.crisisFlag and self._config.crisis_preserve:
            return ForgetDecision(
                memory_id=memory.id,
                action=ForgetAction.PRESERVE,
                reason="Crisis content — preserved indefinitely",
                retention_score=1.0,
            )

        if memory.consolidation.remCycles > self._config.min_rem_cycles:
            return ForgetDecision(
                memory_id=memory.id,
                action=ForgetAction.PRESERVE,
                reason=f"REM cycles remaining ({memory.consolidation.remCycles})",
                retention_score=0.8,
            )

        retention = self._retention_score(memory, now_ms)

        if retention >= self._config.archive_threshold:
            return ForgetDecision(
                memory_id=memory.id,
                action=ForgetAction.PRESERVE,
                reason=f"Retention score {retention:.3f} above archive threshold",
                retention_score=retention,
            )

        if retention >= self._config.delete_threshold:
            return ForgetDecision(
                memory_id=memory.id,
                action=ForgetAction.ARCHIVE,
                reason=f"Retention score {retention:.3f} — archive candidate",
                retention_score=retention,
            )

        # Before deleting, check if memory qualifies for reverie (latent) transition
        if memory.importance.emotionalWeight >= self._config.reverie_eligible_min_emotional_weight:
            return ForgetDecision(
                memory_id=memory.id,
                action=ForgetAction.LATENT,
                reason=f"Retention {retention:.3f} — reverie candidate (emotional weight {memory.importance.emotionalWeight:.1f})",
                retention_score=retention,
            )

        return ForgetDecision(
            memory_id=memory.id,
            action=ForgetAction.DELETE,
            reason=f"Retention score {retention:.3f} — pruning candidate",
            retention_score=retention,
        )

    def batch_evaluate(self, memories: list[MemoryBlock], now_ms: int | None = None) -> list[ForgetDecision]:
        """Evaluate a batch of memories."""
        return [self.evaluate(m, now_ms) for m in memories]

    def get_pruning_candidates(
        self, memories: list[MemoryBlock], now_ms: int | None = None
    ) -> list[tuple[MemoryBlock, ForgetDecision]]:
        """Return memories eligible for pruning, sorted by retention score (lowest first)."""
        decisions = self.batch_evaluate(memories, now_ms)
        candidates = [
            (m, d)
            for m, d in zip(memories, decisions, strict=False)
            if d.action in (ForgetAction.ARCHIVE, ForgetAction.DELETE, ForgetAction.LATENT)
        ]
        candidates.sort(key=lambda x: x[1].retention_score)
        return candidates

    def apply_forgetting(self, memory: MemoryBlock, now_ms: int | None = None) -> MemoryBlock:
        """Apply forgetting decay to a memory's importance."""
        retention = self._retention_score(memory, now_ms)
        decision = self.evaluate(memory, now_ms)
        updated = memory.model_copy(deep=True)
        updated.importance.recency = max(updated.importance.recency * retention, 0.0)
        updated.importance.raw = max(updated.importance.raw * retention, 0.0)

        if decision.action == ForgetAction.LATENT:
            updated.consolidation.phase = ConsolidationPhase.LATENT
            updated.consolidation.reverieEligible = True
            updated.consolidation.reveriePhase = "seeded"

        return updated

    def _retention_score(self, memory: MemoryBlock, now_ms: int | None = None) -> float:
        """Compute retention score using Ebbinghaus curve with therapeutic modifications."""
        now = now_ms or int(time.time() * 1000)
        age_ms = max(now - memory.timestamp, 0)
        age_days = age_ms / (1000 * 86400)

        ebbinghaus = math.exp(-math.log(2) * age_days / self._config.half_life_days)

        importance_factor = memory.importance.raw
        emotional_boost = memory.importance.emotionalWeight / 5.0
        crisis_boost = 1.0 if memory.gating.crisisFlag else 0.0

        retention = 0.4 * ebbinghaus + 0.3 * importance_factor + 0.2 * emotional_boost + 0.1 * crisis_boost
        return min(max(retention, 0.0), 1.0)
