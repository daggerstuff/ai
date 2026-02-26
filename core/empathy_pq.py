"""
Empathy PQ (Perceived Quality) Metric Engine

Calculates therapist effectiveness based on observed shifts in patient
defense maturity levels (PIX-150).
"""

import logging
from typing import List, Optional

from pydantic import BaseModel

from ai.core.gestalt_engine import GestaltState

logger = logging.getLogger(__name__)

# Scoring Constants
_RECOVERY_MULTIPLIER = 2.5  # Positive maturity shift
_INVALIDATION_PENALTY = 4.0  # Negative maturity shift (regression)
_BREAKTHROUGH_THRESHOLD = 6.5  # Level at which we trigger breakthrough bonus
_BREAKTHROUGH_BONUS = 15.0


class PQScore(BaseModel):
    overall_pq: float
    maturity_delta: float
    breakthrough_detected: bool
    invalidation_detected: bool
    session_progression: List[float]  # Maturity levels over time


class EmpathyPQCalculator:
    """
    Orchestrates the calculation of Empathy PQ by analyzing defense shifts.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._history: List[float] = []
        self._total_pq = 50.0  # Start at baseline 50

    def calculate_pq_increment(
        self, current_state: GestaltState, previous_maturity: Optional[float] = None
    ) -> PQScore:
        """
        Calculates the PQ score for a single turn transition.
        """
        cur_m = current_state.defense_maturity
        prev_m = (
            previous_maturity
            if previous_maturity is not None
            else (self._history[-1] if self._history else cur_m)
        )

        if cur_m is None:
            cur_m = prev_m if prev_m is not None else 3.5  # Neutral baseline

        delta = cur_m - prev_m
        breakthrough = False
        invalidation = False

        # Apply multipliers
        if delta > 0:
            score_change = delta * _RECOVERY_MULTIPLIER
            if cur_m >= _BREAKTHROUGH_THRESHOLD and prev_m < _BREAKTHROUGH_THRESHOLD:
                score_change += _BREAKTHROUGH_BONUS
                breakthrough = True
                logger.info(f"✨ Breakthrough detected! Maturity: {cur_m}")
        elif delta < 0:
            score_change = delta * _INVALIDATION_PENALTY
            invalidation = True
            logger.warning(f"⚠️ Invalidation detected. Drop in maturity: {delta}")
        else:
            score_change = 0.0

        self._total_pq = max(0.0, min(100.0, self._total_pq + score_change))
        self._history.append(cur_m)

        return PQScore(
            overall_pq=round(self._total_pq, 2),
            maturity_delta=round(delta, 4),
            breakthrough_detected=breakthrough,
            invalidation_detected=invalidation,
            session_progression=self._history[:],
        )

    def get_session_summary(self) -> dict:
        """Returns a high-level summary of the session's empathy performance."""
        if not self._history:
            return {"status": "no_data"}

        start_m = self._history[0]
        end_m = self._history[-1]
        total_delta = end_m - start_m

        return {
            "final_pq": round(self._total_pq, 2),
            "start_maturity": start_m,
            "end_maturity": end_m,
            "net_growth": round(total_delta, 4),
            "performance_category": (
                "Elite"
                if self._total_pq > 85
                else "Clinical"
                if self._total_pq > 70
                else "Intermediate"
                if self._total_pq > 50
                else "Novice"
            ),
        }
