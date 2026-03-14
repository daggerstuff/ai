"""
Empathy PQ (Perceived Quality) Metric Engine

Calculates therapist effectiveness based on observed shifts in patient
defense maturity levels (PIX-150).
"""

import logging
from statistics import mean
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
    rolling_window_delta: float
    breakthrough_detected: bool
    invalidation_detected: bool
    defensive_instability: float
    session_progression: List[float]  # Maturity levels over time
    window_size: int
    trend_window: List[float]


class EmpathyPQCalculator:
    """
    Orchestrates the calculation of Empathy PQ by analyzing defense shifts.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._history: List[float] = []
        self._trend_window: List[float] = []
        self._total_pq = 50.0  # Start at baseline 50
        self._window_size = 7

    @property
    def window_size(self) -> int:
        return self._window_size

    def calculate_pq_increment(
        self, current_state: GestaltState, previous_maturity: Optional[float] = None
    ) -> PQScore:
        """
        Calculates the PQ score for a single turn transition.
        """
        cur_m = current_state.defense_maturity
        if cur_m is None:
            if previous_maturity is not None:
                cur_m = previous_maturity
            elif self._history:
                cur_m = self._history[-1]
            else:
                cur_m = 3.5  # Neutral baseline

        prev_m = (
            previous_maturity
            if previous_maturity is not None
            else (self._history[-1] if self._history else cur_m)
        )
        delta = round(cur_m - prev_m, 4)

        if self._trend_window:
            trend_baseline = mean(self._trend_window)
        else:
            trend_baseline = prev_m

        rolling_window_delta = round(cur_m - trend_baseline, 4)
        breakthrough = False
        invalidation = False

        # Apply multipliers using rolling-window context
        if rolling_window_delta > 0:
            score_change = rolling_window_delta * _RECOVERY_MULTIPLIER
            if (
                cur_m >= _BREAKTHROUGH_THRESHOLD
                and trend_baseline < _BREAKTHROUGH_THRESHOLD
            ):
                score_change += _BREAKTHROUGH_BONUS
                breakthrough = True
                logger.info(f"✨ Breakthrough detected! Maturity: {cur_m}")
        elif rolling_window_delta < 0:
            score_change = rolling_window_delta * _INVALIDATION_PENALTY
            invalidation = True
            logger.warning(f"⚠️ Invalidation detected. Drop in maturity: {delta}")
        else:
            score_change = 0.0

        self._total_pq = max(0.0, min(100.0, self._total_pq + score_change))
        self._history.append(cur_m)
        self._trend_window.append(cur_m)
        if len(self._trend_window) > self._window_size:
            self._trend_window = self._trend_window[-self._window_size :]

        if self._history:
            instability = self._rolling_instability()
        else:
            instability = 0.0

        return PQScore(
            overall_pq=round(self._total_pq, 2),
            maturity_delta=round(delta, 4),
            rolling_window_delta=rolling_window_delta,
            breakthrough_detected=breakthrough,
            invalidation_detected=invalidation,
            defensive_instability=round(instability, 4),
            session_progression=self._history[:],
            window_size=self._window_size,
            trend_window=self._trend_window[:],
        )

    def _window_delta(self, latest_maturity: float) -> float:
        """Compute the latest delta versus the active rolling window baseline."""
        if len(self._trend_window) <= 1:
            return 0.0
        previous_values = self._trend_window[:-1]
        return round(latest_maturity - mean(previous_values), 4)

    def get_session_summary(self) -> dict:
        """Returns a high-level summary of the session's empathy performance."""
        if not self._history:
            return {"status": "no_data"}

        start_m = self._history[0]
        end_m = self._history[-1]
        total_delta = end_m - start_m
        if len(self._history) > 1:
            deltas = [self._history[i] - self._history[i - 1] for i in range(1, len(self._history))]
            best_gain = max(deltas)
            worst_drop = min(deltas)
            volatility = mean([abs(v) for v in deltas])
        else:
            best_gain = 0.0
            worst_drop = 0.0
            volatility = 0.0

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
            "maturity_window_delta": self._window_delta(self._history[-1]),
            "window_size": self._window_size,
            "turn_count": len(self._history),
            "best_maturity_gain": round(best_gain, 4),
            "worst_maturity_drop": round(worst_drop, 4),
            "window_volatility": round(volatility, 4),
            "defensive_instability": round(self._rolling_instability(), 4),
        }

    def _rolling_instability(self) -> float:
        if len(self._trend_window) < 2:
            return 0.0

        deltas = [
            abs(self._trend_window[i] - self._trend_window[i - 1])
            for i in range(1, len(self._trend_window))
        ]
        return mean(deltas)
