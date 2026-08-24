"""Cross-Session Pattern Detection — Sprint 4, Task 3.

Detects recurring themes, progress trends, trigger correlations,
and intervention effectiveness across multiple sessions.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from ai.research.schema import MemoryBlock

from .session_consolidation import SessionSummary

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecurringTheme:
    theme: str
    frequency: int
    sessions: list[str]
    avg_valence: float
    trend: str


@dataclass(frozen=True)
class ProgressTrend:
    metric: str
    direction: str  # "improving", "declining", "stable"
    confidence: float
    data_points: list[float]
    slope: float


@dataclass(frozen=True)
class TriggerPattern:
    trigger: str
    response: str
    co_occurrence_count: int
    confidence: float
    example_sessions: list[str]


@dataclass(frozen=True)
class InterventionResult:
    intervention: str
    effectiveness_score: float
    sessions_applied: int
    avg_valence_before: float
    avg_valence_after: float


@dataclass
class PatternReport:
    recurring_themes: list[RecurringTheme]
    progress_trends: list[ProgressTrend]
    trigger_patterns: list[TriggerPattern]
    intervention_results: list[InterventionResult]
    elapsed_ms: float


class PatternDetector:
    """Detect patterns across multiple session summaries and memories."""

    def __init__(
        self,
        min_frequency: int = 2,
        min_confidence: float = 0.6,
    ) -> None:
        self._min_freq = min_frequency
        self._min_confidence = min_confidence

    def analyze(
        self,
        sessions: list[SessionSummary],
        raw_memories: list[MemoryBlock] | None = None,
    ) -> PatternReport:
        """Run full pattern analysis across sessions."""
        t0 = time.perf_counter()

        themes = self._detect_recurring_themes(sessions)
        trends = self._detect_progress_trends(sessions)
        triggers = self._detect_triggers(sessions, raw_memories)
        interventions = self._analyze_interventions(sessions)

        elapsed = (time.perf_counter() - t0) * 1000

        report = PatternReport(
            recurring_themes=themes,
            progress_trends=trends,
            trigger_patterns=triggers,
            intervention_results=interventions,
            elapsed_ms=round(elapsed, 2),
        )
        log.info(
            "Pattern analysis: %d sessions → %d themes, %d trends, %d triggers in %.0f ms",
            len(sessions),
            len(themes),
            len(trends),
            len(triggers),
            elapsed,
        )
        return report

    def _detect_recurring_themes(self, sessions: list[SessionSummary]) -> list[RecurringTheme]:
        """Find themes that appear across multiple sessions."""
        theme_sessions: dict[str, list[str]] = defaultdict(list)
        theme_valences: dict[str, list[float]] = defaultdict(list)

        for s in sessions:
            for theme in s.themes:
                theme_sessions[theme].append(s.session_id)
                theme_valences[theme].append(s.emotional_arc.avg_valence)

        results: list[RecurringTheme] = []
        for theme, session_list in theme_sessions.items():
            if len(session_list) < self._min_freq:
                continue
            valences = theme_valences[theme]
            if len(valences) >= 2:
                first_half = sum(valences[: len(valences) // 2]) / (len(valences) // 2)
                second_half = sum(valences[len(valences) // 2 :]) / (len(valences) - len(valences) // 2)
                if second_half > first_half + 0.05:
                    trend = "improving"
                elif second_half < first_half - 0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            results.append(
                RecurringTheme(
                    theme=theme,
                    frequency=len(session_list),
                    sessions=session_list,
                    avg_valence=round(sum(valences) / len(valences), 3),
                    trend=trend,
                )
            )

        results.sort(key=lambda t: t.frequency, reverse=True)
        return results

    def _detect_progress_trends(self, sessions: list[SessionSummary]) -> list[ProgressTrend]:
        """Detect improvement/decline trends in emotional metrics."""
        if len(sessions) < 2:
            return []

        sorted_sessions = sorted(sessions, key=lambda s: s.timestamp_ms)
        valences = [s.emotional_arc.avg_valence for s in sorted_sessions]

        slope = self._linear_slope(valences)
        direction = "improving" if slope > 0.02 else "declining" if slope < -0.02 else "stable"
        confidence = min(abs(slope) * 10, 1.0)

        return [
            ProgressTrend(
                metric="avg_valence",
                direction=direction,
                confidence=round(confidence, 2),
                data_points=valences,
                slope=round(slope, 4),
            )
        ]

    def _detect_triggers(
        self,
        sessions: list[SessionSummary],
        raw_memories: list[MemoryBlock] | None = None,
    ) -> list[TriggerPattern]:
        """Find co-occurring trigger-response patterns."""
        if raw_memories is None:
            return []

        crisis_sessions = set()
        for m in raw_memories:
            if m.gating.crisisFlag:
                crisis_sessions.add(m.sessionId)

        theme_counts: dict[str, int] = Counter()
        theme_crisis: dict[str, int] = Counter()

        for s in sessions:
            for theme in s.themes:
                theme_counts[theme] += 1
                if s.session_id in crisis_sessions:
                    theme_crisis[theme] += 1

        patterns: list[TriggerPattern] = []
        for theme, count in theme_counts.items():
            crisis_count = theme_crisis.get(theme, 0)
            if crisis_count >= self._min_freq:
                confidence = crisis_count / count
                if confidence >= self._min_confidence:
                    patterns.append(
                        TriggerPattern(
                            trigger=theme,
                            response="crisis_escalation",
                            co_occurrence_count=crisis_count,
                            confidence=round(confidence, 2),
                            example_sessions=[
                                s.session_id for s in sessions if theme in s.themes and s.session_id in crisis_sessions
                            ][:3],
                        )
                    )

        return patterns

    def _analyze_interventions(self, sessions: list[SessionSummary]) -> list[InterventionResult]:
        """Track intervention effectiveness via valence changes."""
        if len(sessions) < 2:
            return []

        sorted_sessions = sorted(sessions, key=lambda s: s.timestamp_ms)
        mid = len(sorted_sessions) // 2

        early = sorted_sessions[:mid]
        late = sorted_sessions[mid:]

        early_valence = sum(s.emotional_arc.avg_valence for s in early) / len(early)
        late_valence = sum(s.emotional_arc.avg_valence for s in late) / len(late)

        effectiveness = late_valence - early_valence

        return [
            InterventionResult(
                intervention="ongoing_therapy",
                effectiveness_score=round(effectiveness, 3),
                sessions_applied=len(sessions),
                avg_valence_before=round(early_valence, 3),
                avg_valence_after=round(late_valence, 3),
            )
        ]

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        """Simple linear regression slope."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator
