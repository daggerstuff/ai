"""Dream-Reflection Integration — Sprint 4, Task 4.

Connects dream consolidation output to reflection prompts,
uses schema extraction to inform cross-session patterns,
and creates a feedback loop between reflection and dream priorities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ai.memory.consolidation.rem_dream import DreamResult

from .pattern_detection import PatternReport
from .session_consolidation import SessionSummary

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DreamReflectionInsight:
    insight_id: str
    source: str  # "dream" | "reflection" | "pattern"
    content: str
    confidence: float
    related_memory_ids: list[str]


@dataclass
class DreamReflectionResult:
    insights: list[DreamReflectionInsight]
    dream_priorities: dict[str, float]
    reflection_enhancements: list[str]
    elapsed_ms: float


class DreamReflectionIntegrator:
    """Bridge between dream consolidation and reflection systems."""

    def __init__(self) -> None:
        self._insight_counter = 0

    def integrate(
        self,
        dream_result: DreamResult,
        session_summary: SessionSummary,
        pattern_report: PatternReport | None = None,
    ) -> DreamReflectionResult:
        """Integrate dream insights with session reflection."""
        t0 = time.perf_counter()

        insights: list[DreamReflectionInsight] = []
        insights.extend(self._extract_dream_insights(dream_result))
        insights.extend(self._extract_session_insights(session_summary))
        if pattern_report:
            insights.extend(self._extract_pattern_insights(pattern_report))

        dream_priorities = self._compute_dream_priorities(dream_result, session_summary, pattern_report)
        reflection_enhancements = self._build_reflection_enhancements(insights, session_summary)

        elapsed = (time.perf_counter() - t0) * 1000

        result = DreamReflectionResult(
            insights=insights,
            dream_priorities=dream_priorities,
            reflection_enhancements=reflection_enhancements,
            elapsed_ms=round(elapsed, 2),
        )
        log.info(
            "Dream-reflection integration: %d insights, %d priority updates in %.0f ms",
            len(insights),
            len(dream_priorities),
            elapsed,
        )
        return result

    def _extract_dream_insights(self, dream_result: DreamResult) -> list[DreamReflectionInsight]:
        insights: list[DreamReflectionInsight] = []

        for schema in dream_result.schemas:
            self._insight_counter += 1
            insights.append(
                DreamReflectionInsight(
                    insight_id=f"insight_dream_{self._insight_counter}",
                    source="dream",
                    content=schema.generalization,
                    confidence=schema.confidence,
                    related_memory_ids=schema.source_memory_ids,
                )
            )

        for link in dream_result.cross_links:
            if link.link_type == "emotional_co_occurrence":
                self._insight_counter += 1
                insights.append(
                    DreamReflectionInsight(
                        insight_id=f"insight_dream_{self._insight_counter}",
                        source="dream",
                        content=f"Emotional co-occurrence: memories {link.memory_a_id} and {link.memory_b_id} (similarity {link.similarity:.2f})",
                        confidence=link.similarity,
                        related_memory_ids=[link.memory_a_id, link.memory_b_id],
                    )
                )

        return insights

    def _extract_session_insights(self, summary: SessionSummary) -> list[DreamReflectionInsight]:
        insights: list[DreamReflectionInsight] = []

        if summary.unresolved_topics:
            self._insight_counter += 1
            insights.append(
                DreamReflectionInsight(
                    insight_id=f"insight_session_{self._insight_counter}",
                    source="reflection",
                    content=f"Unresolved topics: {', '.join(summary.unresolved_topics)}",
                    confidence=0.8,
                    related_memory_ids=[],
                )
            )

        arc = summary.emotional_arc
        if arc.trend != "stable":
            self._insight_counter += 1
            insights.append(
                DreamReflectionInsight(
                    insight_id=f"insight_session_{self._insight_counter}",
                    source="reflection",
                    content=f"Emotional trajectory: {arc.trend} ({arc.start_valence} → {arc.end_valence})",
                    confidence=0.7,
                    related_memory_ids=[],
                )
            )

        return insights

    def _extract_pattern_insights(self, report: PatternReport) -> list[DreamReflectionInsight]:
        insights: list[DreamReflectionInsight] = []

        for theme in report.recurring_themes:
            if theme.frequency >= 3:
                self._insight_counter += 1
                insights.append(
                    DreamReflectionInsight(
                        insight_id=f"insight_pattern_{self._insight_counter}",
                        source="pattern",
                        content=f"Recurring theme '{theme.theme}' across {theme.frequency} sessions (trend: {theme.trend})",
                        confidence=min(theme.frequency / 5.0, 1.0),
                        related_memory_ids=[],
                    )
                )

        for trigger in report.trigger_patterns:
            self._insight_counter += 1
            insights.append(
                DreamReflectionInsight(
                    insight_id=f"insight_pattern_{self._insight_counter}",
                    source="pattern",
                    content=f"Trigger pattern: '{trigger.trigger}' → '{trigger.response}' (confidence {trigger.confidence:.2f})",
                    confidence=trigger.confidence,
                    related_memory_ids=[],
                )
            )

        return insights

    def _compute_dream_priorities(
        self,
        dream_result: DreamResult,
        summary: SessionSummary,
        pattern_report: PatternReport | None = None,
    ) -> dict[str, float]:
        """Compute priorities for next dream cycle based on reflection."""
        priorities: dict[str, float] = {}

        for topic in summary.unresolved_topics:
            priorities[topic] = 0.9

        if summary.emotional_arc.trend == "declining":
            priorities["crisis_monitoring"] = 0.95

        if pattern_report:
            for theme in pattern_report.recurring_themes:
                if theme.trend == "declining":
                    priorities[f"theme:{theme.theme}"] = 0.85

        for schema in dream_result.schemas:
            if schema.confidence >= 0.5:
                priorities[f"schema:{schema.schema_id}"] = schema.confidence * 0.7

        return priorities

    def _build_reflection_enhancements(
        self,
        insights: list[DreamReflectionInsight],
        summary: SessionSummary,
    ) -> list[str]:
        enhancements: list[str] = []

        dream_insights = [i for i in insights if i.source == "dream"]
        if dream_insights:
            enhancements.append(f"Dream analysis identified {len(dream_insights)} patterns to reflect on")

        pattern_insights = [i for i in insights if i.source == "pattern"]
        if pattern_insights:
            enhancements.append(
                f"Cross-session patterns detected: {', '.join(i.content[:50] for i in pattern_insights[:3])}"
            )

        if summary.emotional_arc.volatility > 0.3:
            enhancements.append("High emotional volatility detected — prioritize stability in next session")

        return enhancements
