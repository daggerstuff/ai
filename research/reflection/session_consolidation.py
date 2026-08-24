"""Session Consolidation — Sprint 4, Task 2.

End-of-session consolidation: theme extraction, emotional arc computation,
unresolved topic identification, and session summarization.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ai.research.schema import MemoryBlock

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmotionalArc:
    start_valence: float
    end_valence: float
    min_valence: float
    max_valence: float
    avg_valence: float
    trend: str  # "improving", "declining", "stable"
    volatility: float


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    tenant_id: str
    themes: list[str]
    emotional_arc: EmotionalArc
    unresolved_topics: list[str]
    summary_text: str
    memory_count: int
    timestamp_ms: int


SummarizerFn = Callable[[list[MemoryBlock], list[str], EmotionalArc], str]


class SessionConsolidator:
    """Consolidate a session's memories into a structured summary."""

    def __init__(self, summarizer: SummarizerFn | None = None) -> None:
        self._summarizer = summarizer or self._default_summarizer

    def consolidate(self, memories: list[MemoryBlock]) -> SessionSummary:
        """Run full session consolidation."""
        t0 = time.perf_counter()
        if not memories:
            raise ValueError("Cannot consolidate empty memory list")

        tenant_id = memories[0].tenantId
        session_id = memories[0].sessionId
        themes = self._extract_themes(memories)
        emotional_arc = self._compute_emotional_arc(memories)
        unresolved = self._identify_unresolved(memories)
        summary_text = self._summarizer(memories, themes, emotional_arc)

        elapsed = (time.perf_counter() - t0) * 1000

        result = SessionSummary(
            session_id=session_id,
            tenant_id=tenant_id,
            themes=themes,
            emotional_arc=emotional_arc,
            unresolved_topics=unresolved,
            summary_text=summary_text,
            memory_count=len(memories),
            timestamp_ms=int(time.time() * 1000),
        )
        log.info(
            "Session consolidation: %s → %d themes, %d unresolved in %.0f ms",
            session_id,
            len(themes),
            len(unresolved),
            elapsed,
        )
        return result

    @staticmethod
    def _extract_themes(memories: list[MemoryBlock]) -> list[str]:
        """Extract key themes from emotional categories."""
        category_counter: Counter[str] = Counter()
        for m in memories:
            for cat in m.emotions.categories or ["general"]:
                category_counter[cat] += 1
        return [cat for cat, _ in category_counter.most_common(10)]

    @staticmethod
    def _compute_emotional_arc(memories: list[MemoryBlock]) -> EmotionalArc:
        """Compute valence trajectory over time."""
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        valences = [m.emotions.valence for m in sorted_memories]

        if not valences:
            return EmotionalArc(0, 0, 0, 0, 0, "stable", 0)

        start = valences[0]
        end = valences[-1]
        avg = sum(valences) / len(valences)
        min_v = min(valences)
        max_v = max(valences)

        if len(valences) >= 3:
            first_half = sum(valences[: len(valences) // 2]) / (len(valences) // 2)
            second_half = sum(valences[len(valences) // 2 :]) / (len(valences) - len(valences) // 2)
            if second_half > first_half + 0.1:
                trend = "improving"
            elif second_half < first_half - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        volatility = (sum((v - avg) ** 2 for v in valences) / len(valences)) ** 0.5

        return EmotionalArc(
            start_valence=round(start, 3),
            end_valence=round(end, 3),
            min_valence=round(min_v, 3),
            max_valence=round(max_v, 3),
            avg_valence=round(avg, 3),
            trend=trend,
            volatility=round(volatility, 3),
        )

    @staticmethod
    def _identify_unresolved(memories: list[MemoryBlock]) -> list[str]:
        """Identify topics mentioned but not resolved."""
        crisis_memories = [m for m in memories if m.gating.crisisFlag]
        high_arousal = [m for m in memories if m.emotions.arousal > 0.7]
        negative_end = [
            m for m in memories if m.emotions.valence < -0.3 and m.timestamp == max(x.timestamp for x in memories)
        ]

        unresolved: list[str] = []
        if crisis_memories:
            unresolved.append("crisis_content_requires_followup")
        if high_arousal:
            topics = set()
            for m in high_arousal:
                topics.update(m.emotions.categories or ["high_arousal"])
            unresolved.extend([f"high_arousal:{t}" for t in topics])
        if negative_end:
            topics = set()
            for m in negative_end:
                topics.update(m.emotions.categories or ["negative_ending"])
            unresolved.extend([f"unresolved:{t}" for t in topics])

        return list(set(unresolved))

    @staticmethod
    def _default_summarizer(
        memories: list[MemoryBlock],
        themes: list[str],
        arc: EmotionalArc,
    ) -> str:
        top_themes = ", ".join(themes[:3]) if themes else "no clear themes"
        return (
            f"Session with {len(memories)} memories. "
            f"Key themes: {top_themes}. "
            f"Emotional arc: {arc.trend} (valence {arc.start_valence} → {arc.end_valence}). "
            f"Volatility: {arc.volatility:.2f}."
        )
