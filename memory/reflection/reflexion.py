"""Reflexion Framework — Sprint 4, Task 1.

Implements the Reflexion pattern (Shinn et al., 2023):
Action → Feedback → Reflection → Context Update.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

log = logging.getLogger(__name__)


class FeedbackType(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class ActionFeedbackPair:
    action: str
    feedback: str
    feedback_type: FeedbackType
    timestamp_ms: int
    session_id: str


@dataclass(frozen=True)
class VerbalReflection:
    reflection_id: str
    what_went_well: list[str]
    what_went_wrong: list[str]
    what_to_change: list[str]
    source_pairs: list[ActionFeedbackPair]
    confidence: float


@dataclass
class ReflexionResult:
    reflections: list[VerbalReflection]
    context_updates: list[str]
    memories_to_update: list[str]
    elapsed_ms: float


ReflectionGeneratorFn = Callable[[list[ActionFeedbackPair]], str]


class ReflexionEngine:
    """Three-component reflection loop: action, feedback, reflection."""

    def __init__(
        self,
        generator: ReflectionGeneratorFn | None = None,
        min_pairs_for_reflection: int = 3,
    ) -> None:
        self._generator = generator or self._default_generator
        self._min_pairs = min_pairs_for_reflection
        self._trajectories: dict[str, list[ActionFeedbackPair]] = {}

    def record_action(
        self,
        action: str,
        feedback: str,
        feedback_type: FeedbackType,
        session_id: str,
    ) -> None:
        pair = ActionFeedbackPair(
            action=action,
            feedback=feedback,
            feedback_type=feedback_type,
            timestamp_ms=int(time.time() * 1000),
            session_id=session_id,
        )
        self._trajectories.setdefault(session_id, []).append(pair)

    def reflect(self, session_id: str) -> ReflexionResult | None:
        """Generate reflection for a session's trajectory."""
        t0 = time.perf_counter()
        pairs = self._trajectories.get(session_id, [])
        if len(pairs) < self._min_pairs:
            return None

        raw_reflection = self._generator(pairs)
        parsed = self._parse_reflection(raw_reflection, pairs)

        context_updates = []
        for r in [parsed]:
            context_updates.extend(r.what_to_change)

        result = ReflexionResult(
            reflections=[parsed],
            context_updates=context_updates,
            memories_to_update=[],
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        log.info(
            "Reflexion: %d pairs → %d insights in %.0f ms",
            len(pairs),
            len(context_updates),
            result.elapsed_ms,
        )
        return result

    def reflect_all(self) -> dict[str, ReflexionResult]:
        """Generate reflections for all sessions with sufficient trajectory."""
        results: dict[str, ReflexionResult] = {}
        for session_id in self._trajectories:
            result = self.reflect(session_id)
            if result is not None:
                results[session_id] = result
        return results

    def get_trajectory(self, session_id: str) -> list[ActionFeedbackPair]:
        return list(self._trajectories.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        self._trajectories.pop(session_id, None)

    @staticmethod
    def _parse_reflection(raw: str, pairs: list[ActionFeedbackPair]) -> VerbalReflection:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        went_well = []
        went_wrong = []
        to_change = []
        section = None
        for line in lines:
            lower = line.lower()
            if "went well" in lower or "success" in lower or "positive" in lower:
                section = "well"
                continue
            if "went wrong" in lower or "fail" in lower or "negative" in lower:
                section = "wrong"
                continue
            if "change" in lower or "differently" in lower or "improve" in lower:
                section = "change"
                continue
            if section == "well" and line.startswith("-"):
                went_well.append(line.lstrip("- "))
            elif section == "wrong" and line.startswith("-"):
                went_wrong.append(line.lstrip("- "))
            elif section == "change" and line.startswith("-"):
                to_change.append(line.lstrip("- "))

        failures = sum(1 for p in pairs if p.feedback_type == FeedbackType.FAILURE)
        confidence = 1.0 - (failures / len(pairs)) if pairs else 0.5

        return VerbalReflection(
            reflection_id=f"reflexion_{int(time.time() * 1000)}",
            what_went_well=went_well or ["Trajectory completed without major issues"],
            what_went_wrong=went_wrong or [],
            what_to_change=to_change or [],
            source_pairs=pairs,
            confidence=round(confidence, 2),
        )

    @staticmethod
    def _default_generator(pairs: list[ActionFeedbackPair]) -> str:
        successes = [p for p in pairs if p.feedback_type == FeedbackType.SUCCESS]
        failures = [p for p in pairs if p.feedback_type == FeedbackType.FAILURE]
        partials = [p for p in pairs if p.feedback_type == FeedbackType.PARTIAL]

        lines = ["Reflection:"]
        lines.append("What went well:")
        for p in successes:
            lines.append(f"- {p.action}: {p.feedback}")
        if not successes:
            lines.append("- No clear successes identified")

        lines.append("What went wrong:")
        for p in failures:
            lines.append(f"- {p.action}: {p.feedback}")
        if not failures:
            lines.append("- No clear failures identified")

        lines.append("What to change next time:")
        for p in partials:
            lines.append(f"- Refine: {p.action} (was {p.feedback})")
        if not partials:
            lines.append("- Continue current approach")

        return "\n".join(lines)

    @property
    def session_count(self) -> int:
        return len(self._trajectories)

    @property
    def total_pairs(self) -> int:
        return sum(len(p) for p in self._trajectories.values())
