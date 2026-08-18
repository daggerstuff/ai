"""Trauma-trigger filtering gate for memory ingestion.

PIX-511 Sprint 2 Gate 2 detects trauma-related indicators in memory content,
tracks opt-in user-specific triggers, and returns auditable gate decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ai.memory.gates import GateDecision, GateResult
from ai.memory.schema import MemoryGating


@dataclass
class TraumaFilterResult:
    """Result of trauma-trigger filtering for a memory block."""

    triggered: bool
    indicators: list[str] = field(default_factory=list)
    confidence: float = 0.0
    content_warning: str | None = None
    user_specific_matches: list[str] = field(default_factory=list)
    severity: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "indicators": self.indicators,
            "confidence": round(self.confidence, 4),
            "content_warning": self.content_warning,
            "user_specific_matches": self.user_specific_matches,
            "severity": self.severity,
        }


class TraumaFilter:
    """Lexicon and user-trigger based trauma content filter."""

    TRAUMA_LEXICON: ClassVar[dict[str, list[str]]] = {
        "abuse": ["abuse", "abused", "abuser", "molestation", "assault", "violence"],
        "neglect": ["neglect", "neglected", "abandoned", "ignored", "left alone"],
        "trauma": ["trauma", "traumatic", "ptsd", "flashback", "dissociat"],
        "grief": ["grief", "mourning", "loss", "death", "funeral", "bereavement"],
        "sexual": ["sexual abuse", "rape", "molested", "inappropriate touch"],
        "medical": ["medical trauma", "hospital", "surgery", "diagnosis", "cancer"],
    }

    _CONTENT_WARNING = "Content may contain trauma-related material"
    _GATE_NAME = "gate2_trauma"
    _SEVERITY_WEIGHTS: ClassVar[dict[str, float]] = {
        "none": 0.0,
        "low": 0.6,
        "medium": 0.8,
        "high": 1.0,
    }

    # Constants for magic values
    SEMANTIC_OVERLAP_THRESHOLD: ClassVar[float] = 0.75
    HIGH_SEVERITY_THRESHOLD: ClassVar[int] = 3
    MEDIUM_SEVERITY_THRESHOLD: ClassVar[int] = 2

    def __init__(self) -> None:
        self._patterns = {
            category: [self._compile_term(term) for term in terms] for category, terms in self.TRAUMA_LEXICON.items()
        }
        self._user_triggers: dict[str, set[str]] = {}

    def register_user_triggers(self, user_id: str, triggers: list[str]) -> None:
        """Register opt-in, user-specific trauma triggers."""
        if not user_id:
            return

        normalized = {trigger.strip().casefold() for trigger in triggers if trigger.strip()}
        if normalized:
            self._user_triggers.setdefault(user_id, set()).update(normalized)

    def filter(self, content: str, user_id: str | None = None) -> TraumaFilterResult:
        """Evaluate content for trauma indicators and content warning needs."""
        matched_categories: set[str] = set()
        indicators: list[str] = []

        for category, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    matched_categories.add(category)
                    indicators.append(category)
                    break

        user_specific_matches = self._match_user_triggers(content, user_id)
        severity = self._severity_for_categories(len(matched_categories))
        confidence = self._confidence(len(matched_categories), severity)
        triggered = bool(matched_categories or user_specific_matches)

        result = TraumaFilterResult(
            triggered=triggered,
            indicators=sorted(set(indicators)),
            confidence=confidence,
            content_warning=None,
            user_specific_matches=user_specific_matches,
            severity=severity,
        )
        result.content_warning = self.get_content_warning(result)
        return result

    def evaluate(self, content: str, user_id: str | None = None) -> GateResult:
        """Evaluate content through the memory gate interface."""
        result = self.filter(content, user_id=user_id)
        details = result.indicators + [f"user:{match}" for match in result.user_specific_matches]

        if result.severity == "high":
            return GateResult(
                gate=self._GATE_NAME,
                decision=GateDecision.ESCALATE,
                reason="Multiple trauma indicators detected",
                confidence=result.confidence,
                details=details,
            )

        if result.severity == "medium":
            return GateResult(
                gate=self._GATE_NAME,
                decision=GateDecision.PASS,
                reason="Trauma content flagged with content warning",
                confidence=result.confidence,
                details=details,
            )

        return GateResult(
            gate=self._GATE_NAME,
            decision=GateDecision.PASS,
            reason="No significant trauma indicators detected",
            confidence=result.confidence,
            details=details,
        )

    def get_content_warning(self, result: TraumaFilterResult) -> str | None:
        """Return warning text for medium or high severity results."""
        if result.severity in {"medium", "high"}:
            return self._CONTENT_WARNING
        return None

    def to_dict(self, result: TraumaFilterResult) -> dict[str, Any]:
        """Return a JSON-serializable trauma filter result."""
        gating = MemoryGating(traumaIndicators=result.indicators)
        return {
            **result.to_dict(),
            "gating": gating.model_dump(mode="json"),
        }

    @staticmethod
    def _compile_term(term: str) -> re.Pattern[str]:
        escaped = re.escape(term.strip()).replace(r"\ ", r"\s+")
        suffix = r"\w*" if term == "dissociat" else ""
        return re.compile(rf"\b{escaped}{suffix}\b", re.IGNORECASE)

    def _match_user_triggers(self, content: str, user_id: str | None) -> list[str]:
        if user_id is None:
            return []

        triggers = self._user_triggers.get(user_id, set())
        if not triggers:
            return []

        content_casefold = content.casefold()
        content_tokens = self._tokens(content_casefold)
        matches = [
            trigger
            for trigger in triggers
            if self._compile_term(trigger).search(content) or self._semantic_match(trigger, content_tokens)
        ]
        return sorted(matches)

    @staticmethod
    def _semantic_match(trigger: str, content_tokens: set[str]) -> bool:
        trigger_tokens = TraumaFilter._tokens(trigger)
        if not trigger_tokens:
            return False
        overlap = len(trigger_tokens & content_tokens) / len(trigger_tokens)
        return overlap >= TraumaFilter.SEMANTIC_OVERLAP_THRESHOLD

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", value.casefold()))

    @classmethod
    def _severity_for_categories(cls, matched_count: int) -> str:
        if matched_count >= cls.HIGH_SEVERITY_THRESHOLD:
            return "high"
        if matched_count == cls.MEDIUM_SEVERITY_THRESHOLD:
            return "medium"
        if matched_count == 1:
            return "low"
        return "none"

    @classmethod
    def _confidence(cls, matched_count: int, severity: str) -> float:
        category_ratio = matched_count / len(cls.TRAUMA_LEXICON)
        return min(1.0, category_ratio * cls._SEVERITY_WEIGHTS[severity])


__all__ = ["TraumaFilter", "TraumaFilterResult"]
