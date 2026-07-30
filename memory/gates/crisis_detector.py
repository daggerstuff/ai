"""PIX-511 memory-gate crisis detection.

This gate wraps the core crisis intervention detector with lightweight context
handling for negated and historical mentions before mapping findings to the
memory-ingestion gate decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from re import Pattern
from typing import Any, ClassVar

from ai.pkg_mera.core.pipelines.crisis_intervention_detector import (
    CrisisInterventionDetector,
    CrisisInterventionResult,
)
from ai.memory.gates import GateDecision, GateResult


class CrisisTier(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    NONE = "none"


@dataclass
class CrisisDetectionResult:
    tier: CrisisTier
    score: float
    crisis_type: str
    matches: list[str]
    confidence: float
    crisis_flag: bool
    evidence_spans: list[str]
    requires_preservation: bool
    recommendations: list[str] = field(default_factory=list)


class CrisisDetector:
    """Three-tier crisis detector for memory ingestion gating."""

    CRITICAL_THRESHOLD: ClassVar[float] = 0.8
    HIGH_THRESHOLD: ClassVar[float] = 0.5
    MODERATE_THRESHOLD: ClassVar[float] = 0.25

    CRITICAL_KEYWORDS: tuple[str, ...] = (
        "suicide",
        "kill myself",
        "hurt myself",
        "end it all",
        "want to die",
        "no reason to live",
    )
    HIGH_KEYWORDS: tuple[str, ...] = (
        "hopeless",
        "worthless",
        "can't continue",
        "cut",
        "self-harm",
    )
    MODERATE_KEYWORDS: tuple[str, ...] = (
        "anxious",
        "down",
        "struggling",
        "panic",
        "overwhelmed",
    )

    _KEYWORD_SCORES: ClassVar[dict[CrisisTier, float]] = {
        CrisisTier.CRITICAL: CRITICAL_THRESHOLD,
        CrisisTier.HIGH: HIGH_THRESHOLD,
        CrisisTier.MODERATE: MODERATE_THRESHOLD,
    }

    def __init__(self) -> None:
        self._detector = CrisisInterventionDetector()
        self._negation_patterns: list[Pattern[str]] = [
            re.compile(r"\b(no|not|never|denies|denied|without|r/o)\b", re.IGNORECASE)
        ]
        self._temporal_patterns: list[Pattern[str]] = [
            re.compile(
                r"\b(used to|previously|in the past|used to be|was feeling|had been)\b",
                re.IGNORECASE,
            )
        ]
        self._keyword_patterns = self._compile_keyword_patterns()

    def detect(self, content: str) -> CrisisDetectionResult:
        core_result = self._detector.process(content)
        evidence_spans, keyword_tier = self._extract_evidence_spans(content)

        base_score = max(core_result.score, self._KEYWORD_SCORES.get(keyword_tier, 0.0))
        negated = self._has_negated_evidence(content, evidence_spans)
        historical = self._has_temporal_context(content, evidence_spans)

        # When *all* crisis evidence is negated (e.g. "I would never hurt
        # myself"), there is no actual crisis risk — fully suppress.
        if negated and evidence_spans:
            score = 0.0
            confidence = 0.0
            tier = CrisisTier.NONE
            crisis_flag = False
        else:
            confidence = base_score * (0.5 if historical else 1.0)
            score = confidence
            tier = self._tier_for_score(score)
            crisis_flag = tier != CrisisTier.NONE

        matches = evidence_spans or list(core_result.matches)
        crisis_type = self._crisis_type(core_result, keyword_tier, matches)

        return CrisisDetectionResult(
            tier=tier,
            score=round(score, 4),
            crisis_type=crisis_type,
            matches=matches,
            confidence=round(confidence, 4),
            crisis_flag=crisis_flag,
            evidence_spans=evidence_spans,
            requires_preservation=crisis_flag,
            recommendations=self._recommendations(core_result, tier, negated, historical),
        )

    def evaluate(self, content: str) -> GateResult:
        result = self.detect(content)
        details = [
            f"tier: {result.tier.value}",
            f"score: {result.score:.4f}",
            f"crisis_type: {result.crisis_type}",
            f"crisis_flag: {result.crisis_flag}",
            f"requires_preservation: {result.requires_preservation}",
            f"matches: {', '.join(result.matches) or 'none'}",
        ]

        if result.tier == CrisisTier.CRITICAL:
            return GateResult(
                gate="gate1_crisis",
                decision=GateDecision.BLOCK,
                reason="Immediate crisis intervention required",
                confidence=result.confidence,
                details=details,
            )
        if result.tier == CrisisTier.HIGH:
            return GateResult(
                gate="gate1_crisis",
                decision=GateDecision.ESCALATE,
                reason="Clinical review required for high-risk content",
                confidence=result.confidence,
                details=details,
            )
        if result.tier == CrisisTier.MODERATE:
            return GateResult(
                gate="gate1_crisis",
                decision=GateDecision.PASS,
                reason="Moderate risk flagged for monitoring",
                confidence=result.confidence,
                details=details,
            )
        return GateResult(
            gate="gate1_crisis",
            decision=GateDecision.PASS,
            reason="No crisis indicators detected",
            confidence=result.confidence,
            details=details,
        )

    def to_dict(self, result: CrisisDetectionResult) -> dict[str, Any]:
        return {
            "tier": result.tier.value,
            "score": round(result.score, 4),
            "crisis_type": result.crisis_type,
            "matches": list(result.matches),
            "confidence": round(result.confidence, 4),
            "crisis_flag": result.crisis_flag,
            "evidence_spans": list(result.evidence_spans),
            "requires_preservation": result.requires_preservation,
            "recommendations": list(result.recommendations),
        }

    def _compile_keyword_patterns(self) -> dict[CrisisTier, list[Pattern[str]]]:
        return {
            CrisisTier.CRITICAL: [self._keyword_pattern(keyword) for keyword in self.CRITICAL_KEYWORDS],
            CrisisTier.HIGH: [self._keyword_pattern(keyword) for keyword in self.HIGH_KEYWORDS],
            CrisisTier.MODERATE: [self._keyword_pattern(keyword) for keyword in self.MODERATE_KEYWORDS],
        }

    def _keyword_pattern(self, keyword: str) -> Pattern[str]:
        escaped = re.escape(keyword).replace(r"\-", r"[-\s]?")
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    def _extract_evidence_spans(self, content: str) -> tuple[list[str], CrisisTier]:
        spans: list[str] = []
        strongest = CrisisTier.NONE
        for tier in (CrisisTier.CRITICAL, CrisisTier.HIGH, CrisisTier.MODERATE):
            tier_matches: list[str] = []
            for pattern in self._keyword_patterns[tier]:
                tier_matches.extend(match.group(0) for match in pattern.finditer(content))
            if tier_matches and strongest == CrisisTier.NONE:
                strongest = tier
            spans.extend(tier_matches)
        return sorted(set(spans), key=str.lower), strongest

    def _has_negated_evidence(self, content: str, evidence_spans: list[str]) -> bool:
        if not evidence_spans:
            return False
        return all(self._span_has_context(content, span, self._negation_patterns) for span in evidence_spans)

    def _has_temporal_context(self, content: str, evidence_spans: list[str]) -> bool:
        if evidence_spans:
            return any(self._span_has_context(content, span, self._temporal_patterns) for span in evidence_spans)
        return any(pattern.search(content) for pattern in self._temporal_patterns)

    def _span_has_context(
        self,
        content: str,
        span: str,
        patterns: list[Pattern[str]],
        window_size: int = 48,
    ) -> bool:
        for match in re.finditer(re.escape(span), content, flags=re.IGNORECASE):
            start = max(0, match.start() - window_size)
            end = min(len(content), match.end() + window_size)
            window = content[start:end]
            if any(pattern.search(window) for pattern in patterns):
                return True
        return False

    def _tier_for_score(self, score: float) -> CrisisTier:
        if score >= self.CRITICAL_THRESHOLD:
            return CrisisTier.CRITICAL
        if score >= self.HIGH_THRESHOLD:
            return CrisisTier.HIGH
        if score >= self.MODERATE_THRESHOLD:
            return CrisisTier.MODERATE
        return CrisisTier.NONE

    def _crisis_type(
        self,
        core_result: CrisisInterventionResult,
        keyword_tier: CrisisTier,
        matches: list[str],
    ) -> str:
        if core_result.crisis_type != "none":
            return core_result.crisis_type
        if not matches or keyword_tier == CrisisTier.NONE:
            return "none"
        return f"{keyword_tier.value}_risk"

    def _recommendations(
        self,
        core_result: CrisisInterventionResult,
        tier: CrisisTier,
        negated: bool,
        historical: bool,
    ) -> list[str]:
        recommendations = list(core_result.recommendations)
        if tier == CrisisTier.CRITICAL:
            recommendations.append("Block deletion and initiate immediate crisis workflow")
        elif tier == CrisisTier.HIGH:
            recommendations.append("Preserve content and escalate for clinical review")
        elif tier == CrisisTier.MODERATE:
            recommendations.append("Flag for monitoring while allowing ingestion")
        if negated:
            recommendations.append("Crisis language appears negated; reduced confidence")
        if historical:
            recommendations.append("Crisis language appears historical; reduced confidence")
        return sorted(set(recommendations))


__all__ = [
    "CrisisDetectionResult",
    "CrisisDetector",
    "CrisisTier",
]
