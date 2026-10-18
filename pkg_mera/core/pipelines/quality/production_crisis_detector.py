"""Production crisis detector used by core quality pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai.pkg_mera.core.pipelines.crisis_intervention_detector import CrisisInterventionDetector


class CrisisLevel(Enum):
    """Crisis level enumeration."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CrisisResult:
    """Result of crisis detection."""

    crisis_level: CrisisLevel
    confidence_score: float
    crisis_types: list[str]
    recommendations: list[str]


class ProductionCrisisDetector:
    """Production-style detector wrapper over the heuristic detector."""

    def __init__(self, use_strict_mode: bool = True) -> None:
        self.use_strict_mode = use_strict_mode
        self._detector = CrisisInterventionDetector()
        self._levels = {
            "low": CrisisLevel.LOW,
            "moderate": CrisisLevel.MODERATE,
            "elevated": CrisisLevel.HIGH,
            "high": CrisisLevel.HIGH,
            "critical": CrisisLevel.EMERGENCY,
            "emergency": CrisisLevel.EMERGENCY,
        }

    def detect_crisis(self, data: dict[str, Any]) -> CrisisResult:
        result = self._detector.process(data)
        crisis_level = self._levels.get(result.severity, CrisisLevel.NONE)

        recs = list(result.recommendations)
        if self.use_strict_mode and crisis_level in {CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY}:
            recs.append("require_immediate_clinical_review")

        return CrisisResult(
            crisis_level=crisis_level,
            confidence_score=result.score,
            crisis_types=list(result.matches),
            recommendations=recs,
        )

    def _analyze_crisis_indicators(self, text: str) -> dict[str, Any]:
        result = self._detector.process(text)
        return {
            "severity": result.severity,
            "score": result.score,
            "crisis_type": result.crisis_type,
            "matches": result.matches,
        }

    def _calculate_crisis_level_production(self, indicators: dict[str, Any]) -> CrisisLevel:
        return self._levels.get(indicators.get("severity", "none"), CrisisLevel.NONE)


__all__ = ["CrisisLevel", "CrisisResult", "ProductionCrisisDetector"]
