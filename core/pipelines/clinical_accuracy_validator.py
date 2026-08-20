"""Clinical accuracy checks for therapeutic responses and annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ClinicalAccuracyResult:
    score: float
    is_accurate: bool
    issues: list[str]


class ClinicalAccuracyValidator:
    """Lightweight semantic validator for clinical correctness and safe limits."""

    CLINICAL_TERMS = {
        "grounded": ["cognitive behavioral", "behavioral activation", "sleep hygiene", "grounding"],
        "risk": ["self-harm", "suicidal", "crisis", "safe space", "emergency"],
    }
    CONTRAINDICATIONS = ["replace medication", "stop taking", "ignore clinician", "forbidden"]

    def process(self, data: Any) -> ClinicalAccuracyResult:
        if data is None:
            raise ValueError("Clinical validation input cannot be None")
        text = self._extract_text(data)
        if not text:
            raise ValueError("No text content available for validation")

        issues: list[str] = []
        normalized = text.lower()

        has_grounding = any(term in normalized for term in self.CLINICAL_TERMS["grounded"])
        has_risk_context = any(term in normalized for term in self.CLINICAL_TERMS["risk"])
        contraindicated = any(term in normalized for term in self.CONTRAINDICATIONS)

        score = 1.0
        if contraindicated:
            score -= 0.5
            issues.append("Contains contraindicated clinical suggestion")
        if not has_grounding and not has_risk_context:
            score -= 0.25
            issues.append("Limited clinical grounding language")
        if "symptoms" in normalized and "assessment" not in normalized:
            score -= 0.15
            issues.append("Clinical context appears incomplete")

        score = max(0.0, round(score, 3))
        return ClinicalAccuracyResult(
            score=score,
            is_accurate=score >= 0.7,
            issues=issues,
        )

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("text", "response", "content", "message"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
            if isinstance(data.get("messages"), list):
                return " ".join(msg.get("content", "") for msg in data["messages"] if isinstance(msg, dict))
        raise TypeError("Unsupported payload type for ClinicalAccuracyValidator")


__all__ = ["ClinicalAccuracyResult", "ClinicalAccuracyValidator"]
