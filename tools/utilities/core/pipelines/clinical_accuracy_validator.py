"""Clinical accuracy checks for therapeutic responses and annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ClinicalAccuracyResult:
    score: float
    is_accurate: bool
    issues: list[str]


class ClinicalAccuracyValidator:
    """Semantic validator for clinical correctness and safe limits.

    Uses word-boundary matching and negation awareness to avoid false positives
    from naive substring heuristics.
    """

    CLINICAL_TERMS = {
        "grounded": [
            "cognitive behavioral",
            "behavioral activation",
            "sleep hygiene",
            "grounding",
            "mindfulness",
            "deep breathing",
            "progressive muscle",
            "cognitive restructuring",
            "exposure therapy",
            "psychoeducation",
            "coping skills",
            "emotional regulation",
            "distress tolerance",
            "interpersonal effectiveness",
            "radical acceptance",
        ],
        "risk": [
            "self-harm",
            "suicidal",
            "suicide",
            "crisis",
            "safe space",
            "emergency",
            "safety plan",
            "risk assessment",
            "harm reduction",
            "protective factors",
            "warning signs",
            "danger",
            "lethal means",
        ],
    }
    CONTRAINDICATIONS = [
        "replace medication",
        "stop taking",
        "discontinue medication",
        "ignore clinician",
        "ignore doctor",
        "ignore therapist",
        "stop medication",
        "skip medication",
        "forbidden",
        "self-diagnose",
        "no need for professional",
    ]
    NEGATION_TERMS = {
        "not",
        "no",
        "never",
        "do not",
        "don't",
        "does not",
        "doesn't",
        "did not",
        "didn't",
        "should not",
        "shouldn't",
        "must not",
        "mustn't",
        "cannot",
        "can't",
        "won't",
        "will not",
    }
    # Terms that indicate clinical assessment context
    ASSESSMENT_TERMS = [
        "assessment",
        "evaluate",
        "evaluated",
        "screening",
        "screened",
        "diagnosed",
        "diagnosis",
        "examined",
        "assessed",
        "measured",
        "assess",
        "evaluate",
        "screen",
        "examine",
    ]

    def process(self, data: Any) -> ClinicalAccuracyResult:
        if data is None:
            raise ValueError("Clinical validation input cannot be None")
        text = self._extract_text(data)
        if not text:
            raise ValueError("No text content available for validation")

        issues: list[str] = []
        normalized = text.lower()

        has_grounding = self._has_any_term(normalized, self.CLINICAL_TERMS["grounded"])
        has_risk_context = self._has_any_term(normalized, self.CLINICAL_TERMS["risk"])

        contraindicated = False
        for term in self.CONTRAINDICATIONS:
            positions = self._find_term_positions(normalized, term)
            for pos in positions:
                if not self._is_negated(normalized, pos):
                    contraindicated = True
                    break
            if contraindicated:
                break

        has_assessment = self._has_any_term(normalized, self.ASSESSMENT_TERMS)
        has_symptoms = self._has_word(normalized, "symptoms") or self._has_word(normalized, "symptom")

        score = 1.0
        if contraindicated:
            score -= 0.5
            issues.append("Contains contraindicated clinical suggestion")
        if not has_grounding and not has_risk_context:
            score -= 0.25
            issues.append("Limited clinical grounding language")
        if has_symptoms and not has_assessment:
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

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+(?:'\w+)?\b", text)

    def _find_term_positions(self, text: str, term: str) -> list[int]:
        """Return character positions where *term* appears as a whole-phrase match."""
        pattern = r"\b" + re.escape(term) + r"\b"
        return [m.start() for m in re.finditer(pattern, text)]

    def _has_any_term(self, text: str, terms: list[str]) -> bool:
        return any(self._find_term_positions(text, t) for t in terms)

    def _has_word(self, text: str, word: str) -> bool:
        return bool(self._find_term_positions(text, word))

    def _is_negated(self, normalized: str, char_pos: int) -> bool:
        """Check whether a negation term appears within 3 tokens before *char_pos*."""
        preceding = normalized[:char_pos]
        preceding_tokens = self._tokenize(preceding)
        window = preceding_tokens[-3:] if len(preceding_tokens) >= 3 else preceding_tokens
        return any(neg in window for neg in self.NEGATION_TERMS)


__all__ = ["ClinicalAccuracyResult", "ClinicalAccuracyValidator"]
