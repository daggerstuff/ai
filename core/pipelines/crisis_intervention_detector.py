"""Crisis intervention detection and escalation support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class CrisisInterventionResult:
    """Structured crisis assessment output."""

    flagged: bool
    crisis_type: str
    severity: str
    score: float
    matches: list[str]
    recommendations: list[str] = field(default_factory=list)


class CrisisInterventionDetector:
    """Production-oriented heuristic crisis detector with escalation actions."""

    CRISIS_KEYWORDS: dict[str, tuple[float, list[str]]] = {
        "suicidal_ideation": (
            0.95,
            [
                r"\bi want to die\b",
                r"\bi want to kill myself\b",
                r"\bend my life\b",
                r"\bno reason to live\b",
                r"\bcan't go on\b",
            ],
        ),
        "self_harm": (
            0.85,
            [
                r"\bcut(?:ting)?\s+myself\b",
                r"\bhurt(?:ing)?\s+myself\b",
                r"\bself.?harm(?:ing)?\b",
            ],
        ),
        "violence": (
            0.75,
            [
                r"\bhurt(?:ing)?\s+.*\bothers\b",
                r"\bkill(?:ing)?\s+(?:someone|others|people|everybody|everyone|him|her|them|myself)\b",
                r"\battack(?:ing)?\s+(?:someone|others|people)\b",
            ],
        ),
        "substance_abuse": (
            0.55,
            [
                r"\boverdose\b",
                r"\bintoxicated\b",
                r"\b(?:going through|experiencing|having|suffering(?:\s+from)?)\s+withdrawal\b",
                r"\bwithdrawal\s+symptoms?\b",
            ],
        ),
        "panic": (0.45, [r"\bcan't breathe\b", r"\bpanic(?:\s+attack)?\b", r"\bheart is racing\b"]),
    }
    NEGATION_TERMS = frozenset(
        {
            "not",
            "never",
            "don't",
            "dont",
            "won't",
            "wont",
            "can't",
            "cant",
            "couldn't",
            "couldnt",
            "haven't",
            "havent",
            "isn't",
            "isnt",
            "aren't",
            "arent",
            "wasn't",
            "wasnt",
            "weren't",
            "werent",
            "didn't",
            "didnt",
            "wouldn't",
            "wouldnt",
            "shouldn't",
            "shouldnt",
        }
    )
    SEVERITY_BANDS = {
        "critical": 0.8,
        "high": 0.6,
        "elevated": 0.4,
        "moderate": 0.25,
        "low": 0.0,
    }

    def __init__(self) -> None:
        self.escalation_contacts = {
            "critical": ["911", "988"],
            "high": ["988"],
        }

    def process(self, data: dict[str, Any] | str) -> CrisisInterventionResult:
        if data is None:
            raise ValueError("data must not be None")
        text = self._extract_text(data).lower()
        if not text:
            return CrisisInterventionResult(False, "none", "none", 0.0, [])

        matches, score, crisis_type = self._detect(text)
        severity = self._severity_label(score)

        recommendations = []
        if score >= self.SEVERITY_BANDS["critical"]:
            recommendations.append("Immediate human escalation required")
            contacts = self.escalation_contacts.get("critical", [])
            recommendations.append(f"Escalate to: {', '.join(contacts)}")
        elif score >= self.SEVERITY_BANDS["high"]:
            recommendations.append("Queue for clinical review within 10 minutes")
        elif score >= self.SEVERITY_BANDS["elevated"]:
            recommendations.append("Increase monitoring and add safety check")
        else:
            recommendations.append("No immediate intervention required")

        return CrisisInterventionResult(
            flagged=score >= self.SEVERITY_BANDS["elevated"],
            crisis_type=crisis_type,
            severity=severity,
            score=score,
            matches=matches,
            recommendations=recommendations,
        )

    def _extract_text(self, data: dict[str, Any] | str) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise ValueError("data must be mapping or text")

        for key in ("text", "content", "message", "query", "input"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # Fallback flatten conversation-like payloads
        if isinstance(data.get("messages"), list):
            parts: list[str] = []
            for message in data["messages"]:
                if isinstance(message, dict):
                    role_content = message.get("content")
                    if isinstance(role_content, str):
                        parts.append(role_content)
            return " ".join(parts)

        return ""

    def _is_negated(self, text: str, match: re.Match[str]) -> bool:
        """Check whether a regex match is preceded by a negation term."""
        prefix_words = text[: match.start()].split()
        window = prefix_words[-3:] if len(prefix_words) >= 3 else prefix_words
        for word in window:
            stripped = word.strip("'\".,!?;:()[]")
            if stripped.lower() in self.NEGATION_TERMS:
                return True
        return False

    def _detect(self, text: str) -> tuple[list[str], float, str]:
        found: list[str] = []
        max_score = 0.0
        crisis_type = "none"

        for label, (base_score, patterns) in self.CRISIS_KEYWORDS.items():
            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m and not self._is_negated(text, m):
                    found.append(label)
                    max_score = max(max_score, base_score)
                    crisis_type = label
                    break

        return sorted(set(found)), float(max_score), crisis_type if found else "none"

    def _severity_label(self, score: float) -> str:
        if score >= self.SEVERITY_BANDS["critical"]:
            return "critical"
        if score >= self.SEVERITY_BANDS["high"]:
            return "high"
        if score >= self.SEVERITY_BANDS["elevated"]:
            return "elevated"
        if score >= self.SEVERITY_BANDS["moderate"]:
            return "moderate"
        return "low"

    def audit_event(self, result: CrisisInterventionResult) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "crisis_type": result.crisis_type,
            "severity": result.severity,
            "score": result.score,
            "matches": result.matches,
            "flagged": result.flagged,
            "recommendations": list(result.recommendations),
        }


__all__ = ["CrisisInterventionDetector", "CrisisInterventionResult"]
