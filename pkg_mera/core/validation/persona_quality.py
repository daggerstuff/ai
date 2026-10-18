"""Persona style validation for therapeutic dialogues."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PersonaQualityResult:
    score: float
    warnings: list[str] = field(default_factory=list)


class PersonaQuality:
    """Evaluate persona consistency and response tone quality."""

    def __init__(self) -> None:
        self.required_traits = {"empathetic", "non_judgmental", "structured", "safe_boundaries"}

    def evaluate(self, messages: list[str] | str) -> PersonaQualityResult:
        if isinstance(messages, list):
            content = " ".join(messages)
        elif isinstance(messages, str):
            content = messages
        else:
            raise TypeError("messages must be list[str] or str")

        lower = content.lower()
        score = 1.0
        warnings: list[str] = []

        if "i" not in lower and len(content.split()) < 4:
            score -= 0.3
            warnings.append("very_short")
        if "you" not in lower:
            score -= 0.2
            warnings.append("low_directness")
        if "unsafe" in lower or "self-harm" in lower:
            score -= 0.5
            warnings.append("high_risk_term")

        score = max(0.0, round(score, 3))
        return PersonaQualityResult(score=score, warnings=warnings)


__all__ = ["PersonaQuality", "PersonaQualityResult"]
