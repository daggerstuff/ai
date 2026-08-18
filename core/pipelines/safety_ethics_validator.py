"""Safety and ethics validation for training/inference payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyValidationResult:
    is_safe: bool
    score: float
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class SafetyEthicsValidator:
    """Heuristic validator that flags unsafe or non-consented content."""

    def __init__(self) -> None:
        self.prohibited = {
            "self_harm": [r"\bkill myself\b", r"\bno reason to live\b", r"\bharm myself\b"],
            "non_consent": [r"\bdox\b", r"\bsteal\b", r"\bmanipulat\b"],
            "medical_misuse": [r"\bofficial diagnosis\b", r"\bstop med\b", r"\bignore doctor\b"],
            "privacy": [r"\bssn\b", r"\bpassword\b", r"\bcredit card\b"],
        }

    def process(self, data: Any) -> SafetyValidationResult:
        if data is None:
            raise ValueError("Input payload cannot be None")

        text = self._extract_text(data)

        violations: list[str] = []
        for category, patterns in self.prohibited.items():
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                violations.append(category)

        risk = len(violations) / max(len(self.prohibited), 1)
        score = round(1.0 - risk, 3)
        is_safe = len(violations) == 0

        suggestions = []
        if "privacy" in violations:
            suggestions.append("Remove personal identifiers before processing")
        if "medical_misuse" in violations:
            suggestions.append("Route through medical policy review")
        if "self_harm" in violations:
            suggestions.append("Escalate to safety and crisis protocol")

        return SafetyValidationResult(
            is_safe=is_safe,
            score=score,
            violations=violations,
            suggestions=suggestions,
        )

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise TypeError("Unsupported payload type for SafetyEthicsValidator")

        for key in ("text", "content", "message", "prompt", "input"):
            value = data.get(key)
            if isinstance(value, str):
                return value

        if isinstance(data.get("messages"), list):
            chunks = []
            for entry in data["messages"]:
                if isinstance(entry, dict):
                    c = entry.get("content")
                    if isinstance(c, str):
                        chunks.append(c)
            return " ".join(chunks)

        return ""


__all__ = ["SafetyEthicsValidator", "SafetyValidationResult"]
