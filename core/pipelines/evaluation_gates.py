"""Evaluation gate framework used for pre-production quality checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    gate: str
    passed: bool
    score: float
    message: str


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)


class EvaluationGates:
    """Reusable gating system for data and model-response checks."""

    def __init__(self) -> None:
        self._gates: dict[str, Callable[[Any], tuple[bool, float, str]]] = {
            "non_empty": self._gate_non_empty,
            "no_empty_text": self._gate_no_empty_text,
            "min_length": self._gate_min_length,
        }

    def register_gate(self, name: str, gate: Callable[[Any], tuple[bool, float, str]]) -> None:
        self._gates[name] = gate

    def run(self, payload: Any) -> GateReport:
        report = GateReport()
        for name, gate in self._gates.items():
            passed, score, message = gate(payload)
            report.results.append(GateResult(name, bool(passed), float(score), message))
        return report

    def run_gates(self, payload: Any) -> GateReport:
        return self.run(payload)

    def evaluate(self, payload: Any) -> GateReport:
        return self.run(payload)

    def _gate_non_empty(self, payload: Any) -> tuple[bool, float, str]:
        if payload is None:
            return False, 0.0, "Payload is empty"
        if isinstance(payload, (list, dict, str, tuple)) and not payload:
            return False, 0.0, "Payload is empty"
        return True, 1.0, "Payload is present"

    def _gate_no_empty_text(self, payload: Any) -> tuple[bool, float, str]:
        text = self._extract_text(payload)
        if not text.strip():
            return False, 0.0, "No usable text content"
        return True, 1.0, "Text content exists"

    def _gate_min_length(self, payload: Any, minimum: int = 5) -> tuple[bool, float, str]:
        text = self._extract_text(payload)
        ok = len(text) >= minimum
        score = min(len(text), minimum) / max(minimum, 1)
        return ok, float(score), "Length sufficient" if ok else "Text is too short"

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("text", "content", "message", "input"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
            if isinstance(payload.get("messages"), list):
                return " ".join(
                    message.get("content", "") for message in payload["messages"] if isinstance(message, dict)
                )
        return ""


__all__ = ["EvaluationGates", "GateReport", "GateResult"]
