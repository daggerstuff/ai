"""Core evaluation orchestration helpers for datasets and responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation_gates import EvaluationGates


@dataclass
class EvaluationRecord:
    gate_passed: bool
    score: float
    details: dict[str, Any]


class EvaluationSystem:
    """Run quality gates and track evaluation history."""

    def __init__(self) -> None:
        self.gates = EvaluationGates()
        self.history: list[EvaluationRecord] = []

    def evaluate(self, payload: Any) -> EvaluationRecord:
        report = self.gates.evaluate(payload)
        score = sum(item.score for item in report.results) / max(len(report.results), 1)
        passed = report.passed
        details = {
            "gate_count": len(report.results),
            "passes": [item.passed for item in report.results],
            "messages": [item.message for item in report.results],
        }
        record = EvaluationRecord(gate_passed=passed, score=score, details=details)
        self.history.append(record)
        return record

    def evaluate_batch(self, payloads: list[Any]) -> list[EvaluationRecord]:
        return [self.evaluate(payload) for payload in payloads]


__all__ = ["EvaluationRecord", "EvaluationSystem"]
