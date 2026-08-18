"""EARS compliance checks for dataset and pipeline safety-sensitive workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarsValidationResult:
    """Result of EARS validation."""

    is_compliant: bool
    sensitivity: str = "unknown"
    total_items: int = 0
    issues: list[str] = field(default_factory=list)


class EarsComplianceGate:
    """Simple but production-minded EARS gate implementation."""

    def __init__(self) -> None:
        self.crisis_terms = ("kill myself", "suicide", "harm myself", "self-harm", "emergency")
        self.sensitive_terms = ("diagnosis", "medication", "hallucination")

    def validate_dataset(self, dataset_data: list[dict[str, Any]] | list[Any]) -> EarsValidationResult:
        if not dataset_data:
            return EarsValidationResult(
                is_compliant=False,
                sensitivity="unknown",
                total_items=0,
                issues=["empty_dataset"],
            )

        issues: list[str] = []
        total = len(dataset_data)
        for item in dataset_data:
            text = self._extract_text(item).lower()
            if any(term in text for term in self.crisis_terms):
                issues.append("crisis_keyword")
            if any(term in text for term in self.sensitive_terms):
                issues.append("clinical_risk")

        sensitive = "high" if issues else "normal"
        return EarsValidationResult(
            is_compliant=not issues,
            sensitivity=sensitive,
            total_items=total,
            issues=sorted(set(issues)),
        )

    def validate_compliance(self, data: dict[str, Any]) -> EarsValidationResult:
        if not data:
            return EarsValidationResult(False, "unknown", 0, ["empty_payload"])

        return EarsValidationResult(True, "normal", 1, [])

    def check_pipeline_sensitivity(self, pipeline_config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(pipeline_config, dict):
            return {"is_sensitive": False, "sensitivity_level": "unknown", "requires_review": True}

        enabled = pipeline_config.get("enabled", True)
        safety_checks = pipeline_config.get("safety_checks", False)
        sensitivity = "high" if safety_checks else "normal"
        requires_review = bool(enabled and safety_checks)

        return {
            "is_sensitive": bool(safety_checks),
            "sensitivity_level": sensitivity,
            "requires_review": requires_review,
        }

    def _extract_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            return ""
        text = item.get("content")
        if isinstance(text, str):
            return text
        return str(item)


__all__ = ["EarsComplianceGate", "EarsValidationResult"]
