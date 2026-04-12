# Stub for ai.core.pipelines.ears_compliance_gate
# Generated for test compatibility

from dataclasses import dataclass


@dataclass
class EarsValidationResult:
    """Result of EARS validation."""
    is_compliant: bool
    sensitivity: str = "unknown"
    total_items: int = 0
    issues: list = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

class EarsComplianceGate:
    """Stub implementation for EarsComplianceGate."""

    def validate_dataset(self, dataset_data: list) -> EarsValidationResult:
        """Validate a dataset for compliance."""
        if not dataset_data:
            return EarsValidationResult(
                is_compliant=False,
                sensitivity="unknown",
                total_items=0
            )

        # Check for crisis content
        has_crisis = any(
            "hurt" in str(item.get("content", "")).lower() or
            "crisis" in str(item.get("label", "")).lower()
            for item in dataset_data
        )

        return EarsValidationResult(
            is_compliant=not has_crisis,
            sensitivity="high" if has_crisis else "normal",
            total_items=len(dataset_data)
        )

    def validate_compliance(self, data: dict) -> EarsValidationResult:
        """Validate compliance for given data."""
        return EarsValidationResult(
            is_compliant=True,
            sensitivity="normal",
            total_items=1
        )

    def check_pipeline_sensitivity(self, pipeline_config: dict) -> dict:
        """Check pipeline sensitivity settings."""
        return {
            "is_sensitive": False,
            "sensitivity_level": "normal",
            "requires_review": False
        }

__all__ = ["EarsComplianceGate", "EarsValidationResult"]
