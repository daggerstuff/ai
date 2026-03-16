"""
Dataset Validation Module for Bias and Stereotype Detection

Validates therapeutic training data before writing to disk to prevent
ingestion of biased or inappropriate content into the model training pipeline.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    bias_indicators: dict[str, int]
    failed_checks: list[str]


class BiasIndicators:
    """Stereotypes and bias patterns to detect in therapeutic content."""

    # Harmful stereotypes that shouldn't be in training data
    CULTURAL_STEREOTYPES = {
        "model minority": ["model minority", "hardworking asian", "submissive"],
        "lazy": ["lazy", "unmotivated", "shiftless"],
        "aggressive": ["aggressive black", "thug", "criminal"],
        "overly emotional": ["hysterical", "emotional female", "irrational woman"],
        "religious extremism": ["muslim terrorist", "evangelical crazy", "cult member"],
    }

    # Offensive generalizations
    OFFENSIVE_GENERALIZATIONS = {
        "all x are": ["all mexicans are", "all jews are", "all asians are"],
        "women should": ["women should stay home", "women are too emotional"],
        "mental health stigma": ["crazy person", "mentally ill people are dangerous"],
    }

    # Problematic therapy responses
    PROBLEMATIC_RESPONSES = {
        "dismissive": [
            "you're overreacting",
            "that's not real",
            "you're too sensitive",
        ],
        "blame": [
            "it's your fault",
            "you deserve this",
            "you brought this on yourself",
        ],
        "conversion": ["you just need faith", "prayer will fix you", "think positive"],
        "abuse minimization": [
            "it wasn't that bad",
            "move on already",
            "forget about it",
        ],
    }

    # HIPAA/Privacy concerns
    PRIVACY_VIOLATIONS = {
        "identifiable info": ["john smith", "123 main street", "555-1234"],
        "specific institutions": ["st. mary's hospital", "county jail"],
    }


class DatasetValidator:
    """Validates therapeutic training datasets for bias and appropriateness."""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, warnings become validation failures
        """
        self.strict_mode = strict_mode
        self.bias_indicators = BiasIndicators()

    def validate_edge_case(self, edge_case: dict[str, Any]) -> ValidationResult:
        """
        Validate a single edge case scenario.

        Args:
            edge_case: Edge case dictionary to validate

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        bias_counts = {}
        failed_checks = []

        # Check required fields
        if not self._validate_required_fields(edge_case, errors):
            failed_checks.append("missing_required_fields")

        # Check for bias indicators
        if bias_found := self._check_bias_indicators(edge_case, warnings):
            bias_counts |= bias_found
            failed_checks.append("bias_indicators_detected")

        # Check for privacy violations
        if self._check_privacy_violations(edge_case, errors):
            failed_checks.append("privacy_violations")

        # Check therapeutic quality
        if self._check_therapeutic_quality(edge_case, warnings):
            failed_checks.append("therapeutic_quality_issues")

        # Check for harmful content that shouldn't be in training
        if self._check_harmful_content(edge_case, errors):
            failed_checks.append("harmful_content")

        # In strict mode, convert warnings to errors
        if self.strict_mode and warnings:
            errors.extend(warnings)
            warnings = []

        is_valid = not errors

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            bias_indicators=bias_counts,
            failed_checks=failed_checks,
        )

    def _validate_required_fields(self, edge_case: dict, errors: list[str]) -> bool:
        """Check for required fields in edge case."""
        required = {"scenario_id", "edge_case_type", "severity_level", "description"}
        if missing := required - set(edge_case.keys()):
            errors.append(f"Missing required fields: {', '.join(missing)}")
            return False

        return True

    def _check_bias_indicators(
        self, edge_case: dict[str, Any], warnings: list[str]
    ) -> dict[str, int]:
        """Check for bias indicators in edge case content."""
        text = self._extract_text(edge_case).lower()
        bias_found = {}

        for category, patterns in self.bias_indicators.CULTURAL_STEREOTYPES.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    bias_found[f"stereotype_{category}"] = (
                        bias_found.get(f"stereotype_{category}", 0) + 1
                    )
                    warnings.append(
                        f"Potential stereotype detected: {category} "
                        f"(pattern: '{pattern}')"
                    )

        for (
            category,
            patterns,
        ) in self.bias_indicators.OFFENSIVE_GENERALIZATIONS.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    bias_found[f"generalization_{category}"] = (
                        bias_found.get(f"generalization_{category}", 0) + 1
                    )
                    warnings.append(
                        f"Potentially offensive generalization: {category} "
                        f"(pattern: '{pattern}')"
                    )

        return bias_found

    def _check_privacy_violations(
        self, edge_case: dict[str, Any], errors: list[str]
    ) -> bool:
        """Check for privacy violations (PII, specific institutions)."""
        text = self._extract_text(edge_case)
        found = False

        # Check for common name patterns (basic)
        import re

        name_pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"
        names = re.findall(name_pattern, text)

        # Filter common therapeutic terms
        common_terms = {"Client", "Therapist", "Patient", "Person"}
        suspicious_names = [n for n in names if n not in common_terms]

        if suspicious_names:
            errors.append(f"Potentially identifiable names found: {suspicious_names}")
            found = True

        return found

    def _check_therapeutic_quality(
        self, edge_case: dict[str, Any], warnings: list[str]
    ) -> bool:
        """Check for therapeutic quality issues."""
        text = self._extract_text(edge_case).lower()
        issues = False

        for category, patterns in self.bias_indicators.PROBLEMATIC_RESPONSES.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    warnings.append(
                        f"Problematic therapeutic response detected: {category} "
                        f"(pattern: '{pattern}')"
                    )
                    issues = True

        return issues

    def _check_harmful_content(
        self, edge_case: dict[str, Any], errors: list[str]
    ) -> bool:
        """Check for genuinely harmful content that violates training policies."""
        text = self._extract_text(edge_case).lower()
        harmful_keywords = {
            "racial slur": ["n-word", "n word"],
            "extreme graphic violence": ["dismember", "decapitate"],
            "child exploitation": [
                "child abuse graphic",
                "child sexual abuse explicit",
            ],
        }

        found = False
        for category, patterns in harmful_keywords.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    errors.append(f"Harmful content detected: {category}")
                    found = True

        return found

    def _extract_text(self, edge_case: dict[str, Any]) -> str:
        """Extract all text from edge case for analysis."""
        text_parts = []

        text_parts.extend(
            edge_case[key]
            for key in [
                "description",
                "client_presentation",
                "generated_text",
                "template",
            ]
            if key in edge_case and isinstance(edge_case[key], str)
        )
        return " ".join(text_parts)

    def validate_batch(self, edge_cases: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Validate a batch of edge cases.

        Returns:
            Summary of validation results
        """
        results = {
            "total": len(edge_cases),
            "valid": 0,
            "invalid": 0,
            "warnings": 0,
            "details": [],
            "bias_summary": {},
            "failed_cases": [],
        }

        for case in edge_cases:
            validation = self.validate_edge_case(case)

            if validation.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["failed_cases"].append(
                    {
                        "scenario_id": case.get("scenario_id", "unknown"),
                        "errors": validation.errors,
                    }
                )

            if validation.warnings:
                results["warnings"] += 1

            # Aggregate bias indicators
            for bias_type, count in validation.bias_indicators.items():
                results["bias_summary"][bias_type] = (
                    results["bias_summary"].get(bias_type, 0) + count
                )

        results["pass_rate"] = (
            results["valid"] / results["total"] if results["total"] > 0 else 0
        )

        return results


def validate_jsonl_file(filepath: str, strict_mode: bool = True) -> dict[str, Any]:
    """
    Validate a JSONL file of edge cases.

    Args:
        filepath: Path to JSONL file
        strict_mode: If True, warnings become validation failures

    Returns:
        Validation summary
    """
    validator = DatasetValidator(strict_mode=strict_mode)
    cases = []

    try:
        with open(filepath, "r") as f:
            cases.extend(json.loads(line) for line in f if line.strip())
    except Exception as e:
        logger.error(f"Error reading JSONL file {filepath}: {e}")
        return {"error": str(e)}

    return validator.validate_batch(cases)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    validator = DatasetValidator(strict_mode=True)

    test_case = {
        "scenario_id": "test_001",
        "edge_case_type": "crisis_intervention",
        "severity_level": "high",
        "description": "Client expressing suicidal ideation",
    }

    result = validator.validate_edge_case(test_case)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
