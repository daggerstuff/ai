"""
Mental Health Dataset Download Validation

Specialized validation for mental health datasets with ethical compliance,
privacy protection, and therapeutic content verification.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dataset_validator import DatasetValidator, ValidationResult
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class MentalHealthValidationResult(ValidationResult):
    """Extended validation result for mental health datasets."""

    ethical_compliance: bool = False
    privacy_protection: bool = False
    therapeutic_content_score: float = 0.0
    sensitive_content_flags: list[str] = None
    anonymization_score: float = 0.0


class MentalHealthDatasetValidator(DatasetValidator):
    """Specialized validator for mental health datasets."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

        # Mental health specific validation rules
        self.mental_health_rules = {
            "required_anonymization": True,
            "max_personal_identifiers": 0,
            "min_therapeutic_content": 0.8,
            "prohibited_content_types": [
                "self_harm",
                "suicide_methods",
                "illegal_substances",
            ],
            "required_consent_indicators": True,
        }

        # Sensitive content patterns
        self.sensitive_patterns = {
            "personal_identifiers": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
                r"\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b",  # Address
            ],
            "therapeutic_indicators": [
                r"\b(?:therapy|counseling|treatment|session|therapeutic)\b",
                r"\b(?:anxiety|depression|trauma|ptsd|bipolar)\b",
                r"\b(?:coping|healing|recovery|support)\b",
            ],
            "prohibited_content": [
                r"\b(?:suicide|self.harm|cutting|overdose)\b",
                r"\b(?:kill myself|end it all|not worth living)\b",
            ],
        }

        logger.info("MentalHealthDatasetValidator initialized")

    def validate_mental_health_dataset(
        self, dataset_path: str
    ) -> MentalHealthValidationResult:
        """Validate mental health dataset with specialized checks."""
        logger.info(f"Validating mental health dataset: {dataset_path}")

        # Run base validation
        base_result = self.validate_dataset(dataset_path, "mental_health")

        # Perform mental health specific validation
        ethical_compliance = self._check_ethical_compliance(dataset_path)
        privacy_protection = self._check_privacy_protection(dataset_path)
        therapeutic_score = self._calculate_therapeutic_content_score(dataset_path)
        sensitive_flags = self._detect_sensitive_content(dataset_path)
        anonymization_score = self._calculate_anonymization_score(dataset_path)

        # Create extended result
        result = MentalHealthValidationResult(
            dataset_name=base_result.dataset_name,
            is_valid=base_result.is_valid and ethical_compliance and privacy_protection,
            file_count=base_result.file_count,
            total_size=base_result.total_size,
            format_compliance=base_result.format_compliance,
            integrity_check=base_result.integrity_check,
            quality_score=base_result.quality_score * therapeutic_score,
            issues=base_result.issues,
            validation_timestamp=base_result.validation_timestamp,
            ethical_compliance=ethical_compliance,
            privacy_protection=privacy_protection,
            therapeutic_content_score=therapeutic_score,
            sensitive_content_flags=sensitive_flags,
            anonymization_score=anonymization_score,
        )

        # Add mental health specific issues
        if not ethical_compliance:
            result.issues.append("Ethical compliance check failed")
        if not privacy_protection:
            result.issues.append("Privacy protection insufficient")
        if therapeutic_score < self.mental_health_rules["min_therapeutic_content"]:
            result.issues.append(
                f"Therapeutic content score too low: {therapeutic_score:.2f}"
            )
        if sensitive_flags:
            result.issues.extend(
                [f"Sensitive content detected: {flag}" for flag in sensitive_flags]
            )

        logger.info(
            f"Mental health validation completed: {'VALID' if result.is_valid else 'INVALID'}"
        )
        return result

    def _check_ethical_compliance(self, dataset_path: str) -> bool:
        """Check ethical compliance for mental health data."""
        try:
            # Check for consent indicators
            consent_found = False
            ethics_approval = False

            # Look for ethics documentation
            for file_path in Path(dataset_path).rglob("*"):
                if file_path.is_file():
                    filename = file_path.name.lower()
                    if any(
                        term in filename
                        for term in ["consent", "ethics", "irb", "approval"]
                    ):
                        consent_found = True

                    # Check file content for ethics indicators
                    if filename.endswith((".txt", ".md", ".json")):
                        try:
                            content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            ).lower()
                            if any(
                                term in content
                                for term in [
                                    "informed consent",
                                    "ethics approval",
                                    "irb approval",
                                ]
                            ):
                                ethics_approval = True
                        except Exception:
                            continue

            return consent_found or ethics_approval

        except Exception as e:
            logger.error(f"Ethics compliance check failed: {e}")
            return False

    def _check_privacy_protection(self, dataset_path: str) -> bool:
        """Check privacy protection measures."""
        try:
            personal_identifier_count = 0
            total_files_checked = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    total_files_checked += 1
                    try:
                        content = file_path.read_text(encoding="utf-8")

                        # Check for personal identifiers
                        for pattern in self.sensitive_patterns["personal_identifiers"]:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            personal_identifier_count += len(matches)

                    except Exception:
                        continue

            # Privacy protection passes if very few personal identifiers found
            identifier_ratio = personal_identifier_count / max(total_files_checked, 1)
            return (
                identifier_ratio <= self.mental_health_rules["max_personal_identifiers"]
            )

        except Exception as e:
            logger.error(f"Privacy protection check failed: {e}")
            return False

    def _calculate_therapeutic_content_score(self, dataset_path: str) -> float:
        """Calculate therapeutic content relevance score."""
        try:
            therapeutic_indicators = 0
            total_content_blocks = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract text content
                        text_content = self._extract_text_from_json(data)
                        if text_content:
                            total_content_blocks += 1

                            # Count therapeutic indicators
                            for pattern in self.sensitive_patterns[
                                "therapeutic_indicators"
                            ]:
                                if re.search(pattern, text_content, re.IGNORECASE):
                                    therapeutic_indicators += 1
                                    break  # Count once per content block

                    except Exception:
                        continue

            return therapeutic_indicators / max(total_content_blocks, 1)

        except Exception as e:
            logger.error(f"Therapeutic content scoring failed: {e}")
            return 0.0

    def _detect_sensitive_content(self, dataset_path: str) -> list[str]:
        """Detect potentially harmful sensitive content."""
        flags = []

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")

                        # Check for prohibited content
                        for pattern in self.sensitive_patterns["prohibited_content"]:
                            if re.search(pattern, content, re.IGNORECASE):
                                flags.append(f"Prohibited content in {file_path.name}")
                                break

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Sensitive content detection failed: {e}")

        return flags

    def _calculate_anonymization_score(self, dataset_path: str) -> float:
        """Calculate anonymization quality score."""
        try:
            anonymization_indicators = 0
            total_files = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    total_files += 1
                    try:
                        content = file_path.read_text(encoding="utf-8")

                        # Look for anonymization indicators
                        anonymization_patterns = [
                            r"\[REDACTED\]",
                            r"\[REMOVED\]",
                            r"\[ANONYMIZED\]",
                            r"Patient \d+",
                            r"Participant \d+",
                            r"Subject \d+",
                        ]

                        for pattern in anonymization_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                anonymization_indicators += 1
                                break

                    except Exception:
                        continue

            return anonymization_indicators / max(total_files, 1)

        except Exception as e:
            logger.error(f"Anonymization scoring failed: {e}")
            return 0.0

    def _extract_text_from_json(self, data) -> str:
        """Extract text content from JSON data structure."""
        text_parts = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ["content", "text", "message", "conversation"]:
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                text_parts.append(item)
                            elif isinstance(item, dict) and "content" in item:
                                text_parts.append(str(item["content"]))

        return " ".join(text_parts)

    def generate_mental_health_report(
        self,
        results: list[MentalHealthValidationResult],
        output_path: str = "mental_health_validation_report.json",
    ) -> str:
        """Generate specialized mental health validation report."""
        report = {
            "report_type": "Mental Health Dataset Validation",
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(results),
                "ethically_compliant": sum(1 for r in results if r.ethical_compliance),
                "privacy_protected": sum(1 for r in results if r.privacy_protection),
                "average_therapeutic_score": (
                    sum(r.therapeutic_content_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_anonymization_score": (
                    sum(r.anonymization_score for r in results) / len(results)
                    if results
                    else 0
                ),
            },
            "detailed_results": [
                {
                    "dataset_name": r.dataset_name,
                    "is_valid": r.is_valid,
                    "ethical_compliance": r.ethical_compliance,
                    "privacy_protection": r.privacy_protection,
                    "therapeutic_content_score": r.therapeutic_content_score,
                    "anonymization_score": r.anonymization_score,
                    "sensitive_content_flags": r.sensitive_content_flags,
                    "issues": r.issues,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Mental health validation report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    validator = MentalHealthDatasetValidator()

    # Test validation
    result = validator.validate_mental_health_dataset("./test_mental_health_dataset")
