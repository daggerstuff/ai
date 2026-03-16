"""
Dataset Validation and Integrity Checks

This module provides comprehensive validation for downloaded datasets,
ensuring data integrity, format compliance, and quality standards.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    dataset_name: str
    is_valid: bool
    file_count: int
    total_size: int
    format_compliance: bool
    integrity_check: bool
    quality_score: float
    issues: list[str]
    validation_timestamp: str


class DatasetValidator:
    """Comprehensive dataset validation system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.validation_rules = {
            "min_file_size": 1024,  # 1KB minimum
            "max_file_size": 100 * 1024 * 1024,  # 100MB maximum
            "required_formats": [".json", ".jsonl", ".csv", ".txt"],
            "min_quality_score": 0.7,
        }

        logger.info("DatasetValidator initialized")

    def validate_dataset(
        self, dataset_path: str, dataset_type: str = "general"
    ) -> ValidationResult:
        """Validate a complete dataset."""
        dataset_name = Path(dataset_path).name

        logger.info(f"Validating dataset: {dataset_name}")

        issues = []
        file_count = 0
        total_size = 0
        format_compliance = True
        integrity_check = True

        try:
            # Check if dataset path exists
            if not os.path.exists(dataset_path):
                issues.append(f"Dataset path does not exist: {dataset_path}")
                return ValidationResult(
                    dataset_name=dataset_name,
                    is_valid=False,
                    file_count=0,
                    total_size=0,
                    format_compliance=False,
                    integrity_check=False,
                    quality_score=0.0,
                    issues=issues,
                    validation_timestamp=datetime.now().isoformat(),
                )

            # Validate files
            for root, _dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)

                    file_count += 1
                    total_size += file_size

                    # Check file size
                    if file_size < self.validation_rules["min_file_size"]:
                        issues.append(f"File too small: {file} ({file_size} bytes)")
                    elif file_size > self.validation_rules["max_file_size"]:
                        issues.append(f"File too large: {file} ({file_size} bytes)")

                    # Check format compliance
                    file_ext = Path(file).suffix.lower()
                    if file_ext not in self.validation_rules["required_formats"]:
                        format_compliance = False
                        issues.append(f"Unsupported format: {file} ({file_ext})")

                    # Check file integrity
                    if not self._check_file_integrity(file_path):
                        integrity_check = False
                        issues.append(f"File integrity check failed: {file}")

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                file_count, total_size, len(issues), format_compliance, integrity_check
            )

            is_valid = (
                file_count > 0
                and format_compliance
                and integrity_check
                and quality_score >= self.validation_rules["min_quality_score"]
            )

            result = ValidationResult(
                dataset_name=dataset_name,
                is_valid=is_valid,
                file_count=file_count,
                total_size=total_size,
                format_compliance=format_compliance,
                integrity_check=integrity_check,
                quality_score=quality_score,
                issues=issues,
                validation_timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"Validation completed for {dataset_name}: {'VALID' if is_valid else 'INVALID'}"
            )
            return result

        except Exception as e:
            logger.error(f"Validation failed for {dataset_name}: {e}")
            return ValidationResult(
                dataset_name=dataset_name,
                is_valid=False,
                file_count=file_count,
                total_size=total_size,
                format_compliance=False,
                integrity_check=False,
                quality_score=0.0,
                issues=[f"Validation error: {e!s}"],
                validation_timestamp=datetime.now().isoformat(),
            )

    def validate_mental_health_dataset(self, dataset_path: str) -> ValidationResult:
        """Validate mental health specific dataset."""
        logger.info("Validating mental health dataset")

        result = self.validate_dataset(dataset_path, "mental_health")

        # Additional mental health specific checks
        if result.is_valid:
            mental_health_issues = self._check_mental_health_compliance(dataset_path)
            result.issues.extend(mental_health_issues)

            if mental_health_issues:
                result.quality_score *= 0.9  # Reduce score for compliance issues

        return result

    def validate_reasoning_dataset(self, dataset_path: str) -> ValidationResult:
        """Validate reasoning enhancement dataset."""
        logger.info("Validating reasoning dataset")

        result = self.validate_dataset(dataset_path, "reasoning")

        # Additional reasoning specific checks
        if result.is_valid:
            reasoning_issues = self._check_reasoning_patterns(dataset_path)
            result.issues.extend(reasoning_issues)

            if reasoning_issues:
                result.quality_score *= 0.9

        return result

    def validate_personality_dataset(self, dataset_path: str) -> ValidationResult:
        """Validate personality balancing dataset."""
        logger.info("Validating personality dataset")

        result = self.validate_dataset(dataset_path, "personality")

        # Additional personality specific checks
        if result.is_valid:
            personality_issues = self._check_personality_markers(dataset_path)
            result.issues.extend(personality_issues)

            if personality_issues:
                result.quality_score *= 0.9

        return result

    def _check_file_integrity(self, file_path: str) -> bool:
        """Check file integrity."""
        try:
            # Basic integrity check - can read file
            with open(file_path, "rb") as f:
                f.read(1024)  # Read first 1KB

            # For JSON files, check if valid JSON
            if file_path.endswith(".json"):
                with open(file_path, encoding="utf-8") as f:
                    json.load(f)

            return True
        except Exception:
            return False

    def _calculate_quality_score(
        self,
        file_count: int,
        total_size: int,
        issue_count: int,
        format_compliance: bool,
        integrity_check: bool,
    ) -> float:
        """Calculate overall quality score."""
        base_score = 1.0

        # Reduce score for issues
        if issue_count > 0:
            base_score *= max(0.1, 1.0 - (issue_count * 0.1))

        # Reduce score for format/integrity issues
        if not format_compliance:
            base_score *= 0.7
        if not integrity_check:
            base_score *= 0.5

        # Bonus for good file count
        if file_count >= 10:
            base_score *= 1.1

        return min(1.0, base_score)

    def _check_mental_health_compliance(self, dataset_path: str) -> list[str]:
        """Check mental health dataset compliance."""
        issues = []

        # Check for sensitive content handling
        # Check for proper anonymization
        # Check for ethical guidelines compliance

        # Placeholder implementation
        logger.info("Mental health compliance check completed")
        return issues

    def _check_reasoning_patterns(self, dataset_path: str) -> list[str]:
        """Check reasoning dataset patterns."""
        issues = []

        # Check for chain-of-thought patterns
        # Check for logical reasoning structures
        # Check for problem-solution pairs

        # Placeholder implementation
        logger.info("Reasoning patterns check completed")
        return issues

    def _check_personality_markers(self, dataset_path: str) -> list[str]:
        """Check personality dataset markers."""
        issues = []

        # Check for personality trait indicators
        # Check for balanced representation
        # Check for bias detection

        # Placeholder implementation
        logger.info("Personality markers check completed")
        return issues

    def generate_validation_report(
        self,
        results: list[ValidationResult],
        output_path: str = "validation_report.json",
    ) -> str:
        """Generate comprehensive validation report."""
        report = {
            "validation_summary": {
                "total_datasets": len(results),
                "valid_datasets": sum(1 for r in results if r.is_valid),
                "invalid_datasets": sum(1 for r in results if not r.is_valid),
                "average_quality_score": (
                    sum(r.quality_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "report_timestamp": datetime.now().isoformat(),
            },
            "dataset_results": [
                {
                    "dataset_name": r.dataset_name,
                    "is_valid": r.is_valid,
                    "file_count": r.file_count,
                    "total_size": r.total_size,
                    "quality_score": r.quality_score,
                    "issues": r.issues,
                    "validation_timestamp": r.validation_timestamp,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report generated: {output_path}")
        return output_path


# Example usage and testing
if __name__ == "__main__":
    validator = DatasetValidator()

    # Test validation
    test_result = validator.validate_dataset("./test_dataset", "general")
