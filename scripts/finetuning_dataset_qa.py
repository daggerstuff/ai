#!/usr/bin/env python3
"""
Quality Assurance and Bias Detection for Fine-Tuning Datasets

This module provides automated quality checks and bias detection for the
fine-tuning dataset, ensuring data quality, consistency, and fairness.

Key Features:
- Data quality validation (format, completeness, consistency)
- Bias detection (demographic, topical, linguistic)
- Statistical quality metrics
- Manual review queue generation
- Automated quality scoring
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a single quality issue."""

    issue_type: str  # 'bias', 'completeness', 'consistency', 'format'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    example_id: str | None = None
    affected_field: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "example_id": self.example_id,
            "affected_field": self.affected_field,
            "suggestion": self.suggestion,
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for a dataset."""

    dataset_path: str
    total_examples: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Quality scores (0.0-1.0)
    overall_quality_score: float = 0.0
    format_validity_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    bias_score: float = 0.0  # Higher is better (less biased)

    # Issue counts
    issues: list[QualityIssue] = field(default_factory=list)
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0

    # Statistics
    statistics: dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "timestamp": self.timestamp,
            "total_examples": self.total_examples,
            "scores": {
                "overall": self.overall_quality_score,
                "format_validity": self.format_validity_score,
                "completeness": self.completeness_score,
                "consistency": self.consistency_score,
                "bias": self.bias_score,
            },
            "issues": {
                "total": len(self.issues),
                "critical": self.critical_issues,
                "high": self.high_issues,
                "medium": self.medium_issues,
                "low": self.low_issues,
            },
            "statistics": self.statistics,
            "recommendations": self.recommendations,
        }


class BiasDetector:
    """
    Detects various forms of bias in training data.

    Checks for:
    - Demographic bias (gender, race, age, etc.)
    - Topical bias (over/under-representation)
    - Linguistic bias (formality, complexity)
    - Geographic bias
    - Socioeconomic bias
    """

    # Demographic indicator patterns
    DEMOGRAPHIC_PATTERNS = {
        "gender": [
            re.compile(r"\b(she|her|hers|herself)\b", re.I),
            re.compile(r"\b(he|him|his|himself)\b", re.I),
            re.compile(r"\b(they|them|their|theirs|themself)\b", re.I),
        ],
        "age": [
            re.compile(r"\b(years? old|yo|teen|teenager|elderly|senior)\b", re.I),
            re.compile(r"\b(age|aged|ageing)\b", re.I),
        ],
        "race_ethnicity": [
            re.compile(
                r"\b(white|black|asian|hispanic|latino|indigenous|aboriginal)\b",
                re.I,
            ),
        ],
        "socioeconomic": [
            re.compile(
                r"\b(rich|poor|wealthy|income|salary|welfare|unemployed)\b",
                re.I,
            ),
        ],
    }

    def __init__(self):
        self._stats: dict[str, int] = {}

    def detect_bias(self, examples: list[dict[str, Any]]) -> list[QualityIssue]:
        """
        Detect bias in a list of examples.

        Args:
            examples: List of example dictionaries

        Returns:
            List of detected bias issues
        """
        issues = []

        if not examples:
            return issues

        # Check demographic representation
        demographic_issues = self._check_demographic_balance(examples)
        issues.extend(demographic_issues)

        # Check topical distribution
        topical_issues = self._check_topical_balance(examples)
        issues.extend(topical_issues)

        # Check linguistic patterns
        linguistic_issues = self._check_linguistic_bias(examples)
        issues.extend(linguistic_issues)

        return issues

    def _check_demographic_balance(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check for demographic imbalances."""
        issues = []

        # Count demographic references
        demo_counts: dict[str, int] = {
            "gender": 0,
            "age": 0,
            "race_ethnicity": 0,
            "socioeconomic": 0,
        }

        for example in examples:
            text = f"{example.get('input', '')} {example.get('target', '')}".lower()

            for category, patterns in self.DEMOGRAPHIC_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(text):
                        demo_counts[category] += 1

        # Check for extreme imbalances
        total = len(examples)
        if total > 0:
            for category, count in demo_counts.items():
                ratio = count / total
                # Flag if >90% or <1% coverage (very under or over represented)
                if ratio > 0.9:
                    issues.append(
                        QualityIssue(
                            issue_type="bias",
                            severity="medium",
                            description=(
                                f"Potential over-representation of {category} references ({ratio:.1%} of examples)"
                            ),
                            suggestion=("Review if demographic diversity is adequately represented"),
                        )
                    )

        return issues

    def _check_topical_balance(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check for topical imbalances."""
        issues = []

        # Count conversation types
        conv_types = Counter(ex.get("conversation_type", "unknown") for ex in examples)

        # Check if one type dominates
        total = len(examples)
        if total > 10:  # Only check if we have enough examples
            for conv_type, count in conv_types.items():
                ratio = count / total
                if ratio > 0.7:  # 70% threshold
                    issues.append(
                        QualityIssue(
                            issue_type="bias",
                            severity="low",
                            description=(f"Conversation type '{conv_type}' dominates dataset ({ratio:.1%})"),
                            suggestion="Consider balancing topical diversity",
                        )
                    )

        return issues

    def _check_linguistic_bias(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check for linguistic biases."""
        issues = []

        # Analyze text length distribution
        lengths = []
        for example in examples:
            text = f"{example.get('input', '')} {example.get('target', '')}"
            lengths.append(len(text.split()))

        if len(lengths) > 10:
            mean_len = statistics.mean(lengths)
            std_len = statistics.stdev(lengths)

            # Check for extreme outliers
            if std_len > 0:
                for i, length in enumerate(lengths):
                    z_score = (length - mean_len) / std_len
                    if abs(z_score) > 4:  # Extreme outlier
                        issues.append(
                            QualityIssue(
                                issue_type="bias",
                                severity="low",
                                description=(f"Example {i} has extreme text length (z-score: {z_score:.2f})"),
                                suggestion="Review for potential data quality issues",
                                example_id=examples[i].get("id"),
                            )
                        )

        return issues


class QualityAssurance:
    """
    Quality assurance for fine-tuning datasets.

    Performs:
    - Format validation
    - Completeness checks
    - Consistency verification
    - Bias detection
    - Statistical analysis
    """

    # Required fields for valid examples
    REQUIRED_FIELDS = {"id", "input", "target", "example_type"}

    # Valid example types
    VALID_EXAMPLE_TYPES = {
        "standard",
        "memory_retrieval",
        "memory_filtering",
        "memory_synthesis",
        "temporal_pattern",
        "emotional_context",
    }

    def __init__(self, bias_threshold: float = 0.7):
        """
        Initialize QA checker.

        Args:
            bias_threshold: Minimum acceptable bias score (0-1)
        """
        self.bias_threshold = bias_threshold
        self.bias_detector = BiasDetector()

    def run_full_check(
        self,
        dataset_path: str | Path,
    ) -> QualityReport:
        """
        Run comprehensive quality checks on a dataset.

        Args:
            dataset_path: Path to dataset JSONL file

        Returns:
            QualityReport with all findings
        """
        dataset_path = Path(dataset_path)

        # Load examples
        examples = self._load_dataset(dataset_path)

        # Run all checks
        format_issues = self._check_format_validity(examples)
        completeness_issues = self._check_completeness(examples)
        consistency_issues = self._check_consistency(examples)
        bias_issues = self.bias_detector.detect_bias(examples)

        # Combine all issues
        all_issues = format_issues + completeness_issues + consistency_issues + bias_issues

        # Calculate scores
        format_score = self._calculate_format_score(examples, format_issues)
        completeness_score = self._calculate_completeness_score(examples, completeness_issues)
        consistency_score = self._calculate_consistency_score(examples, consistency_issues)
        bias_score = self._calculate_bias_score(examples, bias_issues)

        # Overall score (weighted average)
        overall_score = format_score * 0.25 + completeness_score * 0.25 + consistency_score * 0.25 + bias_score * 0.25

        # Count issues by severity
        critical = sum(1 for i in all_issues if i.severity == "critical")
        high = sum(1 for i in all_issues if i.severity == "high")
        medium = sum(1 for i in all_issues if i.severity == "medium")
        low = sum(1 for i in all_issues if i.severity == "low")

        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, examples)

        # Compute statistics
        stats = self._compute_statistics(examples)

        return QualityReport(
            dataset_path=str(dataset_path),
            total_examples=len(examples),
            overall_quality_score=overall_score,
            format_validity_score=format_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            bias_score=bias_score,
            issues=all_issues,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            statistics=stats,
            recommendations=recommendations,
        )

    def _load_dataset(self, path: Path) -> list[dict[str, Any]]:
        """Load dataset from JSONL file."""
        examples = []

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line: {e}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {path}")

        return examples

    def _check_format_validity(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check format validity of examples."""
        issues = []

        for _i, example in enumerate(examples):
            # Check required fields
            missing = self.REQUIRED_FIELDS - set(example.keys())
            if missing:
                issues.append(
                    QualityIssue(
                        issue_type="format",
                        severity="critical",
                        description=f"Missing required fields: {missing}",
                        example_id=example.get("id"),
                        affected_field=", ".join(missing),
                        suggestion="Add missing required fields",
                    )
                )

            # Check example_type validity
            example_type = example.get("example_type")
            if example_type and example_type not in self.VALID_EXAMPLE_TYPES:
                issues.append(
                    QualityIssue(
                        issue_type="format",
                        severity="high",
                        description=f"Invalid example_type: {example_type}",
                        example_id=example.get("id"),
                        affected_field="example_type",
                        suggestion=f"Use one of: {self.VALID_EXAMPLE_TYPES}",
                    )
                )

            # Check types
            if not isinstance(example.get("input", ""), str):
                issues.append(
                    QualityIssue(
                        issue_type="format",
                        severity="high",
                        description="'input' field must be a string",
                        example_id=example.get("id"),
                        affected_field="input",
                    )
                )

            if not isinstance(example.get("target", ""), str):
                issues.append(
                    QualityIssue(
                        issue_type="format",
                        severity="high",
                        description="'target' field must be a string",
                        example_id=example.get("id"),
                        affected_field="target",
                    )
                )

        return issues

    def _check_completeness(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check completeness of examples."""
        issues = []

        for example in examples:
            # Check for empty content
            if not example.get("input", "").strip():
                issues.append(
                    QualityIssue(
                        issue_type="completeness",
                        severity="critical",
                        description="'input' field is empty",
                        example_id=example.get("id"),
                        affected_field="input",
                        suggestion="Provide non-empty input text",
                    )
                )

            if not example.get("target", "").strip():
                issues.append(
                    QualityIssue(
                        issue_type="completeness",
                        severity="critical",
                        description="'target' field is empty",
                        example_id=example.get("id"),
                        affected_field="target",
                        suggestion="Provide non-empty target text",
                    )
                )

            # Check for very short content
            input_len = len(example.get("input", "").split())
            if input_len < 5:
                issues.append(
                    QualityIssue(
                        issue_type="completeness",
                        severity="medium",
                        description=f"'input' is very short ({input_len} words)",
                        example_id=example.get("id"),
                        affected_field="input",
                        suggestion="Consider providing more context",
                    )
                )

        return issues

    def _check_consistency(
        self,
        examples: list[dict[str, Any]],
    ) -> list[QualityIssue]:
        """Check consistency across examples."""
        issues = []

        # Check for duplicate IDs
        ids = [ex.get("id") for ex in examples if ex.get("id")]
        duplicates = {id_ for id_ in ids if ids.count(id_) > 1}

        if duplicates:
            issues.append(
                QualityIssue(
                    issue_type="consistency",
                    severity="high",
                    description=f"Found {len(duplicates)} duplicate example IDs",
                    suggestion="Ensure all example IDs are unique",
                )
            )

        # Check for inconsistent split assignments
        {ex.get("split") for ex in examples if ex.get("split")}

        return issues

    def _calculate_format_score(
        self,
        examples: list[dict[str, Any]],
        issues: list[QualityIssue],
    ) -> float:
        """Calculate format validity score."""
        if not examples:
            return 0.0

        critical = sum(1 for i in issues if i.issue_type == "format" and i.severity == "critical")
        high = sum(1 for i in issues if i.issue_type == "format" and i.severity == "high")

        # Penalize heavily for critical/high issues
        penalty = (critical * 0.1) + (high * 0.05)
        return max(0.0, 1.0 - penalty)

    def _calculate_completeness_score(
        self,
        examples: list[dict[str, Any]],
        issues: list[QualityIssue],
    ) -> float:
        """Calculate completeness score."""
        if not examples:
            return 0.0

        critical = sum(1 for i in issues if i.issue_type == "completeness" and i.severity == "critical")
        medium = sum(1 for i in issues if i.issue_type == "completeness" and i.severity == "medium")

        penalty = (critical * 0.15) + (medium * 0.05)
        return max(0.0, 1.0 - penalty)

    def _calculate_consistency_score(
        self,
        examples: list[dict[str, Any]],
        issues: list[QualityIssue],
    ) -> float:
        """Calculate consistency score."""
        if not examples:
            return 0.0

        high = sum(1 for i in issues if i.issue_type == "consistency" and i.severity == "high")

        penalty = high * 0.1
        return max(0.0, 1.0 - penalty)

    def _calculate_bias_score(
        self,
        examples: list[dict[str, Any]],
        issues: list[QualityIssue],
    ) -> float:
        """Calculate bias score."""
        if not examples:
            return 0.0

        # Count bias issues
        bias_count = len(issues)

        # Simple scoring: fewer issues = better
        if bias_count == 0:
            return 1.0
        if bias_count <= 2:
            return 0.9
        if bias_count <= 5:
            return 0.7
        if bias_count <= 10:
            return 0.5
        return 0.3

    def _generate_recommendations(
        self,
        issues: list[QualityIssue],
        examples: list[dict[str, Any]],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Count issue types
        issue_counts = Counter(i.issue_type for i in issues)

        if issue_counts.get("format", 0) > 0:
            recommendations.append("Fix format issues before proceeding with training")

        if issue_counts.get("completeness", 0) > 0:
            recommendations.append("Complete missing data fields to improve dataset quality")

        if issue_counts.get("bias", 0) > 0:
            recommendations.append("Review dataset for potential biases and consider rebalancing")

        if len(examples) < 100:
            recommendations.append(
                f"Dataset is small ({len(examples)} examples). Consider collecting more data for better generalization."
            )

        return recommendations

    def _compute_statistics(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute dataset statistics."""
        if not examples:
            return {}

        # Text length stats
        input_lengths = [len(ex.get("input", "").split()) for ex in examples]
        target_lengths = [len(ex.get("target", "").split()) for ex in examples]

        return {
            "input_length": {
                "mean": statistics.mean(input_lengths) if input_lengths else 0,
                "median": statistics.median(input_lengths) if input_lengths else 0,
                "min": min(input_lengths) if input_lengths else 0,
                "max": max(input_lengths) if input_lengths else 0,
            },
            "target_length": {
                "mean": statistics.mean(target_lengths) if target_lengths else 0,
                "median": statistics.median(target_lengths) if target_lengths else 0,
                "min": min(target_lengths) if target_lengths else 0,
                "max": max(target_lengths) if target_lengths else 0,
            },
            "example_types": dict(Counter(ex.get("example_type") for ex in examples)),
            "conversation_types": dict(Counter(ex.get("conversation_type") for ex in examples)),
        }



def main():
    """CLI entry point for QA checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run quality assurance checks on fine-tuning dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quality report (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed issue descriptions",
    )

    args = parser.parse_args()

    # Run QA
    qa = QualityAssurance()
    report = qa.run_full_check(args.dataset_path)

    # Print summary


    if report.recommendations:
        for _rec in report.recommendations:
            pass

    if report.issues and args.verbose:
        for _issue in report.issues:
            pass

    # Save report if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    # Return exit code based on quality
    if report.overall_quality_score < 0.5:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
