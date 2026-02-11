#!/usr/bin/env python3
"""
Sampling Validation System for Task 25
Provides comprehensive validation checks for sampling operations to ensure quality and integrity.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    score: float
    details: dict[str, Any]
    issues: list[str]
    recommendations: list[str]

@dataclass
class SamplingValidationReport:
    """Comprehensive validation report for sampling results"""
    overall_score: float
    passed_checks: int
    total_checks: int
    validation_results: list[ValidationResult]
    summary: dict[str, Any]
    critical_issues: list[str]
    recommendations: list[str]

class SamplingValidator:
    """
    Comprehensive validation system for sampling operations.
    Ensures sampling quality, distribution, and integrity.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the sampling validator with configuration."""
        self.config = config or {}
        self.validation_history = []

        # Validation thresholds
        self.thresholds = self.config.get("thresholds", {
            "distribution_deviation": 0.15,  # Max deviation from expected distribution
            "quality_variance": 0.20,        # Max quality variance within tiers
            "duplicate_rate": 0.05,          # Max duplicate rate
            "coverage_minimum": 0.80,        # Min coverage of quality spectrum
            "tier_balance": 0.10,            # Max imbalance between tiers
            "sample_size_deviation": 0.25    # Max deviation from target sample sizes
        })

        # Critical validation checks
        self.critical_checks = [
            "distribution_validation",
            "quality_consistency",
            "duplicate_detection",
            "tier_balance_check"
        ]

    def validate_sampling_results(self, sampling_results: list[Any],
                                 original_data: dict[str, list[dict]],
                                 target_total: int) -> SamplingValidationReport:
        """
        Perform comprehensive validation of sampling results.

        Args:
            sampling_results: Results from sampling operation
            original_data: Original tier data that was sampled from
            target_total: Target total number of samples

        Returns:
            SamplingValidationReport with detailed validation results
        """
        logger.info("Starting comprehensive sampling validation...")

        validation_results = []
        critical_issues = []
        recommendations = []

        # Run all validation checks
        checks = [
            self._validate_distribution,
            self._validate_quality_consistency,
            self._validate_duplicates,
            self._validate_coverage,
            self._validate_tier_balance,
            self._validate_sample_sizes,
            self._validate_data_integrity,
            self._validate_statistical_properties
        ]

        for check in checks:
            try:
                result = check(sampling_results, original_data, target_total)
                validation_results.append(result)

                if not result.passed and result.check_name in self.critical_checks:
                    critical_issues.extend(result.issues)

                recommendations.extend(result.recommendations)

            except Exception as e:
                logger.error(f"Validation check {check.__name__} failed: {e}")
                validation_results.append(ValidationResult(
                    check_name=check.__name__,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    issues=[f"Validation check failed: {e}"],
                    recommendations=["Review validation check implementation"]
                ))

        # Calculate overall score
        passed_checks = sum(1 for r in validation_results if r.passed)
        total_checks = len(validation_results)
        overall_score = np.mean([r.score for r in validation_results])

        # Generate summary
        summary = self._generate_validation_summary(validation_results, sampling_results, target_total)

        report = SamplingValidationReport(
            overall_score=overall_score,
            passed_checks=passed_checks,
            total_checks=total_checks,
            validation_results=validation_results,
            summary=summary,
            critical_issues=list(set(critical_issues)),
            recommendations=list(set(recommendations))
        )

        self.validation_history.append(report)
        logger.info(f"Validation complete: {passed_checks}/{total_checks} checks passed, "
                   f"overall score: {overall_score:.3f}")

        return report

    def _validate_distribution(self, sampling_results: list[Any],
                              original_data: dict[str, list[dict]],
                              target_total: int) -> ValidationResult:
        """Validate that sampling distribution matches expected weights."""
        issues = []
        recommendations = []
        details = {}

        # Expected weights (from tier configs)
        expected_weights = {
            "tier_1_priority": 0.40,
            "tier_2_professional": 0.25,
            "tier_3_cot": 0.20,
            "tier_4_reddit": 0.10,
            "tier_5_research": 0.04,
            "tier_6_knowledge": 0.01
        }

        # Calculate actual distribution
        total_samples = sum(len(result.samples) for result in sampling_results)
        actual_distribution = {}

        for result in sampling_results:
            tier_id = result.metadata.get("tier_id", "unknown")
            actual_weight = len(result.samples) / total_samples if total_samples > 0 else 0
            actual_distribution[tier_id] = actual_weight

        # Check deviations
        max_deviation = 0.0
        for tier_id, expected_weight in expected_weights.items():
            actual_weight = actual_distribution.get(tier_id, 0.0)
            deviation = abs(actual_weight - expected_weight)
            max_deviation = max(max_deviation, deviation)

            if deviation > self.thresholds["distribution_deviation"]:
                issues.append(f"Tier {tier_id}: expected {expected_weight:.3f}, got {actual_weight:.3f} "
                             f"(deviation: {deviation:.3f})")

        details.update({
            "expected_distribution": expected_weights,
            "actual_distribution": actual_distribution,
            "max_deviation": max_deviation,
            "total_samples": total_samples
        })

        passed = max_deviation <= self.thresholds["distribution_deviation"]
        score = max(0.0, 1.0 - (max_deviation / self.thresholds["distribution_deviation"]))

        if not passed:
            recommendations.append("Adjust sampling weights to better match expected distribution")
            recommendations.append("Review tier availability and quality thresholds")

        return ValidationResult(
            check_name="distribution_validation",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_quality_consistency(self, sampling_results: list[Any],
                                    original_data: dict[str, list[dict]],
                                    target_total: int) -> ValidationResult:
        """Validate quality consistency within and across tiers."""
        issues = []
        recommendations = []
        details = {}

        tier_quality_stats = {}
        all_qualities = []

        for result in sampling_results:
            tier_id = result.metadata.get("tier_id", "unknown")
            tier_qualities = []

            # Calculate quality for each sample (if not already available)
            for sample in result.samples:
                # Use the quality score from the result or calculate it
                quality = getattr(sample, "quality_score", result.quality_score)
                tier_qualities.append(quality)
                all_qualities.append(quality)

            if tier_qualities:
                tier_quality_stats[tier_id] = {
                    "mean": np.mean(tier_qualities),
                    "std": np.std(tier_qualities),
                    "min": np.min(tier_qualities),
                    "max": np.max(tier_qualities),
                    "count": len(tier_qualities)
                }

        # Check quality variance within tiers
        max_variance = 0.0
        for tier_id, stats in tier_quality_stats.items():
            variance = stats["std"]
            max_variance = max(max_variance, variance)

            if variance > self.thresholds["quality_variance"]:
                issues.append(f"High quality variance in {tier_id}: {variance:.3f}")

        # Check overall quality distribution
        if all_qualities:
            overall_mean = np.mean(all_qualities)
            overall_std = np.std(all_qualities)

            details.update({
                "tier_quality_stats": tier_quality_stats,
                "overall_mean_quality": overall_mean,
                "overall_std_quality": overall_std,
                "max_tier_variance": max_variance
            })

        passed = max_variance <= self.thresholds["quality_variance"]
        score = max(0.0, 1.0 - (max_variance / self.thresholds["quality_variance"]))

        if not passed:
            recommendations.append("Review quality assessment consistency across tiers")
            recommendations.append("Consider stratified sampling within tiers for better quality distribution")

        return ValidationResult(
            check_name="quality_consistency",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_duplicates(self, sampling_results: list[Any],
                           original_data: dict[str, list[dict]],
                           target_total: int) -> ValidationResult:
        """Validate that there are no duplicate samples."""
        issues = []
        recommendations = []
        details = {}

        all_sample_ids = []
        tier_duplicates = {}

        for result in sampling_results:
            tier_id = result.metadata.get("tier_id", "unknown")
            tier_sample_ids = []

            for sample in result.samples:
                sample_id = sample.get("id", str(hash(str(sample))))
                all_sample_ids.append(sample_id)
                tier_sample_ids.append(sample_id)

            # Check for duplicates within tier
            tier_duplicate_count = len(tier_sample_ids) - len(set(tier_sample_ids))
            tier_duplicates[tier_id] = tier_duplicate_count

            if tier_duplicate_count > 0:
                issues.append(f"Found {tier_duplicate_count} duplicates in {tier_id}")

        # Check for duplicates across all samples
        total_samples = len(all_sample_ids)
        unique_samples = len(set(all_sample_ids))
        total_duplicates = total_samples - unique_samples
        duplicate_rate = total_duplicates / total_samples if total_samples > 0 else 0

        details.update({
            "total_samples": total_samples,
            "unique_samples": unique_samples,
            "total_duplicates": total_duplicates,
            "duplicate_rate": duplicate_rate,
            "tier_duplicates": tier_duplicates
        })

        passed = duplicate_rate <= self.thresholds["duplicate_rate"]
        score = max(0.0, 1.0 - (duplicate_rate / self.thresholds["duplicate_rate"]))

        if not passed:
            recommendations.append("Implement duplicate detection in sampling process")
            recommendations.append("Review sample ID generation and uniqueness")

        return ValidationResult(
            check_name="duplicate_detection",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_coverage(self, sampling_results: list[Any],
                          original_data: dict[str, list[dict]],
                          target_total: int) -> ValidationResult:
        """Validate coverage of quality spectrum and data diversity."""
        issues = []
        recommendations = []
        details = {}

        # Collect all quality scores
        sampled_qualities = []
        for result in sampling_results:
            for sample in result.samples:
                quality = getattr(sample, "quality_score", result.quality_score)
                sampled_qualities.append(quality)

        if not sampled_qualities:
            return ValidationResult(
                check_name="coverage_validation",
                passed=False,
                score=0.0,
                details={"error": "No quality scores available"},
                issues=["No samples with quality scores found"],
                recommendations=["Ensure quality scores are calculated for all samples"]
            )

        # Calculate coverage metrics
        min_quality = np.min(sampled_qualities)
        max_quality = np.max(sampled_qualities)
        quality_range = max_quality - min_quality

        # Check if we cover the full quality spectrum (0.0 to 1.0)
        expected_range = 1.0
        coverage_ratio = quality_range / expected_range

        # Check distribution across quality bands
        quality_bands = {
            "low": sum(1 for q in sampled_qualities if q < 0.4),
            "medium": sum(1 for q in sampled_qualities if 0.4 <= q < 0.7),
            "high": sum(1 for q in sampled_qualities if q >= 0.7)
        }

        total_samples = len(sampled_qualities)
        band_distribution = {k: v/total_samples for k, v in quality_bands.items()}

        details.update({
            "quality_range": quality_range,
            "min_quality": min_quality,
            "max_quality": max_quality,
            "coverage_ratio": coverage_ratio,
            "quality_bands": quality_bands,
            "band_distribution": band_distribution,
            "total_samples": total_samples
        })

        passed = coverage_ratio >= self.thresholds["coverage_minimum"]
        score = min(1.0, coverage_ratio / self.thresholds["coverage_minimum"])

        if not passed:
            issues.append(f"Quality coverage {coverage_ratio:.3f} below minimum {self.thresholds['coverage_minimum']}")
            recommendations.append("Ensure sampling includes full quality spectrum")
            recommendations.append("Review quality thresholds to allow more diverse sampling")

        return ValidationResult(
            check_name="coverage_validation",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_tier_balance(self, sampling_results: list[Any],
                              original_data: dict[str, list[dict]],
                              target_total: int) -> ValidationResult:
        """Validate balance between tiers."""
        issues = []
        recommendations = []
        details = {}

        # Calculate tier sample counts
        tier_counts = {}
        total_samples = 0

        for result in sampling_results:
            tier_id = result.metadata.get("tier_id", "unknown")
            count = len(result.samples)
            tier_counts[tier_id] = count
            total_samples += count

        # Calculate balance metrics
        if total_samples == 0:
            return ValidationResult(
                check_name="tier_balance_check",
                passed=False,
                score=0.0,
                details={"error": "No samples found"},
                issues=["No samples in any tier"],
                recommendations=["Review sampling process"]
            )

        # Check for extreme imbalances
        tier_proportions = {k: v/total_samples for k, v in tier_counts.items()}
        max_proportion = max(tier_proportions.values()) if tier_proportions else 0
        min_proportion = min(tier_proportions.values()) if tier_proportions else 0
        balance_ratio = min_proportion / max_proportion if max_proportion > 0 else 0

        # Check if any tier is completely missing (when it should have samples)
        missing_tiers = []
        for tier_id in original_data:
            if tier_id not in tier_counts or tier_counts[tier_id] == 0:
                if len(original_data[tier_id]) > 0:  # Only flag if original data exists
                    missing_tiers.append(tier_id)

        details.update({
            "tier_counts": tier_counts,
            "tier_proportions": tier_proportions,
            "balance_ratio": balance_ratio,
            "missing_tiers": missing_tiers,
            "total_samples": total_samples
        })

        # Pass if balance ratio is reasonable and no critical tiers are missing
        passed = balance_ratio >= self.thresholds["tier_balance"] and len(missing_tiers) == 0
        score = balance_ratio * (1.0 - len(missing_tiers) * 0.2)  # Penalty for missing tiers

        if not passed:
            if missing_tiers:
                issues.append(f"Missing samples from tiers: {missing_tiers}")
            if balance_ratio < self.thresholds["tier_balance"]:
                issues.append(f"Poor tier balance ratio: {balance_ratio:.3f}")

            recommendations.append("Review tier availability and quality thresholds")
            recommendations.append("Consider adjusting sampling weights for better balance")

        return ValidationResult(
            check_name="tier_balance_check",
            passed=passed,
            score=max(0.0, score),
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_sample_sizes(self, sampling_results: list[Any],
                              original_data: dict[str, list[dict]],
                              target_total: int) -> ValidationResult:
        """Validate that sample sizes meet expectations."""
        issues = []
        recommendations = []
        details = {}

        total_sampled = sum(len(result.samples) for result in sampling_results)
        size_deviation = abs(total_sampled - target_total) / target_total if target_total > 0 else 1.0

        # Check individual tier sample sizes
        tier_deviations = {}
        for result in sampling_results:
            tier_id = result.metadata.get("tier_id", "unknown")
            target_count = result.metadata.get("target_count", 0)
            actual_count = len(result.samples)

            if target_count > 0:
                deviation = abs(actual_count - target_count) / target_count
                tier_deviations[tier_id] = {
                    "target": target_count,
                    "actual": actual_count,
                    "deviation": deviation
                }

                if deviation > self.thresholds["sample_size_deviation"]:
                    issues.append(f"Tier {tier_id}: target {target_count}, got {actual_count} "
                                 f"(deviation: {deviation:.3f})")

        details.update({
            "target_total": target_total,
            "actual_total": total_sampled,
            "total_deviation": size_deviation,
            "tier_deviations": tier_deviations
        })

        passed = size_deviation <= self.thresholds["sample_size_deviation"]
        score = max(0.0, 1.0 - (size_deviation / self.thresholds["sample_size_deviation"]))

        if not passed:
            recommendations.append("Review sampling algorithm to better meet target sizes")
            recommendations.append("Check data availability constraints")

        return ValidationResult(
            check_name="sample_size_validation",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_data_integrity(self, sampling_results: list[Any],
                                original_data: dict[str, list[dict]],
                                target_total: int) -> ValidationResult:
        """Validate data integrity of sampled data."""
        issues = []
        recommendations = []
        details = {}

        # Check for required fields
        required_fields = ["id", "messages"]
        missing_fields_count = 0
        empty_samples_count = 0
        malformed_samples = []

        total_samples = 0
        for result in sampling_results:
            for sample in result.samples:
                total_samples += 1

                # Check required fields
                for field in required_fields:
                    if field not in sample:
                        missing_fields_count += 1
                        malformed_samples.append(f"Missing {field} in sample {sample.get('id', 'unknown')}")

                # Check for empty messages
                messages = sample.get("messages", [])
                if not messages or len(messages) == 0:
                    empty_samples_count += 1
                    malformed_samples.append(f"Empty messages in sample {sample.get('id', 'unknown')}")

        integrity_score = 1.0
        if total_samples > 0:
            integrity_score -= (missing_fields_count + empty_samples_count) / total_samples

        details.update({
            "total_samples": total_samples,
            "missing_fields_count": missing_fields_count,
            "empty_samples_count": empty_samples_count,
            "malformed_samples": malformed_samples[:10],  # Limit to first 10
            "integrity_score": integrity_score
        })

        passed = integrity_score >= 0.95  # 95% integrity threshold
        score = max(0.0, integrity_score)

        if not passed:
            issues.append(f"Data integrity issues: {missing_fields_count} missing fields, "
                         f"{empty_samples_count} empty samples")
            recommendations.append("Review data preprocessing and validation")
            recommendations.append("Implement stricter data quality checks before sampling")

        return ValidationResult(
            check_name="data_integrity_check",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_statistical_properties(self, sampling_results: list[Any],
                                        original_data: dict[str, list[dict]],
                                        target_total: int) -> ValidationResult:
        """Validate statistical properties of the sample."""
        issues = []
        recommendations = []
        details = {}

        # Collect sample statistics
        sample_lengths = []
        quality_scores = []

        for result in sampling_results:
            for sample in result.samples:
                # Message count
                messages = sample.get("messages", [])
                sample_lengths.append(len(messages))

                # Quality score
                quality = getattr(sample, "quality_score", result.quality_score)
                quality_scores.append(quality)

        if not sample_lengths or not quality_scores:
            return ValidationResult(
                check_name="statistical_properties",
                passed=False,
                score=0.0,
                details={"error": "Insufficient data for statistical analysis"},
                issues=["No samples available for statistical analysis"],
                recommendations=["Ensure samples are properly collected"]
            )

        # Calculate statistics
        length_stats = {
            "mean": np.mean(sample_lengths),
            "std": np.std(sample_lengths),
            "min": np.min(sample_lengths),
            "max": np.max(sample_lengths),
            "median": np.median(sample_lengths)
        }

        quality_stats = {
            "mean": np.mean(quality_scores),
            "std": np.std(quality_scores),
            "min": np.min(quality_scores),
            "max": np.max(quality_scores),
            "median": np.median(quality_scores)
        }

        # Check for reasonable distributions
        reasonable_stats = True

        # Check if message lengths are reasonable (2-20 messages typical)
        if length_stats["mean"] < 2 or length_stats["mean"] > 50:
            issues.append(f"Unusual average message count: {length_stats['mean']:.1f}")
            reasonable_stats = False

        # Check if quality distribution is reasonable
        if quality_stats["std"] < 0.05:  # Too little variation
            issues.append(f"Very low quality variation: {quality_stats['std']:.3f}")
            reasonable_stats = False

        details.update({
            "sample_count": len(sample_lengths),
            "length_statistics": length_stats,
            "quality_statistics": quality_stats,
            "reasonable_distributions": reasonable_stats
        })

        passed = reasonable_stats
        score = 1.0 if reasonable_stats else 0.7  # Partial credit for having data

        if not passed:
            recommendations.append("Review sampling criteria for more realistic distributions")
            recommendations.append("Check data preprocessing for potential issues")

        return ValidationResult(
            check_name="statistical_properties",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations
        )

    def _generate_validation_summary(self, validation_results: list[ValidationResult],
                                   sampling_results: list[Any],
                                   target_total: int) -> dict[str, Any]:
        """Generate a comprehensive validation summary."""
        total_samples = sum(len(result.samples) for result in sampling_results)

        return {
            "validation_timestamp": np.datetime64("now").astype(str),
            "target_samples": target_total,
            "actual_samples": total_samples,
            "validation_checks": len(validation_results),
            "passed_checks": sum(1 for r in validation_results if r.passed),
            "failed_checks": sum(1 for r in validation_results if not r.passed),
            "average_score": np.mean([r.score for r in validation_results]),
            "critical_failures": sum(1 for r in validation_results
                                   if not r.passed and r.check_name in self.critical_checks),
            "tier_count": len(sampling_results),
            "validation_status": "PASSED" if all(r.passed for r in validation_results) else "FAILED"
        }


    def export_validation_report(self, report: SamplingValidationReport,
                               output_path: str) -> None:
        """Export validation report to JSON file."""
        report_dict = {
            "overall_score": report.overall_score,
            "passed_checks": report.passed_checks,
            "total_checks": report.total_checks,
            "summary": report.summary,
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "validation_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "issues": r.issues,
                    "recommendations": r.recommendations
                }
                for r in report.validation_results
            ]
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Validation report exported to {output_path}")


def main():
    """Example usage of the sampling validation system."""
    # This would typically be called after sampling operations
    SamplingValidator()

    # Example validation (would use real sampling results)


if __name__ == "__main__":
    main()
