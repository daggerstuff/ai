#!/usr/bin/env python3
"""
Unit tests for the sampling validation system.
"""

from unittest.mock import Mock

import pytest

from .sampling_validation import SamplingValidationReport, SamplingValidator, ValidationResult


class TestSamplingValidator:
    """Test cases for the SamplingValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SamplingValidator()

        # Create mock sampling results
        self.mock_sampling_results = [
            Mock(
                samples=[
                    {"id": f"tier1_{i}", "messages": [{"content": "test", "role": "user"}], "quality_score": 0.8}
                    for i in range(40)
                ],
                quality_score=0.8,
                metadata={"tier_id": "tier_1_priority", "target_count": 40}
            ),
            Mock(
                samples=[
                    {"id": f"tier2_{i}", "messages": [{"content": "test", "role": "assistant"}], "quality_score": 0.7}
                    for i in range(25)
                ],
                quality_score=0.7,
                metadata={"tier_id": "tier_2_professional", "target_count": 25}
            ),
            Mock(
                samples=[
                    {"id": f"tier3_{i}", "messages": [{"content": "test", "role": "user"}], "quality_score": 0.6}
                    for i in range(20)
                ],
                quality_score=0.6,
                metadata={"tier_id": "tier_3_cot", "target_count": 20}
            )
        ]

        # Create mock original data
        self.mock_original_data = {
            "tier_1_priority": [{"id": f"orig1_{i}", "messages": []} for i in range(100)],
            "tier_2_professional": [{"id": f"orig2_{i}", "messages": []} for i in range(80)],
            "tier_3_cot": [{"id": f"orig3_{i}", "messages": []} for i in range(60)]
        }

    def test_validator_initialization(self):
        """Test validator initialization with default and custom config."""
        # Default initialization
        validator = SamplingValidator()
        assert validator.thresholds["distribution_deviation"] == 0.15
        assert validator.thresholds["quality_variance"] == 0.20
        assert len(validator.critical_checks) == 4

        # Custom configuration
        custom_config = {
            "thresholds": {
                "distribution_deviation": 0.10,
                "quality_variance": 0.15
            }
        }
        validator = SamplingValidator(custom_config)
        assert validator.thresholds["distribution_deviation"] == 0.10
        assert validator.thresholds["quality_variance"] == 0.15

    def test_validate_sampling_results_success(self):
        """Test successful validation of good sampling results."""
        report = self.validator.validate_sampling_results(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert isinstance(report, SamplingValidationReport)
        assert report.total_checks == 8  # All validation checks
        assert report.overall_score > 0.0
        assert "validation_timestamp" in report.summary
        assert report.summary["actual_samples"] == 85

    def test_distribution_validation_pass(self):
        """Test distribution validation with good distribution."""
        result = self.validator._validate_distribution(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "distribution_validation"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert "expected_distribution" in result.details
        assert "actual_distribution" in result.details

    def test_distribution_validation_fail(self):
        """Test distribution validation with poor distribution."""
        # Create imbalanced sampling results
        imbalanced_results = [
            Mock(
                samples=[{"id": f"tier1_{i}", "messages": []} for i in range(90)],
                quality_score=0.8,
                metadata={"tier_id": "tier_1_priority", "target_count": 90}
            ),
            Mock(
                samples=[{"id": f"tier2_{i}", "messages": []} for i in range(5)],
                quality_score=0.7,
                metadata={"tier_id": "tier_2_professional", "target_count": 5}
            )
        ]

        result = self.validator._validate_distribution(
            imbalanced_results,
            self.mock_original_data,
            target_total=95
        )

        assert result.check_name == "distribution_validation"
        assert len(result.issues) > 0
        assert len(result.recommendations) > 0

    def test_quality_consistency_validation(self):
        """Test quality consistency validation."""
        result = self.validator._validate_quality_consistency(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "quality_consistency"
        assert "tier_quality_stats" in result.details
        assert "overall_mean_quality" in result.details
        assert 0.0 <= result.score <= 1.0

    def test_duplicate_detection_no_duplicates(self):
        """Test duplicate detection with no duplicates."""
        result = self.validator._validate_duplicates(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "duplicate_detection"
        assert result.details["duplicate_rate"] == 0.0
        assert result.passed
        assert result.score == 1.0

    def test_duplicate_detection_with_duplicates(self):
        """Test duplicate detection with duplicates present."""
        # Create results with duplicates
        duplicate_results = [
            Mock(
                samples=[
                    {"id": "duplicate_1", "messages": []},
                    {"id": "duplicate_1", "messages": []},  # Duplicate
                    {"id": "unique_1", "messages": []}
                ],
                quality_score=0.8,
                metadata={"tier_id": "tier_1_priority", "target_count": 3}
            )
        ]

        result = self.validator._validate_duplicates(
            duplicate_results,
            self.mock_original_data,
            target_total=3
        )

        assert result.check_name == "duplicate_detection"
        assert result.details["duplicate_rate"] > 0.0
        assert len(result.issues) > 0

    def test_coverage_validation(self):
        """Test coverage validation."""
        result = self.validator._validate_coverage(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "coverage_validation"
        assert "quality_range" in result.details
        assert "quality_bands" in result.details
        assert 0.0 <= result.score <= 1.0

    def test_tier_balance_validation(self):
        """Test tier balance validation."""
        result = self.validator._validate_tier_balance(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "tier_balance_check"
        assert "tier_counts" in result.details
        assert "tier_proportions" in result.details
        assert "balance_ratio" in result.details

    def test_sample_sizes_validation(self):
        """Test sample sizes validation."""
        result = self.validator._validate_sample_sizes(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "sample_size_validation"
        assert "target_total" in result.details
        assert "actual_total" in result.details
        assert result.details["actual_total"] == 85

    def test_data_integrity_validation_pass(self):
        """Test data integrity validation with good data."""
        result = self.validator._validate_data_integrity(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "data_integrity_check"
        assert result.details["missing_fields_count"] == 0
        assert result.details["empty_samples_count"] == 0
        assert result.passed

    def test_data_integrity_validation_fail(self):
        """Test data integrity validation with bad data."""
        # Create results with integrity issues
        bad_results = [
            Mock(
                samples=[
                    {"messages": []},  # Missing 'id'
                    {"id": "test_2"},  # Missing 'messages'
                    {"id": "test_3", "messages": []}  # Empty messages
                ],
                quality_score=0.8,
                metadata={"tier_id": "tier_1_priority", "target_count": 3}
            )
        ]

        result = self.validator._validate_data_integrity(
            bad_results,
            self.mock_original_data,
            target_total=3
        )

        assert result.check_name == "data_integrity_check"
        assert result.details["missing_fields_count"] > 0 or result.details["empty_samples_count"] > 0
        assert len(result.issues) > 0

    def test_statistical_properties_validation(self):
        """Test statistical properties validation."""
        result = self.validator._validate_statistical_properties(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert result.check_name == "statistical_properties"
        assert "length_statistics" in result.details
        assert "quality_statistics" in result.details
        assert "sample_count" in result.details

    def test_validation_with_empty_results(self):
        """Test validation with empty sampling results."""
        empty_results = []

        report = self.validator.validate_sampling_results(
            empty_results,
            self.mock_original_data,
            target_total=100
        )

        assert isinstance(report, SamplingValidationReport)
        assert report.summary["actual_samples"] == 0
        assert len(report.critical_issues) > 0

    def test_validation_history_tracking(self):
        """Test that validation history is properly tracked."""
        initial_history_length = len(self.validator.validation_history)

        self.validator.validate_sampling_results(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        assert len(self.validator.validation_history) == initial_history_length + 1
        assert isinstance(self.validator.validation_history[-1], SamplingValidationReport)

    def test_export_validation_report(self, tmp_path):
        """Test exporting validation report to JSON."""
        report = self.validator.validate_sampling_results(
            self.mock_sampling_results,
            self.mock_original_data,
            target_total=85
        )

        output_file = tmp_path / "validation_report.json"
        self.validator.export_validation_report(report, str(output_file))

        assert output_file.exists()

        # Verify JSON structure
        import json
        with open(output_file) as f:
            exported_data = json.load(f)

        assert "overall_score" in exported_data
        assert "validation_results" in exported_data
        assert "summary" in exported_data
        assert len(exported_data["validation_results"]) == report.total_checks

    def test_critical_checks_identification(self):
        """Test that critical checks are properly identified."""
        # Create results that fail critical checks
        failing_results = [
            Mock(
                samples=[{"id": f"tier1_{i}", "messages": []} for i in range(100)],  # All samples in one tier
                quality_score=0.8,
                metadata={"tier_id": "tier_1_priority", "target_count": 100}
            )
        ]

        report = self.validator.validate_sampling_results(
            failing_results,
            self.mock_original_data,
            target_total=100
        )

        # Should have critical issues due to poor distribution
        assert len(report.critical_issues) > 0
        assert report.summary["critical_failures"] > 0

    def test_validation_result_structure(self):
        """Test that ValidationResult has proper structure."""
        result = ValidationResult(
            check_name="test_check",
            passed=True,
            score=0.85,
            details={"test": "data"},
            issues=[],
            recommendations=[]
        )

        assert result.check_name == "test_check"
        assert result.passed
        assert result.score == 0.85
        assert result.details == {"test": "data"}
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)

    def test_validation_report_structure(self):
        """Test that SamplingValidationReport has proper structure."""
        validation_results = [
            ValidationResult("test1", True, 0.9, {}, [], []),
            ValidationResult("test2", False, 0.5, {}, ["issue"], ["recommendation"])
        ]

        report = SamplingValidationReport(
            overall_score=0.7,
            passed_checks=1,
            total_checks=2,
            validation_results=validation_results,
            summary={"test": "summary"},
            critical_issues=["critical issue"],
            recommendations=["recommendation"]
        )

        assert report.overall_score == 0.7
        assert report.passed_checks == 1
        assert report.total_checks == 2
        assert len(report.validation_results) == 2
        assert len(report.critical_issues) == 1
        assert len(report.recommendations) == 1


def test_main_function():
    """Test the main function runs without error."""
    from .sampling_validation import main

    # Should run without raising exceptions
    main()


if __name__ == "__main__":
    pytest.main([__file__])
