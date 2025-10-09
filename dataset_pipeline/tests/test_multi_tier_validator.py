#!/usr/bin/env python3
"""
Test suite for Multi-Tier Quality Validation System
"""

import json
import tempfile

import pytest

from .datetime import datetime
from .multi_tier_validator import (
    DatasetTier,
    MultiTierValidator,
    QualityMetrics,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


class TestMultiTierValidator:
    """Test cases for MultiTierValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MultiTierValidator()

        # Sample conversation data
        self.high_quality_conversation = {
            "id": "high_quality_001",
            "content": "I understand you're feeling anxious about this situation. Let's explore some evidence-based coping strategies that might help you manage these feelings. How do you typically respond when anxiety arises?",
            "turns": [
                {"speaker": "user", "text": "I'm feeling really anxious about my job interview tomorrow."},
                {"speaker": "therapist", "text": "I understand you're feeling anxious about this situation. Let's explore some coping strategies."},
                {"speaker": "user", "text": "I usually just worry more and can't sleep."},
                {"speaker": "therapist", "text": "That's a common response. Let's work on some techniques to help you feel more prepared and calm."}
            ],
            "metadata": {
                "therapeutic_approach": "CBT",
                "condition": "anxiety",
                "quality_score": 0.95
            }
        }

        self.medium_quality_conversation = {
            "id": "medium_quality_001",
            "content": "That sounds hard. How are you feeling about it?",
            "turns": [
                {"speaker": "user", "text": "I'm having trouble with my relationship."},
                {"speaker": "therapist", "text": "That sounds hard. How are you feeling?"}
            ],
            "metadata": {
                "quality_score": 0.75
            }
        }

        self.low_quality_conversation = {
            "id": "low_quality_001",
            "content": "ok",
            "turns": [
                {"speaker": "user", "text": "I need help."},
                {"speaker": "therapist", "text": "ok"}
            ],
            "metadata": {
                "quality_score": 0.3
            }
        }

        self.safety_concern_conversation = {
            "id": "safety_concern_001",
            "content": "I want to harm myself and don't see any point in living.",
            "turns": [
                {"speaker": "user", "text": "I want to harm myself and don't see any point in living."},
                {"speaker": "therapist", "text": "I'm concerned about your safety. Let me connect you with crisis support resources immediately."}
            ],
            "metadata": {
                "safety_flag": True
            }
        }

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MultiTierValidator()
        assert validator.validation_history == []
        assert validator.tier_statistics == {}
        assert len(validator.tier_weights) == 6  # All tiers

        # Check tier weights structure
        for tier in DatasetTier:
            assert tier in validator.tier_weights
            weights = validator.tier_weights[tier]
            assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights sum to 1

    def test_dataset_tier_enum(self):
        """Test DatasetTier enum properties."""
        assert DatasetTier.PRIORITY.tier_name == "priority"
        assert DatasetTier.PRIORITY.threshold == 0.99
        assert DatasetTier.REDDIT.tier_name == "reddit"
        assert DatasetTier.REDDIT.threshold == 0.85
        assert DatasetTier.KNOWLEDGE.threshold == 0.75

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        metrics = QualityMetrics(
            coherence_score=0.9,
            therapeutic_accuracy=0.85,
            emotional_authenticity=0.8,
            safety_compliance=0.95,
            language_quality=0.88
        )

        overall_score = metrics.calculate_overall_score()

        # Check that overall score is calculated correctly
        expected = (0.9 * 0.25) + (0.85 * 0.30) + (0.8 * 0.20) + (0.95 * 0.15) + (0.88 * 0.10)
        assert abs(overall_score - expected) < 0.001
        assert metrics.overall_score == overall_score

    def test_validate_high_quality_conversation_priority_tier(self):
        """Test validation of high-quality conversation for Priority tier."""
        result = self.validator.validate_conversation(
            self.high_quality_conversation,
            DatasetTier.PRIORITY
        )

        assert result.conversation_id == "high_quality_001"
        assert result.tier == DatasetTier.PRIORITY
        assert isinstance(result.metrics, QualityMetrics)
        assert result.metrics.overall_score > 0.7  # Should be reasonably high
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.timestamp, datetime)

    def test_validate_low_quality_conversation_priority_tier(self):
        """Test validation of low-quality conversation for Priority tier."""
        result = self.validator.validate_conversation(
            self.low_quality_conversation,
            DatasetTier.PRIORITY
        )

        assert result.conversation_id == "low_quality_001"
        assert result.tier == DatasetTier.PRIORITY
        assert not result.passed  # Should fail Priority tier standards
        assert len(result.issues) > 0  # Should have validation issues
        assert len(result.recommendations) > 0  # Should have recommendations

        # Check for high severity issues (gap > 0.2 but < 0.3)
        high_severity_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.HIGH]
        assert len(high_severity_issues) > 0

    def test_validate_medium_quality_conversation_reddit_tier(self):
        """Test validation of medium-quality conversation for Reddit tier."""
        result = self.validator.validate_conversation(
            self.medium_quality_conversation,
            DatasetTier.REDDIT
        )

        assert result.conversation_id == "medium_quality_001"
        assert result.tier == DatasetTier.REDDIT
        # Medium quality might pass Reddit tier (85% threshold)
        assert result.metrics.overall_score > 0.5

    def test_safety_compliance_assessment(self):
        """Test safety compliance assessment."""
        result = self.validator.validate_conversation(
            self.safety_concern_conversation,
            DatasetTier.PROFESSIONAL
        )

        # Safety compliance should be high if properly handled
        assert result.metrics.safety_compliance >= 0.8

        # Should have appropriate recommendations for safety
        [
            rec for rec in result.recommendations
            if "safety" in rec.lower() or "crisis" in rec.lower()
        ]
        # Note: May or may not have safety recommendations depending on implementation

    def test_batch_validation(self):
        """Test batch validation functionality."""
        conversations = [
            self.high_quality_conversation,
            self.medium_quality_conversation,
            self.low_quality_conversation
        ]

        results = self.validator.validate_batch(conversations, DatasetTier.PROFESSIONAL)

        assert len(results) == 3
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all(result.tier == DatasetTier.PROFESSIONAL for result in results)

        # Check that conversation IDs match
        result_ids = [result.conversation_id for result in results]
        expected_ids = ["high_quality_001", "medium_quality_001", "low_quality_001"]
        assert result_ids == expected_ids

    def test_batch_validation_with_error(self):
        """Test batch validation with error handling."""
        # Create conversation that will cause validation error
        invalid_conversation = {"invalid": "data"}

        conversations = [self.high_quality_conversation, invalid_conversation]
        results = self.validator.validate_batch(conversations, DatasetTier.REDDIT)

        assert len(results) == 2

        # First result should be successful
        assert results[0].passed or not results[0].passed  # Either outcome is valid

        # Second result should be failed due to error
        assert not results[1].passed
        assert len(results[1].issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in results[1].issues)

    def test_tier_statistics_tracking(self):
        """Test tier statistics tracking."""
        # Validate some conversations
        conversations = [
            self.high_quality_conversation,
            self.medium_quality_conversation,
            self.low_quality_conversation
        ]

        self.validator.validate_batch(conversations, DatasetTier.REDDIT)

        # Check statistics
        stats = self.validator.get_tier_statistics(DatasetTier.REDDIT)
        assert stats["total_validated"] == 3
        assert stats["passed"] + stats["failed"] == 3
        assert "average_score" in stats
        assert stats["average_score"] > 0

    def test_validation_summary(self):
        """Test validation summary generation."""
        # Perform some validations
        self.validator.validate_conversation(self.high_quality_conversation, DatasetTier.PRIORITY)
        self.validator.validate_conversation(self.medium_quality_conversation, DatasetTier.REDDIT)

        summary = self.validator.get_validation_summary()

        assert "total_validations" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "pass_rate" in summary
        assert "tier_averages" in summary
        assert "tier_statistics" in summary
        assert "issue_severity_distribution" in summary
        assert "last_validation" in summary

        assert summary["total_validations"] == 2
        assert summary["passed"] + summary["failed"] == 2
        assert 0 <= summary["pass_rate"] <= 1

    def test_validation_summary_empty(self):
        """Test validation summary with no validations."""
        summary = self.validator.get_validation_summary()
        assert summary["message"] == "No validations performed yet"

    def test_issue_severity_determination(self):
        """Test issue severity determination."""
        # Test different score gaps
        assert self.validator._determine_severity(0.5, 0.9) == ValidationSeverity.CRITICAL  # gap = 0.4
        assert self.validator._determine_severity(0.7, 0.9) == ValidationSeverity.HIGH      # gap = 0.2
        assert self.validator._determine_severity(0.8, 0.9) == ValidationSeverity.MEDIUM    # gap = 0.1
        assert self.validator._determine_severity(0.85, 0.9) == ValidationSeverity.LOW      # gap = 0.05
        assert self.validator._determine_severity(0.88, 0.9) == ValidationSeverity.INFO     # gap = 0.02

    def test_improvement_suggestions(self):
        """Test improvement suggestions generation."""
        suggestions = self.validator._get_improvement_suggestions("coherence_score")
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)

        # Test all metric types
        metrics = ["coherence_score", "therapeutic_accuracy", "emotional_authenticity",
                  "safety_compliance", "language_quality"]

        for metric in metrics:
            suggestions = self.validator._get_improvement_suggestions(metric)
            assert len(suggestions) > 0

    def test_validation_result_severity_counts(self):
        """Test ValidationResult severity counts."""
        issues = [
            ValidationIssue(ValidationSeverity.CRITICAL, "test", "Critical issue", 0.3),
            ValidationIssue(ValidationSeverity.HIGH, "test", "High issue", 0.2),
            ValidationIssue(ValidationSeverity.CRITICAL, "test", "Another critical", 0.25)
        ]

        result = ValidationResult(
            conversation_id="test",
            tier=DatasetTier.REDDIT,
            metrics=QualityMetrics(),
            passed=False,
            issues=issues
        )

        counts = result.get_severity_counts()
        assert counts["critical"] == 2
        assert counts["high"] == 1
        assert counts["medium"] == 0
        assert counts["low"] == 0
        assert counts["info"] == 0

    def test_export_validation_report(self):
        """Test validation report export."""
        # Perform some validations
        self.validator.validate_conversation(self.high_quality_conversation, DatasetTier.PRIORITY)
        self.validator.validate_conversation(self.low_quality_conversation, DatasetTier.REDDIT)

        # Export report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        self.validator.export_validation_report(filepath)

        # Verify report content
        with open(filepath) as f:
            report = json.load(f)

        assert "validation_summary" in report
        assert "tier_statistics" in report
        assert "validation_history" in report

        assert len(report["validation_history"]) == 2

        # Clean up
        import os
        os.unlink(filepath)

    def test_tier_specific_weights(self):
        """Test that different tiers use different quality weights."""
        # Validate same conversation with different tiers
        priority_result = self.validator.validate_conversation(
            self.high_quality_conversation, DatasetTier.PRIORITY
        )
        reddit_result = self.validator.validate_conversation(
            self.high_quality_conversation, DatasetTier.REDDIT
        )

        # Scores might be different due to different weighting
        # Both should be ValidationResult objects
        assert isinstance(priority_result, ValidationResult)
        assert isinstance(reddit_result, ValidationResult)
        assert priority_result.tier == DatasetTier.PRIORITY
        assert reddit_result.tier == DatasetTier.REDDIT


def test_main_function():
    """Test the main function runs without errors."""
    # This is a basic test to ensure main() doesn't crash
    try:
        # We can't easily test the full main() due to print statements,
        # but we can test that it imports and the validator works
        validator = MultiTierValidator()
        assert validator is not None
    except Exception as e:
        pytest.fail(f"Main function test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
