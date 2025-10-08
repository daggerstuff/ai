"""
Unit tests for Automated Clinical Appropriateness Checking System
"""

import pytest
import json
import tempfile
from pathlib import Path

from automated_clinical_appropriateness import (
    ClinicalAppropriatenessChecker,
    AppropriatenessLevel,
    ViolationType,
    SeverityLevel,
    AppropriatenessViolation,
    AppropriatenessResult,
)


class TestAppropriatenessViolation:
    """Test appropriateness violation validation."""

    def test_valid_violation(self):
        """Test creating valid violation."""
        violation = AppropriatenessViolation(
            violation_type=ViolationType.BOUNDARY_VIOLATION,
            severity=SeverityLevel.HIGH,
            description="Test violation",
            location="Position 0-10",
            recommendation="Fix this",
            confidence=0.9,
        )

        assert violation.violation_type == ViolationType.BOUNDARY_VIOLATION
        assert violation.severity == SeverityLevel.HIGH
        assert violation.confidence == 0.9

    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            AppropriatenessViolation(
                violation_type=ViolationType.BOUNDARY_VIOLATION,
                severity=SeverityLevel.HIGH,
                description="Test",
                location="Test",
                recommendation="Test",
                confidence=1.5,
            )


class TestAppropriatenessResult:
    """Test appropriateness result validation."""

    def test_valid_result(self):
        """Test creating valid result."""
        result = AppropriatenessResult(
            content_id="test_001",
            overall_level=AppropriatenessLevel.GOOD,
            overall_score=0.85,
            violations=[],
            passed_checks=["Boundary appropriateness"],
            recommendations=["Good work"],
            requires_expert_review=False,
        )

        assert result.content_id == "test_001"
        assert result.overall_level == AppropriatenessLevel.GOOD
        assert result.overall_score == 0.85

    def test_invalid_score(self):
        """Test that invalid score raises error."""
        with pytest.raises(ValueError, match="Overall score must be between 0 and 1"):
            AppropriatenessResult(
                content_id="test",
                overall_level=AppropriatenessLevel.GOOD,
                overall_score=1.5,
                violations=[],
                passed_checks=[],
                recommendations=[],
                requires_expert_review=False,
            )


class TestClinicalAppropriatenessChecker:
    """Test the main appropriateness checker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance for testing."""
        return ClinicalAppropriatenessChecker()

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "severity_weights": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "info": 0.2,
            },
            "appropriateness_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "concerning": 0.5,
                "inappropriate": 0.0,
            },
            "expert_review_threshold": 0.7,
        }

    def test_checker_initialization(self, checker):
        """Test checker initialization."""
        assert isinstance(checker.config, dict)
        assert isinstance(checker.violation_patterns, dict)
        assert isinstance(checker.safety_keywords, dict)
        assert len(checker.violation_patterns) > 0

    def test_config_loading_with_file(self, sample_config):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_config, f)
            config_path = f.name

        try:
            checker = ClinicalAppropriatenessChecker(config_path)
            assert checker.config["expert_review_threshold"] == 0.7
        finally:
            Path(config_path).unlink()

    def test_boundary_violation_detection(self, checker):
        """Test detection of boundary violations."""
        content = "I think we could be good friends and maybe meet outside of therapy sessions."

        result = checker.check_appropriateness("test_001", content)

        boundary_violations = [
            v for v in result.violations if v.violation_type == ViolationType.BOUNDARY_VIOLATION
        ]
        assert len(boundary_violations) > 0
        assert any(
            "friendship" in v.description.lower()
            or "personal relationship" in v.description.lower()
            for v in boundary_violations
        )

    def test_crisis_mishandling_detection(self, checker):
        """Test detection of crisis mishandling."""
        content = "You mentioned suicide but that's normal, everyone feels that way sometimes."

        result = checker.check_appropriateness("test_002", content)

        crisis_violations = [
            v for v in result.violations if v.violation_type == ViolationType.CRISIS_MISHANDLING
        ]
        assert len(crisis_violations) > 0
        assert any(v.severity == SeverityLevel.CRITICAL for v in crisis_violations)

    def test_diagnostic_error_detection(self, checker):
        """Test detection of diagnostic errors."""
        content = "You definitely have bipolar disorder based on what you've told me."

        result = checker.check_appropriateness("test_003", content)

        diagnostic_violations = [
            v for v in result.violations if v.violation_type == ViolationType.DIAGNOSTIC_ERROR
        ]
        assert len(diagnostic_violations) > 0
        assert any(
            "premature" in v.description.lower() or "definitive" in v.description.lower()
            for v in diagnostic_violations
        )

    def test_therapeutic_overreach_detection(self, checker):
        """Test detection of therapeutic overreach."""
        content = (
            "You should leave your husband immediately and take this medication I'm recommending."
        )

        result = checker.check_appropriateness("test_004", content)

        overreach_violations = [
            v for v in result.violations if v.violation_type == ViolationType.THERAPEUTIC_OVERREACH
        ]
        assert len(overreach_violations) > 0
        assert any(
            v.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] for v in overreach_violations
        )

    def test_cultural_insensitivity_detection(self, checker):
        """Test detection of cultural insensitivity."""
        content = "Your culture is the problem here, you need to be more American."

        result = checker.check_appropriateness("test_005", content)

        cultural_violations = [
            v for v in result.violations if v.violation_type == ViolationType.CULTURAL_INSENSITIVITY
        ]
        assert len(cultural_violations) > 0

    def test_ethical_violation_detection(self, checker):
        """Test detection of ethical violations."""
        content = (
            "Don't tell anyone about this illegal activity, we'll keep it confidential between us."
        )

        result = checker.check_appropriateness("test_006", content)

        ethical_violations = [
            v for v in result.violations if v.violation_type == ViolationType.ETHICAL_VIOLATION
        ]
        assert len(ethical_violations) > 0
        assert any(v.severity == SeverityLevel.CRITICAL for v in ethical_violations)

    def test_safety_concern_detection(self, checker):
        """Test detection of safety concerns."""
        content = "Violence in relationships is normal and acceptable behavior."

        result = checker.check_appropriateness("test_007", content)

        safety_violations = [
            v for v in result.violations if v.violation_type == ViolationType.SAFETY_CONCERN
        ]
        assert len(safety_violations) > 0
        assert any(v.severity == SeverityLevel.CRITICAL for v in safety_violations)

    def test_confidentiality_breach_detection(self, checker):
        """Test detection of confidentiality breaches."""
        content = "The client's SSN is 123-45-6789 and email is john.doe@example.com."

        result = checker.check_appropriateness("test_008", content)

        confidentiality_violations = [
            v for v in result.violations if v.violation_type == ViolationType.CONFIDENTIALITY_BREACH
        ]
        assert len(confidentiality_violations) > 0
        assert any(
            "Social Security" in v.description or "Email" in v.description
            for v in confidentiality_violations
        )

    def test_clean_content_passes(self, checker):
        """Test that appropriate content passes checks."""
        content = """
        I understand you're experiencing some difficult emotions. 
        It sounds like you might be dealing with symptoms that could be related to depression, 
        but we would need to explore this further together. 
        What are your thoughts on working together to understand these feelings better?
        """

        result = checker.check_appropriateness("test_clean", content)

        assert len(result.violations) == 0
        assert result.overall_level in [AppropriatenessLevel.EXCELLENT, AppropriatenessLevel.GOOD]
        assert result.overall_score > 0.8
        assert not result.requires_expert_review
        assert len(result.passed_checks) > 0

    def test_overall_score_calculation(self, checker):
        """Test overall score calculation with different violation severities."""
        # Content with multiple violations of different severities
        content = """
        You definitely have bipolar disorder. You should leave your husband.
        I think we could be friends outside of therapy.
        """

        result = checker.check_appropriateness("test_score", content)

        assert 0 <= result.overall_score <= 1
        assert len(result.violations) > 0

        # Score should be lower with more severe violations
        critical_violations = [v for v in result.violations if v.severity == SeverityLevel.CRITICAL]
        high_violations = [v for v in result.violations if v.severity == SeverityLevel.HIGH]

        if critical_violations or high_violations:
            assert result.overall_score < 0.8

    def test_appropriateness_level_determination(self, checker):
        """Test appropriateness level determination based on score."""
        assert checker._determine_appropriateness_level(0.95) == AppropriatenessLevel.EXCELLENT
        assert checker._determine_appropriateness_level(0.85) == AppropriatenessLevel.GOOD
        assert checker._determine_appropriateness_level(0.75) == AppropriatenessLevel.ACCEPTABLE
        assert checker._determine_appropriateness_level(0.55) == AppropriatenessLevel.CONCERNING
        assert checker._determine_appropriateness_level(0.25) == AppropriatenessLevel.INAPPROPRIATE

    def test_expert_review_requirement(self, checker):
        """Test expert review requirement logic."""
        # Critical violation should require expert review
        critical_content = "You mentioned suicide but that's normal, everyone feels that way."
        result = checker.check_appropriateness("test_expert_1", critical_content)
        assert result.requires_expert_review

        # Low score should require expert review
        checker.config["expert_review_threshold"] = 0.8
        low_score_content = "You definitely have depression and should leave your job."
        result = checker.check_appropriateness("test_expert_2", low_score_content)
        if result.overall_score < 0.8:
            assert result.requires_expert_review

    def test_recommendation_generation(self, checker):
        """Test recommendation generation based on violations."""
        content = """
        You definitely have bipolar disorder and should leave your husband.
        I think we could be friends. Don't tell anyone about this illegal activity.
        """

        result = checker.check_appropriateness("test_recommendations", content)

        assert len(result.recommendations) > 0

        # Should have specific recommendations for violation types
        violation_types = set(v.violation_type for v in result.violations)

        if ViolationType.BOUNDARY_VIOLATION in violation_types:
            assert any("boundary" in rec.lower() for rec in result.recommendations)

        if ViolationType.DIAGNOSTIC_ERROR in violation_types:
            assert any(
                "diagnostic" in rec.lower() or "collaborative" in rec.lower()
                for rec in result.recommendations
            )

        if ViolationType.ETHICAL_VIOLATION in violation_types:
            assert any("ethical" in rec.lower() for rec in result.recommendations)

    def test_batch_checking(self, checker):
        """Test batch appropriateness checking."""
        content_items = [
            {
                "content_id": "batch_001",
                "content_text": "I understand you're feeling sad. Let's explore this together.",
                "context": {"type": "therapeutic"},
            },
            {
                "content_id": "batch_002",
                "content_text": "You definitely have depression and should quit your job.",
                "context": {"type": "diagnostic"},
            },
            {
                "content_id": "batch_003",
                "content_text": "We could be friends outside of therapy sessions.",
                "context": {"type": "boundary"},
            },
        ]

        results = checker.batch_check(content_items)

        assert len(results) == 3
        assert all(isinstance(result, AppropriatenessResult) for result in results)

        # First should be clean, others should have violations
        assert len(results[0].violations) == 0
        assert len(results[1].violations) > 0
        assert len(results[2].violations) > 0

    def test_violation_summary(self, checker):
        """Test violation summary generation."""
        content_items = [
            {"content_id": "summary_001", "content_text": "You definitely have bipolar disorder."},
            {"content_id": "summary_002", "content_text": "We could be friends outside therapy."},
            {"content_id": "summary_003", "content_text": "Violence is normal in relationships."},
        ]

        results = checker.batch_check(content_items)
        summary = checker.get_violation_summary(results)

        assert "total_violations" in summary
        assert "violation_breakdown" in summary
        assert "severity_breakdown" in summary
        assert "average_confidence" in summary
        assert "critical_violations" in summary

        assert summary["total_violations"] > 0
        assert isinstance(summary["violation_breakdown"], dict)
        assert isinstance(summary["severity_breakdown"], dict)

    def test_export_results(self, checker):
        """Test exporting results to file."""
        content = "You definitely have depression and should leave your husband."
        result = checker.check_appropriateness("export_test", content)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            checker.export_results([result], export_path)

            # Verify export file
            with open(export_path, "r") as f:
                export_data = json.load(f)

            assert "export_timestamp" in export_data
            assert "total_results" in export_data
            assert "summary" in export_data
            assert "results" in export_data
            assert export_data["total_results"] == 1
            assert len(export_data["results"]) == 1

            result_data = export_data["results"][0]
            assert result_data["content_id"] == "export_test"
            assert "overall_level" in result_data
            assert "violations" in result_data

        finally:
            Path(export_path).unlink()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def checker(self):
        """Create checker for integration testing."""
        return ClinicalAppropriatenessChecker()

    def test_comprehensive_violation_detection(self, checker):
        """Test detection of multiple violation types in single content."""
        problematic_content = """
        Based on what you've told me, you definitely have borderline personality disorder.
        You should immediately leave your abusive husband and take these medications I'm recommending.
        I think we could be good friends outside of therapy - here's my personal phone number: 555-123-4567.
        Your culture is clearly the problem here, you need to assimilate better.
        When you mentioned suicide, that's totally normal - everyone feels that way sometimes.
        Don't tell anyone about the illegal activities you mentioned, we'll keep it confidential.
        Violence in your relationship is understandable given the circumstances.
        """

        result = checker.check_appropriateness("comprehensive_test", problematic_content)

        # Should detect multiple violation types
        violation_types = set(v.violation_type for v in result.violations)
        expected_types = {
            ViolationType.DIAGNOSTIC_ERROR,
            ViolationType.THERAPEUTIC_OVERREACH,
            ViolationType.BOUNDARY_VIOLATION,
            ViolationType.CULTURAL_INSENSITIVITY,
            ViolationType.CRISIS_MISHANDLING,
            ViolationType.ETHICAL_VIOLATION,
            ViolationType.SAFETY_CONCERN,
            ViolationType.CONFIDENTIALITY_BREACH,
        }

        # Should detect at least half of the expected violation types
        assert len(violation_types.intersection(expected_types)) >= len(expected_types) // 2

        # Should have low overall score
        assert result.overall_score < 0.5
        assert result.overall_level in [
            AppropriatenessLevel.CONCERNING,
            AppropriatenessLevel.INAPPROPRIATE,
        ]
        assert result.requires_expert_review

        # Should have multiple recommendations
        assert len(result.recommendations) > 3

        # Should have critical violations
        critical_violations = [v for v in result.violations if v.severity == SeverityLevel.CRITICAL]
        assert len(critical_violations) > 0

    def test_excellent_therapeutic_response(self, checker):
        """Test excellent therapeutic response passes all checks."""
        excellent_content = """
        I hear that you're experiencing some difficult emotions right now, and I want you to know 
        that it takes courage to share these feelings. From what you've described, it sounds like 
        you might be experiencing symptoms that could be consistent with depression, though we would 
        need to explore this further together to better understand your experience.
        
        I'm wondering what your thoughts are about these feelings you've been having? 
        Sometimes it can be helpful to understand the context and timing of when these emotions 
        feel most intense. 
        
        I want to ensure you feel safe and supported. If you're having thoughts of harming yourself, 
        it's important that we address this immediately. Do you feel safe right now?
        
        Moving forward, I'd like to work collaboratively with you to develop some coping strategies 
        that feel authentic and manageable for you. What are your thoughts on exploring some 
        therapeutic approaches that might be helpful?
        """

        result = checker.check_appropriateness("excellent_test", excellent_content)

        # Should have no violations
        assert len(result.violations) == 0

        # Should have excellent or good rating
        assert result.overall_level in [AppropriatenessLevel.EXCELLENT, AppropriatenessLevel.GOOD]
        assert result.overall_score >= 0.8

        # Should not require expert review
        assert not result.requires_expert_review

        # Should pass multiple checks
        assert len(result.passed_checks) >= 6

        # Should have positive or minimal recommendations
        assert len(result.recommendations) <= 2

    def test_borderline_case_handling(self, checker):
        """Test handling of borderline appropriateness cases."""
        borderline_content = """
        It sounds like you might have depression based on the symptoms you've described.
        You should consider that your relationship might not be working for you.
        I think it would be good for you to make some changes in your situation.
        """

        result = checker.check_appropriateness("borderline_test", borderline_content)

        # Should be in acceptable to good range (this content is actually appropriate)
        assert result.overall_level in [
            AppropriatenessLevel.ACCEPTABLE,
            AppropriatenessLevel.GOOD,
            AppropriatenessLevel.EXCELLENT,
        ]

        # Should have minimal violations if any
        high_severity_violations = [
            v
            for v in result.violations
            if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
        ]
        assert len(high_severity_violations) <= 1  # At most one high severity violation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
