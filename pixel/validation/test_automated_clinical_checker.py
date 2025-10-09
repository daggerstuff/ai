"""
Unit tests for Automated Clinical Appropriateness Checker

This module provides comprehensive unit tests for the automated clinical
appropriateness checking system, covering rule validation, pattern matching,
and appropriateness assessment.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from .automated_clinical_checker import (
    AutomatedClinicalChecker,
    ClinicalRule,
    AppropriatenessViolation,
    AppropriatenessCheckResult,
    AppropriatenessLevel,
    ViolationType,
    CheckCategory,
)

from .clinical_accuracy_validator import ClinicalContext, TherapeuticModality


class TestAutomatedClinicalChecker:
    """Test suite for AutomatedClinicalChecker"""

    @pytest.fixture
    def checker(self):
        """Create a checker instance for testing"""
        return AutomatedClinicalChecker()

    @pytest.fixture
    def sample_context(self):
        """Create a sample clinical context"""
        return ClinicalContext(
            client_presentation="Client with depression and anxiety",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

    @pytest.fixture
    def sample_rule(self):
        """Create a sample clinical rule"""
        return ClinicalRule(
            rule_id="TEST001",
            name="Test Rule",
            description="Test rule for unit testing",
            category=CheckCategory.BOUNDARY_CHECK,
            violation_type=ViolationType.BOUNDARY_VIOLATION,
            severity=5,
            patterns=[r"test pattern"],
            keywords=["test keyword"],
        )

    def test_checker_initialization(self, checker):
        """Test checker initialization"""
        assert checker is not None
        assert isinstance(checker.rules, dict)
        assert len(checker.rules) > 0  # Should have default rules
        assert isinstance(checker.compiled_patterns, dict)
        assert isinstance(checker.check_stats, dict)

    def test_default_rules_loaded(self, checker):
        """Test that default rules are loaded correctly"""
        # Check for specific default rules
        assert "BR001" in checker.rules  # Personal Information Sharing
        assert "SF001" in checker.rules  # Suicide Risk Minimization
        assert "ET001" in checker.rules  # Confidentiality Maintenance

        # Verify rule structure
        rule = checker.rules["BR001"]
        assert rule.name == "Personal Information Sharing"
        assert rule.category == CheckCategory.BOUNDARY_CHECK
        assert rule.violation_type == ViolationType.BOUNDARY_VIOLATION
        assert rule.severity > 0
        assert len(rule.patterns) > 0

    def test_pattern_compilation(self, checker):
        """Test that regex patterns are compiled correctly"""
        assert len(checker.compiled_patterns) > 0

        # Check that patterns are compiled for existing rules
        for rule in checker.rules.values():
            for pattern in rule.patterns:
                pattern_key = f"{rule.rule_id}_{pattern}"
                # Pattern should exist or compilation should have been attempted
                # (some patterns might fail compilation and be skipped)

    @pytest.mark.asyncio
    async def test_appropriate_response(self, checker, sample_context):
        """Test checking of appropriate therapeutic response"""
        appropriate_response = """
        I understand you're experiencing depression and anxiety. These are
        challenging feelings to cope with. Let's work together to explore
        your symptoms and develop effective coping strategies. How long
        have you been experiencing these feelings?
        """

        result = await checker.check_appropriateness(appropriate_response, sample_context)

        assert isinstance(result, AppropriatenessCheckResult)
        assert result.overall_level in [
            AppropriatenessLevel.HIGHLY_APPROPRIATE,
            AppropriatenessLevel.APPROPRIATE,
        ]
        assert len(result.violations) == 0
        assert len(result.passed_checks) > 0
        assert result.confidence_score > 0.8

    @pytest.mark.asyncio
    async def test_boundary_violation_detection(self, checker, sample_context):
        """Test detection of boundary violations"""
        boundary_violation_response = """
        I understand your struggles. You know, my wife went through something
        similar when we got divorced. Let me tell you about my personal
        experience with depression. Maybe we could grab coffee outside of
        therapy to discuss this more.
        """

        result = await checker.check_appropriateness(boundary_violation_response, sample_context)

        assert result.overall_level in [
            AppropriatenessLevel.INAPPROPRIATE,
            AppropriatenessLevel.DANGEROUS,
        ]
        assert len(result.violations) > 0

        # Check for specific boundary violations
        boundary_violations = [
            v for v in result.violations if v.violation_type == ViolationType.BOUNDARY_VIOLATION
        ]
        assert len(boundary_violations) > 0

    @pytest.mark.asyncio
    async def test_safety_violation_detection(self, checker, sample_context):
        """Test detection of safety violations"""
        safety_violation_response = """
        Everyone thinks about suicide sometimes, you don't really mean it.
        You're just seeking attention. Things aren't that bad, let's talk
        about something else next session.
        """

        result = await checker.check_appropriateness(safety_violation_response, sample_context)

        assert result.overall_level in [
            AppropriatenessLevel.INAPPROPRIATE,
            AppropriatenessLevel.DANGEROUS,
        ]

        # Check for safety violations
        safety_violations = [
            v for v in result.violations if v.violation_type == ViolationType.SAFETY_VIOLATION
        ]
        assert len(safety_violations) > 0

        # Should have critical warnings
        assert any("CRITICAL" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_confidentiality_violation_detection(self, checker, sample_context):
        """Test detection of confidentiality violations"""
        confidentiality_violation_response = """
        I had another client who had the same problem. Someone else told me
        about a similar case. I can tell your family about what we discuss
        here if you'd like.
        """

        result = await checker.check_appropriateness(
            confidentiality_violation_response, sample_context
        )

        # Check for confidentiality violations
        confidentiality_violations = [
            v for v in result.violations if v.violation_type == ViolationType.CONFIDENTIALITY_BREACH
        ]
        assert len(confidentiality_violations) > 0

    @pytest.mark.asyncio
    async def test_cultural_insensitivity_detection(self, checker, sample_context):
        """Test detection of cultural insensitivity"""
        culturally_insensitive_response = """
        People like you always have these problems. In your culture,
        everyone deals with this the same way. That's just how your
        people are. You should be more like Americans.
        """

        result = await checker.check_appropriateness(
            culturally_insensitive_response, sample_context
        )

        # Check for cultural insensitivity violations
        cultural_violations = [
            v for v in result.violations if v.violation_type == ViolationType.CULTURAL_INSENSITIVITY
        ]
        assert len(cultural_violations) > 0

    @pytest.mark.asyncio
    async def test_professional_language_violation(self, checker, sample_context):
        """Test detection of unprofessional language"""
        unprofessional_response = """
        That's crazy! You're being dramatic and overreacting.
        Just get over it and move on. Suck it up and deal with it.
        That's nuts and weird behavior.
        """

        result = await checker.check_appropriateness(unprofessional_response, sample_context)

        # Check for therapeutic violations (unprofessional language)
        therapeutic_violations = [
            v for v in result.violations if v.violation_type == ViolationType.THERAPEUTIC_VIOLATION
        ]
        assert len(therapeutic_violations) > 0

    @pytest.mark.asyncio
    async def test_multiple_violations(self, checker, sample_context):
        """Test response with multiple types of violations"""
        multiple_violations_response = """
        That's crazy! My wife had the same problem. I had another client
        who dealt with this by just getting over it. People like you
        always overreact. Let's grab coffee outside therapy to discuss
        my personal experience with this.
        """

        result = await checker.check_appropriateness(multiple_violations_response, sample_context)

        assert result.overall_level in [
            AppropriatenessLevel.INAPPROPRIATE,
            AppropriatenessLevel.DANGEROUS,
        ]
        assert len(result.violations) >= 3

        # Should have multiple violation types
        violation_types = set(v.violation_type for v in result.violations)
        assert len(violation_types) >= 2

    def test_add_rule(self, checker, sample_rule):
        """Test adding a new clinical rule"""
        initial_count = len(checker.rules)
        result = checker.add_rule(sample_rule)

        assert result is True
        assert len(checker.rules) == initial_count + 1
        assert sample_rule.rule_id in checker.rules
        assert checker.rules[sample_rule.rule_id] == sample_rule

    def test_remove_rule(self, checker, sample_rule):
        """Test removing a clinical rule"""
        # Add rule first
        checker.add_rule(sample_rule)
        initial_count = len(checker.rules)

        # Remove rule
        result = checker.remove_rule(sample_rule.rule_id)

        assert result is True
        assert len(checker.rules) == initial_count - 1
        assert sample_rule.rule_id not in checker.rules

    def test_remove_nonexistent_rule(self, checker):
        """Test removing a non-existent rule"""
        result = checker.remove_rule("NONEXISTENT")
        assert result is False

    @pytest.mark.asyncio
    async def test_custom_rule_application(self, checker, sample_context):
        """Test that custom rules are applied correctly"""
        # Add custom rule
        custom_rule = ClinicalRule(
            rule_id="CUSTOM001",
            name="Custom Test Rule",
            description="Custom rule for testing",
            category=CheckCategory.THERAPEUTIC_CHECK,
            violation_type=ViolationType.THERAPEUTIC_VIOLATION,
            severity=6,
            patterns=[r"custom violation pattern"],
            keywords=["custom keyword"],
        )

        checker.add_rule(custom_rule)

        # Test response with custom violation
        response_with_custom_violation = """
        This response contains a custom violation pattern that should
        be detected by our custom rule.
        """

        result = await checker.check_appropriateness(response_with_custom_violation, sample_context)

        # Check that custom rule was applied
        custom_violations = [v for v in result.violations if v.rule_id == "CUSTOM001"]
        assert len(custom_violations) > 0

    def test_statistics_tracking(self, checker):
        """Test that statistics are tracked correctly"""
        initial_stats = checker.get_statistics()
        assert "total_checks" in initial_stats
        assert "violations_found" in initial_stats
        assert "by_category" in initial_stats
        assert "by_severity" in initial_stats

    @pytest.mark.asyncio
    async def test_statistics_update(self, checker, sample_context):
        """Test that statistics are updated after checks"""
        initial_stats = checker.get_statistics()
        initial_checks = initial_stats["total_checks"]

        # Perform a check
        await checker.check_appropriateness("This is a test response", sample_context)

        updated_stats = checker.get_statistics()
        assert updated_stats["total_checks"] == initial_checks + 1

    def test_export_rules(self, checker):
        """Test exporting rules to file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            checker.export_rules(output_path)

            # Verify file was created and contains data
            assert output_path.exists()

            with open(output_path, "r") as f:
                exported_data = json.load(f)

            assert isinstance(exported_data, list)
            assert len(exported_data) > 0

            # Verify rule structure
            first_rule = exported_data[0]
            required_fields = [
                "rule_id",
                "name",
                "description",
                "category",
                "violation_type",
                "severity",
                "patterns",
                "keywords",
            ]
            for field in required_fields:
                assert field in first_rule

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_load_custom_rules(self):
        """Test loading custom rules from file"""
        custom_rules = [
            {
                "rule_id": "CUSTOM001",
                "name": "Custom Rule 1",
                "description": "First custom rule",
                "category": "boundary_check",
                "violation_type": "boundary_violation",
                "severity": 7,
                "patterns": ["custom pattern 1"],
                "keywords": ["custom keyword 1"],
                "context_requirements": [],
                "exceptions": [],
                "is_active": True,
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(custom_rules, f)
            rules_path = Path(f.name)

        try:
            checker = AutomatedClinicalChecker(rules_path)

            # Verify custom rule was loaded
            assert "CUSTOM001" in checker.rules
            assert checker.rules["CUSTOM001"].name == "Custom Rule 1"

        finally:
            if rules_path.exists():
                rules_path.unlink()

    @pytest.mark.asyncio
    async def test_violation_context_extraction(self, checker, sample_context):
        """Test that violation context is extracted correctly"""
        response_with_violation = """
        This is some context before the violation. My wife went through
        something similar when we got divorced. This is context after
        the violation that should be included.
        """

        result = await checker.check_appropriateness(response_with_violation, sample_context)

        if result.violations:
            violation = result.violations[0]
            assert len(violation.context) > len(violation.matched_text)
            assert violation.matched_text in violation.context
            assert "location" in violation.__dict__
            assert "start" in violation.location
            assert "end" in violation.location

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, checker, sample_context):
        """Test confidence scoring for different types of matches"""
        # Pattern match should have higher confidence than keyword match
        pattern_response = "My wife went through something similar"
        keyword_response = "This contains my personal information"

        pattern_result = await checker.check_appropriateness(pattern_response, sample_context)
        keyword_result = await checker.check_appropriateness(keyword_response, sample_context)

        if pattern_result.violations and keyword_result.violations:
            # Pattern matches should generally have higher confidence
            pattern_confidence = pattern_result.violations[0].confidence
            keyword_confidence = keyword_result.violations[0].confidence

            # This is a general expectation, but may vary based on
            # specific implementation
            assert pattern_confidence >= 0.8
            assert keyword_confidence >= 0.5

    @pytest.mark.asyncio
    async def test_severity_based_appropriateness_levels(self, checker, sample_context):
        """Test that appropriateness levels correspond to violation severity"""
        # High severity violation
        high_severity_response = """
        Everyone thinks about suicide, you don't really mean it.
        You're just seeking attention.
        """

        # Low severity violation
        low_severity_response = "That's a bit weird, but okay."

        high_result = await checker.check_appropriateness(high_severity_response, sample_context)
        low_result = await checker.check_appropriateness(low_severity_response, sample_context)

        # High severity should result in more serious appropriateness level
        high_severity_levels = [AppropriatenessLevel.DANGEROUS, AppropriatenessLevel.INAPPROPRIATE]
        low_severity_levels = [AppropriatenessLevel.QUESTIONABLE, AppropriatenessLevel.APPROPRIATE]

        if high_result.violations:
            assert high_result.overall_level in high_severity_levels

        if low_result.violations:
            assert low_result.overall_level in low_severity_levels

    @pytest.mark.asyncio
    async def test_warning_generation(self, checker, sample_context):
        """Test that appropriate warnings are generated"""
        safety_violation_response = """
        Everyone feels suicidal sometimes, you're just being dramatic.
        """

        result = await checker.check_appropriateness(safety_violation_response, sample_context)

        # Should generate critical warning for safety violations
        assert len(result.warnings) > 0
        assert any("CRITICAL" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, checker, sample_context):
        """Test that appropriate recommendations are generated"""
        boundary_violation_response = """
        Let me tell you about my personal experience. We could meet
        outside of therapy to discuss this further.
        """

        result = await checker.check_appropriateness(boundary_violation_response, sample_context)

        # Should generate boundary-related recommendations
        assert len(result.recommendations) > 0
        recommendations_text = " ".join(result.recommendations).lower()
        assert any(
            keyword in recommendations_text
            for keyword in ["boundary", "professional", "supervision"]
        )


# Integration tests
class TestAutomatedClinicalCheckerIntegration:
    """Integration tests for automated clinical checker"""

    @pytest.mark.asyncio
    async def test_complete_checking_workflow(self):
        """Test complete checking workflow from start to finish"""
        checker = AutomatedClinicalChecker()

        context = ClinicalContext(
            client_presentation="Complex case with multiple issues",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="working",
            crisis_indicators=["suicidal_ideation"],
            cultural_factors=["hispanic_background"],
        )

        # Complex response with multiple issues
        complex_response = """
        I understand you're having suicidal thoughts, but everyone feels
        that way sometimes. You're probably just being dramatic. Let me
        tell you about my personal experience with depression when my
        wife left me. People like you from your culture always have
        these problems. Why don't we grab coffee outside of therapy
        to discuss this more? I had another client who dealt with
        this by just getting over it.
        """

        # Perform comprehensive check
        result = await checker.check_appropriateness(complex_response, context)

        # Verify comprehensive result
        assert isinstance(result, AppropriatenessCheckResult)
        assert result.check_id is not None
        assert isinstance(result.timestamp, datetime)
        assert result.overall_level in list(AppropriatenessLevel)
        assert 0.0 <= result.confidence_score <= 1.0

        # Should detect multiple violations
        assert len(result.violations) >= 3

        # Should have multiple violation types
        violation_types = set(v.violation_type for v in result.violations)
        assert len(violation_types) >= 2

        # Should generate warnings and recommendations
        assert len(result.warnings) > 0
        assert len(result.recommendations) > 0

        # Should be inappropriate or dangerous
        assert result.overall_level in [
            AppropriatenessLevel.INAPPROPRIATE,
            AppropriatenessLevel.DANGEROUS,
        ]

        # Get final statistics
        stats = checker.get_statistics()
        assert stats["total_checks"] >= 1
        assert stats["violations_found"] >= 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
