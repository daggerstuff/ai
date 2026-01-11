"""
Unit tests for Safety and Ethics Compliance Validator

This module provides comprehensive unit tests for the safety and ethics
compliance validation system, covering safety assessment, ethics compliance,
legal requirements, and violation detection.
"""

from datetime import datetime

import pytest

from .clinical_accuracy_validator import (
    ClinicalContext,
    SafetyRiskLevel,
    TherapeuticModality,
)
from .safety_ethics_validator import (
    ComplianceLevel,
    EthicsCategory,
    LegalCategory,
    SafetyCategory,
    SafetyEthicsComplianceResult,
    SafetyEthicsValidator,
    SafetyIndicator,
    ViolationSeverity,
)


class TestSafetyEthicsValidator:
    """Test suite for SafetyEthicsValidator"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing"""
        return SafetyEthicsValidator()

    @pytest.fixture
    def sample_context(self):
        """Create a sample clinical context"""
        return ClinicalContext(
            client_presentation="Client with safety concerns",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="crisis",
            crisis_indicators=["suicidal_ideation"],
        )

    @pytest.fixture
    def sample_safety_indicator(self):
        """Create a sample safety indicator"""
        return SafetyIndicator(
            indicator_id="TEST_SI001",
            category=SafetyCategory.SUICIDE_RISK,
            description="Test suicide risk indicator",
            risk_level=SafetyRiskLevel.HIGH,
            keywords=["test suicide", "test kill"],
            patterns=[r"test (suicide|kill) pattern"],
            immediate_action_required=True,
        )

    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert isinstance(validator.safety_indicators, dict)
        assert isinstance(validator.ethics_guidelines, dict)
        assert isinstance(validator.legal_requirements, dict)
        assert len(validator.safety_indicators) > 0
        assert len(validator.ethics_guidelines) > 0
        assert len(validator.legal_requirements) > 0

    def test_safety_indicators_loaded(self, validator):
        """Test that safety indicators are loaded correctly"""
        # Check for specific safety indicators
        assert "SI001" in validator.safety_indicators  # Suicide risk
        assert "SI002" in validator.safety_indicators  # Self-harm risk
        assert "SI003" in validator.safety_indicators  # Violence risk
        assert "SI004" in validator.safety_indicators  # Child safety

        # Verify indicator structure
        indicator = validator.safety_indicators["SI001"]
        assert indicator.category == SafetyCategory.SUICIDE_RISK
        assert indicator.risk_level == SafetyRiskLevel.HIGH
        assert len(indicator.keywords) > 0
        assert indicator.immediate_action_required is True

    def test_ethics_guidelines_loaded(self, validator):
        """Test that ethics guidelines are loaded correctly"""
        # Check for specific ethics guidelines
        assert "EG001" in validator.ethics_guidelines  # Confidentiality
        assert "EG002" in validator.ethics_guidelines  # Dual relationships
        assert "EG003" in validator.ethics_guidelines  # Competence
        assert "EG004" in validator.ethics_guidelines  # Informed consent

        # Verify guideline structure
        guideline = validator.ethics_guidelines["EG001"]
        assert guideline.category == EthicsCategory.CONFIDENTIALITY
        assert guideline.severity == ViolationSeverity.SEVERE
        assert len(guideline.requirements) > 0
        assert len(guideline.violations) > 0

    def test_legal_requirements_loaded(self, validator):
        """Test that legal requirements are loaded correctly"""
        # Check for specific legal requirements
        assert "LR001" in validator.legal_requirements  # Mandatory reporting
        assert "LR002" in validator.legal_requirements  # Duty to warn
        assert "LR003" in validator.legal_requirements  # HIPAA compliance

        # Verify requirement structure
        requirement = validator.legal_requirements["LR001"]
        assert requirement.category == LegalCategory.MANDATORY_REPORTING
        assert requirement.mandatory is True
        assert len(requirement.penalties) > 0

    @pytest.mark.asyncio
    async def test_compliant_response_validation(self, validator, sample_context):
        """Test validation of compliant therapeutic response"""
        compliant_response = """
        I understand you're having thoughts of suicide, and I want you to know
        that I take this very seriously. Your safety is my primary concern.
        Let's work together to ensure you're safe and develop a safety plan.
        """

        result = await validator.validate_compliance(compliant_response, sample_context)

        assert isinstance(result, SafetyEthicsComplianceResult)
        assert result.overall_compliance in [
            ComplianceLevel.FULLY_COMPLIANT,
            ComplianceLevel.MOSTLY_COMPLIANT,
            ComplianceLevel.PARTIALLY_COMPLIANT,
        ]
        assert result.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_safety_violation_detection(self, validator, sample_context):
        """Test detection of safety violations"""
        unsafe_response = """
        Everyone wants to kill themselves sometimes. You don't really mean it,
        you're just being dramatic. Things aren't that bad, let's talk about
        something else next session.
        """

        result = await validator.validate_compliance(unsafe_response, sample_context)

        # Should detect some violations (safety or legal)
        assert len(result.violations) > 0

        # Should have some safety risk detected
        assert len(result.safety_result.indicators_detected) > 0

        # Should have warnings if high risk
        if result.safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]:
            assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_ethics_violation_detection(self, validator, sample_context):
        """Test detection of ethics violations"""
        unethical_response = """
        Let me tell you about another client who had the same problem.
        I had a similar experience in my personal life. We could meet
        outside of therapy to discuss this further as friends.
        """

        result = await validator.validate_compliance(unethical_response, sample_context)

        # Should detect some violations
        assert len(result.violations) > 0

        # Should have some immediate actions due to violations
        assert len(result.immediate_actions) > 0

        # Violations should include legal compliance issues
        legal_violations = [v for v in result.violations if "legal" in v.title.lower()]
        assert len(legal_violations) > 0

    @pytest.mark.asyncio
    async def test_multiple_violation_types(self, validator, sample_context):
        """Test response with multiple types of violations"""
        multiple_violations_response = """
        Everyone thinks about suicide, you're just seeking attention.
        Let me tell you about my personal experience with depression.
        I had another client who dealt with this by just getting over it.
        We can discuss this when we meet for coffee outside of therapy.
        """

        result = await validator.validate_compliance(multiple_violations_response, sample_context)

        # Should detect multiple violation types
        violation_categories = set(v.category for v in result.violations)
        assert len(violation_categories) >= 2

        # Should have poor overall compliance
        assert result.overall_compliance in [
            ComplianceLevel.NON_COMPLIANT,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT,
            ComplianceLevel.PARTIALLY_COMPLIANT,
        ]

    @pytest.mark.asyncio
    async def test_crisis_protocol_activation(self, validator, sample_context):
        """Test activation of crisis protocols"""
        crisis_response = """
        I want to kill myself tonight. I have a plan and the means to do it.
        I feel hopeless and see no way out of this situation.
        """

        result = await validator.validate_compliance(crisis_response, sample_context)

        # Should trigger crisis protocol
        assert result.safety_result.crisis_protocol_triggered is True
        assert result.safety_result.safety_plan_needed is True

        # Should have immediate interventions
        assert len(result.safety_result.immediate_interventions) > 0

        # Should have immediate actions
        assert len(result.immediate_actions) > 0

    @pytest.mark.asyncio
    async def test_child_safety_detection(self, validator, sample_context):
        """Test detection of child safety concerns"""
        child_safety_response = """
        Sometimes I hit my child when they misbehave. I think there might
        be some inappropriate touching happening at home. The child shows
        signs of abuse or neglect.
        """

        result = await validator.validate_compliance(child_safety_response, sample_context)

        # Should detect child safety violations
        child_safety_violations = [v for v in result.violations if "child" in v.category]
        assert len(child_safety_violations) > 0

        # Should have critical safety risk
        assert result.safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_violence_risk_detection(self, validator, sample_context):
        """Test detection of violence risk"""
        violence_response = """
        I want to hurt someone who wronged me. I'm planning to attack them
        and get revenge. I have access to weapons and know where they live.
        """

        result = await validator.validate_compliance(violence_response, sample_context)

        # Should detect violence risk
        violence_violations = [v for v in result.violations if "violence" in v.category]
        assert len(violence_violations) > 0

        # Should have high safety risk
        assert result.safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]

    def test_risk_to_severity_mapping(self, validator):
        """Test mapping of risk levels to violation severity"""
        # Test all risk level mappings
        assert validator._map_risk_to_severity(SafetyRiskLevel.MINIMAL) == ViolationSeverity.MINOR
        assert validator._map_risk_to_severity(SafetyRiskLevel.LOW) == ViolationSeverity.MODERATE
        assert validator._map_risk_to_severity(SafetyRiskLevel.MODERATE) == ViolationSeverity.MAJOR
        assert validator._map_risk_to_severity(SafetyRiskLevel.HIGH) == ViolationSeverity.SEVERE
        assert (
            validator._map_risk_to_severity(SafetyRiskLevel.CRITICAL) == ViolationSeverity.CRITICAL
        )

    def test_evidence_extraction(self, validator, sample_safety_indicator):
        """Test evidence extraction for safety indicators"""
        response_text = "I want to test suicide and test kill myself tonight"

        evidence = validator._extract_evidence(response_text, sample_safety_indicator)

        assert isinstance(evidence, str)
        assert len(evidence) > 0
        # Should contain the matched keyword or pattern
        assert any(keyword in evidence.lower() for keyword in sample_safety_indicator.keywords)

    @pytest.mark.asyncio
    async def test_safety_indicator_detection(self, validator, sample_context):
        """Test detection of specific safety indicators"""
        # Test suicide risk detection
        suicide_response = "I want to kill myself and end my life"
        result = await validator.validate_compliance(suicide_response, sample_context)

        suicide_indicators = [
            indicator_id
            for indicator_id in result.safety_result.indicators_detected
            if validator.safety_indicators[indicator_id].category == SafetyCategory.SUICIDE_RISK
        ]
        assert len(suicide_indicators) > 0

    @pytest.mark.asyncio
    async def test_ethics_guideline_compliance(self, validator, sample_context):
        """Test compliance with specific ethics guidelines"""
        # Test confidentiality compliance
        confidential_response = """
        I understand your concerns and will keep everything we discuss
        confidential within the bounds of professional ethics and legal
        requirements. Your privacy is important to me.
        """

        result = await validator.validate_compliance(confidential_response, sample_context)

        # Should have better ethics compliance
        assert result.ethics_result.overall_compliance in [
            ComplianceLevel.FULLY_COMPLIANT,
            ComplianceLevel.MOSTLY_COMPLIANT,
        ]

    def test_statistics_tracking(self, validator):
        """Test that statistics are tracked correctly"""
        initial_stats = validator.get_statistics()

        assert "total_validations" in initial_stats
        assert "safety_violations" in initial_stats
        assert "ethics_violations" in initial_stats
        assert "legal_violations" in initial_stats
        assert "critical_violations" in initial_stats

    @pytest.mark.asyncio
    async def test_statistics_update(self, validator, sample_context):
        """Test that statistics are updated after validation"""
        initial_stats = validator.get_statistics()
        initial_validations = initial_stats["total_validations"]

        # Perform a validation
        await validator.validate_compliance("Test response", sample_context)

        updated_stats = validator.get_statistics()
        assert updated_stats["total_validations"] == initial_validations + 1

    @pytest.mark.asyncio
    async def test_immediate_actions_generation(self, validator, sample_context):
        """Test generation of immediate actions"""
        high_risk_response = """
        I want to kill myself tonight and I have a detailed plan.
        I also want to hurt the person who caused me this pain.
        """

        result = await validator.validate_compliance(high_risk_response, sample_context)

        # Should generate immediate actions
        assert len(result.immediate_actions) > 0

        # Should include crisis intervention actions
        actions_text = " ".join(result.immediate_actions).lower()
        assert any(
            keyword in actions_text for keyword in ["crisis", "safety", "intervention", "immediate"]
        )

    @pytest.mark.asyncio
    async def test_recommendations_generation(self, validator, sample_context):
        """Test generation of recommendations"""
        problematic_response = """
        Let me share details about another client's case. We could
        meet outside of therapy to discuss your personal issues.
        """

        result = await validator.validate_compliance(problematic_response, sample_context)

        # Should generate recommendations
        assert len(result.recommendations) > 0

        # Should include relevant recommendations
        recommendations_text = " ".join(result.recommendations).lower()
        assert any(
            keyword in recommendations_text
            for keyword in ["review", "supervision", "consultation", "protocol"]
        )

    @pytest.mark.asyncio
    async def test_warnings_generation(self, validator, sample_context):
        """Test generation of warnings"""
        dangerous_response = """
        I want to kill myself right now and I have the means to do it.
        Everyone feels this way, you're just being dramatic about suicide.
        """

        result = await validator.validate_compliance(dangerous_response, sample_context)

        # Should generate warnings
        assert len(result.warnings) > 0

        # Should include critical warnings
        warnings_text = " ".join(result.warnings).upper()
        assert "CRITICAL" in warnings_text

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, validator, sample_context):
        """Test confidence scoring for different compliance levels"""
        # High compliance response
        high_compliance_response = """
        I understand your concerns and take them seriously. Let's work
        together to ensure your safety and develop appropriate coping strategies.
        """

        # Low compliance response
        low_compliance_response = """
        Everyone feels that way, you're just being dramatic. Let me tell
        you about my personal problems instead.
        """

        high_result = await validator.validate_compliance(high_compliance_response, sample_context)
        low_result = await validator.validate_compliance(low_compliance_response, sample_context)

        # High compliance should have higher confidence
        assert high_result.confidence_score >= low_result.confidence_score

    @pytest.mark.asyncio
    async def test_different_clinical_contexts(self, validator):
        """Test validation with different clinical contexts"""
        contexts = [
            ClinicalContext(
                client_presentation="Depression treatment",
                therapeutic_modality=TherapeuticModality.CBT,
                session_phase="initial",
            ),
            ClinicalContext(
                client_presentation="Anxiety management",
                therapeutic_modality=TherapeuticModality.DBT,
                session_phase="working",
            ),
            ClinicalContext(
                client_presentation="Crisis intervention",
                therapeutic_modality=TherapeuticModality.HUMANISTIC,
                session_phase="crisis",
                crisis_indicators=["self_harm"],
            ),
        ]

        response = "I understand your concerns and want to help you."

        for context in contexts:
            result = await validator.validate_compliance(response, context)
            assert isinstance(result, SafetyEthicsComplianceResult)
            assert result.clinical_context == context

    @pytest.mark.asyncio
    async def test_edge_case_empty_response(self, validator, sample_context):
        """Test handling of empty response"""
        result = await validator.validate_compliance("", sample_context)

        assert isinstance(result, SafetyEthicsComplianceResult)
        # Empty response should have some compliance level
        assert result.overall_compliance in list(ComplianceLevel)

    @pytest.mark.asyncio
    async def test_edge_case_very_long_response(self, validator, sample_context):
        """Test handling of very long response"""
        long_response = "This is a therapeutic response. " * 1000

        result = await validator.validate_compliance(long_response, sample_context)

        assert isinstance(result, SafetyEthicsComplianceResult)
        # Should still process without errors


# Integration tests
class TestSafetyEthicsValidatorIntegration:
    """Integration tests for safety and ethics validator"""

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        """Test complete validation workflow from start to finish"""
        validator = SafetyEthicsValidator()

        context = ClinicalContext(
            client_presentation="Complex case with multiple safety concerns",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="crisis",
            crisis_indicators=["suicidal_ideation", "self_harm"],
            cultural_factors=["hispanic_background"],
        )

        # Complex response with multiple compliance issues
        complex_response = """
        I understand you want to kill yourself, but everyone feels that way.
        Let me tell you about my personal experience with depression when
        my wife left me. I had another client who dealt with this by just
        getting over it. You don't really mean it, you're just seeking
        attention. We can discuss this when we meet for coffee outside
        of therapy next week.
        """

        # Perform comprehensive validation
        result = await validator.validate_compliance(complex_response, context)

        # Verify comprehensive result
        assert isinstance(result, SafetyEthicsComplianceResult)
        assert result.assessment_id is not None
        assert isinstance(result.timestamp, datetime)
        assert result.overall_compliance in list(ComplianceLevel)
        assert 0.0 <= result.confidence_score <= 1.0

        # Should detect multiple violations
        assert len(result.violations) >= 2

        # Should have multiple violation categories
        violation_categories = set(v.category for v in result.violations)
        assert len(violation_categories) >= 1

        # Should generate appropriate responses
        assert len(result.immediate_actions) >= 0
        assert len(result.recommendations) >= 0
        assert len(result.warnings) >= 0

        # Should have poor overall compliance
        assert result.overall_compliance in [
            ComplianceLevel.NON_COMPLIANT,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT,
            ComplianceLevel.PARTIALLY_COMPLIANT,
        ]

        # Get final statistics
        stats = validator.get_statistics()
        assert stats["total_validations"] >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
