"""
Unit tests for Clinical Accuracy Assessment Framework

This module provides comprehensive unit tests for the clinical accuracy
validator, covering all assessment components and edge cases.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from .clinical_accuracy_validator import (
    ClinicalAccuracyLevel,
    ClinicalAccuracyResult,
    ClinicalAccuracyValidator,
    ClinicalContext,
    DSM5Assessment,
    PDM2Assessment,
    SafetyAssessment,
    SafetyRiskLevel,
    TherapeuticAppropriatenessScore,
    TherapeuticModality,
)


class TestClinicalAccuracyValidator:
    """Test suite for ClinicalAccuracyValidator"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing"""
        return ClinicalAccuracyValidator()

    @pytest.fixture
    def sample_context(self):
        """Create a sample clinical context"""
        return ClinicalContext(
            client_presentation="Client reports depression and anxiety symptoms",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
            crisis_indicators=["sleep_disturbance"],
            cultural_factors=["hispanic_background"],
            contraindications=[],
        )

    @pytest.fixture
    def sample_response(self):
        """Create a sample clinical response"""
        return """
        I understand you're experiencing depression and anxiety. Let's explore 
        your symptoms and develop coping strategies. Are you having any thoughts 
        of self-harm? We'll use cognitive-behavioral techniques to help you.
        """

    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert validator.config is not None
        assert validator.dsm5_criteria is not None
        assert validator.pdm2_frameworks is not None
        assert validator.safety_protocols is not None
        assert validator.therapeutic_guidelines is not None
        assert len(validator.accuracy_thresholds) == 5

    def test_config_loading_default(self):
        """Test default configuration loading"""
        validator = ClinicalAccuracyValidator()

        assert validator.config["dsm5_weight"] == 0.3
        assert validator.config["pdm2_weight"] == 0.2
        assert validator.config["therapeutic_weight"] == 0.3
        assert validator.config["safety_weight"] == 0.2
        assert validator.config["expert_validation_threshold"] == 0.7

    def test_config_loading_custom(self):
        """Test custom configuration loading"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            custom_config = {"dsm5_weight": 0.4, "safety_weight": 0.3}
            json.dump(custom_config, f)
            config_path = Path(f.name)

        try:
            validator = ClinicalAccuracyValidator(config_path)
            assert validator.config["dsm5_weight"] == 0.4
            assert validator.config["safety_weight"] == 0.3
            assert validator.config["pdm2_weight"] == 0.2  # Default value
        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_assess_clinical_accuracy_basic(self, validator, sample_context, sample_response):
        """Test basic clinical accuracy assessment"""
        result = await validator.assess_clinical_accuracy(sample_response, sample_context)

        assert isinstance(result, ClinicalAccuracyResult)
        assert result.assessment_id.startswith("ca_")
        assert isinstance(result.timestamp, datetime)
        assert result.clinical_context == sample_context
        assert isinstance(result.dsm5_assessment, DSM5Assessment)
        assert isinstance(result.pdm2_assessment, PDM2Assessment)
        assert isinstance(result.therapeutic_appropriateness, TherapeuticAppropriatenessScore)
        assert isinstance(result.safety_assessment, SafetyAssessment)
        assert isinstance(result.overall_accuracy, ClinicalAccuracyLevel)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.expert_validation_needed, bool)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)

    @pytest.mark.asyncio
    async def test_dsm5_assessment_depression(self, validator, sample_context):
        """Test DSM-5 assessment for depression indicators"""
        response = """
        You're experiencing depressed mood, loss of interest, sleep problems,
        fatigue, worthlessness, and concentration difficulties for over two weeks.
        """

        dsm5_result = await validator._assess_dsm5_compliance(response, sample_context)

        assert dsm5_result.primary_diagnosis == "Major Depressive Disorder"
        assert dsm5_result.diagnostic_confidence > 0.0
        assert len(dsm5_result.criteria_met) > 0
        assert "depressed" in dsm5_result.criteria_met

    @pytest.mark.asyncio
    async def test_dsm5_assessment_anxiety(self, validator, sample_context):
        """Test DSM-5 assessment for anxiety indicators"""
        response = """
        You're experiencing excessive worry, anxiety, restlessness, and 
        difficulty concentrating. These symptoms are affecting your daily life.
        """

        dsm5_result = await validator._assess_dsm5_compliance(response, sample_context)

        assert dsm5_result.primary_diagnosis == "Generalized Anxiety Disorder"
        assert dsm5_result.diagnostic_confidence > 0.0

    @pytest.mark.asyncio
    async def test_dsm5_assessment_comorbid(self, validator, sample_context):
        """Test DSM-5 assessment for comorbid conditions"""
        response = """
        You're experiencing depression with sadness, hopelessness, and sleep issues,
        along with anxiety, worry, and restlessness affecting your functioning.
        """

        dsm5_result = await validator._assess_dsm5_compliance(response, sample_context)

        assert dsm5_result.primary_diagnosis is not None
        assert (
            len(dsm5_result.secondary_diagnoses) > 0
            or "anxiety" in dsm5_result.primary_diagnosis.lower()
        )

    @pytest.mark.asyncio
    async def test_pdm2_assessment(self, validator, sample_context):
        """Test PDM-2 psychodynamic assessment"""
        response = """
        You show patterns of self-criticism, feelings of inadequacy, and 
        grandiose expectations that aren't met, leading to disappointment.
        """

        pdm2_result = await validator._assess_pdm2_compliance(response, sample_context)

        assert isinstance(pdm2_result.personality_patterns, list)
        assert isinstance(pdm2_result.mental_functioning, dict)
        assert len(pdm2_result.mental_functioning) > 0
        assert all(0.0 <= score <= 1.0 for score in pdm2_result.mental_functioning.values())

    @pytest.mark.asyncio
    async def test_therapeutic_appropriateness_initial_phase(self, validator):
        """Test therapeutic appropriateness for initial phase"""
        context = ClinicalContext(
            client_presentation="New client seeking help",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        response = """
        Welcome, let's start by building rapport and understanding your concerns.
        I'll assess your situation and provide some psychoeducation about your symptoms.
        """

        therapeutic_result = await validator._assess_therapeutic_appropriateness(response, context)

        assert therapeutic_result.intervention_appropriateness > 0.0
        assert therapeutic_result.timing_appropriateness > 0.0
        assert therapeutic_result.overall_score > 0.0
        assert len(therapeutic_result.rationale) > 0

    @pytest.mark.asyncio
    async def test_therapeutic_appropriateness_cultural_sensitivity(
        self, validator, sample_context
    ):
        """Test therapeutic appropriateness with cultural considerations"""
        response = """
        I want to understand your cultural background and how your family 
        and beliefs might influence your experience. Let's respect your values.
        """

        therapeutic_result = await validator._assess_therapeutic_appropriateness(
            response, sample_context
        )

        assert therapeutic_result.cultural_sensitivity > 0.5

    @pytest.mark.asyncio
    async def test_therapeutic_appropriateness_boundary_violations(self, validator, sample_context):
        """Test detection of therapeutic boundary violations"""
        response = """
        You remind me of my personal friend. Let's meet outside the office 
        for coffee to discuss your relationship problems.
        """

        therapeutic_result = await validator._assess_therapeutic_appropriateness(
            response, sample_context
        )

        assert therapeutic_result.boundary_maintenance < 0.5
        assert therapeutic_result.overall_score < 0.7

    @pytest.mark.asyncio
    async def test_safety_assessment_low_risk(self, validator, sample_context):
        """Test safety assessment for low-risk scenario"""
        response = """
        I understand you're feeling sad and overwhelmed. Let's work together 
        to develop healthy coping strategies and build your support system.
        """

        safety_result = await validator._assess_safety_compliance(response, sample_context)

        assert safety_result.suicide_risk == SafetyRiskLevel.MINIMAL
        assert safety_result.overall_risk == SafetyRiskLevel.MINIMAL
        assert not safety_result.safety_plan_needed
        assert len(safety_result.immediate_interventions) == 0

    @pytest.mark.asyncio
    async def test_safety_assessment_high_risk(self, validator, sample_context):
        """Test safety assessment for high-risk scenario"""
        response = """
        I feel hopeless and want to die. I think about suicide every day and 
        have a plan to end it all. I can't see any reason to keep living.
        """

        safety_result = await validator._assess_safety_compliance(response, sample_context)

        assert safety_result.suicide_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.MODERATE]
        assert safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.MODERATE]
        assert len(safety_result.immediate_interventions) > 0

    @pytest.mark.asyncio
    async def test_safety_assessment_self_harm(self, validator, sample_context):
        """Test safety assessment for self-harm indicators"""
        response = """
        Sometimes I cut myself when I feel overwhelmed. It helps me feel 
        something other than emotional pain. I hurt myself to punish myself.
        """

        safety_result = await validator._assess_safety_compliance(response, sample_context)

        assert safety_result.self_harm_risk in [SafetyRiskLevel.MODERATE, SafetyRiskLevel.LOW]

    def test_calculate_overall_accuracy_excellent(self, validator):
        """Test overall accuracy calculation for excellent performance"""
        dsm5 = DSM5Assessment(diagnostic_confidence=0.95)
        pdm2 = PDM2Assessment(mental_functioning={"domain1": 0.9, "domain2": 0.95})
        therapeutic = TherapeuticAppropriatenessScore(overall_score=0.92)
        safety = SafetyAssessment(overall_risk=SafetyRiskLevel.MINIMAL)

        accuracy, confidence = validator._calculate_overall_accuracy(
            dsm5, pdm2, therapeutic, safety
        )

        assert accuracy == ClinicalAccuracyLevel.EXCELLENT
        assert confidence >= 0.9

    def test_calculate_overall_accuracy_concerning(self, validator):
        """Test overall accuracy calculation for concerning performance"""
        dsm5 = DSM5Assessment(diagnostic_confidence=0.3)
        pdm2 = PDM2Assessment(mental_functioning={"domain1": 0.4, "domain2": 0.3})
        therapeutic = TherapeuticAppropriatenessScore(overall_score=0.4)
        safety = SafetyAssessment(overall_risk=SafetyRiskLevel.HIGH)

        accuracy, confidence = validator._calculate_overall_accuracy(
            dsm5, pdm2, therapeutic, safety
        )

        assert accuracy in [ClinicalAccuracyLevel.CONCERNING, ClinicalAccuracyLevel.DANGEROUS]
        assert confidence < 0.7

    def test_generate_recommendations_low_confidence(self, validator):
        """Test recommendation generation for low confidence assessment"""
        dsm5 = DSM5Assessment(diagnostic_confidence=0.5)
        pdm2 = PDM2Assessment()
        therapeutic = TherapeuticAppropriatenessScore(overall_score=0.6, cultural_sensitivity=0.4)
        safety = SafetyAssessment(overall_risk=SafetyRiskLevel.MODERATE, safety_plan_needed=True)

        recommendations = validator._generate_recommendations(dsm5, pdm2, therapeutic, safety)

        assert len(recommendations) > 0
        assert any("diagnostic assessment" in rec for rec in recommendations)
        assert any("cultural sensitivity" in rec for rec in recommendations)
        assert any("safety" in rec for rec in recommendations)

    def test_generate_warnings_high_risk(self, validator):
        """Test warning generation for high-risk scenarios"""
        safety = SafetyAssessment(overall_risk=SafetyRiskLevel.HIGH)
        accuracy = ClinicalAccuracyLevel.DANGEROUS

        warnings = validator._generate_warnings(safety, accuracy)

        assert len(warnings) > 0
        assert any("HIGH RISK" in warning for warning in warnings)
        assert any("DANGEROUS" in warning for warning in warnings)

    def test_generate_warnings_concerning(self, validator):
        """Test warning generation for concerning accuracy"""
        safety = SafetyAssessment(overall_risk=SafetyRiskLevel.LOW)
        accuracy = ClinicalAccuracyLevel.CONCERNING

        warnings = validator._generate_warnings(safety, accuracy)

        assert len(warnings) > 0
        assert any("CONCERNING" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_expert_validation_needed_low_confidence(self, validator, sample_context):
        """Test expert validation requirement for low confidence"""
        # Create a response that would result in low confidence
        response = "I don't know what to say about your situation."

        result = await validator.assess_clinical_accuracy(response, sample_context)

        # Should require expert validation due to low confidence
        assert result.expert_validation_needed or result.confidence_score < 0.7

    @pytest.mark.asyncio
    async def test_expert_validation_needed_high_risk(self, validator, sample_context):
        """Test expert validation requirement for high-risk scenarios"""
        response = """
        I want to kill myself and have a detailed plan. I feel hopeless 
        and see no way out. I'm going to end my life tonight.
        """

        result = await validator.assess_clinical_accuracy(response, sample_context)

        # Should require expert validation due to high safety risk
        assert result.expert_validation_needed

    def test_export_assessment(self, validator, sample_context):
        """Test assessment export functionality"""
        # Create a sample result
        result = ClinicalAccuracyResult(
            assessment_id="test_001",
            timestamp=datetime.now(),
            clinical_context=sample_context,
            dsm5_assessment=DSM5Assessment(primary_diagnosis="Test Disorder"),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.8),
            safety_assessment=SafetyAssessment(),
            overall_accuracy=ClinicalAccuracyLevel.GOOD,
            confidence_score=0.85,
            expert_validation_needed=False,
            recommendations=["Test recommendation"],
            warnings=[],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            validator.export_assessment(result, output_path)

            # Verify file was created and contains expected data
            assert output_path.exists()

            with open(output_path, "r") as f:
                exported_data = json.load(f)

            assert exported_data["assessment_id"] == "test_001"
            assert exported_data["overall_accuracy"] == "good"
            assert exported_data["confidence_score"] == 0.85
            assert exported_data["expert_validation_needed"] is False
            assert len(exported_data["recommendations"]) == 1

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_dsm5_criteria_loading(self, validator):
        """Test DSM-5 criteria loading"""
        assert "major_depressive_disorder" in validator.dsm5_criteria
        assert "generalized_anxiety_disorder" in validator.dsm5_criteria

        mdd_criteria = validator.dsm5_criteria["major_depressive_disorder"]
        assert "criteria" in mdd_criteria
        assert "minimum_criteria" in mdd_criteria
        assert "duration_requirement" in mdd_criteria
        assert len(mdd_criteria["criteria"]) > 0

    def test_pdm2_frameworks_loading(self, validator):
        """Test PDM-2 frameworks loading"""
        assert "personality_patterns" in validator.pdm2_frameworks
        assert "mental_functioning_domains" in validator.pdm2_frameworks
        assert len(validator.pdm2_frameworks["personality_patterns"]) > 0
        assert len(validator.pdm2_frameworks["mental_functioning_domains"]) > 0

    def test_safety_protocols_loading(self, validator):
        """Test safety protocols loading"""
        assert "suicide_risk_factors" in validator.safety_protocols
        assert "crisis_interventions" in validator.safety_protocols
        assert len(validator.safety_protocols["suicide_risk_factors"]) > 0
        assert len(validator.safety_protocols["crisis_interventions"]) > 0

    def test_therapeutic_guidelines_loading(self, validator):
        """Test therapeutic guidelines loading"""
        assert "intervention_timing" in validator.therapeutic_guidelines
        assert "cultural_considerations" in validator.therapeutic_guidelines

        timing = validator.therapeutic_guidelines["intervention_timing"]
        assert "initial_phase" in timing
        assert "working_phase" in timing
        assert "termination_phase" in timing

    @pytest.mark.asyncio
    async def test_edge_case_empty_response(self, validator, sample_context):
        """Test handling of empty response"""
        result = await validator.assess_clinical_accuracy("", sample_context)

        assert isinstance(result, ClinicalAccuracyResult)
        assert result.confidence_score < 0.5
        assert result.expert_validation_needed

    @pytest.mark.asyncio
    async def test_edge_case_very_long_response(self, validator, sample_context):
        """Test handling of very long response"""
        long_response = "This is a test response. " * 1000

        result = await validator.assess_clinical_accuracy(long_response, sample_context)

        assert isinstance(result, ClinicalAccuracyResult)
        # Should still process without errors

    @pytest.mark.asyncio
    async def test_different_therapeutic_modalities(self, validator):
        """Test assessment with different therapeutic modalities"""
        modalities = [
            TherapeuticModality.CBT,
            TherapeuticModality.DBT,
            TherapeuticModality.PSYCHODYNAMIC,
            TherapeuticModality.HUMANISTIC,
        ]

        for modality in modalities:
            context = ClinicalContext(
                client_presentation="Test presentation",
                therapeutic_modality=modality,
                session_phase="working",
            )

            response = "Appropriate therapeutic response for this modality."
            result = await validator.assess_clinical_accuracy(response, context)

            assert isinstance(result, ClinicalAccuracyResult)
            assert result.clinical_context.therapeutic_modality == modality

    @pytest.mark.asyncio
    async def test_different_session_phases(self, validator):
        """Test assessment across different session phases"""
        phases = ["initial", "working", "termination"]

        for phase in phases:
            context = ClinicalContext(
                client_presentation="Test presentation",
                therapeutic_modality=TherapeuticModality.CBT,
                session_phase=phase,
            )

            response = f"Response appropriate for {phase} phase."
            result = await validator.assess_clinical_accuracy(response, context)

            assert isinstance(result, ClinicalAccuracyResult)
            assert result.clinical_context.session_phase == phase


# Integration tests
class TestClinicalAccuracyIntegration:
    """Integration tests for clinical accuracy assessment"""

    @pytest.mark.asyncio
    async def test_complete_assessment_workflow(self):
        """Test complete assessment workflow from start to finish"""
        validator = ClinicalAccuracyValidator()

        context = ClinicalContext(
            client_presentation="Client with depression and anxiety seeking help",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
            crisis_indicators=["sleep_disturbance", "appetite_changes"],
            cultural_factors=["latino_background", "religious_beliefs"],
            contraindications=[],
        )

        response = """
        Thank you for coming in today. I can see that you're dealing with some 
        difficult feelings of depression and anxiety. Let's start by understanding 
        your symptoms better and ensuring your safety. Are you having any thoughts 
        of harming yourself? I want to respect your cultural background and beliefs 
        as we work together using cognitive-behavioral approaches to help you develop 
        effective coping strategies.
        """

        # Perform assessment
        result = await validator.assess_clinical_accuracy(response, context)

        # Verify comprehensive result
        assert result.assessment_id is not None
        assert result.overall_accuracy in list(ClinicalAccuracyLevel)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)

        # Export and verify
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            validator.export_assessment(result, output_path)
            assert output_path.exists()

            with open(output_path, "r") as f:
                exported_data = json.load(f)

            assert "assessment_id" in exported_data
            assert "overall_accuracy" in exported_data
            assert "confidence_score" in exported_data

        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
