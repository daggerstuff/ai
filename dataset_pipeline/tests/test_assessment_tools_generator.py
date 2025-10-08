"""
Tests for Assessment Tools and Diagnostic Conversation Templates System

This module provides comprehensive tests for the assessment tools generator,
covering clinical interview frameworks, diagnostic conversations, and export functionality.
"""

import json
import os
import tempfile

import pytest

from .assessment_tools_generator import (
    AssessmentDomain,
    AssessmentToolsGenerator,
    AssessmentType,
    DiagnosticConversation,
)
from .client_scenario_generator import (
    ClientScenario,
    ClinicalFormulation,
    ScenarioType,
    SeverityLevel,
)


class TestAssessmentToolsGenerator:
    """Test suite for AssessmentToolsGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return AssessmentToolsGenerator()

    @pytest.fixture
    def sample_client_scenario(self):
        """Create sample client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder", "Generalized Anxiety Disorder"],
            attachment_style="secure",
            defense_mechanisms=["intellectualization"],
            psychodynamic_themes=["abandonment fears"]
        )

        return ClientScenario(
            id="test_assessment_001",
            scenario_type=ScenarioType.DIAGNOSTIC_INTERVIEW,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="Depression and anxiety affecting work and relationships",
            clinical_formulation=clinical_formulation,
            demographics={"age": 28, "gender": "female"},
            session_context="Initial diagnostic assessment"
        )

    @pytest.fixture
    def severe_client_scenario(self):
        """Create severe client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder with suicidal ideation"],
            attachment_style="anxious",
            defense_mechanisms=["denial"],
            psychodynamic_themes=["hopelessness"]
        )

        return ClientScenario(
            id="severe_assessment_001",
            scenario_type=ScenarioType.DIAGNOSTIC_INTERVIEW,
            severity_level=SeverityLevel.SEVERE,
            presenting_problem="Severe depression with suicidal thoughts",
            clinical_formulation=clinical_formulation,
            demographics={"age": 35, "gender": "male"},
            session_context="Crisis assessment"
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "assessment_tools")
        assert hasattr(generator, "assessment_protocols")
        assert hasattr(generator, "question_banks")

        # Check that tools are loaded
        assert len(generator.assessment_tools) > 0
        assert AssessmentType.MENTAL_STATUS_EXAM in generator.assessment_tools
        assert AssessmentType.DIAGNOSTIC_INTERVIEW in generator.assessment_tools
        assert AssessmentType.RISK_ASSESSMENT in generator.assessment_tools

    def test_assessment_tool_structure(self, generator):
        """Test assessment tool data structure."""
        mse_tools = generator.assessment_tools[AssessmentType.MENTAL_STATUS_EXAM]
        assert len(mse_tools) > 0

        tool = mse_tools[0]
        assert hasattr(tool, "id")
        assert hasattr(tool, "name")
        assert hasattr(tool, "assessment_type")
        assert hasattr(tool, "domains_covered")
        assert hasattr(tool, "target_population")
        assert hasattr(tool, "administration_time")
        assert hasattr(tool, "required_training")
        assert hasattr(tool, "validity_reliability")
        assert hasattr(tool, "scoring_method")
        assert hasattr(tool, "interpretation_guidelines")
        assert hasattr(tool, "clinical_cutoffs")
        assert hasattr(tool, "contraindications")
        assert hasattr(tool, "cultural_considerations")

        # Validate data types
        assert isinstance(tool.domains_covered, list)
        assert isinstance(tool.target_population, list)
        assert isinstance(tool.administration_time, tuple)
        assert len(tool.administration_time) == 2
        assert isinstance(tool.validity_reliability, dict)
        assert isinstance(tool.interpretation_guidelines, list)
        assert isinstance(tool.contraindications, list)
        assert isinstance(tool.cultural_considerations, list)

    def test_assessment_protocols_initialization(self, generator):
        """Test assessment protocols initialization."""
        protocols = generator.assessment_protocols
        assert len(protocols) > 0

        # Check protocol structure
        protocol = protocols[0]
        assert hasattr(protocol, "id")
        assert hasattr(protocol, "name")
        assert hasattr(protocol, "assessment_type")
        assert hasattr(protocol, "target_conditions")
        assert hasattr(protocol, "protocol_steps")
        assert hasattr(protocol, "assessment_questions")
        assert hasattr(protocol, "decision_points")
        assert hasattr(protocol, "outcome_criteria")
        assert hasattr(protocol, "documentation_requirements")
        assert hasattr(protocol, "follow_up_recommendations")

        # Validate data types
        assert isinstance(protocol.target_conditions, list)
        assert isinstance(protocol.protocol_steps, list)
        assert isinstance(protocol.assessment_questions, list)
        assert isinstance(protocol.decision_points, list)
        assert isinstance(protocol.outcome_criteria, dict)
        assert isinstance(protocol.documentation_requirements, list)
        assert isinstance(protocol.follow_up_recommendations, list)

    def test_question_banks_initialization(self, generator):
        """Test question banks initialization."""
        question_banks = generator.question_banks
        assert len(question_banks) > 0

        # Check that key domains are covered
        assert AssessmentDomain.PRESENTING_PROBLEM in question_banks
        assert AssessmentDomain.MENTAL_STATUS in question_banks
        assert AssessmentDomain.RISK_FACTORS in question_banks

        # Check question structure
        presenting_questions = question_banks[AssessmentDomain.PRESENTING_PROBLEM]
        assert len(presenting_questions) > 0

        question = presenting_questions[0]
        assert hasattr(question, "id")
        assert hasattr(question, "domain")
        assert hasattr(question, "technique")
        assert hasattr(question, "question_text")
        assert hasattr(question, "follow_up_questions")
        assert hasattr(question, "clinical_rationale")
        assert hasattr(question, "expected_responses")
        assert hasattr(question, "red_flag_indicators")
        assert hasattr(question, "scoring_criteria")

        # Validate data types
        assert isinstance(question.follow_up_questions, list)
        assert isinstance(question.expected_responses, list)
        assert isinstance(question.red_flag_indicators, list)

    def test_generate_diagnostic_conversation(self, generator, sample_client_scenario):
        """Test diagnostic conversation generation."""
        conversation = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.DIAGNOSTIC_INTERVIEW, num_exchanges=10
        )

        assert isinstance(conversation, DiagnosticConversation)
        assert conversation.id is not None
        assert conversation.assessment_protocol is not None
        assert conversation.client_scenario == sample_client_scenario
        assert len(conversation.conversation_exchanges) == 10
        assert isinstance(conversation.domains_assessed, list)
        assert isinstance(conversation.techniques_used, list)
        assert isinstance(conversation.clinical_observations, list)
        assert isinstance(conversation.assessment_findings, dict)
        assert isinstance(conversation.diagnostic_impressions, list)
        assert isinstance(conversation.recommendations, list)
        assert conversation.risk_level in ["low", "moderate", "high"]
        assert conversation.follow_up_plan is not None

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 10

        # Should alternate between therapist and client
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "assessment_domain" in exchange
                assert "interview_technique" in exchange
                assert "clinical_purpose" in exchange
            else:
                assert exchange["speaker"] == "client"
                assert "emotional_state" in exchange
                assert "cooperation_level" in exchange
                assert "information_quality" in exchange

    def test_depression_assessment_conversation(self, generator, sample_client_scenario):
        """Test depression-specific assessment conversation."""
        conversation = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.DIAGNOSTIC_INTERVIEW
        )

        # Should include depression-related diagnostic impressions
        impressions_text = " ".join(conversation.diagnostic_impressions).lower()
        assert "depression" in impressions_text or "depressive" in impressions_text

        # Should include appropriate recommendations
        recommendations_text = " ".join(conversation.recommendations).lower()
        assert "cbt" in recommendations_text or "cognitive" in recommendations_text

        # Should assess relevant domains
        domain_values = [domain.value for domain in conversation.domains_assessed]
        assert any("presenting" in domain for domain in domain_values)

    def test_severe_case_assessment(self, generator, severe_client_scenario):
        """Test assessment of severe case with risk factors."""
        conversation = generator.generate_diagnostic_conversation(
            severe_client_scenario, AssessmentType.RISK_ASSESSMENT
        )

        # Should identify high risk level
        assert conversation.risk_level == "high"

        # Should include safety planning in recommendations
        recommendations_text = " ".join(conversation.recommendations).lower()
        assert "safety" in recommendations_text

        # Should include risk factors in assessment findings
        risk_factors = conversation.assessment_findings.get("risk_factors", [])
        assert len(risk_factors) > 0
        assert any("suicide" in factor.lower() for factor in risk_factors)

    def test_mental_status_exam_conversation(self, generator, sample_client_scenario):
        """Test mental status examination conversation."""
        conversation = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.MENTAL_STATUS_EXAM
        )

        # Should assess mental status domain
        domain_values = [domain.value for domain in conversation.domains_assessed]
        assert "mental_status" in domain_values

        # Should include cognitive assessment techniques
        technique_values = [technique.value for technique in conversation.techniques_used]
        assert any("structured" in technique for technique in technique_values)

    def test_client_response_generation(self, generator, sample_client_scenario):
        """Test client response generation based on scenario."""
        # Test depression response
        response = generator._generate_client_response_content(sample_client_scenario, 0)
        assert isinstance(response, str)
        assert len(response) > 0

        # Should be appropriate for depression scenario
        response_lower = response.lower()
        assert any(word in response_lower for word in ["down", "sad", "depressed", "joy", "feeling"])

    def test_clinical_observations_generation(self, generator, sample_client_scenario):
        """Test clinical observations generation."""
        # Mock exchanges
        exchanges = [
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "test", "emotional_state": "sad", "cooperation_level": "high"},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "test", "emotional_state": "sad", "cooperation_level": "high"}
        ]

        observations = generator._generate_clinical_observations(exchanges, sample_client_scenario)

        assert isinstance(observations, list)
        assert len(observations) > 0

        # Should include emotional state observation
        observations_text = " ".join(observations).lower()
        assert "sad" in observations_text or "affect" in observations_text

    def test_assessment_findings_generation(self, generator, sample_client_scenario):
        """Test assessment findings generation."""
        # Mock exchanges and protocol
        exchanges = []
        protocol = generator.assessment_protocols[0]

        findings = generator._generate_assessment_findings(exchanges, protocol, sample_client_scenario)

        assert isinstance(findings, dict)
        assert "presenting_symptoms" in findings
        assert "symptom_duration" in findings
        assert "functional_impairment" in findings
        assert "risk_factors" in findings
        assert "protective_factors" in findings

        # Should include DSM-5 considerations
        assert findings["presenting_symptoms"] == sample_client_scenario.clinical_formulation.dsm5_considerations

    def test_diagnostic_impressions_generation(self, generator, sample_client_scenario):
        """Test diagnostic impressions generation."""
        findings = {"presenting_symptoms": ["Major Depressive Disorder"]}

        impressions = generator._generate_diagnostic_impressions(findings, sample_client_scenario)

        assert isinstance(impressions, list)
        assert len(impressions) > 0

        # Should include depression diagnosis
        impressions_text = " ".join(impressions).lower()
        assert "depression" in impressions_text or "depressive" in impressions_text

    def test_recommendations_generation(self, generator, sample_client_scenario):
        """Test treatment recommendations generation."""
        findings = {"risk_factors": ["Depressive symptoms"]}

        recommendations = generator._generate_recommendations(findings, sample_client_scenario)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should include evidence-based treatments
        recommendations_text = " ".join(recommendations).lower()
        assert "cbt" in recommendations_text or "therapy" in recommendations_text

    def test_risk_level_assessment(self, generator, severe_client_scenario):
        """Test risk level assessment."""
        findings = {"risk_factors": ["Potential suicide risk", "High symptom severity"]}

        risk_level = generator._assess_risk_level(findings, severe_client_scenario)

        assert risk_level == "high"

    def test_export_diagnostic_conversations(self, generator, sample_client_scenario):
        """Test conversation export functionality."""
        conversations = []

        # Generate multiple conversations
        for _i in range(3):
            conv = generator.generate_diagnostic_conversation(
                sample_client_scenario, AssessmentType.DIAGNOSTIC_INTERVIEW, num_exchanges=6
            )
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = generator.export_diagnostic_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 3
            assert export_result["output_file"] == temp_file
            assert "assessment_types_covered" in export_result
            assert "domains_covered" in export_result

            # Check exported file
            assert os.path.exists(temp_file)
            with open(temp_file, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "metadata" in exported_data
            assert "conversations" in exported_data
            assert exported_data["metadata"]["total_conversations"] == 3
            assert len(exported_data["conversations"]) == 3

            # Check metadata
            metadata = exported_data["metadata"]
            assert "assessment_types" in metadata
            assert "domains_covered" in metadata
            assert "average_exchanges" in metadata
            assert isinstance(metadata["assessment_types"], list)
            assert isinstance(metadata["domains_covered"], list)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_enum_conversion_for_export(self, generator):
        """Test enum conversion for JSON serialization."""
        # Test conversion
        test_dict = {
            "assessment_type": AssessmentType.DIAGNOSTIC_INTERVIEW,
            "domain": AssessmentDomain.PRESENTING_PROBLEM
        }
        generator._convert_enums_to_strings(test_dict)

        assert test_dict["assessment_type"] == "diagnostic_interview"
        assert test_dict["domain"] == "presenting_problem"

    def test_different_assessment_types(self, generator, sample_client_scenario):
        """Test different assessment types."""
        # Test diagnostic interview
        diagnostic_conv = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.DIAGNOSTIC_INTERVIEW
        )
        assert diagnostic_conv.assessment_protocol.assessment_type == AssessmentType.DIAGNOSTIC_INTERVIEW

        # Test mental status exam
        mse_conv = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.MENTAL_STATUS_EXAM
        )
        assert mse_conv.assessment_protocol.assessment_type == AssessmentType.MENTAL_STATUS_EXAM

        # Test risk assessment
        risk_conv = generator.generate_diagnostic_conversation(
            sample_client_scenario, AssessmentType.RISK_ASSESSMENT
        )
        assert risk_conv.assessment_protocol.assessment_type == AssessmentType.RISK_ASSESSMENT


if __name__ == "__main__":
    pytest.main([__file__])
