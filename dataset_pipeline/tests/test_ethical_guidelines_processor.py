"""
Tests for Ethical Guidelines and Professional Boundaries Processing System

This module provides comprehensive tests for the ethical guidelines processor,
covering ethical scenario generation, boundary-setting conversations, and export functionality.
"""

import json
import os
import tempfile

import pytest

from .ethical_guidelines_processor import (
    EthicalConversation,
    EthicalDilemmaType,
    EthicalGuidelinesProcessor,
    EthicalPrinciple,
    EthicalScenario,
    EthicalSeverity,
    ProfessionalBoundary,
)


class TestEthicalGuidelinesProcessor:
    """Test suite for EthicalGuidelinesProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor instance for testing."""
        return EthicalGuidelinesProcessor()

    @pytest.fixture
    def sample_ethical_scenario(self):
        """Create sample ethical scenario for testing."""
        return EthicalScenario(
            id="test_confidentiality",
            dilemma_type=EthicalDilemmaType.CONFIDENTIALITY_CONFLICT,
            severity=EthicalSeverity.MODERATE,
            situation_description="Family member requesting client information",
            ethical_principles_involved=[EthicalPrinciple.AUTONOMY, EthicalPrinciple.FIDELITY],
            boundary_concerns=[ProfessionalBoundary.COMMUNICATION_BOUNDARIES],
            stakeholders=["client", "family member", "therapist"],
            potential_consequences=["breach of confidentiality", "family conflict"],
            decision_factors=["client consent", "legal requirements"],
            recommended_actions=["explain confidentiality", "obtain consent"],
            consultation_needed=False,
            documentation_required=True
        )

    @pytest.fixture
    def crisis_ethical_scenario(self):
        """Create crisis ethical scenario for testing."""
        return EthicalScenario(
            id="test_duty_to_warn",
            dilemma_type=EthicalDilemmaType.DUTY_TO_WARN,
            severity=EthicalSeverity.MAJOR,
            situation_description="Client threatens specific person",
            ethical_principles_involved=[EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.FIDELITY],
            boundary_concerns=[ProfessionalBoundary.COMMUNICATION_BOUNDARIES],
            stakeholders=["client", "potential victim", "therapist"],
            potential_consequences=["breach of confidentiality", "legal liability"],
            decision_factors=["imminent danger", "specific threat"],
            recommended_actions=["assess threat", "contact authorities"],
            consultation_needed=True,
            documentation_required=True
        )

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert hasattr(processor, "ethical_guidelines")
        assert hasattr(processor, "boundary_frameworks")
        assert hasattr(processor, "ethical_scenarios")

        # Check that guidelines are loaded
        assert len(processor.ethical_guidelines) > 0
        assert EthicalPrinciple.BENEFICENCE in processor.ethical_guidelines
        assert EthicalPrinciple.NON_MALEFICENCE in processor.ethical_guidelines
        assert EthicalPrinciple.AUTONOMY in processor.ethical_guidelines

    def test_ethical_guideline_structure(self, processor):
        """Test ethical guideline data structure."""
        beneficence_guidelines = processor.ethical_guidelines[EthicalPrinciple.BENEFICENCE]
        assert len(beneficence_guidelines) > 0

        guideline = beneficence_guidelines[0]
        assert hasattr(guideline, "id")
        assert hasattr(guideline, "principle")
        assert hasattr(guideline, "title")
        assert hasattr(guideline, "description")
        assert hasattr(guideline, "applicable_situations")
        assert hasattr(guideline, "implementation_steps")
        assert hasattr(guideline, "warning_signs")
        assert hasattr(guideline, "corrective_actions")
        assert hasattr(guideline, "documentation_requirements")
        assert hasattr(guideline, "supervision_triggers")
        assert hasattr(guideline, "legal_considerations")

        # Validate data types
        assert isinstance(guideline.applicable_situations, list)
        assert isinstance(guideline.implementation_steps, list)
        assert isinstance(guideline.warning_signs, list)
        assert isinstance(guideline.corrective_actions, list)
        assert isinstance(guideline.documentation_requirements, list)
        assert isinstance(guideline.supervision_triggers, list)
        assert isinstance(guideline.legal_considerations, list)

    def test_boundary_frameworks(self, processor):
        """Test boundary frameworks initialization."""
        frameworks = processor.boundary_frameworks

        assert ProfessionalBoundary.DUAL_RELATIONSHIPS in frameworks
        assert ProfessionalBoundary.PHYSICAL_BOUNDARIES in frameworks
        assert ProfessionalBoundary.EMOTIONAL_BOUNDARIES in frameworks

        dual_rel_framework = frameworks[ProfessionalBoundary.DUAL_RELATIONSHIPS]
        assert "definition" in dual_rel_framework
        assert "risk_factors" in dual_rel_framework
        assert "prevention_strategies" in dual_rel_framework
        assert "warning_signs" in dual_rel_framework
        assert "intervention_steps" in dual_rel_framework

        # Validate data types
        assert isinstance(dual_rel_framework["risk_factors"], list)
        assert isinstance(dual_rel_framework["prevention_strategies"], list)
        assert isinstance(dual_rel_framework["warning_signs"], list)
        assert isinstance(dual_rel_framework["intervention_steps"], list)

    def test_ethical_scenarios_initialization(self, processor):
        """Test ethical scenarios initialization."""
        scenarios = processor.ethical_scenarios
        assert len(scenarios) > 0

        # Check for different scenario types
        scenario_types = [scenario.dilemma_type for scenario in scenarios]
        assert EthicalDilemmaType.CONFIDENTIALITY_CONFLICT in scenario_types
        assert EthicalDilemmaType.DUTY_TO_WARN in scenario_types
        assert EthicalDilemmaType.COMPETENCE_LIMITS in scenario_types

        # Validate scenario structure
        scenario = scenarios[0]
        assert hasattr(scenario, "id")
        assert hasattr(scenario, "dilemma_type")
        assert hasattr(scenario, "severity")
        assert hasattr(scenario, "situation_description")
        assert hasattr(scenario, "ethical_principles_involved")
        assert hasattr(scenario, "boundary_concerns")
        assert hasattr(scenario, "stakeholders")
        assert hasattr(scenario, "potential_consequences")
        assert hasattr(scenario, "decision_factors")
        assert hasattr(scenario, "recommended_actions")
        assert hasattr(scenario, "consultation_needed")
        assert hasattr(scenario, "documentation_required")

    def test_generate_ethical_conversation(self, processor, sample_ethical_scenario):
        """Test ethical conversation generation."""
        conversation = processor.generate_ethical_conversation(
            sample_ethical_scenario, num_exchanges=6
        )

        assert isinstance(conversation, EthicalConversation)
        assert conversation.id is not None
        assert conversation.scenario == sample_ethical_scenario
        assert len(conversation.conversation_exchanges) == 6
        assert isinstance(conversation.ethical_principles_demonstrated, list)
        assert isinstance(conversation.boundaries_addressed, list)
        assert isinstance(conversation.teaching_points, list)
        assert isinstance(conversation.follow_up_actions, list)

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 6

        # Should alternate between therapist and other party
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "ethical_principle" in exchange
                assert "boundary_addressed" in exchange
                assert "professional_action" in exchange
            else:
                assert exchange["speaker"] in ["client", "family_member"]
                assert "emotional_state" in exchange
                assert "understanding_level" in exchange

    def test_confidentiality_scenario_conversation(self, processor, sample_ethical_scenario):
        """Test confidentiality scenario conversation generation."""
        conversation = processor.generate_ethical_conversation(sample_ethical_scenario)

        # Should demonstrate autonomy and fidelity principles
        principles_demonstrated = conversation.ethical_principles_demonstrated
        assert EthicalPrinciple.AUTONOMY in principles_demonstrated or EthicalPrinciple.FIDELITY in principles_demonstrated

        # Should address communication boundaries
        assert ProfessionalBoundary.COMMUNICATION_BOUNDARIES in conversation.boundaries_addressed

        # Should include confidentiality teaching points
        teaching_points_text = " ".join(conversation.teaching_points).lower()
        assert "confidentiality" in teaching_points_text

        # Should include documentation in follow-up actions
        follow_up_text = " ".join(conversation.follow_up_actions).lower()
        assert "document" in follow_up_text

    def test_duty_to_warn_scenario_conversation(self, processor, crisis_ethical_scenario):
        """Test duty to warn scenario conversation generation."""
        conversation = processor.generate_ethical_conversation(crisis_ethical_scenario)

        # Should demonstrate non-maleficence principle
        principles_demonstrated = conversation.ethical_principles_demonstrated
        assert EthicalPrinciple.NON_MALEFICENCE in principles_demonstrated or EthicalPrinciple.FIDELITY in principles_demonstrated

        # Should include supervision notes for major severity
        assert conversation.supervision_notes is not None
        assert "duty_to_warn" in conversation.supervision_notes.lower()

        # Should include consultation in follow-up actions
        follow_up_text = " ".join(conversation.follow_up_actions).lower()
        assert "consultation" in follow_up_text or "supervision" in follow_up_text

    def test_ethical_content_generation(self, processor, sample_ethical_scenario):
        """Test ethical content generation for different principles."""
        # Test autonomy principle content
        autonomy_content = processor._generate_ethical_content(
            sample_ethical_scenario, EthicalPrinciple.AUTONOMY, 0
        )
        assert isinstance(autonomy_content, str)
        assert len(autonomy_content) > 0

        # Test non-maleficence principle content
        non_maleficence_content = processor._generate_ethical_content(
            sample_ethical_scenario, EthicalPrinciple.NON_MALEFICENCE, 0
        )
        assert isinstance(non_maleficence_content, str)
        assert len(non_maleficence_content) > 0

        # Content should be different for different principles
        assert autonomy_content != non_maleficence_content

    def test_stakeholder_response_generation(self, processor, sample_ethical_scenario):
        """Test stakeholder response generation."""
        response = processor._generate_stakeholder_response(sample_ethical_scenario, 0)

        assert isinstance(response, str)
        assert len(response) > 0

        # Should be appropriate for confidentiality scenario
        response_lower = response.lower()
        assert any(word in response_lower for word in ["worried", "family", "information", "okay"])

    def test_teaching_points_generation(self, processor, sample_ethical_scenario):
        """Test teaching points generation."""
        # Mock exchanges
        exchanges = [
            {"speaker": "therapist", "ethical_principle": "autonomy"},
            {"speaker": "family_member", "content": "test"}
        ]

        teaching_points = processor._generate_teaching_points(sample_ethical_scenario, exchanges)

        assert isinstance(teaching_points, list)
        assert len(teaching_points) > 0

        # Should include confidentiality-related teaching points
        teaching_text = " ".join(teaching_points).lower()
        assert "confidentiality" in teaching_text

    def test_export_ethical_conversations(self, processor, sample_ethical_scenario):
        """Test conversation export functionality."""
        conversations = []

        # Generate multiple conversations
        for _i in range(3):
            conv = processor.generate_ethical_conversation(sample_ethical_scenario, num_exchanges=4)
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = processor.export_ethical_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 3
            assert export_result["output_file"] == temp_file
            assert "principles_covered" in export_result
            assert "boundaries_addressed" in export_result

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
            assert "ethical_principles_covered" in metadata
            assert "boundary_types_addressed" in metadata
            assert isinstance(metadata["ethical_principles_covered"], list)
            assert isinstance(metadata["boundary_types_addressed"], list)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_enum_conversion_for_export(self, processor, sample_ethical_scenario):
        """Test enum conversion for JSON serialization."""
        processor.generate_ethical_conversation(sample_ethical_scenario)

        # Test conversion
        test_dict = {"principle": EthicalPrinciple.AUTONOMY, "boundary": ProfessionalBoundary.COMMUNICATION_BOUNDARIES}
        processor._convert_enums_to_strings(test_dict)

        assert test_dict["principle"] == "autonomy"
        assert test_dict["boundary"] == "communication_boundaries"

    def test_different_severity_levels(self, processor):
        """Test handling of different ethical severity levels."""
        # Create scenarios with different severity levels
        minor_scenario = EthicalScenario(
            id="minor_test",
            dilemma_type=EthicalDilemmaType.CONFIDENTIALITY_CONFLICT,
            severity=EthicalSeverity.MINOR,
            situation_description="Minor boundary question",
            ethical_principles_involved=[EthicalPrinciple.AUTONOMY],
            boundary_concerns=[],
            stakeholders=["client", "therapist"],
            potential_consequences=["minor confusion"],
            decision_factors=["client preference"],
            recommended_actions=["clarify boundaries"],
            consultation_needed=False,
            documentation_required=False
        )

        major_scenario = EthicalScenario(
            id="major_test",
            dilemma_type=EthicalDilemmaType.DUTY_TO_WARN,
            severity=EthicalSeverity.MAJOR,
            situation_description="Serious threat situation",
            ethical_principles_involved=[EthicalPrinciple.NON_MALEFICENCE],
            boundary_concerns=[],
            stakeholders=["client", "potential victim"],
            potential_consequences=["serious harm"],
            decision_factors=["imminent danger"],
            recommended_actions=["immediate action"],
            consultation_needed=True,
            documentation_required=True
        )

        minor_conv = processor.generate_ethical_conversation(minor_scenario)
        major_conv = processor.generate_ethical_conversation(major_scenario)

        # Major scenario should have supervision notes
        assert minor_conv.supervision_notes is None
        assert major_conv.supervision_notes is not None


if __name__ == "__main__":
    pytest.main([__file__])
