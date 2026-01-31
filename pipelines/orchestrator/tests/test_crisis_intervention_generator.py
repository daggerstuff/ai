"""
Tests for Crisis Intervention and Safety Protocol Conversations System

This module provides comprehensive tests for the crisis intervention generator,
covering crisis protocols, safety planning, and emergency response conversations.
"""

import json
import os
import tempfile

import pytest

from .client_scenario_generator import (
    ClientScenario,
    ClinicalFormulation,
    ScenarioType,
    SeverityLevel,
)
from .crisis_intervention_generator import (
    CrisisConversation,
    CrisisInterventionGenerator,
    CrisisType,
    CrisisUrgency,
    InterventionTechnique,
    SafetyPlan,
    SafetyProtocol,
)


class TestCrisisInterventionGenerator:
    """Test suite for CrisisInterventionGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CrisisInterventionGenerator()

    @pytest.fixture
    def suicide_scenario(self):
        """Create suicide ideation scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder with suicidal ideation"],
            attachment_style="anxious",
            defense_mechanisms=["denial"],
            psychodynamic_themes=["hopelessness"]
        )

        return ClientScenario(
            id="suicide_crisis_001",
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.SEVERE,
            presenting_problem="Suicidal ideation with plan",
            clinical_formulation=clinical_formulation,
            demographics={"age": 32, "gender": "female"},
            session_context="Crisis intervention session"
        )

    @pytest.fixture
    def violence_scenario(self):
        """Create violence threat scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Intermittent Explosive Disorder"],
            attachment_style="disorganized",
            defense_mechanisms=["projection"],
            psychodynamic_themes=["anger_management"]
        )

        return ClientScenario(
            id="violence_crisis_001",
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.SEVERE,
            presenting_problem="Threats of violence toward others",
            clinical_formulation=clinical_formulation,
            demographics={"age": 28, "gender": "male"},
            session_context="Emergency intervention"
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "crisis_protocols")
        assert hasattr(generator, "safety_frameworks")
        assert hasattr(generator, "intervention_strategies")

        # Check that protocols are loaded
        assert len(generator.crisis_protocols) > 0
        assert CrisisType.SUICIDE_IDEATION in generator.crisis_protocols
        assert CrisisType.VIOLENCE_THREAT in generator.crisis_protocols

    def test_crisis_protocol_structure(self, generator):
        """Test crisis protocol data structure."""
        suicide_protocol = generator.crisis_protocols[CrisisType.SUICIDE_IDEATION]

        assert hasattr(suicide_protocol, "id")
        assert hasattr(suicide_protocol, "crisis_type")
        assert hasattr(suicide_protocol, "urgency_level")
        assert hasattr(suicide_protocol, "name")
        assert hasattr(suicide_protocol, "description")
        assert hasattr(suicide_protocol, "immediate_actions")
        assert hasattr(suicide_protocol, "assessment_priorities")
        assert hasattr(suicide_protocol, "intervention_techniques")
        assert hasattr(suicide_protocol, "safety_protocols")
        assert hasattr(suicide_protocol, "de_escalation_steps")
        assert hasattr(suicide_protocol, "risk_factors")
        assert hasattr(suicide_protocol, "protective_factors")
        assert hasattr(suicide_protocol, "emergency_criteria")
        assert hasattr(suicide_protocol, "follow_up_requirements")
        assert hasattr(suicide_protocol, "documentation_needs")
        assert hasattr(suicide_protocol, "legal_considerations")

        # Validate data types
        assert isinstance(suicide_protocol.immediate_actions, list)
        assert isinstance(suicide_protocol.assessment_priorities, list)
        assert isinstance(suicide_protocol.intervention_techniques, list)
        assert isinstance(suicide_protocol.safety_protocols, list)
        assert isinstance(suicide_protocol.de_escalation_steps, list)
        assert isinstance(suicide_protocol.risk_factors, list)
        assert isinstance(suicide_protocol.protective_factors, list)
        assert isinstance(suicide_protocol.emergency_criteria, list)
        assert isinstance(suicide_protocol.follow_up_requirements, list)
        assert isinstance(suicide_protocol.documentation_needs, list)
        assert isinstance(suicide_protocol.legal_considerations, list)

    def test_safety_frameworks_initialization(self, generator):
        """Test safety frameworks initialization."""
        frameworks = generator.safety_frameworks

        assert SafetyProtocol.SAFETY_PLANNING in frameworks
        assert SafetyProtocol.MEANS_RESTRICTION in frameworks
        assert SafetyProtocol.SUPPORT_ACTIVATION in frameworks

        safety_planning = frameworks[SafetyProtocol.SAFETY_PLANNING]
        assert "components" in safety_planning
        assert "implementation_steps" in safety_planning
        assert "effectiveness_factors" in safety_planning

        # Validate data types
        assert isinstance(safety_planning["components"], list)
        assert isinstance(safety_planning["implementation_steps"], list)
        assert isinstance(safety_planning["effectiveness_factors"], list)

    def test_intervention_strategies_initialization(self, generator):
        """Test intervention strategies initialization."""
        strategies = generator.intervention_strategies

        assert InterventionTechnique.DE_ESCALATION in strategies
        assert InterventionTechnique.GROUNDING in strategies
        assert InterventionTechnique.CRISIS_COUNSELING in strategies

        de_escalation = strategies[InterventionTechnique.DE_ESCALATION]
        assert "principles" in de_escalation
        assert "verbal_techniques" in de_escalation
        assert "non_verbal_techniques" in de_escalation

        # Validate data types
        assert isinstance(de_escalation["principles"], list)
        assert isinstance(de_escalation["verbal_techniques"], list)
        assert isinstance(de_escalation["non_verbal_techniques"], list)

    def test_generate_crisis_conversation_suicide(self, generator, suicide_scenario):
        """Test crisis conversation generation for suicide ideation."""
        conversation = generator.generate_crisis_conversation(
            suicide_scenario, CrisisType.SUICIDE_IDEATION, num_exchanges=12
        )

        assert isinstance(conversation, CrisisConversation)
        assert conversation.id is not None
        assert conversation.crisis_protocol.crisis_type == CrisisType.SUICIDE_IDEATION
        assert conversation.client_scenario == suicide_scenario
        assert len(conversation.conversation_exchanges) == 12
        assert isinstance(conversation.crisis_assessment, dict)
        assert isinstance(conversation.intervention_techniques_used, list)
        assert isinstance(conversation.de_escalation_success, bool)
        assert conversation.risk_level_initial in ["low", "moderate", "high"]
        assert conversation.risk_level_final in ["low", "moderate", "high"]
        assert isinstance(conversation.emergency_actions_taken, list)
        assert isinstance(conversation.follow_up_scheduled, bool)
        assert isinstance(conversation.clinical_notes, list)

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 12

        # Should alternate between therapist and client
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "intervention_technique" in exchange
                assert "crisis_phase" in exchange
                assert "safety_focus" in exchange
            else:
                assert exchange["speaker"] == "client"
                assert "emotional_state" in exchange
                assert "cooperation_level" in exchange
                assert "risk_indicators" in exchange

    def test_generate_crisis_conversation_violence(self, generator, violence_scenario):
        """Test crisis conversation generation for violence threat."""
        conversation = generator.generate_crisis_conversation(
            violence_scenario, CrisisType.VIOLENCE_THREAT, num_exchanges=10
        )

        assert isinstance(conversation, CrisisConversation)
        assert conversation.crisis_protocol.crisis_type == CrisisType.VIOLENCE_THREAT
        assert conversation.crisis_protocol.urgency_level == CrisisUrgency.IMMEDIATE

        # Should have high initial risk for violence threat
        assert conversation.risk_level_initial in ["moderate", "high"]

        # Should include de-escalation techniques
        technique_values = [t.value for t in conversation.intervention_techniques_used]
        assert "de_escalation" in technique_values or "active_listening" in technique_values

    def test_safety_assessment_questions(self, generator):
        """Test safety assessment question generation."""
        suicide_question = generator._generate_safety_assessment_question(CrisisType.SUICIDE_IDEATION)
        violence_question = generator._generate_safety_assessment_question(CrisisType.VIOLENCE_THREAT)

        assert isinstance(suicide_question, str)
        assert isinstance(violence_question, str)
        assert "hurt yourself" in suicide_question.lower() or "ending your life" in suicide_question.lower()
        assert "hurt someone" in violence_question.lower() or "hurting someone" in violence_question.lower()

    def test_client_crisis_response_generation(self, generator, suicide_scenario):
        """Test client crisis response generation."""
        response = generator._generate_client_crisis_content(
            suicide_scenario, CrisisType.SUICIDE_IDEATION, 0
        )

        assert isinstance(response, str)
        assert len(response) > 0

        # Should be appropriate for suicide crisis
        response_lower = response.lower()
        assert any(word in response_lower for word in ["hopeless", "can't", "pain", "take", "anymore"])

    def test_risk_indicator_identification(self, generator):
        """Test risk indicator identification from client responses."""
        # Test suicide risk indicators
        suicide_content = "I have a plan. I've been thinking about this for weeks."
        suicide_indicators = generator._identify_risk_indicators(suicide_content, CrisisType.SUICIDE_IDEATION)

        assert "specific_plan" in suicide_indicators
        assert "chronic_ideation" in suicide_indicators

        # Test violence risk indicators
        violence_content = "I'm so angry I could hurt someone. They deserve what's coming."
        violence_indicators = generator._identify_risk_indicators(violence_content, CrisisType.VIOLENCE_THREAT)

        assert "violence_intent" in violence_indicators
        assert "high_anger" in violence_indicators

    def test_crisis_assessment_generation(self, generator, suicide_scenario):
        """Test crisis assessment from conversation."""
        # Mock exchanges with risk indicators
        exchanges = [
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "test", "risk_indicators": ["specific_plan", "hopelessness"]},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "test", "risk_indicators": ["chronic_ideation"]}
        ]

        protocol = generator.crisis_protocols[CrisisType.SUICIDE_IDEATION]
        assessment = generator._conduct_crisis_assessment(exchanges, protocol, suicide_scenario)

        assert isinstance(assessment, dict)
        assert "crisis_type" in assessment
        assert "risk_indicators" in assessment
        assert "current_risk_level" in assessment
        assert "protective_factors" in assessment
        assert "support_systems" in assessment
        assert "immediate_safety" in assessment
        assert "intervention_needed" in assessment
        assert "emergency_criteria_met" in assessment

        # Should identify high risk due to specific plan
        assert assessment["current_risk_level"] == "high"
        assert "specific_plan" in assessment["risk_indicators"]

    def test_safety_plan_creation(self, generator, suicide_scenario):
        """Test safety plan creation."""
        # Mock crisis assessment
        crisis_assessment = {
            "immediate_safety": True,
            "current_risk_level": "moderate"
        }

        safety_plan = generator._create_safety_plan(
            crisis_assessment, suicide_scenario, CrisisType.SUICIDE_IDEATION
        )

        assert isinstance(safety_plan, SafetyPlan)
        assert safety_plan.id is not None
        assert safety_plan.client_id == suicide_scenario.id
        assert safety_plan.crisis_type == CrisisType.SUICIDE_IDEATION
        assert isinstance(safety_plan.warning_signs, list)
        assert isinstance(safety_plan.coping_strategies, list)
        assert isinstance(safety_plan.support_contacts, list)
        assert isinstance(safety_plan.emergency_contacts, list)
        assert isinstance(safety_plan.environmental_safety, list)
        assert isinstance(safety_plan.means_restriction, list)
        assert isinstance(safety_plan.reasons_for_living, list)
        assert isinstance(safety_plan.professional_contacts, list)

        # Check content appropriateness
        assert len(safety_plan.warning_signs) > 0
        assert len(safety_plan.coping_strategies) > 0
        assert len(safety_plan.emergency_contacts) > 0

        # Should include suicide-specific warning signs
        warning_signs_text = " ".join(safety_plan.warning_signs).lower()
        assert "hopeless" in warning_signs_text or "death" in warning_signs_text

    def test_de_escalation_assessment(self, generator):
        """Test de-escalation success assessment."""
        # Mock successful de-escalation
        successful_exchanges = [
            {"speaker": "client", "emotional_state": "hopeless"},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "emotional_state": "desperate"},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "emotional_state": "calmer"}
        ]

        success = generator._assess_de_escalation_success(successful_exchanges)
        assert success

        # Mock unsuccessful de-escalation
        unsuccessful_exchanges = [
            {"speaker": "client", "emotional_state": "hopeless"},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "emotional_state": "desperate"}
        ]

        no_success = generator._assess_de_escalation_success(unsuccessful_exchanges)
        assert not no_success

    def test_export_crisis_conversations(self, generator, suicide_scenario):
        """Test conversation export functionality."""
        conversations = []

        # Generate multiple conversations
        for _i in range(3):
            conv = generator.generate_crisis_conversation(
                suicide_scenario, CrisisType.SUICIDE_IDEATION, num_exchanges=8
            )
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = generator.export_crisis_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 3
            assert export_result["output_file"] == temp_file
            assert "crisis_types_covered" in export_result
            assert "success_rate" in export_result

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
            assert "crisis_types" in metadata
            assert "de_escalation_success_rate" in metadata
            assert "average_risk_reduction" in metadata
            assert isinstance(metadata["crisis_types"], list)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_different_crisis_types(self, generator, suicide_scenario, violence_scenario):
        """Test handling of different crisis types."""
        # Test suicide ideation
        suicide_conv = generator.generate_crisis_conversation(
            suicide_scenario, CrisisType.SUICIDE_IDEATION
        )
        assert suicide_conv.crisis_protocol.crisis_type == CrisisType.SUICIDE_IDEATION

        # Test violence threat
        violence_conv = generator.generate_crisis_conversation(
            violence_scenario, CrisisType.VIOLENCE_THREAT
        )
        assert violence_conv.crisis_protocol.crisis_type == CrisisType.VIOLENCE_THREAT
        assert violence_conv.crisis_protocol.urgency_level == CrisisUrgency.IMMEDIATE

        # Test generic crisis type (should create fallback protocol)
        panic_conv = generator.generate_crisis_conversation(
            suicide_scenario, CrisisType.PANIC_ATTACK
        )
        assert panic_conv.crisis_protocol.crisis_type == CrisisType.PANIC_ATTACK


if __name__ == "__main__":
    pytest.main([__file__])
