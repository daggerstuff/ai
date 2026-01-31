"""
Tests for Therapeutic Alliance and Rapport-Building Conversations System

This module provides comprehensive tests for the therapeutic alliance generator,
covering alliance components, rapport-building techniques, alliance assessment,
and conversation generation.
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
from .therapeutic_alliance_generator import (
    AllianceAssessment,
    AllianceComponent,
    AllianceConversation,
    AllianceRuptureType,
    AllianceStage,
    AllianceStrength,
    RapportTechnique,
    TherapeuticAllianceGenerator,
)


class TestTherapeuticAllianceGenerator:
    """Test suite for TherapeuticAllianceGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return TherapeuticAllianceGenerator()

    @pytest.fixture
    def basic_scenario(self):
        """Create basic client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Generalized Anxiety Disorder"],
            attachment_style="secure",
            defense_mechanisms=["intellectualization"],
            psychodynamic_themes=["control_and_autonomy"]
        )

        return ClientScenario(
            id="alliance_client_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MILD,
            presenting_problem="Anxiety about work performance",
            clinical_formulation=clinical_formulation,
            demographics={"age": 28, "gender": "female"},
            session_context="Initial therapy session"
        )

    @pytest.fixture
    def cultural_scenario(self):
        """Create culturally diverse client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Adjustment Disorder"],
            attachment_style="anxious",
            defense_mechanisms=["avoidance"],
            psychodynamic_themes=["cultural_identity"]
        )

        return ClientScenario(
            id="cultural_client_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="Cultural adaptation stress",
            clinical_formulation=clinical_formulation,
            demographics={"age": 35, "gender": "male", "ethnicity": "Latino"},
            session_context="Cross-cultural therapy session"
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "alliance_strategies")
        assert hasattr(generator, "rapport_frameworks")
        assert hasattr(generator, "rupture_repair_protocols")
        assert hasattr(generator, "cultural_alliance_considerations")

        # Check that strategies are loaded
        assert len(generator.alliance_strategies) > 0
        assert AllianceComponent.TASK_ALLIANCE in generator.alliance_strategies
        assert AllianceComponent.BOND_ALLIANCE in generator.alliance_strategies
        assert AllianceComponent.GOAL_ALLIANCE in generator.alliance_strategies

    def test_alliance_strategies_structure(self, generator):
        """Test alliance strategies data structure."""
        task_strategies = generator.alliance_strategies[AllianceComponent.TASK_ALLIANCE]

        assert len(task_strategies) > 0
        strategy = task_strategies[0]

        assert hasattr(strategy, "id")
        assert hasattr(strategy, "alliance_component")
        assert hasattr(strategy, "target_stage")
        assert hasattr(strategy, "rapport_techniques")
        assert hasattr(strategy, "cultural_adaptations")
        assert hasattr(strategy, "specific_interventions")
        assert hasattr(strategy, "expected_outcomes")
        assert hasattr(strategy, "success_indicators")
        assert hasattr(strategy, "potential_challenges")
        assert hasattr(strategy, "monitoring_plan")

        # Validate data types
        assert isinstance(strategy.rapport_techniques, list)
        assert isinstance(strategy.cultural_adaptations, list)
        assert isinstance(strategy.specific_interventions, list)
        assert isinstance(strategy.expected_outcomes, list)
        assert isinstance(strategy.success_indicators, list)
        assert isinstance(strategy.potential_challenges, list)
        assert isinstance(strategy.monitoring_plan, list)

    def test_rapport_frameworks_initialization(self, generator):
        """Test rapport frameworks initialization."""
        frameworks = generator.rapport_frameworks

        assert RapportTechnique.EMPATHIC_REFLECTION in frameworks
        assert RapportTechnique.ACTIVE_LISTENING in frameworks
        assert RapportTechnique.GENUINENESS in frameworks
        assert RapportTechnique.CULTURAL_RESPONSIVENESS in frameworks

        empathy_framework = frameworks[RapportTechnique.EMPATHIC_REFLECTION]
        assert "description" in empathy_framework
        assert "key_elements" in empathy_framework
        assert "verbal_techniques" in empathy_framework
        assert "nonverbal_elements" in empathy_framework
        assert "cultural_considerations" in empathy_framework

        # Validate data types
        assert isinstance(empathy_framework["key_elements"], list)
        assert isinstance(empathy_framework["verbal_techniques"], list)
        assert isinstance(empathy_framework["nonverbal_elements"], list)
        assert isinstance(empathy_framework["cultural_considerations"], list)

    def test_rupture_repair_protocols_initialization(self, generator):
        """Test rupture repair protocols initialization."""
        protocols = generator.rupture_repair_protocols

        assert AllianceRuptureType.WITHDRAWAL_RUPTURE in protocols
        assert AllianceRuptureType.CONFRONTATION_RUPTURE in protocols
        assert AllianceRuptureType.CULTURAL_MISUNDERSTANDING in protocols

        withdrawal_protocol = protocols[AllianceRuptureType.WITHDRAWAL_RUPTURE]
        assert "description" in withdrawal_protocol
        assert "warning_signs" in withdrawal_protocol
        assert "repair_strategies" in withdrawal_protocol
        assert "repair_interventions" in withdrawal_protocol
        assert "prevention_strategies" in withdrawal_protocol

        # Validate data types
        assert isinstance(withdrawal_protocol["warning_signs"], list)
        assert isinstance(withdrawal_protocol["repair_strategies"], list)
        assert isinstance(withdrawal_protocol["repair_interventions"], list)
        assert isinstance(withdrawal_protocol["prevention_strategies"], list)

    def test_cultural_considerations_initialization(self, generator):
        """Test cultural considerations initialization."""
        considerations = generator.cultural_alliance_considerations

        assert "communication_styles" in considerations
        assert "relationship_patterns" in considerations
        assert "trust_building" in considerations

        communication = considerations["communication_styles"]
        assert "direct_vs_indirect" in communication
        assert "high_context_vs_low_context" in communication

        # Validate data types
        assert isinstance(communication["direct_vs_indirect"], list)
        assert isinstance(communication["high_context_vs_low_context"], list)

    def test_generate_task_alliance_conversation(self, generator, basic_scenario):
        """Test task alliance conversation generation."""
        conversation = generator.generate_alliance_conversation(
            basic_scenario, AllianceComponent.TASK_ALLIANCE, num_exchanges=8
        )

        assert isinstance(conversation, AllianceConversation)
        assert conversation.id is not None
        assert conversation.alliance_focus == AllianceComponent.TASK_ALLIANCE
        assert conversation.stage == AllianceStage.INITIAL_ENGAGEMENT
        assert conversation.client_scenario == basic_scenario
        assert len(conversation.conversation_exchanges) == 8
        assert isinstance(conversation.rapport_techniques_used, list)
        assert isinstance(conversation.cultural_adaptations_applied, list)
        assert isinstance(conversation.alliance_building_moments, list)
        assert isinstance(conversation.alliance_assessment, AllianceAssessment)

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 8

        # Should alternate between therapist and client
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "rapport_technique" in exchange
                assert "alliance_component" in exchange
                assert "alliance_building_element" in exchange
                assert "cultural_adaptation" in exchange
                assert "nonverbal_focus" in exchange
            else:
                assert exchange["speaker"] == "client"
                assert "engagement_level" in exchange
                assert "trust_indicators" in exchange
                assert "alliance_response" in exchange
                assert "emotional_openness" in exchange

    def test_generate_bond_alliance_conversation(self, generator, basic_scenario):
        """Test bond alliance conversation generation."""
        conversation = generator.generate_alliance_conversation(
            basic_scenario, AllianceComponent.BOND_ALLIANCE, num_exchanges=6
        )

        assert isinstance(conversation, AllianceConversation)
        assert conversation.alliance_focus == AllianceComponent.BOND_ALLIANCE

        # Should include bond-building techniques
        technique_values = [t.value for t in conversation.rapport_techniques_used]
        assert any(technique in technique_values for technique in [
            "empathic_reflection", "genuineness", "unconditional_positive_regard"
        ])

    def test_generate_goal_alliance_conversation(self, generator, basic_scenario):
        """Test goal alliance conversation generation."""
        conversation = generator.generate_alliance_conversation(
            basic_scenario, AllianceComponent.GOAL_ALLIANCE, num_exchanges=6
        )

        assert isinstance(conversation, AllianceConversation)
        assert conversation.alliance_focus == AllianceComponent.GOAL_ALLIANCE

        # Should include collaborative and validation techniques
        technique_values = [t.value for t in conversation.rapport_techniques_used]
        assert any(technique in technique_values for technique in [
            "collaborative_approach", "validation"
        ])

    def test_alliance_content_generation(self, generator):
        """Test alliance-specific content generation."""
        # Test task alliance content
        task_content = generator._generate_alliance_content(
            AllianceComponent.TASK_ALLIANCE, RapportTechnique.COLLABORATIVE_APPROACH, 0
        )
        assert isinstance(task_content, str)
        assert len(task_content) > 0
        assert any(word in task_content.lower() for word in ["together", "work", "approach"])

        # Test bond alliance content
        bond_content = generator._generate_alliance_content(
            AllianceComponent.BOND_ALLIANCE, RapportTechnique.EMPATHIC_REFLECTION, 0
        )
        assert isinstance(bond_content, str)
        assert any(word in bond_content.lower() for word in ["hear", "feel", "understand"])

        # Test goal alliance content
        goal_content = generator._generate_alliance_content(
            AllianceComponent.GOAL_ALLIANCE, RapportTechnique.VALIDATION, 0
        )
        assert isinstance(goal_content, str)
        assert any(word in goal_content.lower() for word in ["goals", "important", "change"])

    def test_client_alliance_response_progression(self, generator, basic_scenario):
        """Test client alliance response progression over exchanges."""
        strategy = generator.alliance_strategies[AllianceComponent.BOND_ALLIANCE][0]

        # Test progression from cautious to engaged
        response_0 = generator._generate_client_alliance_content(basic_scenario, strategy, 0)
        response_3 = generator._generate_client_alliance_content(basic_scenario, strategy, 3)

        assert isinstance(response_0, str)
        assert isinstance(response_3, str)

        # Later responses should show more engagement/trust
        assert response_0 != response_3

    def test_alliance_assessment_generation(self, generator, basic_scenario):
        """Test alliance assessment from conversation."""
        # Mock exchanges with positive alliance indicators
        exchanges = [
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "I feel comfortable here",
             "trust_indicators": ["explicit_trust_expression"],
             "alliance_response": "positive_bond_response",
             "emotional_openness": "high_openness"},
            {"speaker": "therapist", "content": "test"},
            {"speaker": "client", "content": "This makes sense to me",
             "trust_indicators": ["positive_regard"],
             "alliance_response": "positive_task_response",
             "emotional_openness": "moderate_openness"}
        ]

        strategy = generator.alliance_strategies[AllianceComponent.BOND_ALLIANCE][0]
        assessment = generator._assess_alliance_strength(exchanges, strategy, basic_scenario)

        assert isinstance(assessment, AllianceAssessment)
        assert assessment.id is not None
        assert isinstance(assessment.overall_strength, AllianceStrength)
        assert 0 <= assessment.task_alliance_score <= 10
        assert 0 <= assessment.bond_alliance_score <= 10
        assert 0 <= assessment.goal_alliance_score <= 10
        assert isinstance(assessment.rapport_indicators, list)
        assert isinstance(assessment.rupture_indicators, list)
        assert isinstance(assessment.cultural_factors, list)
        assert isinstance(assessment.improvement_areas, list)
        assert isinstance(assessment.strengths, list)

        # Should show positive indicators
        assert "explicit_trust_expression" in assessment.rapport_indicators

    def test_cultural_adaptation_identification(self, generator, cultural_scenario):
        """Test cultural adaptation identification."""
        conversation = generator.generate_alliance_conversation(
            cultural_scenario, AllianceComponent.BOND_ALLIANCE, num_exchanges=6
        )

        # Should include cultural adaptations for Latino client
        assert len(conversation.cultural_adaptations_applied) > 0
        assert any("cultural" in adaptation.lower() for adaptation in conversation.cultural_adaptations_applied)

    def test_rapport_technique_identification(self, generator, basic_scenario):
        """Test rapport technique identification from conversation."""
        # Mock exchanges with specific techniques
        exchanges = [
            {"speaker": "therapist", "rapport_technique": "empathic_reflection"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "rapport_technique": "active_listening"},
            {"speaker": "client", "content": "test"}
        ]

        techniques = generator._identify_rapport_techniques_used(exchanges)

        assert isinstance(techniques, list)
        assert RapportTechnique.EMPATHIC_REFLECTION in techniques
        assert RapportTechnique.ACTIVE_LISTENING in techniques

    def test_alliance_building_moments_identification(self, generator):
        """Test alliance building moments identification."""
        # Mock exchanges with alliance building elements
        exchanges = [
            {"speaker": "therapist", "alliance_building_element": "emotional_attunement"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "alliance_building_element": "partnership_building"},
            {"speaker": "client", "content": "test"}
        ]

        moments = generator._identify_alliance_building_moments(exchanges)

        assert isinstance(moments, list)
        assert len(moments) == 2
        assert "emotional_attunement" in moments[0]
        assert "partnership_building" in moments[1]

    def test_trust_indicators_identification(self, generator):
        """Test trust indicators identification from client content."""
        # Test explicit trust expression
        trust_content = "I feel comfortable sharing this with you"
        trust_indicators = generator._identify_trust_indicators(trust_content)

        assert isinstance(trust_indicators, list)
        assert "explicit_trust_expression" in trust_indicators

        # Test collaborative language
        collab_content = "I think we can work together on this"
        collab_indicators = generator._identify_trust_indicators(collab_content)

        assert "collaborative_language" in collab_indicators

    def test_emotional_openness_assessment(self, generator):
        """Test emotional openness assessment."""
        # Test high openness
        emotional_content = "I feel scared and worried about this situation"
        high_openness = generator._assess_emotional_openness(emotional_content, 3)
        assert high_openness == "high_openness"

        # Test moderate openness
        moderate_content = "I feel uncertain about this"
        moderate_openness = generator._assess_emotional_openness(moderate_content, 1)
        assert moderate_openness == "moderate_openness"

        # Test guarded
        guarded_content = "I don't know"
        guarded_openness = generator._assess_emotional_openness(guarded_content, 0)
        assert guarded_openness == "guarded"

    def test_export_alliance_conversations(self, generator, basic_scenario):
        """Test conversation export functionality."""
        conversations = []

        # Generate multiple conversations
        for component in [AllianceComponent.TASK_ALLIANCE, AllianceComponent.BOND_ALLIANCE]:
            conv = generator.generate_alliance_conversation(
                basic_scenario, component, num_exchanges=6
            )
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = generator.export_alliance_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 2
            assert export_result["output_file"] == temp_file
            assert "alliance_components" in export_result
            assert "rapport_techniques" in export_result
            assert "average_strength" in export_result

            # Check exported file
            assert os.path.exists(temp_file)
            with open(temp_file, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "metadata" in exported_data
            assert "conversations" in exported_data
            assert exported_data["metadata"]["total_conversations"] == 2
            assert len(exported_data["conversations"]) == 2

            # Check metadata
            metadata = exported_data["metadata"]
            assert "alliance_components_covered" in metadata
            assert "rapport_techniques_used" in metadata
            assert "average_alliance_strength" in metadata
            assert isinstance(metadata["alliance_components_covered"], list)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_different_alliance_stages(self, generator, basic_scenario):
        """Test handling of different alliance stages."""
        # Test initial engagement
        initial_conv = generator.generate_alliance_conversation(
            basic_scenario, AllianceComponent.BOND_ALLIANCE, AllianceStage.INITIAL_ENGAGEMENT
        )
        assert initial_conv.stage == AllianceStage.INITIAL_ENGAGEMENT

        # Test working alliance
        working_conv = generator.generate_alliance_conversation(
            basic_scenario, AllianceComponent.TASK_ALLIANCE, AllianceStage.WORKING_ALLIANCE
        )
        assert working_conv.stage == AllianceStage.WORKING_ALLIANCE


if __name__ == "__main__":
    pytest.main([__file__])
