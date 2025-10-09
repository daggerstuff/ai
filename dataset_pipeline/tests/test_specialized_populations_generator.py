"""
Tests for Specialized Populations Training Data System

This module provides comprehensive tests for the specialized populations generator,
covering trauma-informed care, addiction treatment, LGBTQ+ affirmative therapy,
and other specialized population approaches.
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
from .specialized_populations_generator import (
    CulturalFactor,
    PopulationConversation,
    SpecializedPopulation,
    SpecializedPopulationsGenerator,
)


class TestSpecializedPopulationsGenerator:
    """Test suite for SpecializedPopulationsGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return SpecializedPopulationsGenerator()

    @pytest.fixture
    def trauma_scenario(self):
        """Create trauma survivor scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Post-Traumatic Stress Disorder", "Complex Trauma"],
            attachment_style="disorganized",
            defense_mechanisms=["dissociation", "avoidance"],
            psychodynamic_themes=["safety_and_trust"]
        )

        return ClientScenario(
            id="trauma_survivor_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="PTSD symptoms following childhood trauma",
            clinical_formulation=clinical_formulation,
            demographics={"age": 29, "gender": "female", "ethnicity": "Hispanic"},
            session_context="Trauma-focused therapy session"
        )

    @pytest.fixture
    def addiction_scenario(self):
        """Create addiction recovery scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Alcohol Use Disorder", "Major Depressive Disorder"],
            attachment_style="anxious",
            defense_mechanisms=["denial", "rationalization"],
            psychodynamic_themes=["control_and_powerlessness"]
        )

        return ClientScenario(
            id="addiction_recovery_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="Alcohol addiction with depression",
            clinical_formulation=clinical_formulation,
            demographics={"age": 35, "gender": "male"},
            session_context="Addiction recovery session"
        )

    @pytest.fixture
    def lgbtq_scenario(self):
        """Create LGBTQ+ scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Adjustment Disorder", "Gender Dysphoria"],
            attachment_style="secure",
            defense_mechanisms=["intellectualization"],
            psychodynamic_themes=["identity_and_authenticity"]
        )

        return ClientScenario(
            id="lgbtq_client_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MILD,
            presenting_problem="Gender identity exploration and family acceptance",
            clinical_formulation=clinical_formulation,
            demographics={"age": 22, "gender": "non-binary"},
            session_context="Identity-affirming therapy session"
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert hasattr(generator, "population_characteristics")
        assert hasattr(generator, "specialized_protocols")
        assert hasattr(generator, "cultural_frameworks")
        assert hasattr(generator, "trauma_informed_principles")

        # Check that characteristics are loaded
        assert len(generator.population_characteristics) > 0
        assert SpecializedPopulation.TRAUMA_SURVIVORS in generator.population_characteristics
        assert SpecializedPopulation.ADDICTION_RECOVERY in generator.population_characteristics
        assert SpecializedPopulation.LGBTQ_PLUS in generator.population_characteristics

    def test_population_characteristics_structure(self, generator):
        """Test population characteristics data structure."""
        trauma_chars = generator.population_characteristics[SpecializedPopulation.TRAUMA_SURVIVORS]

        assert hasattr(trauma_chars, "id")
        assert hasattr(trauma_chars, "population")
        assert hasattr(trauma_chars, "name")
        assert hasattr(trauma_chars, "description")
        assert hasattr(trauma_chars, "common_presentations")
        assert hasattr(trauma_chars, "unique_challenges")
        assert hasattr(trauma_chars, "cultural_considerations")
        assert hasattr(trauma_chars, "trauma_considerations")
        assert hasattr(trauma_chars, "preferred_modalities")
        assert hasattr(trauma_chars, "contraindicated_approaches")
        assert hasattr(trauma_chars, "assessment_modifications")
        assert hasattr(trauma_chars, "therapeutic_goals")
        assert hasattr(trauma_chars, "treatment_considerations")

        # Validate data types
        assert isinstance(trauma_chars.common_presentations, list)
        assert isinstance(trauma_chars.unique_challenges, list)
        assert isinstance(trauma_chars.cultural_considerations, list)
        assert isinstance(trauma_chars.trauma_considerations, list)
        assert isinstance(trauma_chars.preferred_modalities, list)
        assert isinstance(trauma_chars.contraindicated_approaches, list)
        assert isinstance(trauma_chars.assessment_modifications, list)
        assert isinstance(trauma_chars.therapeutic_goals, list)
        assert isinstance(trauma_chars.treatment_considerations, list)

    def test_specialized_protocols_initialization(self, generator):
        """Test specialized protocols initialization."""
        protocols = generator.specialized_protocols

        assert SpecializedPopulation.TRAUMA_SURVIVORS in protocols
        assert SpecializedPopulation.ADDICTION_RECOVERY in protocols
        assert SpecializedPopulation.LGBTQ_PLUS in protocols

        # Check trauma protocol structure
        trauma_protocols = protocols[SpecializedPopulation.TRAUMA_SURVIVORS]
        assert len(trauma_protocols) > 0

        protocol = trauma_protocols[0]
        assert hasattr(protocol, "id")
        assert hasattr(protocol, "population")
        assert hasattr(protocol, "protocol_name")
        assert hasattr(protocol, "evidence_base")
        assert hasattr(protocol, "target_symptoms")
        assert hasattr(protocol, "intervention_phases")
        assert hasattr(protocol, "specialized_techniques")
        assert hasattr(protocol, "cultural_adaptations")
        assert hasattr(protocol, "safety_considerations")
        assert hasattr(protocol, "outcome_measures")
        assert hasattr(protocol, "session_structure")
        assert hasattr(protocol, "therapist_requirements")

        # Validate data types
        assert isinstance(protocol.target_symptoms, list)
        assert isinstance(protocol.intervention_phases, list)
        assert isinstance(protocol.specialized_techniques, list)
        assert isinstance(protocol.cultural_adaptations, list)
        assert isinstance(protocol.safety_considerations, list)
        assert isinstance(protocol.outcome_measures, list)
        assert isinstance(protocol.session_structure, list)
        assert isinstance(protocol.therapist_requirements, list)

    def test_cultural_frameworks_initialization(self, generator):
        """Test cultural frameworks initialization."""
        frameworks = generator.cultural_frameworks

        assert CulturalFactor.LANGUAGE_BARRIERS in frameworks
        assert CulturalFactor.RELIGIOUS_BELIEFS in frameworks
        assert CulturalFactor.FAMILY_DYNAMICS in frameworks

        language_framework = frameworks[CulturalFactor.LANGUAGE_BARRIERS]
        assert "considerations" in language_framework
        assert "interventions" in language_framework
        assert "assessment_modifications" in language_framework

        # Validate data types
        assert isinstance(language_framework["considerations"], list)
        assert isinstance(language_framework["interventions"], list)
        assert isinstance(language_framework["assessment_modifications"], list)

    def test_trauma_informed_principles_initialization(self, generator):
        """Test trauma-informed principles initialization."""
        principles = generator.trauma_informed_principles

        assert "safety" in principles
        assert "trustworthiness" in principles
        assert "choice" in principles
        assert "collaboration" in principles
        assert "cultural_humility" in principles

        safety_principle = principles["safety"]
        assert "physical_safety" in safety_principle
        assert "psychological_safety" in safety_principle

        # Validate data types
        assert isinstance(safety_principle["physical_safety"], list)
        assert isinstance(safety_principle["psychological_safety"], list)

    def test_generate_trauma_conversation(self, generator, trauma_scenario):
        """Test trauma survivor conversation generation."""
        conversation = generator.generate_specialized_conversation(
            trauma_scenario, SpecializedPopulation.TRAUMA_SURVIVORS, num_exchanges=10
        )

        assert isinstance(conversation, PopulationConversation)
        assert conversation.id is not None
        assert conversation.population_characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS
        assert conversation.client_scenario == trauma_scenario
        assert len(conversation.conversation_exchanges) == 10
        assert isinstance(conversation.cultural_adaptations_used, list)
        assert isinstance(conversation.trauma_informed_elements, list)
        assert isinstance(conversation.population_specific_techniques, list)
        assert isinstance(conversation.therapeutic_alliance_factors, list)
        assert isinstance(conversation.cultural_competency_demonstrated, list)

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 10

        # Should alternate between therapist and client
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "specialized_technique" in exchange
                assert "population_focus" in exchange
                assert "cultural_adaptation" in exchange
                assert "trauma_informed_element" in exchange
            else:
                assert exchange["speaker"] == "client"
                assert "emotional_state" in exchange
                assert "cultural_expression" in exchange
                assert "population_specific_concerns" in exchange
                assert "engagement_level" in exchange

    def test_generate_addiction_conversation(self, generator, addiction_scenario):
        """Test addiction recovery conversation generation."""
        conversation = generator.generate_specialized_conversation(
            addiction_scenario, SpecializedPopulation.ADDICTION_RECOVERY, num_exchanges=8
        )

        assert isinstance(conversation, PopulationConversation)
        assert conversation.population_characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY

        # Should include addiction-specific elements
        assert len(conversation.population_specific_techniques) > 0

        # Check for motivational interviewing elements
        therapist_content = " ".join([
            e.get("content", "") for e in conversation.conversation_exchanges
            if e.get("speaker") == "therapist"
        ]).lower()

        # Should include supportive, non-confrontational language
        assert any(word in therapist_content for word in ["support", "journey", "change", "appreciate"])

    def test_generate_lgbtq_conversation(self, generator, lgbtq_scenario):
        """Test LGBTQ+ conversation generation."""
        conversation = generator.generate_specialized_conversation(
            lgbtq_scenario, SpecializedPopulation.LGBTQ_PLUS, num_exchanges=8
        )

        assert isinstance(conversation, PopulationConversation)
        assert conversation.population_characteristics.population == SpecializedPopulation.LGBTQ_PLUS

        # Should include affirming elements
        therapist_content = " ".join([
            e.get("content", "") for e in conversation.conversation_exchanges
            if e.get("speaker") == "therapist"
        ]).lower()

        # Should include affirming language
        assert any(word in therapist_content for word in ["affirming", "authentic", "identity", "valid"])

    def test_population_opening_generation(self, generator):
        """Test population-specific opening generation."""
        trauma_chars = generator.population_characteristics[SpecializedPopulation.TRAUMA_SURVIVORS]
        addiction_chars = generator.population_characteristics[SpecializedPopulation.ADDICTION_RECOVERY]
        lgbtq_chars = generator.population_characteristics[SpecializedPopulation.LGBTQ_PLUS]

        trauma_opening = generator._generate_population_opening(trauma_chars)
        addiction_opening = generator._generate_population_opening(addiction_chars)
        lgbtq_opening = generator._generate_population_opening(lgbtq_chars)

        # Should be different for different populations
        assert trauma_opening != addiction_opening
        assert addiction_opening != lgbtq_opening

        # Should include population-appropriate language
        assert "safe" in trauma_opening.lower()
        assert "journey" in addiction_opening.lower()
        assert "affirming" in lgbtq_opening.lower()

    def test_client_response_generation(self, generator, trauma_scenario):
        """Test population-specific client response generation."""
        trauma_chars = generator.population_characteristics[SpecializedPopulation.TRAUMA_SURVIVORS]

        response = generator._generate_population_client_content(
            trauma_scenario, trauma_chars, 0
        )

        assert isinstance(response, str)
        assert len(response) > 0

        # Should be appropriate for trauma population
        response_lower = response.lower()
        assert any(word in response_lower for word in ["raw", "happened", "trust", "normal", "threat"])

    def test_cultural_adaptation_identification(self, generator, trauma_scenario):
        """Test cultural adaptation identification."""
        # Mock exchanges with cultural adaptations
        exchanges = [
            {"speaker": "therapist", "cultural_adaptation": "family_inclusive_approach"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "cultural_adaptation": "spiritual_integration"},
            {"speaker": "client", "content": "test"}
        ]

        trauma_chars = generator.population_characteristics[SpecializedPopulation.TRAUMA_SURVIVORS]
        adaptations = generator._identify_cultural_adaptations(exchanges, trauma_chars)

        assert isinstance(adaptations, list)
        assert "family_inclusive_approach" in adaptations
        assert "spiritual_integration" in adaptations

    def test_trauma_informed_elements_identification(self, generator):
        """Test trauma-informed elements identification."""
        # Mock exchanges with trauma-informed elements
        exchanges = [
            {"speaker": "therapist", "trauma_informed_element": "safety"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "trauma_informed_element": "choice"},
            {"speaker": "client", "content": "test"}
        ]

        elements = generator._identify_trauma_informed_elements(exchanges)

        assert isinstance(elements, list)
        assert "safety" in elements
        assert "choice" in elements

    def test_therapeutic_alliance_assessment(self, generator, trauma_scenario):
        """Test therapeutic alliance factors assessment."""
        # Mock exchanges with alliance-building elements
        exchanges = [
            {"speaker": "therapist", "content": "This is a safe space for you"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "content": "We'll go at your pace"},
            {"speaker": "client", "content": "test"},
            {"speaker": "therapist", "content": "I want to understand your experience"},
            {"speaker": "client", "content": "test"}
        ]

        trauma_chars = generator.population_characteristics[SpecializedPopulation.TRAUMA_SURVIVORS]
        factors = generator._assess_therapeutic_alliance_factors(exchanges, trauma_chars)

        assert isinstance(factors, list)
        assert "safety_establishment" in factors
        assert "client_control" in factors
        assert "empathic_understanding" in factors

    def test_export_specialized_conversations(self, generator, trauma_scenario):
        """Test conversation export functionality."""
        conversations = []

        # Generate multiple conversations
        for _i in range(3):
            conv = generator.generate_specialized_conversation(
                trauma_scenario, SpecializedPopulation.TRAUMA_SURVIVORS, num_exchanges=6
            )
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = generator.export_specialized_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 3
            assert export_result["output_file"] == temp_file
            assert "populations_covered" in export_result
            assert "cultural_adaptations" in export_result
            assert "trauma_informed_elements" in export_result

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
            assert "populations_covered" in metadata
            assert "cultural_adaptations_used" in metadata
            assert "trauma_informed_coverage" in metadata
            assert isinstance(metadata["populations_covered"], list)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_different_specialized_populations(self, generator, trauma_scenario, addiction_scenario, lgbtq_scenario):
        """Test handling of different specialized populations."""
        # Test trauma survivors
        trauma_conv = generator.generate_specialized_conversation(
            trauma_scenario, SpecializedPopulation.TRAUMA_SURVIVORS
        )
        assert trauma_conv.population_characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS

        # Test addiction recovery
        addiction_conv = generator.generate_specialized_conversation(
            addiction_scenario, SpecializedPopulation.ADDICTION_RECOVERY
        )
        assert addiction_conv.population_characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY

        # Test LGBTQ+
        lgbtq_conv = generator.generate_specialized_conversation(
            lgbtq_scenario, SpecializedPopulation.LGBTQ_PLUS
        )
        assert lgbtq_conv.population_characteristics.population == SpecializedPopulation.LGBTQ_PLUS

        # Test generic population (should create fallback)
        veterans_conv = generator.generate_specialized_conversation(
            trauma_scenario, SpecializedPopulation.VETERANS
        )
        assert veterans_conv.population_characteristics.population == SpecializedPopulation.VETERANS


if __name__ == "__main__":
    pytest.main([__file__])
