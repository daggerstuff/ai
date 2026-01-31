"""
Unit tests for therapeutic response generator.

Tests the comprehensive therapeutic response generation functionality including
technique selection, modality matching, response content generation, and
conversation creation with clinical accuracy.
"""

import json
import tempfile
import unittest
from pathlib import Path

from .client_scenario_generator import (
    ClientScenarioGenerator,
    DemographicCategory,
    ScenarioType,
    SeverityLevel,
)
from .conversation_schema import Conversation
from .therapeutic_response_generator import (
    ResponseContext,
    ResponseType,
    TherapeuticModality,
    TherapeuticResponse,
    TherapeuticResponseGenerator,
    TherapeuticTechnique,
)


class TestTherapeuticResponseGenerator(unittest.TestCase):
    """Test therapeutic response generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = TherapeuticResponseGenerator()
        self.scenario_generator = ClientScenarioGenerator()

        # Create test scenario
        self.test_scenario = self.scenario_generator.generate_client_scenario(
            scenario_type=ScenarioType.INITIAL_ASSESSMENT,
            severity_level=SeverityLevel.MODERATE,
            demographic_category=DemographicCategory.YOUNG_ADULT
        )

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.dsm5_parser is not None
        assert self.generator.pdm2_parser is not None
        assert self.generator.big_five_processor is not None
        assert self.generator.technique_templates is not None
        assert self.generator.modality_approaches is not None
        assert self.generator.crisis_responses is not None

    def test_therapeutic_technique_enum(self):
        """Test TherapeuticTechnique enum values."""
        expected_techniques = {
            "active_listening", "empathic_reflection", "open_ended_questioning",
            "summarization", "clarification", "validation", "psychoeducation",
            "cognitive_restructuring", "behavioral_activation", "mindfulness",
            "grounding_techniques", "safety_planning", "therapeutic_challenge"
        }
        actual_techniques = {technique.value for technique in TherapeuticTechnique}
        assert expected_techniques == actual_techniques

    def test_response_type_enum(self):
        """Test ResponseType enum values."""
        expected_types = {
            "assessment_question", "empathic_response", "psychoeducational",
            "intervention_suggestion", "crisis_response", "therapeutic_challenge",
            "supportive_statement", "treatment_planning"
        }
        actual_types = {response_type.value for response_type in ResponseType}
        assert expected_types == actual_types

    def test_therapeutic_modality_enum(self):
        """Test TherapeuticModality enum values."""
        expected_modalities = {
            "cognitive_behavioral", "psychodynamic", "humanistic",
            "dialectical_behavioral", "acceptance_commitment", "mindfulness_based",
            "trauma_informed", "solution_focused"
        }
        actual_modalities = {modality.value for modality in TherapeuticModality}
        assert expected_modalities == actual_modalities

    def test_generate_therapeutic_response(self):
        """Test basic therapeutic response generation."""
        context = ResponseContext(client_scenario=self.test_scenario)
        response = self.generator.generate_therapeutic_response(context)

        # Check response structure
        assert isinstance(response, TherapeuticResponse)
        assert isinstance(response.id, str)
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.technique in list(TherapeuticTechnique)
        assert response.response_type in list(ResponseType)
        assert response.modality in list(TherapeuticModality)
        assert isinstance(response.clinical_rationale, str)
        assert isinstance(response.target_symptoms, list)
        assert isinstance(response.contraindications, list)
        assert isinstance(response.follow_up_suggestions, list)
        assert isinstance(response.effectiveness_indicators, list)

    def test_technique_selection_for_crisis(self):
        """Test appropriate technique selection for crisis scenarios."""
        crisis_scenario = self.scenario_generator.generate_client_scenario(
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.CRISIS
        )

        context = ResponseContext(client_scenario=crisis_scenario)
        technique = self.generator._select_appropriate_technique(context)

        # Should select crisis-appropriate techniques
        crisis_techniques = [
            TherapeuticTechnique.SAFETY_PLANNING,
            TherapeuticTechnique.GROUNDING_TECHNIQUES,
            TherapeuticTechnique.VALIDATION
        ]
        assert technique in crisis_techniques

    def test_technique_selection_for_initial_assessment(self):
        """Test appropriate technique selection for initial assessment."""
        context = ResponseContext(client_scenario=self.test_scenario)
        technique = self.generator._select_appropriate_technique(context)

        # Should select assessment-appropriate techniques
        assessment_techniques = [
            TherapeuticTechnique.ACTIVE_LISTENING,
            TherapeuticTechnique.OPEN_ENDED_QUESTIONING,
            TherapeuticTechnique.EMPATHIC_REFLECTION
        ]
        assert technique in assessment_techniques

    def test_modality_selection_based_on_dsm5(self):
        """Test modality selection based on DSM-5 considerations."""
        # Create scenario with anxiety-related DSM-5 considerations
        anxiety_scenario = self.test_scenario
        anxiety_scenario.clinical_formulation.dsm5_considerations = ["Consider Generalized Anxiety Disorder"]

        context = ResponseContext(client_scenario=anxiety_scenario)
        modality = self.generator._select_appropriate_modality(context)

        # Should select anxiety-appropriate modalities
        anxiety_modalities = [
            TherapeuticModality.COGNITIVE_BEHAVIORAL,
            TherapeuticModality.MINDFULNESS_BASED
        ]
        assert modality in anxiety_modalities

    def test_response_content_generation(self):
        """Test response content generation with template filling."""
        context = ResponseContext(client_scenario=self.test_scenario)
        technique = TherapeuticTechnique.EMPATHIC_REFLECTION
        modality = TherapeuticModality.HUMANISTIC

        content = self.generator._generate_response_content(context, technique, modality)

        assert isinstance(content, str)
        assert len(content) > 10  # Reasonable length
        assert not content.startswith(" ")  # No leading spaces
        assert not content.endswith(" ")  # No trailing spaces

    def test_crisis_response_content(self):
        """Test crisis-specific response content generation."""
        crisis_scenario = self.scenario_generator.generate_client_scenario(
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.CRISIS
        )
        crisis_scenario.session_context["crisis_type"] = "suicidal_ideation"

        context = ResponseContext(client_scenario=crisis_scenario)
        technique = TherapeuticTechnique.SAFETY_PLANNING
        modality = TherapeuticModality.TRAUMA_INFORMED

        content = self.generator._generate_response_content(context, technique, modality)

        # Should contain crisis-appropriate language
        crisis_keywords = ["safety", "safe", "concerned", "together", "help"]
        assert any(keyword in content.lower() for keyword in crisis_keywords)

    def test_template_variable_extraction(self):
        """Test extraction of template variables from context."""
        context = ResponseContext(client_scenario=self.test_scenario)
        variables = self.generator._extract_template_variables(context)

        assert isinstance(variables, dict)
        assert "client_concern" in variables
        assert "emotion" in variables
        assert "intensity_word" in variables

        # Check severity-based intensity
        if self.test_scenario.severity_level == SeverityLevel.MODERATE:
            assert variables["intensity_word"] == "challenging"

    def test_modality_adjustment(self):
        """Test content adjustment based on therapeutic modality."""
        base_content = "I understand what you're saying."

        # Test CBT adjustment
        cbt_content = self.generator._adjust_for_modality(base_content, TherapeuticModality.COGNITIVE_BEHAVIORAL)
        assert "thoughts" in cbt_content.lower()

        # Test psychodynamic adjustment
        psychodynamic_content = self.generator._adjust_for_modality(base_content, TherapeuticModality.PSYCHODYNAMIC)
        assert "pattern" in psychodynamic_content.lower()

        # Test humanistic adjustment
        humanistic_content = self.generator._adjust_for_modality(base_content, TherapeuticModality.HUMANISTIC)
        assert any(word in humanistic_content.lower() for word in ["feel", "experience", "sense"])

    def test_response_type_determination(self):
        """Test response type determination based on technique."""
        context = ResponseContext(client_scenario=self.test_scenario)

        # Test psychoeducation
        response_type = self.generator._determine_response_type(context, TherapeuticTechnique.PSYCHOEDUCATION)
        assert response_type == ResponseType.PSYCHOEDUCATIONAL

        # Test empathic reflection
        response_type = self.generator._determine_response_type(context, TherapeuticTechnique.EMPATHIC_REFLECTION)
        assert response_type == ResponseType.EMPATHIC_RESPONSE

        # Test open-ended questioning
        response_type = self.generator._determine_response_type(context, TherapeuticTechnique.OPEN_ENDED_QUESTIONING)
        assert response_type == ResponseType.ASSESSMENT_QUESTION

    def test_clinical_rationale_generation(self):
        """Test clinical rationale generation."""
        context = ResponseContext(client_scenario=self.test_scenario)
        technique = TherapeuticTechnique.VALIDATION
        modality = TherapeuticModality.HUMANISTIC

        rationale = self.generator._generate_clinical_rationale(context, technique, modality)

        assert isinstance(rationale, str)
        assert len(rationale) > 10
        # Check for validation-related terms
        validation_terms = ["validates", "normalizes", "reduces shame", "client experience"]
        assert any(term in rationale.lower() for term in validation_terms)

    def test_contraindications_identification(self):
        """Test identification of technique contraindications."""
        crisis_scenario = self.scenario_generator.generate_client_scenario(
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.CRISIS
        )

        context = ResponseContext(client_scenario=crisis_scenario)
        contraindications = self.generator._identify_contraindications(context, TherapeuticTechnique.COGNITIVE_RESTRUCTURING)

        assert isinstance(contraindications, list)
        # Should identify crisis as contraindication for cognitive restructuring
        assert any("crisis" in contra.lower() for contra in contraindications)

    def test_follow_up_suggestions(self):
        """Test follow-up suggestion generation."""
        context = ResponseContext(client_scenario=self.test_scenario)
        follow_ups = self.generator._suggest_follow_ups(context, TherapeuticTechnique.EMPATHIC_REFLECTION)

        assert isinstance(follow_ups, list)
        assert len(follow_ups) > 0
        for follow_up in follow_ups:
            assert isinstance(follow_up, str)
            assert len(follow_up) > 0

    def test_effectiveness_indicators(self):
        """Test effectiveness indicator definition."""
        indicators = self.generator._define_effectiveness_indicators(TherapeuticTechnique.VALIDATION)

        assert isinstance(indicators, list)
        assert len(indicators) > 0
        for indicator in indicators:
            assert isinstance(indicator, str)
            assert len(indicator) > 0

    def test_generate_conversation_with_responses(self):
        """Test complete conversation generation with therapeutic responses."""
        conversation = self.generator.generate_conversation_with_responses(
            self.test_scenario,
            num_exchanges=3,
            target_modality=TherapeuticModality.COGNITIVE_BEHAVIORAL
        )

        # Check conversation structure
        assert isinstance(conversation, Conversation)
        assert isinstance(conversation.messages, list)
        assert len(conversation.messages) == 7  # 1 opening + 3 exchanges (6) + 1 = 7
        assert conversation.source == "therapeutic_response_generator"

        # Check message alternation
        assert conversation.messages[0].role == "therapist"
        assert conversation.messages[1].role == "client"
        assert conversation.messages[2].role == "therapist"

        # Check therapist message metadata
        therapist_messages = [msg for msg in conversation.messages if msg.role == "therapist"]
        for msg in therapist_messages:
            assert "technique" in msg.meta
            assert "response_type" in msg.meta
            assert "modality" in msg.meta
            assert "clinical_rationale" in msg.meta

    def test_client_response_generation(self):
        """Test realistic client response generation."""
        # Test initial response
        initial_response = self.generator._generate_initial_client_response(self.test_scenario)
        assert isinstance(initial_response, str)
        assert len(initial_response) > 10

        # Should contain personal language
        personal_words = ["i", "me", "my", "i'm", "i've"]
        assert any(word in initial_response.lower() for word in personal_words)

    def test_generate_response_batch(self):
        """Test batch response generation."""
        scenarios = [self.test_scenario]
        responses = self.generator.generate_response_batch(scenarios, responses_per_scenario=3)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, TherapeuticResponse)

    def test_export_responses_to_json(self):
        """Test exporting responses to JSON."""
        context = ResponseContext(client_scenario=self.test_scenario)
        responses = [self.generator.generate_therapeutic_response(context) for _ in range(2)]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_responses.json"

            # Test successful export
            result = self.generator.export_responses_to_json(responses, output_path)
            assert result
            assert output_path.exists()

            # Verify exported content
            with open(output_path, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "responses" in exported_data
            assert "metadata" in exported_data
            assert exported_data["metadata"]["total_responses"] == len(responses)

            # Check response structure
            assert len(exported_data["responses"]) == len(responses)
            for response_data in exported_data["responses"]:
                assert "id" in response_data
                assert "content" in response_data
                assert "technique" in response_data
                assert "clinical_rationale" in response_data

    def test_get_statistics(self):
        """Test response statistics generation."""
        context = ResponseContext(client_scenario=self.test_scenario)
        responses = [self.generator.generate_therapeutic_response(context) for _ in range(5)]

        stats = self.generator.get_statistics(responses)

        expected_keys = {
            "total_responses", "techniques_used", "response_types",
            "modalities_used", "average_contraindications", "average_follow_ups"
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_responses"] == 5
        assert isinstance(stats["techniques_used"], dict)
        assert isinstance(stats["response_types"], dict)
        assert isinstance(stats["modalities_used"], dict)
        assert isinstance(stats["average_contraindications"], (int, float))
        assert isinstance(stats["average_follow_ups"], (int, float))


if __name__ == "__main__":
    unittest.main()
