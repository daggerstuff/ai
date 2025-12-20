"""
Unit tests for client scenario generator.

Tests the comprehensive client scenario generation functionality including
demographics, presenting problems, clinical formulations, and conversation generation.
"""

import json
import tempfile
import unittest
from pathlib import Path

from ai.dataset_pipeline.generation.client_scenario_generator import (
    ClientDemographics,
    ClientScenario,
    ClientScenarioGenerator,
    ClinicalFormulation,
    DemographicCategory,
    PresentingProblem,
    ScenarioType,
    SeverityLevel,
)
from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message


class TestClientScenarioGenerator(unittest.TestCase):
    """Test client scenario generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ClientScenarioGenerator()

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.dsm5_parser is not None
        assert self.generator.pdm2_parser is not None
        assert self.generator.big_five_processor is not None
        assert self.generator.demographic_templates is not None

        # Check demographic templates structure
        assert DemographicCategory.YOUNG_ADULT in self.generator.demographic_templates
        assert DemographicCategory.MIDDLE_AGED in self.generator.demographic_templates
        assert DemographicCategory.OLDER_ADULT in self.generator.demographic_templates

    def test_scenario_types_enum(self):
        """Test ScenarioType enum values."""
        expected_types = {
            "initial_assessment", "diagnostic_interview", "therapeutic_session",
            "crisis_intervention", "follow_up_session"
        }
        actual_types = {scenario_type.value for scenario_type in ScenarioType}
        assert expected_types == actual_types

    def test_severity_levels_enum(self):
        """Test SeverityLevel enum values."""
        expected_levels = {"mild", "moderate", "severe", "crisis"}
        actual_levels = {level.value for level in SeverityLevel}
        assert expected_levels == actual_levels

    def test_demographic_categories_enum(self):
        """Test DemographicCategory enum values."""
        expected_categories = {
            "young_adult", "middle_aged", "older_adult", "adolescent",
            "college_student", "working_professional", "retired"
        }
        actual_categories = {category.value for category in DemographicCategory}
        assert expected_categories == actual_categories

    def test_generate_demographics(self):
        """Test demographic generation."""
        demographics = self.generator._generate_demographics(DemographicCategory.YOUNG_ADULT)

        assert isinstance(demographics, ClientDemographics)
        assert isinstance(demographics.age, int)
        assert demographics.age >= 18
        assert demographics.age <= 25
        assert isinstance(demographics.gender, str)
        assert isinstance(demographics.occupation, str)
        assert isinstance(demographics.education_level, str)
        assert isinstance(demographics.cultural_background, str)
        assert isinstance(demographics.previous_therapy, bool)

    def test_generate_client_scenario(self):
        """Test complete client scenario generation."""
        scenario = self.generator.generate_client_scenario(
            scenario_type=ScenarioType.INITIAL_ASSESSMENT,
            severity_level=SeverityLevel.MODERATE,
            demographic_category=DemographicCategory.YOUNG_ADULT
        )

        # Check scenario structure
        assert isinstance(scenario, ClientScenario)
        assert isinstance(scenario.id, str)
        assert scenario.scenario_type == ScenarioType.INITIAL_ASSESSMENT
        assert scenario.severity_level == SeverityLevel.MODERATE

        # Check demographics
        assert isinstance(scenario.demographics, ClientDemographics)
        assert scenario.demographics.age >= 18
        assert scenario.demographics.age <= 25

        # Check presenting problem
        assert isinstance(scenario.presenting_problem, PresentingProblem)
        assert isinstance(scenario.presenting_problem.primary_concern, str)
        assert len(scenario.presenting_problem.primary_concern) > 0
        assert isinstance(scenario.presenting_problem.symptoms, list)
        assert isinstance(scenario.presenting_problem.triggers, list)

        # Check clinical formulation
        assert isinstance(scenario.clinical_formulation, ClinicalFormulation)
        assert isinstance(scenario.clinical_formulation.personality_profile, dict)
        assert isinstance(scenario.clinical_formulation.treatment_goals, list)

        # Check additional fields
        assert isinstance(scenario.learning_objectives, list)
        assert isinstance(scenario.therapeutic_considerations, list)
        assert isinstance(scenario.complexity_factors, list)
        assert isinstance(scenario.created_at, str)

    def test_generate_presenting_problem(self):
        """Test presenting problem generation."""
        # Create test demographics
        demographics = ClientDemographics(
            age=25, gender="Female", occupation="student", education_level="Bachelor's degree",
            relationship_status="Single", living_situation="with parents",
            cultural_background="Caucasian American", socioeconomic_status="Middle income"
        )

        # Get a test disorder
        disorders = self.generator.dsm5_parser.get_disorders()
        disorder = disorders[0] if disorders else None

        presenting_problem = self.generator._generate_presenting_problem(
            disorder, SeverityLevel.MODERATE, demographics
        )

        assert isinstance(presenting_problem, PresentingProblem)
        assert isinstance(presenting_problem.primary_concern, str)
        assert len(presenting_problem.primary_concern) > 0
        assert isinstance(presenting_problem.duration, str)
        assert isinstance(presenting_problem.onset, str)
        assert isinstance(presenting_problem.symptoms, list)
        assert isinstance(presenting_problem.triggers, list)
        assert isinstance(presenting_problem.functional_impact, list)
        assert isinstance(presenting_problem.current_stressors, list)

    def test_generate_clinical_formulation(self):
        """Test clinical formulation generation."""
        # Create test data
        demographics = ClientDemographics(
            age=30, gender="Male", occupation="teacher", education_level="Master's degree",
            relationship_status="Married", living_situation="with spouse",
            cultural_background="Hispanic/Latino", socioeconomic_status="Middle income"
        )

        presenting_problem = PresentingProblem(
            primary_concern="Anxiety symptoms",
            duration="3 months",
            onset="Gradual",
            symptoms=["worry", "restlessness"],
            triggers=["work stress"],
            functional_impact=["difficulty concentrating"]
        )

        disorders = self.generator.dsm5_parser.get_disorders()
        disorder = disorders[0] if disorders else None

        formulation = self.generator._generate_clinical_formulation(
            disorder, demographics, presenting_problem
        )

        assert isinstance(formulation, ClinicalFormulation)
        assert isinstance(formulation.dsm5_considerations, list)
        assert isinstance(formulation.personality_profile, dict)
        assert isinstance(formulation.psychodynamic_themes, list)
        assert isinstance(formulation.risk_factors, list)
        assert isinstance(formulation.protective_factors, list)
        assert isinstance(formulation.treatment_goals, list)

        # Check personality profile structure
        assert len(formulation.personality_profile) > 0
        for _factor, level in formulation.personality_profile.items():
            assert level in ["low", "average", "high"]

    def test_generate_scenario_batch(self):
        """Test batch scenario generation."""
        scenarios = self.generator.generate_scenario_batch(count=5)

        assert len(scenarios) == 5

        for scenario in scenarios:
            assert isinstance(scenario, ClientScenario)
            assert isinstance(scenario.id, str)
            assert scenario.scenario_type in list(ScenarioType)
            assert scenario.severity_level in list(SeverityLevel)

        # Check diversity
        scenario_types = {scenario.scenario_type for scenario in scenarios}
        assert len(scenario_types) >= 1  # At least some diversity

    def test_export_scenarios_to_json(self):
        """Test exporting scenarios to JSON."""
        scenarios = self.generator.generate_scenario_batch(count=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_scenarios.json"

            # Test successful export
            result = self.generator.export_scenarios_to_json(scenarios, output_path)
            assert result
            assert output_path.exists()

            # Verify exported content
            with open(output_path, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "scenarios" in exported_data
            assert "metadata" in exported_data
            assert exported_data["metadata"]["total_scenarios"] == len(scenarios)

            # Check scenario structure
            assert len(exported_data["scenarios"]) == len(scenarios)
            for scenario_data in exported_data["scenarios"]:
                assert "id" in scenario_data
                assert "scenario_type" in scenario_data
                assert "severity_level" in scenario_data
                assert "demographics" in scenario_data
                assert "presenting_problem" in scenario_data
                assert "clinical_formulation" in scenario_data

    def test_generate_conversation_templates(self):
        """Test conversation template generation from scenarios."""
        scenario = self.generator.generate_client_scenario(
            scenario_type=ScenarioType.INITIAL_ASSESSMENT,
            severity_level=SeverityLevel.MODERATE
        )

        conversations = self.generator.generate_conversation_templates(scenario)

        assert len(conversations) > 0

        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert isinstance(conversation.conversation_id, str)
            assert isinstance(conversation.messages, list)
            assert len(conversation.messages) > 0
            assert conversation.source == "client_scenario_generator"

            # Check metadata
            assert "scenario_id" in conversation.metadata
            assert conversation.metadata["scenario_id"] == scenario.id

            # Check messages
            for message in conversation.messages:
                assert isinstance(message, Message)
                assert message.role in ["therapist", "client"]
                assert isinstance(message.content, str)
                assert len(message.content) > 0
                assert "scenario_id" in message.metadata

    def test_get_statistics(self):
        """Test scenario statistics generation."""
        scenarios = self.generator.generate_scenario_batch(count=10)
        stats = self.generator.get_statistics(scenarios)

        expected_keys = {
            "total_scenarios", "scenario_types", "severity_levels",
            "age_distribution", "cultural_backgrounds", "disorders_represented",
            "average_complexity_factors"
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_scenarios"] == 10
        assert isinstance(stats["scenario_types"], dict)
        assert isinstance(stats["severity_levels"], dict)
        assert isinstance(stats["age_distribution"], dict)
        assert isinstance(stats["cultural_backgrounds"], dict)
        assert isinstance(stats["average_complexity_factors"], (int, float))

    def test_validate_scenario_quality(self):
        """Test scenario quality validation."""
        scenario = self.generator.generate_client_scenario()
        validation = self.generator.validate_scenario_quality(scenario)

        expected_keys = {"is_valid", "quality_score", "issues", "strengths"}
        assert set(validation.keys()) == expected_keys

        assert isinstance(validation["is_valid"], bool)
        assert isinstance(validation["quality_score"], float)
        assert validation["quality_score"] >= 0.0
        assert validation["quality_score"] <= 1.0
        assert isinstance(validation["issues"], list)
        assert isinstance(validation["strengths"], list)

    def test_generate_scenario_variations(self):
        """Test scenario variation generation."""
        base_scenario = self.generator.generate_client_scenario(
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MILD
        )

        variations = self.generator.generate_scenario_variations(base_scenario, count=3)

        assert len(variations) == 3

        for i, variation in enumerate(variations):
            assert isinstance(variation, ClientScenario)
            assert variation.scenario_type == base_scenario.scenario_type
            assert variation.severity_level != base_scenario.severity_level
            assert f"_var_{i + 1}" in variation.id

    def test_crisis_scenario_generation(self):
        """Test crisis intervention scenario generation."""
        scenario = self.generator.generate_client_scenario(
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.CRISIS
        )

        assert scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION
        assert scenario.severity_level == SeverityLevel.CRISIS

        # Check session context for crisis-specific elements
        assert "urgency_level" in scenario.session_context
        assert "safety_assessment_needed" in scenario.session_context
        assert "crisis_type" in scenario.session_context

        # Generate conversation to check crisis-specific content
        conversations = self.generator.generate_conversation_templates(scenario)
        crisis_conversation = conversations[0]

        # First message should address crisis
        first_message = crisis_conversation.messages[0]
        assert first_message.role == "therapist"
        assert "crisis" in first_message.metadata.get("type", "")


if __name__ == "__main__":
    unittest.main()
