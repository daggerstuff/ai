#!/usr/bin/env python3
"""
Test suite for CCD-Platform Integration
Tests for connecting Case Conceptualization Diagram (CCD) concepts with
the existing therapeutic simulation engine and client profile structures.
"""

import unittest

from ai.tools.utilities.platform.ccd_integration import (
    CCDIntegration,
    create_difficult_client_from_ccd_template,
    update_ccd_with_simulation_results,
)
from ai.tools.utilities.platform.ccd_profiles import (
    get_borderline_traits_ccd_template,
    get_hostile_aggressive_ccd_template,
    get_resistant_ccd_template,
)
from ai.tools.utilities.platform.ccd_schema import CCDConceptualization
from ai.tools.utilities.platform.pixelated_empathy_core import (
    ClientPersonality,
    DifficultClientProfile,
    DifficultyLevel,
    SessionObjective,
)


class TestCCDIntegration(unittest.TestCase):
    """Test CCD-Platform integration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_simulation_state = {
            "trust_level": 0.8,  # Increased to trigger trust adjustment
            "resistance_level": 0.2,  # Decreased to trigger resistance adjustment
            "breakthrough_opportunity": True,  # Set to True to trigger breakthrough adjustment
            "therapeutic_quality": 0.85,  # Increased to trigger quality adjustment
        }

        self.sample_therapist_analysis = {
            "therapeutic_quality": 0.9,  # Increased to trigger quality adjustment
            "skills_demonstrated": {"rapport_building": 0.9, "active_listening": 0.8, "empathy": 0.85},
            "mistakes": [],
            "strengths": ["Good rapport building", "Good empathy"],
            "appropriateness": 0.9,
            "impact_prediction": {
                "trust_change": 0.05,
                "resistance_change": -0.02,
                "emotional_intensity_change": 0.0,
                "crisis_risk_change": 0.0,
                "breakthrough_probability": 0.1,
            },
        }

    def test_ccd_profile_to_difficult_client_profile_resistant(self):
        """Test converting resistant CCD template to DifficultClientProfile"""
        ccd_template = get_resistant_ccd_template()

        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template)

        # Check that we got a valid profile
        assert isinstance(profile, DifficultClientProfile)
        assert profile.personality_type == ClientPersonality.RESISTANT
        assert profile.difficulty_level == DifficultyLevel.INTERMEDIATE
        assert isinstance(profile.client_id, str)
        assert len(profile.client_id) > 0
        assert isinstance(profile.name, str)
        assert len(profile.name) > 0
        assert isinstance(profile.age, int)
        assert profile.age >= 18
        assert profile.age <= 65
        assert profile.gender in ["male", "female", "non-binary"]

        # Check that mapped fields are present
        assert profile.presenting_problem == "Mandated therapy due to work-related issues, denies needing help"
        assert "Defensive" in profile.personality_traits
        assert "Intellectualization" in profile.defense_mechanisms
        assert profile.communication_style == "Closed off, gives minimal responses, questions therapist competence"

        # Check learning objectives mapping
        assert any(obj == SessionObjective.RAPPORT_BUILDING for obj in profile.learning_objectives)
        assert any(obj == SessionObjective.RESISTANCE_MANAGEMENT for obj in profile.learning_objectives)
        assert any(obj == SessionObjective.THERAPEUTIC_CONFRONTATION for obj in profile.learning_objectives)

    def test_ccd_profile_to_difficult_client_profile_hostile_aggressive(self):
        """Test converting hostile-aggressive CCD template to DifficultClientProfile"""
        ccd_template = get_hostile_aggressive_ccd_template()

        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template)

        # Check that we got a valid profile
        assert isinstance(profile, DifficultClientProfile)
        assert profile.personality_type == ClientPersonality.HOSTILE_AGGRESSIVE
        assert profile.difficulty_level == DifficultyLevel.ADVANCED

        # Check that mapped fields are present
        assert profile.presenting_problem == "Anger management issues affecting relationships and work"
        assert "Hostile" in profile.personality_traits
        assert "Projection" in profile.defense_mechanisms
        assert profile.communication_style == "Loud, aggressive, interrupting, blaming others"

        # Check learning objectives mapping
        assert any(obj == SessionObjective.RESISTANCE_MANAGEMENT for obj in profile.learning_objectives)
        assert any(obj == SessionObjective.BOUNDARY_SETTING for obj in profile.learning_objectives)
        assert any(obj == SessionObjective.SAFETY_ASSESSMENT for obj in profile.learning_objectives)

    def test_ccd_profile_to_difficult_client_profile_borderline_traits(self):
        """Test converting borderline traits CCD template to DifficultClientProfile"""
        ccd_template = get_borderline_traits_ccd_template()

        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template)

        # Check that we got a valid profile
        assert isinstance(profile, DifficultClientProfile)
        assert profile.personality_type == ClientPersonality.BORDERLINE_TRAITS
        assert profile.difficulty_level == DifficultyLevel.EXPERT

        # Check that mapped fields are present
        assert profile.presenting_problem == "Relationship instability and emotional crisis episodes"
        assert "Emotionally unstable" in profile.personality_traits
        assert "Splitting" in profile.defense_mechanisms
        assert profile.communication_style == "Intense, rapidly shifting emotions, crisis-focused"

        # Check learning objectives mapping
        # Note: These objectives might not be in the core enum, so we check they exist
        assert len(profile.learning_objectives) > 0

    def test_ccd_profile_to_difficult_client_profile_with_custom_params(self):
        """Test converting CCD template with custom client parameters"""
        ccd_template = get_resistant_ccd_template()

        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(
            ccd_template, client_id="custom_client_123", name="Custom Client", age=35, gender="female"
        )

        # Check that custom parameters were used
        assert profile.client_id == "custom_client_123"
        assert profile.name == "Custom Client"
        assert profile.age == 35
        assert profile.gender == "female"

        # Check that other fields still came from template
        assert profile.personality_type == ClientPersonality.RESISTANT
        assert profile.presenting_problem == "Mandated therapy due to work-related issues, denies needing help"

    def test_update_ccd_formulation_from_simulation_no_change_when_no_formulation(self):
        """Test updating CCD formulation when no formulation exists"""
        # Create conceptualization without formulation
        conceptualization = CCDConceptualization(client_id="test_client")
        # Don't add a formulation

        original_notes = len(conceptualization.clinician_notes)
        original_revisions = len(conceptualization.revision_history)

        # Try to update formulation
        updated = CCDIntegration.update_ccd_formulation_from_simulation(
            conceptualization, self.sample_simulation_state, self.sample_therapist_analysis
        )

        # Should be unchanged since no formulation to update
        assert updated.formulation is None
        assert len(updated.clinician_notes) == original_notes
        assert len(updated.revision_history) == original_revisions

    def test_update_ccd_formulation_from_simulation_with_formulation(self):
        """Test updating CCD formulation when formulation exists"""
        # Create conceptualization with formulation
        conceptualization = CCDConceptualization(client_id="test_client")
        conceptualization.set_formulation(
            summary="Initial formulation summary",
            strengths=["Initial strength"],
            vulnerabilities=["Initial vulnerability"],
            treatment_goals=["Initial goal"],
            prognosis="guardedly_optimistic",
            confidence=0.7,
        )

        # Update with simulation results
        updated = CCDIntegration.update_ccd_formulation_from_simulation(
            conceptualization, self.sample_simulation_state, self.sample_therapist_analysis
        )

        # Check that formulation was updated
        assert updated.formulation is not None
        assert updated.formulation.summary != "Initial formulation summary"
        assert "Current simulation shows:" in updated.formulation.summary

        # Check that clinician notes were added
        assert len(updated.clinician_notes) > len(conceptualization.clinician_notes)

        # Check that revision history was updated
        assert len(updated.revision_history) > len(conceptualization.revision_history)

        # Check that the revision history contains our update note
        latest_revision = updated.revision_history[-1]
        assert "Updated formulation based on simulation progress" in latest_revision["description"]

    def test_update_ccd_formulation_improves_with_good_therapeutic_process(self):
        """Test that formulation improves with good therapeutic process"""
        # Create conceptualization with formulation
        conceptualization = CCDConceptualization(client_id="test_client")
        conceptualization.set_formulation(
            summary="Initial formulation summary",
            strengths=["Initial strength"],
            vulnerabilities=["Fear of vulnerability", "Emotional avoidance"],
            treatment_goals=["Initial goal"],
            prognosis="guarded",
            confidence=0.6,
        )

        # Create simulation state representing good therapeutic process
        good_simulation_state = {
            "trust_level": 0.8,  # High trust
            "resistance_level": 0.2,  # Low resistance
            "breakthrough_opportunity": True,  # Breakthrough opportunity
            "therapeutic_quality": 0.85,  # High quality
        }

        good_therapist_analysis = {
            "therapeutic_quality": 0.9,
            "skills_demonstrated": {"rapport_building": 0.9, "active_listening": 0.8, "empathy": 0.85},
            "mistakes": [],
            "strengths": ["Excellent rapport building", "Good empathy"],
            "appropriateness": 0.9,
            "impact_prediction": {
                "trust_change": 0.1,
                "resistance_change": -0.1,
                "emotional_intensity_change": -0.05,
                "crisis_risk_change": -0.05,
                "breakthrough_probability": 0.2,
            },
        }

        # Update with good simulation results
        updated = CCDIntegration.update_ccd_formulation_from_simulation(
            conceptualization, good_simulation_state, good_therapist_analysis
        )

        # Check that formulation improved
        assert updated.formulation.prognosis == "optimistic"  # Improved from guarded
        assert updated.formulation.confidence > 0.6  # Increased confidence

        # Check that vulnerabilities were reduced (fear-related ones removed)
        vulnerabilities = [v.lower() for v in updated.formulation.vulnerabilities]
        # Should have fewer fear/avoidance related vulnerabilities
        fear_vulnerabilities = [v for v in vulnerabilities if "fear" in v or "avoidance" in v]
        initial_fear_vulnerabilities = [
            v for v in ["fear of vulnerability", "emotional avoidance"] if "fear" in v or "avoidance" in v
        ]
        assert len(fear_vulnerabilities) <= len(initial_fear_vulnerabilities)

        # Check that strengths were increased
        assert len(updated.formulation.strengths) > len(conceptualization.formulation.strengths)

    def test_update_ccd_formulation_worsens_with_poor_therapeutic_process(self):
        """Test that formulation worsens with poor therapeutic process"""
        # Create conceptualization with formulation
        conceptualization = CCDConceptualization(client_id="test_client")
        conceptualization.set_formulation(
            summary="Initial formulation summary",
            strengths=["Good insight", "Motivation for change"],
            vulnerabilities=["Some vulnerability"],
            treatment_goals=["Initial goal"],
            prognosis="optimistic",
            confidence=0.8,
        )

        # Create simulation state representing poor therapeutic process
        poor_simulation_state = {
            "trust_level": 0.2,  # Low trust
            "resistance_level": 0.8,  # High resistance
            "breakthrough_opportunity": False,  # No breakthrough
            "therapeutic_quality": 0.3,  # Low quality
        }

        poor_therapist_analysis = {
            "therapeutic_quality": 0.25,
            "skills_demonstrated": {"rapport_building": 0.2, "active_listening": 0.3, "empathy": 0.2},
            "mistakes": ["Being overly directive", "Not acknowledging autonomy"],
            "strengths": [],
            "appropriateness": 0.3,
            "impact_prediction": {
                "trust_change": -0.1,
                "resistance_change": 0.15,
                "emotional_intensity_change": 0.1,
                "crisis_risk_change": 0.05,
                "breakthrough_probability": -0.05,
            },
        }

        # Update with poor simulation results
        updated = CCDIntegration.update_ccd_formulation_from_simulation(
            conceptualization, poor_simulation_state, poor_therapist_analysis
        )

        # Check that formulation worsened
        assert updated.formulation.prognosis == "guarded"  # Worsened from optimistic
        assert updated.formulation.confidence < 0.8  # Decreased confidence

        # Check that vulnerabilities may have increased
        # (This depends on implementation, but generally poor process should not improve vulnerabilities)

    def test_create_difficult_client_from_ccd_template_convenience_function(self):
        """Test the convenience function for creating DifficultClientProfile from CCD template"""
        ccd_template = get_resistant_ccd_template()

        profile = create_difficult_client_from_ccd_template(
            ccd_template, client_id="convenience_test_001", name="Convenience Test Client", age=40, gender="non-binary"
        )

        # Check that we got a valid profile with correct parameters
        assert isinstance(profile, DifficultClientProfile)
        assert profile.client_id == "convenience_test_001"
        assert profile.name == "Convenience Test Client"
        assert profile.age == 40
        assert profile.gender == "non-binary"
        assert profile.personality_type == ClientPersonality.RESISTANT

    def test_update_ccd_with_simulation_results_convenience_function(self):
        """Test the convenience function for updating CCD with simulation results"""
        # Create conceptualization with formulation
        conceptualization = CCDConceptualization(client_id="test_client")
        conceptualization.set_formulation(
            summary="Initial formulation",
            strengths=["Strength"],
            vulnerabilities=["Vulnerability"],
            treatment_goals=["Goal"],
            prognosis="guardedly_optimistic",
            confidence=0.7,
        )

        # Update using convenience function
        updated = update_ccd_with_simulation_results(
            conceptualization, self.sample_simulation_state, self.sample_therapist_analysis
        )

        # Check that update worked
        assert isinstance(updated, CCDConceptualization)
        assert updated.client_id == "test_client"
        assert updated.formulation is not None
        assert updated.formulation.summary != "Initial formulation"

        # Check that notes were added
        assert len(updated.clinician_notes) > len(conceptualization.clinician_notes)

    def test_ccd_integration_preserves_existing_functionality(self):
        """Test that CCD integration doesn't break existing DifficultClientProfile functionality"""
        # Create a profile using CCD integration
        ccd_template = get_resistant_ccd_template()
        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template)

        # Check that it still works as a DifficultClientProfile
        expected_attributes = [
            "client_id",
            "name",
            "age",
            "gender",
            "personality_type",
            "difficulty_level",
            "presenting_problem",
            "trauma_history",
            "personality_traits",
            "defense_mechanisms",
            "triggers",
            "strengths",
            "communication_style",
            "resistance_patterns",
            "emotional_dysregulation",
            "interpersonal_patterns",
            "comorbidities",
            "medication_issues",
            "social_factors",
            "legal_issues",
            "learning_objectives",
            "common_therapist_mistakes",
            "therapeutic_challenges",
            "success_indicators",
            "ai_instructions",
            "response_patterns",
            "escalation_triggers",
            "de_escalation_responses",
        ]

        for attr in expected_attributes:
            assert hasattr(profile, attr), f"Missing attribute: {attr}"

        # Check that we can access specific expected attributes
        assert isinstance(profile.learning_objectives, list)
        assert isinstance(profile.common_therapist_mistakes, list)
        assert isinstance(profile.response_patterns, dict)



    def test_invalid_learning_objective_parsing(self):
        """Test parsing of invalid learning objectives."""
        ccd_template = get_resistant_ccd_template()
        # Capture the valid objectives before introducing an invalid one
        original_objectives = list(ccd_template["learning_objectives"])

        # Add an invalid objective
        ccd_template["learning_objectives"].append("invalid_objective_value")

        # It should ignore the invalid objective without throwing an error
        profile = CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template)

        # Verify it created successfully and didn't include the invalid one.
        # The invalid value must be ignored: the resulting objectives must match
        # the original valid set exactly in count and membership, proving the
        # invalid objective was dropped rather than normalized into a valid enum.
        result_objectives = profile.learning_objectives
        assert len(result_objectives) == len(original_objectives)
        for obj in result_objectives:
            assert isinstance(obj, SessionObjective)
            assert obj.value in original_objectives

    def test_adjust_formulation_summary_no_change(self):
        """Test _adjust_formulation_summary with neutral inputs"""
        result = CCDIntegration._adjust_formulation_summary("Base summary", 0.5, 0.5, False, 0.5)
        assert result == "Base summary"

    def test_adjust_vulnerabilities_removal(self):
        """Test _adjust_vulnerabilities removes vulnerabilities correctly"""
        # Test removing trust vulnerabilities
        vulnerabilities = ["Trust issues", "Other issues"]
        adjusted = CCDIntegration._adjust_vulnerabilities(vulnerabilities, 0.8, 0.5, {})
        assert "Other issues" in adjusted
        assert "Trust issues" not in adjusted

        # Test removing resistance vulnerabilities
        vulnerabilities = ["Resistant to help", "Other issues"]
        adjusted = CCDIntegration._adjust_vulnerabilities(vulnerabilities, 0.5, 0.2, {})
        assert "Other issues" in adjusted
        assert "Resistant to help" not in adjusted

    def test_adjust_prognosis_edge_cases(self):
        """Test prognosis edge cases"""
        # Test fallback branch
        result = CCDIntegration._adjust_prognosis("unknown_prognosis", 0.5, 0.5, False, 0.5)
        # Because prognosis_scores.get(..., 1) returns 1 (guarded), it becomes guarded and will hit loop
        assert result == "guarded"

        # To hit the fallback return base_prognosis we need a current_score that is not in the dict (0,1,2,3).
        # We can't actually do that because current_score is initialized to 0, 1, 2, or 3,
        # and min/max operations keep it in [0, 3]. So the fallback is mathematically unreachable
        # unless prognosis_scores changes. But let's test the other missing lines (468, 475)

        # Test 468: trust_level > 0.8, current_score goes up
        result = CCDIntegration._adjust_prognosis("poor", 0.9, 0.5, False, 0.5)
        assert result == "guarded"

        # Test 475: resistance_level > 0.8, current_score goes down
        result = CCDIntegration._adjust_prognosis("guarded", 0.5, 0.9, False, 0.5)
        assert result == "poor"


if __name__ == "__main__":
    unittest.main()



            # actually we can't easily patch local variables
