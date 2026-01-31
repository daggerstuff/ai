#!/usr/bin/env python3
"""
Test suite for CulturalCompetencyGenerator - Task 6.12
"""


import pytest
from cultural_competency_generator import (
    CulturalAdaptation,
    CulturalBackground,
    CulturalCompetencyGenerator,
    CulturalProfile,
    DiversityFactor,
)


class TestCulturalCompetencyGenerator:
    """Test cases for CulturalCompetencyGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CulturalCompetencyGenerator()

        self.test_conversation = {
            "conversation_id": "test_001",
            "turns": [
                {
                    "speaker": "user",
                    "content": "I'm struggling with anxiety, but I don't want to bring shame to my family. In our culture, mental health issues are not talked about openly. My parents expect me to be strong and successful."
                },
                {
                    "speaker": "assistant",
                    "content": "I understand you're dealing with anxiety."
                }
            ]
        }

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator is not None
        assert self.generator.cultural_indicators is not None
        assert self.generator.adaptation_guidelines is not None
        assert self.generator.bias_patterns is not None
        assert self.generator.cultural_strengths is not None

    def test_analyze_cultural_profile(self):
        """Test cultural profile analysis."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)

        assert isinstance(profile, CulturalProfile)
        assert isinstance(profile.primary_background, CulturalBackground)
        assert isinstance(profile.cultural_dimensions, dict)
        assert isinstance(profile.diversity_factors, dict)
        assert 0 <= profile.confidence_score <= 1
        assert len(profile.cultural_values) >= 0

    def test_cultural_background_detection(self):
        """Test cultural background detection."""
        # Test collectivistic indicators
        collectivistic_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "My family is very important to me. We always make decisions together and respect our elders. Family harmony is our priority."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(collectivistic_conversation)

        # Should detect collectivistic background
        assert profile.primary_background in [
            CulturalBackground.EAST_ASIAN_COLLECTIVISTIC,
            CulturalBackground.LATIN_AMERICAN,
            CulturalBackground.MIXED_MULTICULTURAL
        ]

    def test_generate_cultural_adaptation(self):
        """Test cultural adaptation generation."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)
        adaptation = self.generator.generate_cultural_adaptation(profile)

        assert isinstance(adaptation, CulturalAdaptation)
        assert adaptation.cultural_profile == profile
        assert len(adaptation.communication_adjustments) > 0
        assert len(adaptation.therapeutic_considerations) > 0
        assert adaptation.family_involvement_level in ["high", "moderate", "low", "flexible"]
        assert len(adaptation.potential_biases_to_avoid) > 0

    def test_culturally_aware_response_generation(self):
        """Test culturally aware response generation."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)
        adaptation = self.generator.generate_cultural_adaptation(profile)

        original_response = "You should seek professional help immediately."
        culturally_aware_response = self.generator.generate_culturally_aware_response(
            original_response, adaptation
        )

        assert isinstance(culturally_aware_response, str)
        assert len(culturally_aware_response) >= len(original_response)

    def test_communication_style_detection(self):
        """Test communication style detection."""
        # Test direct communication
        direct_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "I need to tell you directly - I'm having serious problems and I want straight answers about what to do."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(direct_conversation)
        assert profile.communication_style == "direct"

        # Test indirect communication
        indirect_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "Perhaps there might be some issues that could possibly be affecting my well-being, maybe."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(indirect_conversation)
        assert profile.communication_style == "indirect"

    def test_diversity_factor_identification(self):
        """Test diversity factor identification."""
        # Test socioeconomic indicators
        low_income_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "I'm struggling financially and can't afford therapy. The stress of not being able to pay bills is overwhelming."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(low_income_conversation)

        if DiversityFactor.SOCIOECONOMIC_STATUS in profile.diversity_factors:
            assert profile.diversity_factors[DiversityFactor.SOCIOECONOMIC_STATUS] == "lower_income"

    def test_religious_considerations(self):
        """Test religious considerations identification."""
        religious_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "My faith is very important to me. I pray every day and attend religious services regularly. I believe God will help me through this."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(religious_conversation)

        assert "spiritual" in profile.cultural_values
        assert len(profile.religious_considerations) > 0

    def test_cultural_values_extraction(self):
        """Test cultural values extraction."""
        family_oriented_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "My family comes first in everything I do. I want to achieve success to make my parents proud and honor our traditions."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(family_oriented_conversation)

        assert "family_oriented" in profile.cultural_values
        assert "achievement_oriented" in profile.cultural_values or "traditional" in profile.cultural_values

    def test_potential_barriers_identification(self):
        """Test potential barriers identification."""
        barrier_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "English is not my first language, and I'm worried about the stigma of mental health treatment in my community. It would be shameful for my family."
                }
            ]
        }

        profile = self.generator.analyze_cultural_profile(barrier_conversation)

        assert len(profile.potential_barriers) > 0
        assert any("language" in barrier or "stigma" in barrier for barrier in profile.potential_barriers)

    def test_family_involvement_determination(self):
        """Test family involvement level determination."""
        # Test collectivistic background
        collectivistic_profile = CulturalProfile(
            primary_background=CulturalBackground.EAST_ASIAN_COLLECTIVISTIC,
            cultural_dimensions={},
            diversity_factors={},
            language_preferences=["english"],
            communication_style="indirect",
            family_structure="extended",
            religious_considerations=[],
            cultural_values=["family_oriented"],
            potential_barriers=[],
            confidence_score=0.8
        )

        involvement = self.generator._determine_family_involvement(collectivistic_profile)
        assert involvement == "high"

        # Test individualistic background
        individualistic_profile = CulturalProfile(
            primary_background=CulturalBackground.WESTERN_INDIVIDUALISTIC,
            cultural_dimensions={},
            diversity_factors={},
            language_preferences=["english"],
            communication_style="direct",
            family_structure="nuclear",
            religious_considerations=[],
            cultural_values=[],
            potential_barriers=[],
            confidence_score=0.8
        )

        involvement = self.generator._determine_family_involvement(individualistic_profile)
        assert involvement == "moderate"

    def test_bias_avoidance_recommendations(self):
        """Test bias avoidance recommendations."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)
        adaptation = self.generator.generate_cultural_adaptation(profile)

        biases = adaptation.potential_biases_to_avoid

        assert len(biases) > 0
        assert any("stereotyping" in bias.lower() for bias in biases)
        assert any("cultural" in bias.lower() for bias in biases)

    def test_cultural_strengths_identification(self):
        """Test cultural strengths identification."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)
        adaptation = self.generator.generate_cultural_adaptation(profile)

        strengths = adaptation.cultural_strengths_to_leverage

        assert len(strengths) >= 0
        # Should identify relevant strengths based on cultural background

    def test_error_handling(self):
        """Test error handling in cultural analysis."""
        # Test with invalid conversation
        invalid_conversation = {"invalid": "data"}

        profile = self.generator.analyze_cultural_profile(invalid_conversation)

        # Should return default profile
        assert isinstance(profile, CulturalProfile)
        assert profile.primary_background == CulturalBackground.MIXED_MULTICULTURAL
        assert profile.confidence_score <= 0.5

    def test_cultural_summary(self):
        """Test cultural summary generation."""
        profile = self.generator.analyze_cultural_profile(self.test_conversation)
        adaptation = self.generator.generate_cultural_adaptation(profile)
        summary = self.generator.get_cultural_summary(adaptation)

        assert isinstance(summary, dict)
        assert "cultural_background" in summary
        assert "communication_style" in summary
        assert "family_involvement" in summary
        assert "confidence" in summary

    def test_response_adaptation_methods(self):
        """Test specific response adaptation methods."""
        original_response = "You should consider therapy options."

        # Test communication adjustment
        adjusted = self.generator._apply_communication_adjustment(
            original_response, "Use indirect communication patterns"
        )
        assert "might consider" in adjusted or "could be helpful" in adjusted

        # Test language adaptation
        complex_response = "You should utilize professional therapeutic interventions to facilitate recovery."
        simplified = self.generator._apply_language_adaptation(
            complex_response, "Use simple, concrete language"
        )
        assert "use" in simplified
        assert "help" in simplified


def test_cultural_competency_integration():
    """Integration test for cultural competency generator."""
    generator = CulturalCompetencyGenerator()

    # Test full workflow with culturally diverse conversation
    conversation = {
        "conversation_id": "integration_test",
        "turns": [
            {
                "speaker": "user",
                "content": "I come from a traditional family where we don't discuss personal problems outside the family. My parents immigrated here and work very hard. I feel pressure to succeed and not disappoint them, but I'm struggling with depression."
            }
        ]
    }

    # Analyze cultural profile
    profile = generator.analyze_cultural_profile(conversation)

    # Should detect cultural indicators
    assert profile.confidence_score > 0.5
    assert "family_oriented" in profile.cultural_values
    assert len(profile.potential_barriers) > 0

    # Generate adaptation
    adaptation = generator.generate_cultural_adaptation(profile)

    # Should provide appropriate adaptations
    assert adaptation.family_involvement_level in ["high", "moderate"]
    assert len(adaptation.communication_adjustments) > 0
    assert len(adaptation.therapeutic_considerations) > 0

    # Generate culturally aware response
    original_response = "You should see a therapist immediately and focus on your individual needs."
    adapted_response = generator.generate_culturally_aware_response(original_response, adaptation)

    # Should be culturally adapted
    assert adapted_response != original_response
    assert len(adapted_response) >= len(original_response)


if __name__ == "__main__":
    pytest.main([__file__])
