#!/usr/bin/env python3
"""
Test suite for PersonalityAdapter - Task 6.11
"""


import pytest
from personality_adapter import (
    AdaptedResponse,
    CommunicationStyle,
    PersonalityAdaptation,
    PersonalityAdapter,
    TherapeuticApproach,
)


class TestPersonalityAdapter:
    """Test cases for PersonalityAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = PersonalityAdapter()

        self.test_conversation = {
            "conversation_id": "test_001",
            "turns": [
                {
                    "speaker": "user",
                    "content": "I like to have everything organized and planned out. I prefer working alone and thinking things through carefully before making decisions. I get stressed when things are chaotic."
                },
                {
                    "speaker": "assistant",
                    "content": "I understand you prefer structure and organization."
                }
            ]
        }

    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter is not None
        assert self.adapter.adaptation_rules is not None
        assert self.adapter.communication_patterns is not None
        assert self.adapter.therapeutic_mappings is not None

    def test_analyze_personality_for_adaptation(self):
        """Test personality analysis for adaptation."""
        adaptation = self.adapter.analyze_personality_for_adaptation(self.test_conversation)

        assert isinstance(adaptation, PersonalityAdaptation)
        assert adaptation.personality_profile is not None
        assert isinstance(adaptation.communication_style, CommunicationStyle)
        assert isinstance(adaptation.therapeutic_approach, TherapeuticApproach)
        assert 0 <= adaptation.adaptation_confidence <= 1
        assert len(adaptation.recommendations) > 0

    def test_adapt_response(self):
        """Test response adaptation."""
        adaptation = self.adapter.analyze_personality_for_adaptation(self.test_conversation)
        original_response = "Maybe you could try to be more flexible with your schedule."

        adapted_response = self.adapter.adapt_response(original_response, adaptation)

        assert isinstance(adapted_response, AdaptedResponse)
        assert adapted_response.original_response == original_response
        assert adapted_response.adapted_response != original_response
        assert len(adapted_response.personality_factors) > 0
        assert 0 <= adapted_response.confidence_score <= 1

    def test_communication_style_adaptation(self):
        """Test different communication style adaptations."""
        # Test direct communication
        response = "Maybe you should consider therapy."
        direct_adapted = self.adapter._make_more_direct(response)
        assert "maybe" not in direct_adapted.lower()

        # Test supportive communication
        supportive_adapted = self.adapter._make_more_supportive(response)
        assert any(word in supportive_adapted.lower() for word in ["understand", "normal", "strength"])

        # Test analytical communication
        analytical_adapted = self.adapter._make_more_analytical(response)
        assert any(phrase in analytical_adapted.lower() for phrase in ["examine", "analyze", "systematically"])

    def test_personality_trait_mapping(self):
        """Test personality trait to adaptation mapping."""
        # Test high conscientiousness conversation
        high_c_conversation = {
            "turns": [
                {
                    "speaker": "user",
                    "content": "I always make detailed plans and stick to my schedule. I believe in hard work and achieving my goals through discipline and organization."
                }
            ]
        }

        adaptation = self.adapter.analyze_personality_for_adaptation(high_c_conversation)

        # Should prefer structured approaches
        assert adaptation.structure_preference in ["high", "moderate"]
        assert adaptation.therapeutic_approach in [TherapeuticApproach.CBT, TherapeuticApproach.BEHAVIORAL]

    def test_error_handling(self):
        """Test error handling in adaptation."""
        # Test with invalid conversation
        invalid_conversation = {"invalid": "data"}

        adaptation = self.adapter.analyze_personality_for_adaptation(invalid_conversation)

        # Should return default adaptation
        assert isinstance(adaptation, PersonalityAdaptation)
        assert adaptation.adaptation_confidence <= 0.5

    def test_adaptation_summary(self):
        """Test adaptation summary generation."""
        adaptation = self.adapter.analyze_personality_for_adaptation(self.test_conversation)
        summary = self.adapter.get_adaptation_summary(adaptation)

        assert isinstance(summary, dict)
        assert "personality_scores" in summary
        assert "adaptation_settings" in summary
        assert "confidence" in summary
        assert "recommendations" in summary

    def test_default_adaptation(self):
        """Test default adaptation fallback."""
        default_adaptation = self.adapter._get_default_adaptation()

        assert isinstance(default_adaptation, PersonalityAdaptation)
        assert default_adaptation.communication_style == CommunicationStyle.SUPPORTIVE
        assert default_adaptation.therapeutic_approach == TherapeuticApproach.HUMANISTIC
        assert len(default_adaptation.recommendations) > 0

    def test_emotional_support_adjustment(self):
        """Test emotional support level adjustments."""
        original_response = "You need to work on this issue."

        # Test high emotional support
        high_support_response = self.adapter._increase_emotional_support(original_response)
        assert "not alone" in high_support_response.lower() or "support" in high_support_response.lower()

    def test_detail_level_adjustment(self):
        """Test detail level adjustments."""
        original_response = "Try meditation."

        # Test high detail level
        detailed_response = self.adapter._add_more_detail(original_response)
        assert len(detailed_response) > len(original_response)

        # Test low detail level (simplification)
        complex_response = "You should consider implementing a comprehensive mindfulness-based stress reduction program that incorporates various meditation techniques and breathing exercises."
        simplified_response = self.adapter._simplify_response(complex_response)
        assert len(simplified_response.split()) < len(complex_response.split())


def test_personality_adapter_integration():
    """Integration test for personality adapter."""
    adapter = PersonalityAdapter()

    # Test full workflow
    conversation = {
        "conversation_id": "integration_test",
        "turns": [
            {
                "speaker": "user",
                "content": "I'm very outgoing and love being around people. I make decisions quickly and prefer to talk through problems with others. I'm optimistic and energetic."
            }
        ]
    }

    # Analyze personality
    adaptation = adapter.analyze_personality_for_adaptation(conversation)

    # Should detect extraverted personality
    assert adaptation.personality_profile.extraversion > 0.5
    assert adaptation.communication_style in [CommunicationStyle.COLLABORATIVE, CommunicationStyle.DIRECT]

    # Adapt response
    original_response = "You might want to spend some time alone to reflect on this."
    adapted_response = adapter.adapt_response(original_response, adaptation)

    # Should adapt for extraverted preference
    assert adapted_response.adapted_response != original_response
    assert adapted_response.confidence_score > 0


if __name__ == "__main__":
    pytest.main([__file__])
