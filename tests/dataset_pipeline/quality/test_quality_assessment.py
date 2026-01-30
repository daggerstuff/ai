import unittest

from ai.dataset_pipeline.quality.quality_assessment_framework import (
    QualityAssessmentFramework,
    QualityTier,
)


class TestQualityAssessment(unittest.TestCase):
    """
    Test suite for Task 5.7.1.1: Component Unit Tests (Quality focus)
    Validates hierarchical quality assessment framework.
    """

    def setUp(self):
        """Set up the assessment framework."""
        self.framework = QualityAssessmentFramework()

        # Sample high-quality therapeutic conversation
        self.high_quality_conv = {
            "conversation_id": "test_high_001",
            "messages": [
                {"role": "therapist", "content": "Hello, how are you feeling today?"},
                {
                    "role": "client",
                    "content": "I've been feeling quite anxious lately, especially at night.",
                },
                {
                    "role": "therapist",
                    "content": "I understand. Anxiety at night can be very difficult. Can you tell me more about what those feelings are like for you?",
                },
                {
                    "role": "client",
                    "content": "It feels like my heart starts racing and I can't stop thinking about all the things that could go wrong.",
                },
            ],
        }

        # Sample low-quality/broken conversation
        self.low_quality_conv = {
            "conversation_id": "test_low_001",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "asdf qwerty test placeholder"},
                {"role": "user", "content": "???"},
            ],
        }

    def test_assess_therapeutic_relevance(self):
        """Test detection of therapeutic keywords and relevance."""
        # Note: _assess_therapeutic_relevance takes content as string
        content = "I'm feeling depressed and anxious about my trauma history. It is affecting my mental health and relationships."
        score = self.framework._assess_therapeutic_relevance(content)
        # With improved scoring, this should be higher
        self.assertGreater(score, 0.2)

        junk_content = "123 456 789 000"
        junk_score = self.framework._assess_therapeutic_relevance(junk_content)
        self.assertLess(junk_score, 0.2)

    def test_assess_safety_compliance(self):
        """Test safety compliance scoring."""
        safe_content = "I had a good day going for a walk in the park."
        safe_score = self.framework._assess_safety_compliance(safe_content)
        self.assertEqual(safe_score, 1.0)

        # Testing a scenario that might be considered a safety risk (should score lower)
        risk_content = "I want to hurt myself, I have no reason to go on."
        risk_score = self.framework._assess_safety_compliance(risk_content)
        self.assertLess(risk_score, 1.0)

    def test_linguistic_quality(self):
        """Test linguistic quality assessment (grammar, length, repeated chars)."""
        good_text = "This is a well-structured sentence with appropriate length. It communicates clearly and effectively."
        good_score = self.framework._assess_linguistic_quality(good_text)
        self.assertGreater(good_score, 0.4)

        bad_text = "a" * 1000  # repetitive or very long without structure
        bad_score = self.framework._assess_linguistic_quality(bad_text)
        self.assertLess(bad_score, 0.5)

    def test_overall_assessment(self):
        """Test end-to-end assessment of a conversation."""
        # Convert our sample to the format expected by the framework if necessary
        # The framework seems to expect 'messages' or 'conversations'

        # We need to check if the framework uses '.messages' or '["messages"]'
        # Looking at _extract_content in the outline, it takes the dict.

        assessment = self.framework.assess_conversation(self.high_quality_conv)

        self.assertEqual(assessment.conversation_id, "test_high_001")
        self.assertGreater(assessment.metrics.overall_score, 0.7)
        self.assertIsInstance(assessment.assigned_tier, QualityTier)
        self.assertTrue(len(assessment.quality_strengths) > 0)

    def test_low_quality_tier_assignment(self):
        """Test that low-quality conversations are assigned to lower tiers."""
        assessment = self.framework.assess_conversation(self.low_quality_conv)

        # Depending on specific thresholds in QualityTier:
        # PRIORITY (0.99), PROFESSIONAL (0.90), CORE (0.80), BASIC (0.70), ARCHIVE (0.60)
        # Low quality should likely be ARCHIVE or FAIL (if there is one)
        self.assertTrue(assessment.metrics.overall_score < 0.7)
        self.assertEqual(assessment.assigned_tier.tier_name, "archive")


if __name__ == "__main__":
    unittest.main()
