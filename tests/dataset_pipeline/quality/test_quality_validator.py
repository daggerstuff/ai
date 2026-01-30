import unittest

from ai.dataset_pipeline.quality.conversation_schema import Conversation
from ai.dataset_pipeline.quality.quality_validator import QualityValidator


class TestQualityValidator(unittest.TestCase):
    """
    Test suite for Task 5.7.1.1: Component Unit Tests (Quality focus)
    Validates real-time quality validator metrics.
    """

    def setUp(self):
        """Set up the quality validator."""
        self.validator = QualityValidator()

        # High quality conversation (messages alternating, therapeutic content)
        self.high_quality_conv = Conversation(conversation_id="high_001")
        self.high_quality_conv.add_message(
            "therapist", "I hear you. I think it is important. Yes."
        )
        self.high_quality_conv.add_message(
            "client", "It feels hard. I feel fear. I think."
        )
        self.high_quality_conv.add_message(
            "therapist", "I feel for you. I believe we can help. Yes."
        )
        self.high_quality_conv.add_message(
            "client", "Yes, thank you. I appreciate it. I feel better."
        )

        # Low quality (single speaker, repetitive, junk)
        self.low_quality_conv = Conversation(conversation_id="low_001")
        self.low_quality_conv.add_message("user", "test test test test test test")
        self.low_quality_conv.add_message("user", "placeholder content")

    def test_validate_structure(self):
        """Test basic structure validation (length, speakers)."""
        issues = []
        strengths = []
        # _validate_structure returns a score and populates issues/strengths
        score = self.validator._validate_structure(
            self.high_quality_conv, issues, strengths
        )
        self.assertGreaterEqual(score, 0.7)
        self.assertTrue(any("speakers" in s.lower() for s in strengths))

        low_issues = []
        low_strengths = []
        low_score = self.validator._validate_structure(
            self.low_quality_conv, low_issues, low_strengths
        )
        self.assertLessEqual(low_score, 0.8)
        self.assertTrue(any("single speaker" in i.lower() for i in low_issues))

    def test_validate_content(self):
        """Test content quality (therapeutic patterns, length)."""
        issues = []
        strengths = []
        score = self.validator._validate_content(
            self.high_quality_conv, issues, strengths
        )
        self.assertGreaterEqual(score, 0.9)
        self.assertTrue(any("therapeutic" in s.lower() for s in strengths))

    def test_validate_coherence(self):
        """Test conversation coherence and flow."""
        issues = []
        strengths = []
        score = self.validator._validate_coherence(
            self.high_quality_conv, issues, strengths
        )
        self.assertGreaterEqual(score, 0.6)
        self.assertTrue(any("flow" in s.lower() for s in strengths))

    def test_validate_authenticity(self):
        """Test natural language vs formal/repetitive language."""
        issues = []
        strengths = []
        score = self.validator._validate_authenticity(
            self.high_quality_conv, issues, strengths
        )
        self.assertGreaterEqual(score, 0.8)

        # Repetitive case
        rep_conv = Conversation(conversation_id="rep_001")
        rep_conv.add_message("user", "Repetitive sentence here.")
        rep_conv.add_message("assistant", "Repetitive sentence here.")
        rep_conv.add_message("user", "Repetitive sentence here.")
        rep_issues = []
        rep_strengths = []
        rep_score = self.validator._validate_authenticity(
            rep_conv, rep_issues, rep_strengths
        )
        self.assertLessEqual(rep_score, 0.85)
        self.assertTrue(any("repetitive" in i.lower() for i in rep_issues))

    def test_full_validation(self):
        """Test end-to-end validation resulting in QualityResult."""
        result = self.validator.validate_conversation(self.high_quality_conv)
        self.assertEqual(result.conversation_id, "high_001")
        self.assertGreater(result.overall_score, 0.6)
        self.assertGreater(result.coherence_score, 0.6)
        self.assertTrue(len(result.strengths) > 0)


if __name__ == "__main__":
    unittest.main()
