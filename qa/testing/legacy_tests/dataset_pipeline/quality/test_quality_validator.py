import unittest

from ai.tools.utilities.pipelines.schemas.conversation_schema import Conversation
from ai.tools.utilities.pipelines.quality.quality_validator import QualityValidator


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
        self.high_quality_conv.add_message("therapist", "I hear you. I think it is important. Yes.")
        self.high_quality_conv.add_message("client", "It feels hard. I feel fear. I think.")
        self.high_quality_conv.add_message("therapist", "I feel for you. I believe we can help. Yes.")
        self.high_quality_conv.add_message("client", "Yes, thank you. I appreciate it. I feel better.")

        # Low quality (single speaker, repetitive, junk)
        self.low_quality_conv = Conversation(conversation_id="low_001")
        self.low_quality_conv.add_message("user", "test test test test test test")
        self.low_quality_conv.add_message("user", "placeholder content")

    def test_validate_structure(self):
        """Test basic structure validation (length, speakers)."""
        issues = []
        strengths = []
        # _validate_structure returns a score and populates issues/strengths
        score = self.validator._validate_structure(self.high_quality_conv, issues, strengths)
        assert score >= 0.7
        assert any("speakers" in s.lower() for s in strengths)

        low_issues = []
        low_strengths = []
        low_score = self.validator._validate_structure(self.low_quality_conv, low_issues, low_strengths)
        assert low_score <= 0.8
        assert any("single speaker" in i.lower() for i in low_issues)

    def test_validate_content(self):
        """Test content quality (therapeutic patterns, length)."""
        issues = []
        strengths = []
        score = self.validator._validate_content(self.high_quality_conv, issues, strengths)
        assert score >= 0.9
        assert any("therapeutic" in s.lower() for s in strengths)

    def test_validate_coherence(self):
        """Test conversation coherence and flow."""
        issues = []
        strengths = []
        score = self.validator._validate_coherence(self.high_quality_conv, issues, strengths)
        assert score >= 0.6
        assert any("flow" in s.lower() for s in strengths)

    def test_validate_authenticity(self):
        """Test natural language vs formal/repetitive language."""
        issues = []
        strengths = []
        score = self.validator._validate_authenticity(self.high_quality_conv, issues, strengths)
        assert score >= 0.8

        # Repetitive case
        rep_conv = Conversation(conversation_id="rep_001")
        rep_conv.add_message("user", "Repetitive sentence here.")
        rep_conv.add_message("assistant", "Repetitive sentence here.")
        rep_conv.add_message("user", "Repetitive sentence here.")
        rep_issues = []
        rep_strengths = []
        rep_score = self.validator._validate_authenticity(rep_conv, rep_issues, rep_strengths)
        assert rep_score <= 0.85
        assert any("repetitive" in i.lower() for i in rep_issues)

    def test_full_validation(self):
        """Test end-to-end validation resulting in QualityResult."""
        result = self.validator.validate_conversation(self.high_quality_conv)
        assert result.conversation_id == "high_001"
        assert result.overall_score > 0.6
        assert result.coherence_score > 0.6
        assert len(result.strengths) > 0


if __name__ == "__main__":
    unittest.main()
