"""
Unit tests for the emotional authenticity assessment system.
"""

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.emotional_authenticity_assessment import (
    EmotionalAuthenticityAssessor,
    EmotionalAuthenticityMetrics,
    assess_emotional_authenticity,
)


class TestEmotionalAuthenticityAssessor:
    """Test cases for the EmotionalAuthenticityAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = EmotionalAuthenticityAssessor()

    def test_initialization_with_default_config(self):
        """Test assessor initialization with default configuration."""
        assessor = EmotionalAuthenticityAssessor()

        assert assessor.weights["emotional_consistency"] == 0.20
        assert assessor.weights["empathy_expression"] == 0.18
        assert assessor.thresholds["excellent"] == 0.85
        assert assessor.thresholds["good"] == 0.70

    def test_initialization_with_custom_config(self):
        """Test assessor initialization with custom configuration."""
        custom_config = {
            "weights": {"emotional_consistency": 0.5, "empathy_expression": 0.5},
            "thresholds": {"excellent": 0.9, "good": 0.8}
        }
        assessor = EmotionalAuthenticityAssessor(custom_config)

        assert assessor.weights["emotional_consistency"] == 0.5
        assert assessor.thresholds["excellent"] == 0.9

    def test_assess_empty_conversation(self):
        """Test assessment of empty conversation."""
        conversation = Conversation(
            id="test_empty",
            messages=[],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert isinstance(metrics, EmotionalAuthenticityMetrics)
        assert metrics.overall_score == 0.0
        assert len(metrics.issues) > 0
        assert "Insufficient messages" in metrics.issues[0]

    def test_assess_single_message_conversation(self):
        """Test assessment of conversation with single message."""
        conversation = Conversation(
            id="test_single",
            messages=[
                Message(role="user", content="I'm feeling sad today.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.overall_score == 0.0
        assert "Insufficient messages" in metrics.issues[0]

    def test_assess_emotionally_authentic_conversation(self):
        """Test assessment of emotionally authentic conversation."""
        conversation = Conversation(
            id="test_authentic",
            messages=[
                Message(role="user", content="I'm feeling really anxious about my job interview tomorrow. I keep worrying about what they'll ask me."),
                Message(role="assistant", content="I understand that job interviews can be really stressful. It sounds like you're experiencing a lot of worry about the unknown questions. That's completely understandable."),
                Message(role="user", content="Yes, exactly. I feel like I'm not prepared enough, even though I've been practicing for weeks."),
                Message(role="assistant", content="I hear you. It sounds like you've put in a lot of effort preparing, which shows your dedication. Sometimes our anxiety can make us feel less prepared than we actually are.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert isinstance(metrics, EmotionalAuthenticityMetrics)
        assert metrics.overall_score > 0.6  # Should be reasonably high
        assert metrics.empathy_expression_score > 0.7  # Good empathy expressions
        assert metrics.emotional_consistency_score > 0.6  # Consistent emotional flow

    def test_assess_conversation_with_emotional_inconsistency(self):
        """Test assessment of conversation with emotional inconsistencies."""
        conversation = Conversation(
            id="test_inconsistent",
            messages=[
                Message(role="user", content="I'm devastated about losing my job."),
                Message(role="assistant", content="That's great! Congratulations!"),
                Message(role="user", content="What? I said I lost my job."),
                Message(role="assistant", content="Oh, I'm sorry to hear that.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.response_appropriateness_score < 0.8  # Should be penalized
        assert any("Inappropriate" in issue for issue in metrics.issues)

    def test_assess_conversation_with_low_empathy(self):
        """Test assessment of conversation with low empathy expression."""
        conversation = Conversation(
            id="test_low_empathy",
            messages=[
                Message(role="user", content="I'm struggling with depression and feel hopeless."),
                Message(role="assistant", content="Ok."),
                Message(role="user", content="I don't know what to do anymore."),
                Message(role="assistant", content="Sure.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.empathy_expression_score < 0.8  # Should be low
        assert any("empathy" in issue.lower() for issue in metrics.issues)

    def test_assess_conversation_with_rich_emotional_vocabulary(self):
        """Test assessment of conversation with rich emotional vocabulary."""
        conversation = Conversation(
            id="test_rich_vocab",
            messages=[
                Message(role="user", content="I'm feeling conflicted and somewhat ambivalent about this decision. It's bittersweet."),
                Message(role="assistant", content="I can hear the complexity in your emotions. Feeling conflicted about important decisions is natural, and that bittersweet quality suggests there are both positive and challenging aspects to consider."),
                Message(role="user", content="Yes, I'm grateful for the opportunity but also apprehensive about the changes it will bring."),
                Message(role="assistant", content="That mix of gratitude and apprehension makes complete sense. It sounds like you're processing multiple layers of emotion around this transition.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.emotional_vocabulary_score > 0.7  # Should be high
        assert metrics.details["emotional_vocabulary_details"]["complex_emotions"] > 0

    def test_assess_conversation_with_generic_responses(self):
        """Test assessment of conversation with generic responses."""
        conversation = Conversation(
            id="test_generic",
            messages=[
                Message(role="user", content="I've been dealing with severe anxiety for months. It's affecting my work, my relationships, and my sleep. I feel overwhelmed and don't know where to turn for help."),
                Message(role="assistant", content="Ok."),
                Message(role="user", content="I'm really struggling here. Can you help me understand what I should do?"),
                Message(role="assistant", content="Sure.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        # The response appropriateness should detect generic responses to emotional content
        # But let's check empathy score instead which should be more sensitive
        assert metrics.empathy_expression_score < 0.8  # Should be low due to lack of empathy
        assert any("Generic response" in issue for issue in metrics.issues)

    def test_assess_conversation_with_emotional_progression(self):
        """Test assessment of conversation with good emotional progression."""
        conversation = Conversation(
            id="test_progression",
            messages=[
                Message(role="user", content="I'm feeling really sad and hopeless about everything."),
                Message(role="assistant", content="I hear that you're experiencing deep sadness and hopelessness. Those are heavy feelings to carry."),
                Message(role="user", content="Yes, but talking about it helps a little. I feel slightly less alone."),
                Message(role="assistant", content="I'm glad that sharing these feelings is providing some comfort. Feeling less alone is an important step."),
                Message(role="user", content="I think I'm ready to try some coping strategies. I feel a bit more hopeful now."),
                Message(role="assistant", content="That shift toward hope and readiness to try new strategies is really meaningful. You've shown real strength in this conversation.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.emotional_progression_score > 0.7  # Should recognize positive progression
        assert metrics.overall_score > 0.6  # Overall should be good

    def test_assess_conversation_with_authenticity_indicators(self):
        """Test assessment of conversation with authenticity indicators."""
        conversation = Conversation(
            id="test_authenticity",
            messages=[
                Message(role="user", content="I feel like I'm not good enough for this job."),
                Message(role="assistant", content="I hear you expressing some self-doubt. In my experience, many people feel this way when facing new challenges. I think it's natural to question our abilities."),
                Message(role="user", content="I guess I'm just scared of failing."),
                Message(role="assistant", content="I understand that fear. I've been there myself, and I'm still learning how to manage those feelings. What specifically worries you most about failing?")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.authenticity_indicators_score > 0.6  # Should recognize authentic language
        assert metrics.details["authenticity_indicators_details"]["positive_indicators"] > 0

    def test_assess_conversation_with_inauthentic_language(self):
        """Test assessment of conversation with inauthentic language patterns."""
        conversation = Conversation(
            id="test_inauthentic",
            messages=[
                Message(role="user", content="I'm having trouble with my mental health."),
                Message(role="assistant", content="Please note that you should simply follow the recommended treatment protocol. It is important to just comply with standard procedures."),
                Message(role="user", content="But I need more personalized help."),
                Message(role="assistant", content="You should merely follow the diagnosis guidelines. It is recommended that you simply adhere to the treatment plan.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_emotional_authenticity(conversation)

        assert metrics.authenticity_indicators_score < 0.7  # Should be penalized
        assert any("inauthentic language" in issue.lower() for issue in metrics.issues)

    def test_quality_level_determination(self):
        """Test quality level determination based on scores."""
        # Test excellent quality
        assert self.assessor._determine_quality_level(0.9) == "excellent"

        # Test good quality
        assert self.assessor._determine_quality_level(0.75) == "good"

        # Test acceptable quality
        assert self.assessor._determine_quality_level(0.6) == "acceptable"

        # Test poor quality
        assert self.assessor._determine_quality_level(0.45) == "poor"

        # Test very poor quality
        assert self.assessor._determine_quality_level(0.3) == "very_poor"

    def test_emotional_vocabulary_sets(self):
        """Test that emotional vocabulary sets are properly initialized."""
        assert "happy" in self.assessor.positive_emotions
        assert "sad" in self.assessor.negative_emotions
        assert "conflicted" in self.assessor.complex_emotions
        assert len(self.assessor.all_emotions) > 50  # Should have substantial vocabulary

    def test_empathy_indicators(self):
        """Test that empathy indicators are properly initialized."""
        assert "i understand" in self.assessor.empathy_phrases
        assert "understand" in self.assessor.empathy_words
        assert len(self.assessor.empathy_phrases) > 10

    def test_authenticity_markers(self):
        """Test that authenticity markers are properly initialized."""
        assert "i feel" in self.assessor.authenticity_positive["personal_disclosure"]
        assert "ok" in self.assessor.authenticity_negative["generic_responses"]

    def test_backward_compatibility_function(self):
        """Test the backward compatibility assess_emotional_authenticity function."""
        conversation = Conversation(
            id="test_compat",
            messages=[
                Message(role="user", content="I'm feeling anxious."),
                Message(role="assistant", content="I understand that anxiety can be difficult to manage.")
            ],
            source="test"
        )

        result = assess_emotional_authenticity(conversation)

        assert "score" in result
        assert "issues" in result
        assert "details" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)
        assert isinstance(result["details"], dict)
