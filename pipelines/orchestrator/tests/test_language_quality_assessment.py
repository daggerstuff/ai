"""
Unit tests for language quality assessment system.
"""

from conversation_schema import Conversation, Message
from language_quality_assessment import (
    LanguageComplexity,
    LanguageQualityAssessor,
    LanguageQualityMetrics,
    assess_language_quality,
)


class TestLanguageQualityAssessor:
    """Test cases for LanguageQualityAssessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = LanguageQualityAssessor()

    def test_assess_high_quality_conversation(self):
        """Test assessment of high-quality conversation with sophisticated language."""
        conversation = Conversation(
            id="test_high_quality",
            messages=[
                Message(role="user", content="I'm experiencing significant anxiety about my upcoming presentation. Could you help me understand effective strategies for managing this stress?"),
                Message(role="assistant", content="Certainly. Anxiety regarding public speaking is quite common and understandable. Let's explore several evidence-based techniques that can help you manage these feelings effectively."),
                Message(role="user", content="That sounds helpful. What specific approaches would you recommend?"),
                Message(role="assistant", content="First, consider implementing progressive muscle relaxation techniques. Additionally, cognitive restructuring can help you identify and challenge negative thought patterns. Furthermore, practicing your presentation multiple times will build confidence and familiarity with the material.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.overall_score > 0.7
        assert metrics.readability_score > 0.6
        assert metrics.lexical_diversity_score > 0.6
        assert metrics.vocabulary_appropriateness_score > 0.7
        assert metrics.sentence_complexity_score > 0.6
        assert metrics.complexity_level in [LanguageComplexity.MODERATE, LanguageComplexity.COMPLEX]
        assert metrics.quality_level in ["good", "excellent"]
        assert len(metrics.issues) == 0

    def test_assess_poor_quality_conversation(self):
        """Test assessment of poor-quality conversation with simple language and errors."""
        conversation = Conversation(
            id="test_poor_quality",
            messages=[
                Message(role="user", content="i dont feel good"),
                Message(role="assistant", content="ok what wrong"),
                Message(role="user", content="dunno just bad"),
                Message(role="assistant", content="thats not good you should of told someone")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.overall_score < 0.5
        assert metrics.grammar_quality_score < 0.7  # Should detect "should of" error
        assert metrics.vocabulary_appropriateness_score < 0.6  # Informal language
        assert metrics.complexity_level in [LanguageComplexity.VERY_SIMPLE, LanguageComplexity.SIMPLE]
        assert metrics.quality_level in ["poor", "very_poor"]
        assert len(metrics.issues) > 0 or len(metrics.warnings) > 0

    def test_assess_conversation_with_grammar_errors(self):
        """Test detection of grammar errors."""
        conversation = Conversation(
            id="test_grammar_errors",
            messages=[
                Message(role="user", content="I am went to the store yesterday and he don't like it."),
                Message(role="assistant", content="I understand. You would of had a better experience if the service was improved.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.grammar_quality_score < 0.7
        assert any("grammar" in issue.lower() for issue in metrics.issues + metrics.warnings)
        assert metrics.details["grammar_quality_details"]["error_count"] > 0

    def test_assess_conversation_with_informal_language(self):
        """Test detection of informal language."""
        conversation = Conversation(
            id="test_informal",
            messages=[
                Message(role="user", content="Yeah, I'm gonna tell you about this awesome thing that happened."),
                Message(role="assistant", content="Cool! That sounds really neat. Tell me more about this super cool stuff.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.vocabulary_appropriateness_score < 0.8
        assert any("informal" in warning.lower() for warning in metrics.warnings + metrics.issues)
        assert metrics.details["vocabulary_appropriateness_details"]["informal_word_ratio"] > 0.1

    def test_assess_conversation_with_complex_sentences(self):
        """Test assessment of sentence complexity."""
        conversation = Conversation(
            id="test_complex",
            messages=[
                Message(role="user", content="Although I understand the theoretical framework, I'm struggling with the practical implementation because the documentation is incomplete."),
                Message(role="assistant", content="While documentation gaps can be frustrating, we can work through this systematically. First, let's identify the specific areas where you need clarification, and then we'll develop a step-by-step approach that addresses each component methodically.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.sentence_complexity_score > 0.6
        assert metrics.complexity_level in [LanguageComplexity.MODERATE, LanguageComplexity.COMPLEX, LanguageComplexity.VERY_COMPLEX]
        assert metrics.details["sentence_complexity_details"]["complex_sentence_ratio"] > 0.5

    def test_assess_conversation_with_poor_readability(self):
        """Test detection of poor readability."""
        conversation = Conversation(
            id="test_readability",
            messages=[
                Message(role="user", content="The implementation of the comprehensive therapeutic intervention methodology requires extensive consideration of multifaceted psychological paradigms and interdisciplinary collaborative frameworks that encompass various theoretical orientations and evidence-based practices within the contemporary mental health treatment landscape."),
                Message(role="assistant", content="The aforementioned therapeutic modalities necessitate sophisticated understanding of complex psychodynamic processes and cognitive-behavioral interventions that integrate seamlessly with humanistic approaches while maintaining adherence to established clinical protocols and professional ethical standards.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        # Very long sentences should reduce readability
        assert metrics.details["readability_details"]["avg_sentence_length"] > 25
        assert any("long sentences" in warning.lower() for warning in metrics.warnings + metrics.issues)

    def test_assess_conversation_with_low_lexical_diversity(self):
        """Test detection of low lexical diversity."""
        conversation = Conversation(
            id="test_diversity",
            messages=[
                Message(role="user", content="I feel bad. I feel really bad. I feel very bad about this."),
                Message(role="assistant", content="You feel bad. I understand you feel bad. Feeling bad is normal when you feel bad.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.lexical_diversity_score < 0.6
        assert any("diversity" in issue.lower() or "repetitive" in issue.lower()
                  for issue in metrics.issues + metrics.warnings)
        assert metrics.details["lexical_diversity_details"]["type_token_ratio"] < 0.5

    def test_assess_conversation_with_good_coherence(self):
        """Test assessment of conversation coherence."""
        conversation = Conversation(
            id="test_coherence",
            messages=[
                Message(role="user", content="I'm having trouble with my anxiety."),
                Message(role="assistant", content="I understand that anxiety can be challenging. Let's explore this together."),
                Message(role="user", content="It affects my daily activities."),
                Message(role="assistant", content="That sounds difficult. Can you tell me more about how it impacts your routine?")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.coherence_score > 0.5
        assert metrics.details["coherence_details"]["pronoun_reference_score"] > 0.5

    def test_assess_empty_conversation(self):
        """Test assessment of empty or very short conversation."""
        conversation = Conversation(
            id="test_empty",
            messages=[
                Message(role="user", content="Hi")
            ],
            source="test"
        )

        metrics = self.assessor.assess_language_quality(conversation)

        assert metrics.overall_score == 0.0
        assert "Insufficient messages" in metrics.issues[0]
        assert metrics.quality_level == "very_poor"

    def test_syllable_counting(self):
        """Test syllable counting functionality."""
        test_cases = [
            ("hello", 2),
            ("cat", 1),
            ("beautiful", 3),
            ("understanding", 4),
            ("a", 1),
            ("", 0)
        ]

        for word, expected in test_cases:
            result = self.assessor._count_syllables(word)
            assert result == expected, f"Expected {expected} syllables for '{word}', got {result}"

    def test_complexity_level_determination(self):
        """Test complexity level determination."""
        # Test very simple
        complexity = self.assessor._determine_complexity_level(
            {"avg_sentence_length": 8},
            {"type_token_ratio": 0.4},
            {"complex_sentence_ratio": 0.1}
        )
        assert complexity in [LanguageComplexity.VERY_SIMPLE, LanguageComplexity.SIMPLE]

        # Test complex
        complexity = self.assessor._determine_complexity_level(
            {"avg_sentence_length": 30},
            {"type_token_ratio": 0.9},
            {"complex_sentence_ratio": 0.8}
        )
        assert complexity in [LanguageComplexity.COMPLEX, LanguageComplexity.VERY_COMPLEX]

    def test_backward_compatibility_function(self):
        """Test backward compatibility function."""
        conversation = Conversation(
            id="test_compat",
            messages=[
                Message(role="user", content="I need help with my anxiety."),
                Message(role="assistant", content="I understand. Let's work through this together.")
            ],
            source="test"
        )

        result = assess_language_quality(conversation)

        assert "score" in result
        assert "issues" in result
        assert "warnings" in result
        assert "complexity_level" in result
        assert "details" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["details"], dict)

    def test_custom_configuration(self):
        """Test assessor with custom configuration."""
        custom_config = {
            "weights": {
                "readability": 0.3,
                "lexical_diversity": 0.2,
                "grammar_quality": 0.2,
                "vocabulary_appropriateness": 0.1,
                "sentence_complexity": 0.1,
                "coherence": 0.1
            },
            "thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.6,
                "poor": 0.4
            }
        }

        assessor = LanguageQualityAssessor(custom_config)

        assert assessor.weights["readability"] == 0.3
        assert assessor.thresholds["excellent"] == 0.9

        conversation = Conversation(
            id="test_custom",
            messages=[
                Message(role="user", content="Test message."),
                Message(role="assistant", content="Response message.")
            ],
            source="test"
        )

        metrics = assessor.assess_language_quality(conversation)
        assert isinstance(metrics, LanguageQualityMetrics)
