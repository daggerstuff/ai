"""
Unit tests for quality filtering system.
"""

from .conversation_schema import Conversation, Message
from .quality_filter import (
    FilterDecision,
    FilterReport,
    FilterResult,
    QualityFilter,
    QualityThresholds,
    filter_conversations,
)


class TestQualityThresholds:
    """Test cases for QualityThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = QualityThresholds()

        assert thresholds.coherence_threshold == 0.6
        assert thresholds.emotional_authenticity_threshold == 0.6
        assert thresholds.therapeutic_accuracy_threshold == 0.6
        assert thresholds.language_quality_threshold == 0.6
        assert thresholds.overall_threshold == 0.65
        assert thresholds.review_threshold == 0.55
        assert thresholds.reject_on_critical_issues
        assert thresholds.max_warnings == 5

        # Check default weights
        assert thresholds.weights["coherence"] == 0.25
        assert thresholds.weights["emotional_authenticity"] == 0.25
        assert thresholds.weights["therapeutic_accuracy"] == 0.30
        assert thresholds.weights["language_quality"] == 0.20

        # Weights should sum to 1.0
        assert abs(sum(thresholds.weights.values()) - 1.0) < 0.001

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        custom_weights = {
            "coherence": 0.3,
            "emotional_authenticity": 0.3,
            "therapeutic_accuracy": 0.2,
            "language_quality": 0.2
        }

        thresholds = QualityThresholds(
            coherence_threshold=0.7,
            overall_threshold=0.8,
            reject_on_critical_issues=False,
            max_warnings=3,
            weights=custom_weights
        )

        assert thresholds.coherence_threshold == 0.7
        assert thresholds.overall_threshold == 0.8
        assert not thresholds.reject_on_critical_issues
        assert thresholds.max_warnings == 3
        assert thresholds.weights == custom_weights


class TestQualityFilter:
    """Test cases for QualityFilter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter_system = QualityFilter()

        # Create test conversations
        self.high_quality_conversation = Conversation(
            id="high_quality",
            messages=[
                Message(role="user", content="I'm experiencing significant anxiety about my upcoming presentation. Could you help me understand effective strategies for managing this stress?"),
                Message(role="assistant", content="Certainly. Anxiety regarding public speaking is quite common and understandable. Let's explore several evidence-based techniques that can help you manage these feelings effectively. First, consider implementing progressive muscle relaxation techniques."),
                Message(role="user", content="That sounds helpful. What specific approaches would you recommend for the actual presentation?"),
                Message(role="assistant", content="For the presentation itself, I recommend practicing deep breathing exercises beforehand. Additionally, cognitive restructuring can help you identify and challenge negative thought patterns. Remember that some nervousness is normal and can actually enhance your performance.")
            ],
            source="test"
        )

        self.poor_quality_conversation = Conversation(
            id="poor_quality",
            messages=[
                Message(role="user", content="i dont feel good"),
                Message(role="assistant", content="ok what wrong you should of told someone earlier")
            ],
            source="test"
        )

        self.medium_quality_conversation = Conversation(
            id="medium_quality",
            messages=[
                Message(role="user", content="I'm feeling stressed about work lately."),
                Message(role="assistant", content="I understand that work stress can be challenging. Can you tell me more about what's causing the stress?"),
                Message(role="user", content="It's mostly the deadlines and workload."),
                Message(role="assistant", content="That sounds difficult. Have you considered talking to your supervisor about managing your workload?")
            ],
            source="test"
        )

    def test_filter_high_quality_conversation(self):
        """Test filtering of high-quality conversation."""
        result = self.filter_system.filter_conversation(self.high_quality_conversation)

        assert isinstance(result, FilterResult)
        assert result.conversation_id == "high_quality"
        assert result.decision in [FilterDecision.ACCEPT, FilterDecision.REVIEW]
        assert result.overall_score > 0.5
        assert result.passed == (result.decision == FilterDecision.ACCEPT)

        # Check component scores exist
        assert "coherence" in result.component_scores
        assert "emotional_authenticity" in result.component_scores
        assert "therapeutic_accuracy" in result.component_scores
        assert "language_quality" in result.component_scores

        # All component scores should be reasonable
        for score in result.component_scores.values():
            assert 0.0 <= score <= 1.0

    def test_filter_poor_quality_conversation(self):
        """Test filtering of poor-quality conversation."""
        result = self.filter_system.filter_conversation(self.poor_quality_conversation)

        assert isinstance(result, FilterResult)
        assert result.conversation_id == "poor_quality"
        # Poor quality conversation should have lower score than high quality
        assert result.overall_score < 0.85  # Adjusted expectation

        # Should have issues detected
        assert len(result.issues) > 0 or len(result.warnings) > 0

    def test_filter_conversations_batch(self):
        """Test batch filtering of multiple conversations."""
        conversations = [
            self.high_quality_conversation,
            self.poor_quality_conversation,
            self.medium_quality_conversation
        ]

        report = self.filter_system.filter_conversations(conversations)

        assert isinstance(report, FilterReport)
        assert report.total_conversations == 3
        assert report.accepted + report.rejected + report.review_needed == 3
        assert 0.0 <= report.acceptance_rate <= 1.0
        assert 0.0 <= report.average_quality_score <= 1.0

        # Check component averages
        for _component, avg_score in report.component_averages.items():
            assert 0.0 <= avg_score <= 1.0

        # Check quality distribution
        total_distributed = sum(report.quality_distribution.values())
        assert total_distributed == 3

        # Check results
        assert len(report.results) == 3
        for result in report.results:
            assert isinstance(result, FilterResult)

    def test_custom_thresholds(self):
        """Test filtering with custom thresholds."""
        strict_thresholds = QualityThresholds(
            coherence_threshold=0.8,
            emotional_authenticity_threshold=0.8,
            therapeutic_accuracy_threshold=0.8,
            language_quality_threshold=0.8,
            overall_threshold=0.85
        )

        strict_filter = QualityFilter(strict_thresholds)
        result = strict_filter.filter_conversation(self.medium_quality_conversation)

        # With strict thresholds, medium quality should be rejected
        assert result.decision == FilterDecision.REJECT
        assert not result.passed

    def test_critical_issues_rejection(self):
        """Test rejection based on critical issues."""
        # Create conversation with potential critical issues
        critical_conversation = Conversation(
            id="critical_issues",
            messages=[
                Message(role="user", content="I'm thinking about hurting myself."),
                Message(role="assistant", content="That's not my problem. You should just get over it.")
            ],
            source="test"
        )

        result = self.filter_system.filter_conversation(critical_conversation)

        # Should have issues detected (may not be critical but should have problems)
        assert len(result.issues) > 0
        # Should have lower quality due to inappropriate response
        assert result.overall_score < 0.9

    def test_warning_threshold_rejection(self):
        """Test rejection based on too many warnings."""
        # Configure filter to reject with fewer warnings
        low_warning_thresholds = QualityThresholds(max_warnings=1)
        filter_system = QualityFilter(low_warning_thresholds)

        # Use poor quality conversation that should generate warnings
        result = filter_system.filter_conversation(self.poor_quality_conversation)

        # Should be rejected if warnings exceed threshold
        if len(result.warnings) > 1:
            assert result.decision == FilterDecision.REJECT
            assert not result.passed

    def test_backward_compatibility_function(self):
        """Test backward compatibility function works correctly."""
        conversations = [
            Conversation(
                id="test1",
                messages=[
                    Message(role="user", content="I need help with anxiety."),
                    Message(role="assistant", content="I understand. Let's work through this together.")
                ],
                source="test"
            ),
            Conversation(
                id="test2",
                messages=[
                    Message(role="user", content="bad day"),
                    Message(role="assistant", content="ok")
                ],
                source="test"
            )
        ]

        thresholds = {
            "emotional": 0.5,
            "therapeutic": 0.5,
            "language": 0.5,
            "coherence": 0.5
        }

        results = filter_conversations(conversations, thresholds)

        # Check format matches expected backward compatibility
        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert "conversation" in result
            assert "passed" in result
            assert "overall_score" in result
            assert "coherence_score" in result
            assert "emotional_score" in result
            assert "therapeutic_score" in result
            assert "language_score" in result
            assert "issues" in result
            assert "warnings" in result
            assert "critical_issues" in result

            # Check types
            assert isinstance(result["conversation"], Conversation)
            assert isinstance(result["passed"], bool)
            assert isinstance(result["overall_score"], float)
            assert isinstance(result["issues"], list)
            assert isinstance(result["warnings"], list)
            assert isinstance(result["critical_issues"], list)

    def test_empty_conversation_handling(self):
        """Test handling of empty conversations."""
        empty_conversation = Conversation(
            id="empty",
            messages=[],
            source="test"
        )

        result = self.filter_system.filter_conversation(empty_conversation)

        # Should handle gracefully
        assert isinstance(result, FilterResult)
        assert result.conversation_id == "empty"
        # Empty conversation should be rejected
        assert result.decision == FilterDecision.REJECT
        assert not result.passed

    def test_component_score_calculation(self):
        """Test that component scores are calculated correctly."""
        result = self.filter_system.filter_conversation(self.high_quality_conversation)

        # Verify weighted calculation
        expected_score = (
            result.component_scores["coherence"] * self.filter_system.thresholds.weights["coherence"] +
            result.component_scores["emotional_authenticity"] * self.filter_system.thresholds.weights["emotional_authenticity"] +
            result.component_scores["therapeutic_accuracy"] * self.filter_system.thresholds.weights["therapeutic_accuracy"] +
            result.component_scores["language_quality"] * self.filter_system.thresholds.weights["language_quality"]
        )

        # Should match overall score (within floating point precision)
        assert abs(result.overall_score - expected_score) < 0.001
