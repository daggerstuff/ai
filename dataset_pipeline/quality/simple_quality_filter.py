"""
Simple Quality Filter

Simplified quality filtering system for conversation datasets with
basic quality assessment and filtering capabilities.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality levels for conversations."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""

    overall_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    safety_score: float = 0.0
    length_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR


@dataclass
class QualityConfig:
    """Configuration for quality filtering."""

    min_overall_score: float = 0.6
    min_coherence_score: float = 0.5
    min_safety_score: float = 0.8
    min_message_length: int = 5
    max_message_length: int = 2000
    min_conversation_length: int = 2
    max_conversation_length: int = 50
    required_roles: list[str] = field(default_factory=lambda: ["user", "assistant"])


class SimpleQualityFilter:
    """Simple quality filtering system."""

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.logger = get_logger(__name__)

        # Quality assessment patterns
        self.quality_patterns = self._initialize_quality_patterns()

        logger.info("SimpleQualityFilter initialized")

    def _initialize_quality_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for quality assessment."""
        return {
            "positive_indicators": [
                r"\b(?:helpful|informative|clear|detailed|specific)\b",
                r"\b(?:understand|explain|clarify|elaborate)\b",
                r"\b(?:thank you|thanks|appreciate|grateful)\b",
            ],
            "negative_indicators": [
                r"\b(?:unclear|confusing|vague|unhelpful)\b",
                r"\b(?:don\'t know|not sure|maybe|perhaps)\b",
                r"\b(?:spam|advertisement|promotion|buy now)\b",
            ],
            "safety_concerns": [
                r"\b(?:harmful|dangerous|illegal|violent)\b",
                r"\b(?:suicide|self.harm|kill|death)\b",
                r"\b(?:hate|discrimination|racist|sexist)\b",
            ],
            "coherence_indicators": [
                r"\b(?:because|therefore|however|although|since)\b",
                r"\b(?:first|second|next|then|finally)\b",
                r"\b(?:in conclusion|to summarize|overall)\b",
            ],
        }

    def assess_quality(self, conversation: Conversation) -> QualityMetrics:
        """Assess the quality of a conversation."""
        metrics = QualityMetrics()

        # Assess individual components
        metrics.coherence_score = self._assess_coherence(conversation)
        metrics.completeness_score = self._assess_completeness(conversation)
        metrics.relevance_score = self._assess_relevance(conversation)
        metrics.safety_score = self._assess_safety(conversation)
        metrics.length_score = self._assess_length(conversation)

        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)

        # Determine quality level
        metrics.quality_level = self._determine_quality_level(metrics.overall_score)

        return metrics

    def _assess_coherence(self, conversation: Conversation) -> float:
        """Assess conversation coherence."""
        if not conversation.messages:
            return 0.0

        coherence_score = 0.5  # Base score

        # Check for coherence indicators
        full_text = " ".join(msg.content for msg in conversation.messages)
        coherence_patterns = self.quality_patterns["coherence_indicators"]

        coherence_matches = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE))
            for pattern in coherence_patterns
        )

        # Normalize coherence score
        coherence_score += min(0.4, coherence_matches * 0.1)

        # Check message flow
        if len(conversation.messages) >= 2:
            # Simple check: alternating roles
            roles = [msg.role for msg in conversation.messages]
            alternating = all(
                roles[i] != roles[i + 1]
                for i in range(len(roles) - 1)
                if i + 1 < len(roles)
            )
            if alternating:
                coherence_score += 0.1

        return min(1.0, coherence_score)

    def _assess_completeness(self, conversation: Conversation) -> float:
        """Assess conversation completeness."""
        if not conversation.messages:
            return 0.0

        completeness_score = 0.0

        # Check message count
        message_count = len(conversation.messages)
        if (
            self.config.min_conversation_length
            <= message_count
            <= self.config.max_conversation_length
        ):
            completeness_score += 0.4

        # Check role diversity
        roles = {msg.role for msg in conversation.messages}
        required_roles = set(self.config.required_roles)
        if required_roles.issubset(roles):
            completeness_score += 0.3

        # Check message content length
        valid_messages = sum(
            1
            for msg in conversation.messages
            if self.config.min_message_length
            <= len(msg.content)
            <= self.config.max_message_length
        )

        if message_count > 0:
            completeness_score += (valid_messages / message_count) * 0.3

        return min(1.0, completeness_score)

    def _assess_relevance(self, conversation: Conversation) -> float:
        """Assess conversation relevance."""
        if not conversation.messages:
            return 0.0

        relevance_score = 0.5  # Base score

        full_text = " ".join(msg.content for msg in conversation.messages)

        # Check for positive indicators
        positive_patterns = self.quality_patterns["positive_indicators"]
        positive_matches = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE))
            for pattern in positive_patterns
        )

        # Check for negative indicators
        negative_patterns = self.quality_patterns["negative_indicators"]
        negative_matches = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE))
            for pattern in negative_patterns
        )

        # Adjust score
        relevance_score += min(0.3, positive_matches * 0.05)
        relevance_score -= min(0.4, negative_matches * 0.1)

        return max(0.0, min(1.0, relevance_score))

    def _assess_safety(self, conversation: Conversation) -> float:
        """Assess conversation safety."""
        if not conversation.messages:
            return 1.0  # Empty conversation is safe

        safety_score = 1.0  # Start with perfect safety

        full_text = " ".join(msg.content for msg in conversation.messages)

        # Check for safety concerns
        safety_patterns = self.quality_patterns["safety_concerns"]
        safety_violations = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE))
            for pattern in safety_patterns
        )

        # Reduce score for each safety violation
        safety_score -= safety_violations * 0.2

        return max(0.0, safety_score)

    def _assess_length(self, conversation: Conversation) -> float:
        """Assess conversation length appropriateness."""
        if not conversation.messages:
            return 0.0

        length_score = 0.0

        # Check overall conversation length
        message_count = len(conversation.messages)
        if (
            self.config.min_conversation_length
            <= message_count
            <= self.config.max_conversation_length
        ):
            length_score += 0.5

        # Check individual message lengths
        valid_length_messages = 0
        for msg in conversation.messages:
            if (
                self.config.min_message_length
                <= len(msg.content)
                <= self.config.max_message_length
            ):
                valid_length_messages += 1

        if message_count > 0:
            length_score += (valid_length_messages / message_count) * 0.5

        return length_score

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        weights = {
            "coherence": 0.25,
            "completeness": 0.25,
            "relevance": 0.20,
            "safety": 0.20,
            "length": 0.10,
        }

        overall_score = (
            metrics.coherence_score * weights["coherence"]
            + metrics.completeness_score * weights["completeness"]
            + metrics.relevance_score * weights["relevance"]
            + metrics.safety_score * weights["safety"]
            + metrics.length_score * weights["length"]
        )

        return min(1.0, overall_score)

    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level from overall score."""
        if overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        if overall_score >= 0.8:
            return QualityLevel.GOOD
        if overall_score >= 0.6:
            return QualityLevel.ACCEPTABLE
        if overall_score >= 0.3:
            return QualityLevel.POOR
        return QualityLevel.REJECTED

    def filter_conversation(
        self, conversation: Conversation
    ) -> tuple[bool, QualityMetrics]:
        """Filter a conversation based on quality metrics."""
        metrics = self.assess_quality(conversation)

        # Apply filtering criteria
        passes_filter = (
            metrics.overall_score >= self.config.min_overall_score
            and metrics.coherence_score >= self.config.min_coherence_score
            and metrics.safety_score >= self.config.min_safety_score
        )

        return passes_filter, metrics

    def filter_conversations(
        self, conversations: list[Conversation]
    ) -> tuple[list[Conversation], list[QualityMetrics]]:
        """Filter a list of conversations."""
        filtered_conversations = []
        all_metrics = []

        for conversation in conversations:
            passes_filter, metrics = self.filter_conversation(conversation)
            all_metrics.append(metrics)

            if passes_filter:
                filtered_conversations.append(conversation)

        logger.info(
            f"Filtered {len(filtered_conversations)}/{len(conversations)} conversations"
        )

        return filtered_conversations, all_metrics

    def get_quality_summary(self, metrics_list: list[QualityMetrics]) -> dict[str, Any]:
        """Get summary statistics for quality metrics."""
        if not metrics_list:
            return {}

        # Calculate averages
        avg_overall = sum(m.overall_score for m in metrics_list) / len(metrics_list)
        avg_coherence = sum(m.coherence_score for m in metrics_list) / len(metrics_list)
        avg_safety = sum(m.safety_score for m in metrics_list) / len(metrics_list)

        # Count by quality level
        quality_counts = {}
        for level in QualityLevel:
            quality_counts[level.value] = sum(
                1 for m in metrics_list if m.quality_level == level
            )

        return {
            "total_conversations": len(metrics_list),
            "average_scores": {
                "overall": avg_overall,
                "coherence": avg_coherence,
                "safety": avg_safety,
            },
            "quality_distribution": quality_counts,
            "pass_rate": sum(
                1
                for m in metrics_list
                if m.overall_score >= self.config.min_overall_score
            )
            / len(metrics_list),
        }


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Create quality filter
    quality_filter = SimpleQualityFilter()

    # Test conversation
    test_conversation = Conversation(
        id="test_1",
        messages=[
            Message(
                role="user",
                content="I'm feeling anxious about my upcoming presentation.",
            ),
            Message(
                role="assistant",
                content="I understand that presentations can be nerve-wracking. Can you tell me more about what specifically is making you feel anxious?",
            ),
            Message(
                role="user",
                content="I'm worried I'll forget what to say or that people will judge me.",
            ),
            Message(
                role="assistant",
                content="Those are very common concerns. Let's work on some strategies to help you feel more confident.",
            ),
        ],
        created_at=datetime.now(),
    )

    # Assess quality
    metrics = quality_filter.assess_quality(test_conversation)

    # Filter conversation
    passes_filter, _ = quality_filter.filter_conversation(test_conversation)

    # Get summary
    summary = quality_filter.get_quality_summary([metrics])
