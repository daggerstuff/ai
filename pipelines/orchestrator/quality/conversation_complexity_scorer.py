"""
Conversation complexity scorer for training allocation.
Scores conversation complexity for appropriate training data allocation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for conversations."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ComplexityScore:
    """Complexity scoring result."""
    conversation_id: str
    overall_score: float
    complexity_level: ComplexityLevel
    dimension_scores: dict[str, float]
    reasoning: list[str]


class ConversationComplexityScorer:
    """
    Scores conversation complexity for training allocation.

    Assesses multiple dimensions of therapeutic conversation complexity
    to ensure appropriate training data distribution.
    """

    def __init__(self):
        """Initialize the conversation complexity scorer."""
        self.logger = get_logger(__name__)

        self.complexity_dimensions = {
            "therapeutic_depth": {
                "indicators": ["insight", "interpretation", "transference", "unconscious"],
                "weight": 0.25
            },
            "emotional_intensity": {
                "indicators": ["crisis", "trauma", "intense", "overwhelming"],
                "weight": 0.20
            },
            "clinical_sophistication": {
                "indicators": ["diagnosis", "assessment", "formulation", "differential"],
                "weight": 0.25
            },
            "intervention_complexity": {
                "indicators": ["technique", "intervention", "strategy", "approach"],
                "weight": 0.15
            },
            "ethical_considerations": {
                "indicators": ["ethical", "boundary", "confidentiality", "duty"],
                "weight": 0.15
            }
        }

        self.complexity_thresholds = {
            ComplexityLevel.BASIC: (0.0, 0.3),
            ComplexityLevel.INTERMEDIATE: (0.3, 0.6),
            ComplexityLevel.ADVANCED: (0.6, 0.8),
            ComplexityLevel.EXPERT: (0.8, 1.0)
        }

        self.logger.info("ConversationComplexityScorer initialized")

    def score_conversation(self, conversation: Conversation) -> ComplexityScore:
        """Score the complexity of a single conversation."""
        try:
            dimension_scores = {}
            reasoning = []

            # Combine all message content
            content = " ".join([msg.content.lower() for msg in conversation.messages])

            # Score each dimension
            for dimension, config in self.complexity_dimensions.items():
                score = self._score_dimension(content, config["indicators"])
                dimension_scores[dimension] = score

                if score > 0.5:
                    reasoning.append(f"High {dimension.replace('_', ' ')}: {score:.2f}")

            # Calculate weighted overall score
            overall_score = sum(
                dimension_scores[dim] * config["weight"]
                for dim, config in self.complexity_dimensions.items()
            )

            # Determine complexity level
            complexity_level = self._determine_complexity_level(overall_score)

            # Add conversation-specific factors
            overall_score = self._adjust_for_conversation_factors(conversation, overall_score, reasoning)

            return ComplexityScore(
                conversation_id=conversation.id,
                overall_score=overall_score,
                complexity_level=complexity_level,
                dimension_scores=dimension_scores,
                reasoning=reasoning
            )

        except Exception as e:
            self.logger.warning(f"Could not score conversation complexity: {e}")
            return ComplexityScore(
                conversation_id=conversation.id,
                overall_score=0.5,
                complexity_level=ComplexityLevel.INTERMEDIATE,
                dimension_scores={},
                reasoning=["Error in scoring"]
            )

    def _score_dimension(self, content: str, indicators: list[str]) -> float:
        """Score a specific complexity dimension."""
        matches = sum(1 for indicator in indicators if indicator in content)
        return min(matches / len(indicators), 1.0)

    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score."""
        for level, (min_score, max_score) in self.complexity_thresholds.items():
            if min_score <= score < max_score:
                return level
        return ComplexityLevel.EXPERT  # For scores >= 0.8

    def _adjust_for_conversation_factors(self, conversation: Conversation, base_score: float, reasoning: list[str]) -> float:
        """Adjust score based on conversation-specific factors."""
        adjusted_score = base_score

        # Message count factor
        message_count = len(conversation.messages)
        if message_count > 10:
            adjusted_score += 0.1
            reasoning.append("Long conversation (+0.1)")
        elif message_count < 4:
            adjusted_score -= 0.1
            reasoning.append("Short conversation (-0.1)")

        # Safety-critical factor
        if conversation.metadata.get("safety_critical", False):
            adjusted_score += 0.2
            reasoning.append("Safety-critical scenario (+0.2)")

        # Edge case factor
        if "edge_case" in conversation.tags:
            adjusted_score += 0.15
            reasoning.append("Edge case scenario (+0.15)")

        # Crisis factor
        if "crisis" in conversation.tags:
            adjusted_score += 0.25
            reasoning.append("Crisis intervention (+0.25)")

        return min(adjusted_score, 1.0)  # Cap at 1.0

    def score_conversations(self, conversations: list[Conversation]) -> list[ComplexityScore]:
        """Score multiple conversations."""
        scores = []

        for conversation in conversations:
            score = self.score_conversation(conversation)
            scores.append(score)

        self.logger.info(f"Scored {len(scores)} conversations for complexity")
        return scores

    def get_complexity_distribution(self, scores: list[ComplexityScore]) -> dict[str, Any]:
        """Get distribution statistics for complexity scores."""
        if not scores:
            return {"error": "No scores to analyze"}

        level_counts = {}
        for score in scores:
            level = score.complexity_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        avg_score = sum(score.overall_score for score in scores) / len(scores)

        return {
            "total_conversations": len(scores),
            "average_complexity": round(avg_score, 3),
            "level_distribution": level_counts,
            "level_percentages": {
                level: round((count / len(scores)) * 100, 1)
                for level, count in level_counts.items()
            }
        }

    def recommend_training_allocation(self, scores: list[ComplexityScore]) -> dict[str, Any]:
        """Recommend training data allocation based on complexity."""
        distribution = self.get_complexity_distribution(scores)

        # Recommended distribution for balanced training
        target_distribution = {
            "basic": 0.30,      # 30% basic conversations
            "intermediate": 0.40, # 40% intermediate conversations
            "advanced": 0.25,    # 25% advanced conversations
            "expert": 0.05       # 5% expert conversations
        }

        recommendations = {}
        current_percentages = {k: v/100 for k, v in distribution.get("level_percentages", {}).items()}

        for level, target_pct in target_distribution.items():
            current_pct = current_percentages.get(level, 0)
            if current_pct < target_pct:
                recommendations[level] = f"Need {target_pct - current_pct:.1%} more"
            elif current_pct > target_pct + 0.1:  # 10% tolerance
                recommendations[level] = f"Have {current_pct - target_pct:.1%} excess"
            else:
                recommendations[level] = "Balanced"

        return {
            "target_distribution": target_distribution,
            "current_distribution": current_percentages,
            "recommendations": recommendations,
            "overall_balance": sum(abs(current_percentages.get(level, 0) - target_pct)
                                 for level, target_pct in target_distribution.items()) / len(target_distribution)
        }


def validate_conversation_complexity_scorer():
    """Validate the ConversationComplexityScorer functionality."""
    try:
        scorer = ConversationComplexityScorer()
        assert hasattr(scorer, "score_conversation")
        assert hasattr(scorer, "complexity_dimensions")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_conversation_complexity_scorer():
        pass
    else:
        pass
