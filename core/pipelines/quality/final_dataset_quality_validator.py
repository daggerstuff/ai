"""
Final dataset quality validator for production readiness.
Comprehensive quality validation before dataset deployment.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Quality validation result."""
    passed: bool
    overall_score: float
    dimension_scores: dict[str, float]
    issues: list[str]
    recommendations: list[str]
    validation_report: dict[str, Any]


class FinalDatasetQualityValidator:
    """
    Final quality validation for production datasets.

    Performs comprehensive quality checks to ensure datasets
    meet production standards for therapeutic AI training.
    """

    def __init__(self):
        """Initialize the final quality validator."""
        self.logger = get_logger(__name__)

        self.quality_dimensions = {
            "content_quality": 0.25,      # 25% weight
            "therapeutic_accuracy": 0.30,  # 30% weight
            "conversation_coherence": 0.20, # 20% weight
            "safety_compliance": 0.25      # 25% weight
        }

        self.minimum_thresholds = {
            "content_quality": 0.7,
            "therapeutic_accuracy": 0.8,
            "conversation_coherence": 0.8,
            "safety_compliance": 0.9,
            "overall_quality": 0.8
        }

        self.safety_keywords = {
            "suicide", "self-harm", "violence", "abuse", "danger",
            "weapon", "kill", "hurt", "harm", "threat"
        }

        self.therapeutic_indicators = {
            "empathy", "understanding", "support", "validation", "reflection",
            "exploration", "insight", "coping", "therapeutic", "healing"
        }

        self.common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "i", "you", "we", "they", "it", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must"}

        self.logger.info("FinalDatasetQualityValidator initialized")

    def validate_dataset(self, conversations: list[Conversation]) -> ValidationResult:
        """Perform comprehensive quality validation."""
        self.logger.info(f"Validating dataset quality for {len(conversations)} conversations")

        # Calculate dimension scores
        dimension_scores = {}
        dimension_scores["content_quality"] = self._assess_content_quality(conversations)
        dimension_scores["therapeutic_accuracy"] = self._assess_therapeutic_accuracy(conversations)
        dimension_scores["conversation_coherence"] = self._assess_conversation_coherence(conversations)
        dimension_scores["safety_compliance"] = self._assess_safety_compliance(conversations)

        # Calculate overall score
        overall_score = sum(
            score * self.quality_dimensions[dimension]
            for dimension, score in dimension_scores.items()
        )

        # Check if validation passed
        passed = all(
            dimension_scores[dim] >= self.minimum_thresholds[dim]
            for dim in dimension_scores
        ) and overall_score >= self.minimum_thresholds["overall_quality"]

        # Identify issues and recommendations
        issues = self._identify_issues(dimension_scores)
        recommendations = self._generate_recommendations(dimension_scores, issues)

        # Generate detailed validation report
        validation_report = self._generate_validation_report(
            conversations, dimension_scores, overall_score, issues, recommendations
        )

        self.logger.info(f"Validation complete: {'PASSED' if passed else 'FAILED'} (score: {overall_score:.3f})")

        return ValidationResult(
            passed=passed,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            recommendations=recommendations,
            validation_report=validation_report
        )

    def _assess_content_quality(self, conversations: list[Conversation]) -> float:
        """Assess content quality dimension."""
        if not conversations:
            return 0.0

        quality_scores = []

        for conv in conversations:
            # Check message count
            message_count_score = min(len(conv.messages) / 4, 1.0)  # Optimal: 4+ messages

            # Check content length
            avg_length = sum(len(msg.content) for msg in conv.messages) / len(conv.messages)
            length_score = min(avg_length / 100, 1.0)  # Optimal: 100+ chars per message

            # Check content diversity (unique words)
            all_content = " ".join(msg.content.lower() for msg in conv.messages)
            words = all_content.split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            diversity_score = min(unique_ratio * 2, 1.0)  # Scale to 0-1

            conv_quality = (message_count_score + length_score + diversity_score) / 3
            quality_scores.append(conv_quality)

        return sum(quality_scores) / len(quality_scores)

    def _assess_therapeutic_accuracy(self, conversations: list[Conversation]) -> float:
        """Assess therapeutic accuracy dimension."""
        if not conversations:
            return 0.0

        accuracy_scores = []

        for conv in conversations:
            # Check for therapeutic language in assistant messages
            assistant_content = " ".join([
                msg.content.lower() for msg in conv.messages
                if msg.role == "assistant"
            ])

            if not assistant_content:
                accuracy_scores.append(0.0)
                continue

            # Count therapeutic indicators
            words = assistant_content.split()
            indicator_count = sum(
                1 for word in words
                if word in self.therapeutic_indicators
            )

            # Score based on indicator density
            indicator_density = indicator_count / len(words) if words else 0
            accuracy_score = min(indicator_density * 50, 1.0)  # Scale appropriately

            accuracy_scores.append(accuracy_score)

        return sum(accuracy_scores) / len(accuracy_scores)

    def _assess_conversation_coherence(self, conversations: list[Conversation]) -> float:
        """Assess conversation coherence dimension."""
        if not conversations:
            return 0.0

        coherence_scores = []

        for conv in conversations:
            if len(conv.messages) < 2:
                coherence_scores.append(0.0)
                continue

            # Check for proper turn-taking
            roles = [msg.role for msg in conv.messages]
            proper_alternation = all(
                roles[i] != roles[i+1] for i in range(len(roles)-1)
            ) if len(roles) > 1 else False

            alternation_score = 1.0 if proper_alternation else 0.5

            # Check for contextual continuity (simple heuristic)
            user_messages = [msg.content.lower() for msg in conv.messages if msg.role == "user"]
            assistant_messages = [msg.content.lower() for msg in conv.messages if msg.role == "assistant"]

            # Look for response relevance (shared keywords)
            relevance_scores = []
            for i, user_msg in enumerate(user_messages):
                if i < len(assistant_messages):
                    user_words = set(user_msg.split())
                    assistant_words = set(assistant_messages[i].split())

                    # Remove common words
                    user_words -= self.common_words
                    assistant_words -= self.common_words

                    if user_words and assistant_words:
                        overlap = len(user_words & assistant_words) / len(user_words | assistant_words)
                        relevance_scores.append(overlap)

            relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5

            coherence_score = (alternation_score + relevance_score) / 2
            coherence_scores.append(coherence_score)

        return sum(coherence_scores) / len(coherence_scores)

    def _assess_safety_compliance(self, conversations: list[Conversation]) -> float:
        """Assess safety compliance dimension."""
        if not conversations:
            return 1.0  # No conversations = no safety issues

        safety_scores = []

        for conv in conversations:
            all_content = " ".join(msg.content.lower() for msg in conv.messages)
            words = set(all_content.split())

            # Check for safety keywords
            safety_violations = sum(1 for keyword in self.safety_keywords if keyword in words)

            # Check if safety issues are handled appropriately
            if safety_violations > 0:
                # Look for appropriate safety responses
                assistant_content = " ".join([
                    msg.content.lower() for msg in conv.messages
                    if msg.role == "assistant"
                ])

                safety_responses = {
                    "safety", "help", "support", "crisis", "emergency", "professional",
                    "counselor", "therapist", "hotline", "resources"
                }

                assistant_words = set(assistant_content.split())
                appropriate_responses = sum(
                    1 for response in safety_responses
                    if response in assistant_words
                )

                # Score based on appropriate handling
                safety_score = min(appropriate_responses / safety_violations, 1.0)
            else:
                safety_score = 1.0  # No safety issues

            safety_scores.append(safety_score)

        return sum(safety_scores) / len(safety_scores)

    def _identify_issues(self, dimension_scores: dict[str, float]) -> list[str]:
        """Identify quality issues based on dimension scores."""
        issues = []

        for dimension, score in dimension_scores.items():
            threshold = self.minimum_thresholds[dimension]
            if score < threshold:
                issues.append(f"{dimension.replace('_', ' ').title()}: {score:.3f} < {threshold:.3f} (below threshold)")

        return issues

    def _generate_recommendations(self, dimension_scores: dict[str, float], issues: list[str]) -> list[str]:
        """Generate recommendations for improving quality."""
        recommendations = []

        if dimension_scores.get("content_quality", 1.0) < self.minimum_thresholds["content_quality"]:
            recommendations.append("Improve content quality: Add more substantial conversations with longer, more diverse content")

        if dimension_scores.get("therapeutic_accuracy", 1.0) < self.minimum_thresholds["therapeutic_accuracy"]:
            recommendations.append("Enhance therapeutic accuracy: Include more therapeutic language and evidence-based interventions")

        if dimension_scores.get("conversation_coherence", 1.0) < self.minimum_thresholds["conversation_coherence"]:
            recommendations.append("Improve conversation coherence: Ensure proper turn-taking and contextual relevance")

        if dimension_scores.get("safety_compliance", 1.0) < self.minimum_thresholds["safety_compliance"]:
            recommendations.append("Address safety compliance: Ensure appropriate handling of crisis and safety-related content")

        if not recommendations:
            recommendations.append("Dataset meets all quality standards - ready for production")

        return recommendations

    def _generate_validation_report(self, conversations: list[Conversation],
                                  dimension_scores: dict[str, float],
                                  overall_score: float,
                                  issues: list[str],
                                  recommendations: list[str]) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "validation_summary": {
                "total_conversations": len(conversations),
                "overall_score": overall_score,
                "validation_status": "PASSED" if overall_score >= self.minimum_thresholds["overall_quality"] else "FAILED",
                "validated_at": datetime.now().isoformat()
            },
            "dimension_analysis": {
                dimension: {
                    "score": score,
                    "threshold": self.minimum_thresholds[dimension],
                    "status": "PASS" if score >= self.minimum_thresholds[dimension] else "FAIL",
                    "weight": self.quality_dimensions[dimension]
                }
                for dimension, score in dimension_scores.items()
            },
            "quality_issues": issues,
            "recommendations": recommendations,
            "dataset_statistics": {
                "total_messages": sum(len(conv.messages) for conv in conversations),
                "average_messages_per_conversation": sum(len(conv.messages) for conv in conversations) / len(conversations) if conversations else 0,
                "total_characters": sum(sum(len(msg.content) for msg in conv.messages) for conv in conversations),
                "unique_conversation_ids": len({conv.id for conv in conversations})
            },
            "validation_criteria": {
                "minimum_thresholds": self.minimum_thresholds,
                "dimension_weights": self.quality_dimensions
            }
        }


def validate_final_dataset_quality_validator():
    """Validate the FinalDatasetQualityValidator functionality."""
    try:
        validator = FinalDatasetQualityValidator()
        assert hasattr(validator, "validate_dataset")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_final_dataset_quality_validator():
        pass
    else:
        pass
