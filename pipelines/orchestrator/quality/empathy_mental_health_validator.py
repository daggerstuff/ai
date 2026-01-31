"""
Empathy Mental Health Validator

Implements empathy scoring framework based on the EMNLP 2020 paper:
"A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support"
by Sharma et al. (behavioral-data/Empathy-Mental-Health)

This module provides comprehensive empathy assessment for therapeutic conversations,
measuring three key dimensions:
1. Emotional Reactions (ER) - Expressing warmth, compassion, concern
2. Interpretations (IP) - Understanding the seeker's feelings/situation
3. Explorations (EX) - Probing to improve understanding

Part of the Pixelated Empathy AI dataset pipeline quality framework.
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Handle imports
quality_path = Path(__file__).parent
pipeline_root = quality_path.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

try:
    from logger import get_logger

    logger = get_logger("dataset_pipeline.empathy_validator")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class EmpathyDimension(Enum):
    """Dimensions of empathy based on EMNLP 2020 framework."""

    EMOTIONAL_REACTIONS = "emotional_reactions"  # ER
    INTERPRETATIONS = "interpretations"  # IP
    EXPLORATIONS = "explorations"  # EX


class EmpathyLevel(Enum):
    """Levels of empathy expression."""

    NO_EMPATHY = 0  # No empathy communication
    WEAK_EMPATHY = 1  # Weak empathy communication
    STRONG_EMPATHY = 2  # Strong empathy communication


@dataclass
class EmpathyIndicators:
    """Linguistic indicators for empathy detection."""

    # Emotional Reactions (ER) - warmth, compassion, concern
    emotional_reaction_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(i'?m\s+(?:so\s+)?sorry)\b",
            r"\b(that\s+(?:sounds?|must\s+be)\s+(?:really\s+)?(?:hard|difficult|tough|painful))\b",
            r"\b(i\s+(?:can\s+)?(?:feel|sense|understand)\s+(?:your|the)\s+(?:pain|sadness|frustration))\b",
            r"\b(my\s+heart\s+goes\s+out)\b",
            r"\b(i'?m\s+here\s+(?:for\s+you|to\s+(?:help|support)))\b",
            r"\b(you'?re\s+not\s+alone)\b",
            r"\b(that\s+(?:sounds?|must\s+(?:be|have\s+been))\s+(?:very\s+)?(?:overwhelming|exhausting|devastating))\b",
            r"\b(i\s+(?:really\s+)?(?:appreciate|admire)\s+(?:you|your))\b",
            r"\b(sending\s+(?:you\s+)?(?:love|hugs|support))\b",
            r"\b((?:that's|it's)\s+(?:completely\s+)?(?:understandable|valid|okay))\b",
        ]
    )

    # Interpretations (IP) - understanding feelings/situation
    interpretation_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(it\s+(?:sounds?|seems?)\s+like\s+you)\b",
            r"\b((?:i\s+)?(?:can\s+)?(?:imagine|picture)\s+(?:how|that))\b",
            r"\b(you\s+(?:must|might)\s+(?:be\s+)?(?:feeling|experiencing))\b",
            r"\b((?:i\s+)?(?:sense|perceive|notice)\s+(?:that\s+)?you)\b",
            r"\b(what\s+(?:i'?m?\s+)?hear(?:ing)?\s+is)\b",
            r"\b(so\s+(?:essentially|basically)\s+you)\b",
            r"\b(if\s+i\s+understand\s+correctly)\b",
            r"\b(it\s+appears\s+(?:that\s+)?you)\b",
            r"\b(you\s+seem\s+(?:to\s+be\s+)?(?:dealing|struggling)\s+with)\b",
            r"\b(that\s+(?:experience|situation)\s+(?:sounds?|seems?)\s+like)\b",
        ]
    )

    # Explorations (EX) - probing to improve understanding
    exploration_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(can\s+you\s+(?:tell|share|describe)\s+(?:me\s+)?more)\b",
            r"\b(what\s+(?:do\s+you\s+)?(?:mean|think|feel)\s+(?:by|about|when))\b",
            r"\b(how\s+(?:does|did|do)\s+(?:that|this|it)\s+(?:make\s+you\s+)?(?:feel|affect))\b",
            r"\b((?:could|would)\s+you\s+(?:help\s+me\s+)?(?:understand|explain))\b",
            r"\b(what\s+(?:was|is)\s+(?:that|it)\s+like\s+(?:for\s+you)?)\b",
            r"\b(tell\s+me\s+(?:more\s+)?about)\b",
            r"\b(what\s+(?:happened|occurs?|comes\s+up)\s+(?:next|then|when))\b",
            r"\b(how\s+(?:long|often)\s+(?:have\s+you|has\s+this))\b",
            r"\b(what\s+(?:thoughts?|feelings?)\s+(?:come|came)\s+up)\b",
            r"\b((?:i'?m?\s+)?(?:curious|wondering)\s+(?:about|if|what))\b",
        ]
    )

    # Validation markers - acknowledging emotions
    validation_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(your\s+feelings\s+(?:are|make)\s+(?:valid|sense))\b",
            r"\b(it'?s\s+(?:okay|alright|normal)\s+to\s+(?:feel|be))\b",
            r"\b(anyone\s+(?:would|might)\s+(?:feel|react)\s+(?:the\s+)?(?:same|similarly))\b",
            r"\b(that\s+makes\s+(?:complete\s+)?sense)\b",
            r"\b(of\s+course\s+you\s+(?:feel|are))\b",
            r"\b((?:that's|it's)\s+a\s+(?:natural|normal|human)\s+(?:response|reaction))\b",
        ]
    )

    # Anti-empathy patterns (reduce score)
    anti_empathy_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(you\s+should(?:n't)?)\b",
            r"\b(just\s+(?:get\s+over|move\s+on|forget|stop))\b",
            r"\b(it'?s\s+not\s+(?:that\s+)?(?:bad|big\s+(?:of\s+)?a\s+deal))\b",
            r"\b((?:don't|do\s+not)\s+(?:feel|be|worry))\b",
            r"\b(you'?re\s+(?:just|being)\s+(?:too|over))\b",
            r"\b(at\s+least\s+(?:you|it))\b",
            r"\b(others?\s+(?:have|had)\s+it\s+worse)\b",
            r"\b((?:stop|quit)\s+(?:being|feeling))\b",
        ]
    )


@dataclass
class EmpathyScore:
    """Empathy assessment score for a single response."""

    emotional_reactions: float = 0.0  # 0-1 score
    interpretations: float = 0.0  # 0-1 score
    explorations: float = 0.0  # 0-1 score
    validation: float = 0.0  # 0-1 score (bonus dimension)
    anti_empathy_penalty: float = 0.0  # Penalty for anti-empathetic language
    overall_score: float = 0.0  # Weighted overall score
    level: EmpathyLevel = EmpathyLevel.NO_EMPATHY
    matched_patterns: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ConversationEmpathyAssessment:
    """Complete empathy assessment for a conversation."""

    conversation_id: str
    turn_scores: list[EmpathyScore]
    average_emotional_reactions: float
    average_interpretations: float
    average_explorations: float
    average_validation: float
    overall_empathy_score: float
    empathy_level: EmpathyLevel
    empathy_progression: list[float]  # How empathy changes through conversation
    issues: list[str]
    strengths: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class EmpathyMentalHealthValidator:
    """
    Comprehensive empathy validator for mental health conversations.

    Based on the EMNLP 2020 framework for understanding empathy in
    text-based mental health support.

    Measures three key dimensions:
    - Emotional Reactions (ER): Warmth, compassion, concern
    - Interpretations (IP): Understanding seeker's feelings/situation
    - Explorations (EX): Probing to improve understanding

    Additionally tracks:
    - Validation: Acknowledging and validating emotions
    - Anti-empathy: Penalizing dismissive or minimizing language
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        min_empathy_threshold: float = 0.5,
        enable_pattern_matching: bool = True,
    ):
        """
        Initialize the empathy validator.

        Args:
            weights: Custom weights for empathy dimensions
            min_empathy_threshold: Minimum acceptable empathy score
            enable_pattern_matching: Enable regex pattern matching
        """
        self.indicators = EmpathyIndicators()

        # Default weights based on EMNLP 2020 findings
        self.weights = weights or {
            "emotional_reactions": 0.30,
            "interpretations": 0.30,
            "explorations": 0.25,
            "validation": 0.15,
        }

        self.min_empathy_threshold = min_empathy_threshold
        self.enable_pattern_matching = enable_pattern_matching

        # Compile regex patterns for efficiency
        self._compile_patterns()

        # Emotion lexicon for sentiment analysis
        self._initialize_emotion_lexicon()

        logger.info(
            f"EmpathyMentalHealthValidator initialized: "
            f"threshold={min_empathy_threshold}, weights={self.weights}"
        )

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {
            "emotional_reactions": [
                re.compile(p, re.IGNORECASE) for p in self.indicators.emotional_reaction_patterns
            ],
            "interpretations": [
                re.compile(p, re.IGNORECASE) for p in self.indicators.interpretation_patterns
            ],
            "explorations": [
                re.compile(p, re.IGNORECASE) for p in self.indicators.exploration_patterns
            ],
            "validation": [
                re.compile(p, re.IGNORECASE) for p in self.indicators.validation_patterns
            ],
            "anti_empathy": [
                re.compile(p, re.IGNORECASE) for p in self.indicators.anti_empathy_patterns
            ],
        }

    def _initialize_emotion_lexicon(self) -> None:
        """Initialize emotion word lexicon for analysis."""
        self.emotion_lexicon = {
            "positive_emotions": {
                "happy", "hopeful", "grateful", "relieved", "proud", "confident",
                "peaceful", "content", "excited", "loved", "supported", "understood"
            },
            "negative_emotions": {
                "sad", "anxious", "depressed", "angry", "frustrated", "scared",
                "lonely", "hopeless", "overwhelmed", "exhausted", "hurt", "worried",
                "guilty", "ashamed", "confused", "lost", "empty", "numb"
            },
            "empathy_words": {
                "understand", "hear", "feel", "sense", "appreciate", "acknowledge",
                "recognize", "validate", "support", "care", "compassion", "empathy"
            },
        }

    def validate_conversation(
        self, conversation: Conversation
    ) -> ConversationEmpathyAssessment:
        """
        Validate empathy in a complete conversation.

        Args:
            conversation: Conversation object to validate

        Returns:
            ConversationEmpathyAssessment with detailed scores
        """
        turn_scores = []
        empathy_progression = []

        # Analyze each therapist/assistant turn
        for i, message in enumerate(conversation.messages):
            if message.role in ["assistant", "therapist", "counselor"]:
                # Get context from previous user message if available
                context = ""
                if i > 0 and conversation.messages[i - 1].role in ["user", "client", "seeker"]:
                    context = conversation.messages[i - 1].content

                score = self._score_single_response(message.content, context)
                turn_scores.append(score)
                empathy_progression.append(score.overall_score)

        # Calculate aggregates
        if not turn_scores:
            return self._create_empty_assessment(conversation.conversation_id)

        avg_er = sum(s.emotional_reactions for s in turn_scores) / len(turn_scores)
        avg_ip = sum(s.interpretations for s in turn_scores) / len(turn_scores)
        avg_ex = sum(s.explorations for s in turn_scores) / len(turn_scores)
        avg_val = sum(s.validation for s in turn_scores) / len(turn_scores)
        overall = sum(s.overall_score for s in turn_scores) / len(turn_scores)

        # Determine overall empathy level
        if overall >= 0.7:
            level = EmpathyLevel.STRONG_EMPATHY
        elif overall >= 0.4:
            level = EmpathyLevel.WEAK_EMPATHY
        else:
            level = EmpathyLevel.NO_EMPATHY

        # Generate insights
        issues, strengths, recommendations = self._generate_insights(
            turn_scores, avg_er, avg_ip, avg_ex, avg_val
        )

        return ConversationEmpathyAssessment(
            conversation_id=conversation.conversation_id,
            turn_scores=turn_scores,
            average_emotional_reactions=round(avg_er, 3),
            average_interpretations=round(avg_ip, 3),
            average_explorations=round(avg_ex, 3),
            average_validation=round(avg_val, 3),
            overall_empathy_score=round(overall, 3),
            empathy_level=level,
            empathy_progression=empathy_progression,
            issues=issues,
            strengths=strengths,
            recommendations=recommendations,
            metadata={
                "num_therapist_turns": len(turn_scores),
                "empathy_trend": self._calculate_trend(empathy_progression),
                "dimension_balance": self._calculate_dimension_balance(avg_er, avg_ip, avg_ex),
            },
        )

    def _score_single_response(
        self, response: str, context: str = ""
    ) -> EmpathyScore:
        """
        Score empathy in a single response.

        Args:
            response: The therapist/assistant response text
            context: Optional context from user's message

        Returns:
            EmpathyScore for the response
        """
        matched_patterns: dict[str, list[str]] = {
            "emotional_reactions": [],
            "interpretations": [],
            "explorations": [],
            "validation": [],
            "anti_empathy": [],
        }

        # Pattern matching for each dimension
        er_score = self._calculate_dimension_score(
            response, "emotional_reactions", matched_patterns
        )
        ip_score = self._calculate_dimension_score(
            response, "interpretations", matched_patterns
        )
        ex_score = self._calculate_dimension_score(
            response, "explorations", matched_patterns
        )
        val_score = self._calculate_dimension_score(
            response, "validation", matched_patterns
        )

        # Anti-empathy penalty
        anti_count = len(matched_patterns.get("anti_empathy", []))
        anti_penalty = min(anti_count * 0.15, 0.5)  # Max 50% penalty

        # Contextual analysis bonus
        context_bonus = self._contextual_analysis(response, context) if context else 0

        # Calculate overall score
        raw_score = (
            self.weights["emotional_reactions"] * er_score
            + self.weights["interpretations"] * ip_score
            + self.weights["explorations"] * ex_score
            + self.weights["validation"] * val_score
            + context_bonus
        )

        # Apply penalty and clamp
        overall = max(0, min(1, raw_score - anti_penalty))

        # Determine level
        if overall >= 0.7:
            level = EmpathyLevel.STRONG_EMPATHY
        elif overall >= 0.4:
            level = EmpathyLevel.WEAK_EMPATHY
        else:
            level = EmpathyLevel.NO_EMPATHY

        return EmpathyScore(
            emotional_reactions=round(er_score, 3),
            interpretations=round(ip_score, 3),
            explorations=round(ex_score, 3),
            validation=round(val_score, 3),
            anti_empathy_penalty=round(anti_penalty, 3),
            overall_score=round(overall, 3),
            level=level,
            matched_patterns=matched_patterns,
        )

    def _calculate_dimension_score(
        self,
        text: str,
        dimension: str,
        matched_patterns: dict[str, list[str]],
    ) -> float:
        """Calculate score for a specific empathy dimension."""
        if not self.enable_pattern_matching:
            return 0.5  # Default neutral score

        patterns = self.compiled_patterns.get(dimension, [])
        matches = []

        for pattern in patterns:
            found = pattern.findall(text)
            if found:
                matches.extend([str(m) if isinstance(m, tuple) else m for m in found])

        matched_patterns[dimension] = matches

        # Score based on number of matches (with diminishing returns)
        if len(matches) == 0:
            return 0.0
        elif len(matches) == 1:
            return 0.5
        elif len(matches) == 2:
            return 0.7
        elif len(matches) >= 3:
            return min(0.7 + len(matches) * 0.1, 1.0)

        return 0.0

    def _contextual_analysis(self, response: str, context: str) -> float:
        """
        Analyze response in context of user's message.

        Bonus for:
        - Referencing emotions mentioned by user
        - Acknowledging specific situations
        - Building on user's narrative
        """
        bonus = 0.0

        # Check if response references emotions from context
        context_lower = context.lower()
        response_lower = response.lower()

        # Find emotions mentioned by user
        user_emotions = set()
        for emotion in self.emotion_lexicon["negative_emotions"]:
            if emotion in context_lower:
                user_emotions.add(emotion)

        # Check if therapist acknowledges these emotions
        for emotion in user_emotions:
            if emotion in response_lower:
                bonus += 0.05

        # Check for reflection of content
        context_words = set(context_lower.split())
        response_words = set(response_lower.split())
        overlap = context_words & response_words

        # Remove common words
        common_words = {"i", "you", "the", "a", "an", "is", "are", "was", "were", "to", "and"}
        meaningful_overlap = overlap - common_words

        if len(meaningful_overlap) >= 3:
            bonus += 0.05

        return min(bonus, 0.15)  # Cap contextual bonus

    def _calculate_trend(self, progression: list[float]) -> str:
        """Calculate empathy trend through conversation."""
        if len(progression) < 2:
            return "stable"

        first_half = sum(progression[: len(progression) // 2]) / max(1, len(progression) // 2)
        second_half = sum(progression[len(progression) // 2 :]) / max(
            1, len(progression) - len(progression) // 2
        )

        diff = second_half - first_half
        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        return "stable"

    def _calculate_dimension_balance(
        self, er: float, ip: float, ex: float
    ) -> str:
        """Assess balance across empathy dimensions."""
        scores = {"emotional_reactions": er, "interpretations": ip, "explorations": ex}
        max_dim = max(scores, key=lambda k: scores[k])
        min_dim = min(scores, key=lambda k: scores[k])

        if scores[max_dim] - scores[min_dim] > 0.3:
            return f"unbalanced (strong: {max_dim}, weak: {min_dim})"
        return "balanced"

    def _generate_insights(
        self,
        turn_scores: list[EmpathyScore],
        avg_er: float,
        avg_ip: float,
        avg_ex: float,
        avg_val: float,
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate issues, strengths, and recommendations."""
        issues = []
        strengths = []
        recommendations = []

        # Analyze each dimension
        if avg_er < 0.3:
            issues.append("Low emotional reactions - lacks warmth and compassion")
            recommendations.append(
                "Increase expressions of care and emotional acknowledgment"
            )
        elif avg_er >= 0.6:
            strengths.append("Strong emotional reactions - demonstrates warmth")

        if avg_ip < 0.3:
            issues.append("Low interpretations - doesn't demonstrate understanding")
            recommendations.append(
                "Add more reflective statements showing understanding of client's situation"
            )
        elif avg_ip >= 0.6:
            strengths.append("Strong interpretations - shows good understanding")

        if avg_ex < 0.3:
            issues.append("Low explorations - doesn't probe to understand better")
            recommendations.append(
                "Include more open-ended questions to explore client's experience"
            )
        elif avg_ex >= 0.6:
            strengths.append("Strong explorations - actively seeks to understand")

        if avg_val < 0.3:
            issues.append("Low validation - doesn't acknowledge feelings as valid")
            recommendations.append("Add more validation of client's emotional experience")
        elif avg_val >= 0.6:
            strengths.append("Strong validation - effectively acknowledges emotions")

        # Check for anti-empathy
        avg_penalty = sum(s.anti_empathy_penalty for s in turn_scores) / len(turn_scores)
        if avg_penalty > 0.1:
            issues.append("Contains dismissive or minimizing language")
            recommendations.append(
                "Remove language that minimizes or dismisses client's experience"
            )

        return issues, strengths, recommendations

    def _create_empty_assessment(
        self, conversation_id: str
    ) -> ConversationEmpathyAssessment:
        """Create an empty assessment for conversations without therapist turns."""
        return ConversationEmpathyAssessment(
            conversation_id=conversation_id,
            turn_scores=[],
            average_emotional_reactions=0.0,
            average_interpretations=0.0,
            average_explorations=0.0,
            average_validation=0.0,
            overall_empathy_score=0.0,
            empathy_level=EmpathyLevel.NO_EMPATHY,
            empathy_progression=[],
            issues=["No therapist/assistant turns found in conversation"],
            strengths=[],
            recommendations=["Add therapist responses to evaluate empathy"],
            metadata={"num_therapist_turns": 0},
        )

    def batch_validate(
        self, conversations: list[Conversation]
    ) -> list[ConversationEmpathyAssessment]:
        """
        Validate empathy in multiple conversations.

        Args:
            conversations: List of conversations to validate

        Returns:
            List of ConversationEmpathyAssessment objects
        """
        assessments = []
        for conv in conversations:
            assessment = self.validate_conversation(conv)
            assessments.append(assessment)

        logger.info(
            f"Batch validated {len(conversations)} conversations, "
            f"average empathy: {sum(a.overall_empathy_score for a in assessments) / len(assessments):.2f}"
        )

        return assessments

    def filter_by_empathy(
        self,
        conversations: list[Conversation],
        min_score: Optional[float] = None,
        min_level: Optional[EmpathyLevel] = None,
    ) -> list[Conversation]:
        """
        Filter conversations by empathy criteria.

        Args:
            conversations: List of conversations to filter
            min_score: Minimum overall empathy score
            min_level: Minimum empathy level

        Returns:
            Filtered list of conversations
        """
        min_score = min_score or self.min_empathy_threshold
        min_level = min_level or EmpathyLevel.WEAK_EMPATHY

        filtered = []
        for conv in conversations:
            assessment = self.validate_conversation(conv)
            if (
                assessment.overall_empathy_score >= min_score
                and assessment.empathy_level.value >= min_level.value
            ):
                filtered.append(conv)

        logger.info(
            f"Filtered conversations: {len(filtered)}/{len(conversations)} passed empathy criteria"
        )

        return filtered

    def get_empathy_statistics(
        self, assessments: list[ConversationEmpathyAssessment]
    ) -> dict[str, Any]:
        """
        Calculate aggregate statistics from multiple assessments.

        Args:
            assessments: List of empathy assessments

        Returns:
            Dictionary with aggregate statistics
        """
        if not assessments:
            return {"error": "No assessments provided"}

        return {
            "total_conversations": len(assessments),
            "average_scores": {
                "emotional_reactions": sum(a.average_emotional_reactions for a in assessments)
                / len(assessments),
                "interpretations": sum(a.average_interpretations for a in assessments)
                / len(assessments),
                "explorations": sum(a.average_explorations for a in assessments)
                / len(assessments),
                "validation": sum(a.average_validation for a in assessments) / len(assessments),
                "overall": sum(a.overall_empathy_score for a in assessments) / len(assessments),
            },
            "level_distribution": {
                "no_empathy": sum(
                    1 for a in assessments if a.empathy_level == EmpathyLevel.NO_EMPATHY
                ),
                "weak_empathy": sum(
                    1 for a in assessments if a.empathy_level == EmpathyLevel.WEAK_EMPATHY
                ),
                "strong_empathy": sum(
                    1 for a in assessments if a.empathy_level == EmpathyLevel.STRONG_EMPATHY
                ),
            },
            "common_issues": self._get_common_items([a.issues for a in assessments]),
            "common_strengths": self._get_common_items([a.strengths for a in assessments]),
        }

    def _get_common_items(self, item_lists: list[list[str]], top_n: int = 5) -> list[str]:
        """Get most common items across multiple lists."""
        from collections import Counter

        all_items = [item for sublist in item_lists for item in sublist]
        counter = Counter(all_items)
        return [item for item, _ in counter.most_common(top_n)]


# Integration with quality assessment framework
def integrate_with_quality_framework(
    validator: EmpathyMentalHealthValidator,
    quality_metrics: dict[str, float],
    conversation: Conversation,
) -> dict[str, float]:
    """
    Integrate empathy scores with existing quality framework.

    Args:
        validator: EmpathyMentalHealthValidator instance
        quality_metrics: Existing quality metrics dictionary
        conversation: Conversation to assess

    Returns:
        Updated quality metrics with empathy scores
    """
    assessment = validator.validate_conversation(conversation)

    # Add empathy metrics to quality framework
    quality_metrics.update({
        "empathy_score": assessment.overall_empathy_score,
        "empathy_emotional_reactions": assessment.average_emotional_reactions,
        "empathy_interpretations": assessment.average_interpretations,
        "empathy_explorations": assessment.average_explorations,
        "empathy_validation": assessment.average_validation,
        "empathy_level": assessment.empathy_level.name,
    })

    # Adjust overall quality based on empathy
    if "overall_score" in quality_metrics:
        # Weight empathy at 15% of overall quality
        empathy_weight = 0.15
        quality_metrics["overall_score"] = (
            quality_metrics["overall_score"] * (1 - empathy_weight)
            + assessment.overall_empathy_score * empathy_weight
        )

    return quality_metrics


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Empathy Mental Health Validator")
    print("=" * 50)

    # Create validator
    validator = EmpathyMentalHealthValidator()

    # Test conversation
    test_conversation = Conversation(
        conversation_id="test_001",
        source="test",
        messages=[
            Message(role="user", content="I've been feeling really anxious lately. I can't sleep and I'm constantly worried about everything."),
            Message(
                role="assistant",
                content="I'm so sorry to hear you've been struggling with anxiety. That sounds really overwhelming. "
                "It makes sense that you're having trouble sleeping when your mind is racing with worries. "
                "Can you tell me more about what's been on your mind?",
            ),
            Message(role="user", content="It's mainly work stuff. I'm afraid I'm going to get fired."),
            Message(
                role="assistant",
                content="I can understand why that fear about your job would be keeping you up at night. "
                "Your feelings are completely valid - job security is so important. "
                "What's happening at work that's making you feel this way?",
            ),
        ],
    )

    # Validate
    assessment = validator.validate_conversation(test_conversation)

    print(f"\nConversation: {assessment.conversation_id}")
    print(f"Overall Empathy Score: {assessment.overall_empathy_score}")
    print(f"Empathy Level: {assessment.empathy_level.name}")
    print(f"\nDimension Scores:")
    print(f"  Emotional Reactions: {assessment.average_emotional_reactions}")
    print(f"  Interpretations: {assessment.average_interpretations}")
    print(f"  Explorations: {assessment.average_explorations}")
    print(f"  Validation: {assessment.average_validation}")
    print(f"\nStrengths: {assessment.strengths}")
    print(f"Issues: {assessment.issues}")
    print(f"Recommendations: {assessment.recommendations}")

