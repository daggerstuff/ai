"""
Conversation Complexity Scorer

Assesses the complexity of therapeutic conversations based on multiple dimensions:
- Emotional depth and range
- Clinical reasoning complexity
- Therapeutic technique sophistication
- Crisis intervention requirements
- Cultural sensitivity needs
"""

import logging
from typing import Dict, List

from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class ConversationComplexityScorer:
    """
    Scores conversation complexity across multiple therapeutic dimensions.

    Complexity scoring helps with:
    - Training data stratification (beginner → advanced)
    - Quality assessment
    - Curriculum design for The Empathy Gym™
    - Adaptive difficulty adjustment
    """

    # Complexity dimension weights
    DIMENSION_WEIGHTS = {
        "emotional_depth": 0.25,
        "clinical_reasoning": 0.25,
        "therapeutic_technique": 0.20,
        "crisis_level": 0.15,
        "cultural_sensitivity": 0.15,
    }

    # Complexity thresholds
    COMPLEXITY_LEVELS = {
        "beginner": (0.0, 0.3),
        "intermediate": (0.3, 0.6),
        "advanced": (0.6, 0.8),
        "expert": (0.8, 1.0),
    }

    def __init__(self):
        """Initialize the complexity scorer."""
        logger.info("Initialized ConversationComplexityScorer")

    def score_conversation(self, conversation: Conversation) -> Dict[str, float]:
        """
        Score a conversation's complexity across multiple dimensions.

        Args:
            conversation: Conversation to score

        Returns:
            Dictionary with dimension scores and overall complexity
        """
        scores = {
            "emotional_depth": self._score_emotional_depth(conversation),
            "clinical_reasoning": self._score_clinical_reasoning(conversation),
            "therapeutic_technique": self._score_therapeutic_technique(conversation),
            "crisis_level": self._score_crisis_level(conversation),
            "cultural_sensitivity": self._score_cultural_sensitivity(conversation),
        }

        # Calculate weighted overall complexity
        overall_complexity = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim] for dim in scores
        )

        # Determine complexity level
        complexity_level = self._get_complexity_level(overall_complexity)

        return {
            **scores,
            "overall_complexity": overall_complexity,
            "complexity_level": complexity_level,
        }

    def _score_emotional_depth(self, conversation: Conversation) -> float:
        """
        Score emotional depth based on:
        - Emotional vocabulary richness
        - Emotional range (number of different emotions)
        - Emotional intensity
        - Emotional transitions

        Args:
            conversation: Conversation to analyze

        Returns:
            Emotional depth score (0.0-1.0)
        """
        # Extract emotional indicators from messages
        emotional_keywords = {
            "high": ["devastated", "overwhelmed", "terrified", "hopeless", "suicidal"],
            "medium": ["anxious", "depressed", "angry", "sad", "worried", "stressed"],
            "low": ["concerned", "uneasy", "bothered", "uncomfortable"],
        }

        high_count = 0
        medium_count = 0
        low_count = 0

        for message in conversation.messages:
            content_lower = message.content.lower()
            for word in emotional_keywords["high"]:
                if word in content_lower:
                    high_count += 1
            for word in emotional_keywords["medium"]:
                if word in content_lower:
                    medium_count += 1
            for word in emotional_keywords["low"]:
                if word in content_lower:
                    low_count += 1

        # Calculate score based on emotional intensity distribution
        total_emotional_words = high_count + medium_count + low_count
        if total_emotional_words == 0:
            return 0.2  # Minimal emotional content

        # Weight high-intensity emotions more heavily
        weighted_score = (
            (high_count * 1.0) + (medium_count * 0.6) + (low_count * 0.3)
        ) / (total_emotional_words * 1.0)

        return min(weighted_score, 1.0)

    def _score_clinical_reasoning(self, conversation: Conversation) -> float:
        """
        Score clinical reasoning complexity based on:
        - Diagnostic thinking
        - Treatment planning
        - Risk assessment
        - Differential diagnosis

        Args:
            conversation: Conversation to analyze

        Returns:
            Clinical reasoning score (0.0-1.0)
        """
        # Check for clinical reasoning indicators
        clinical_indicators = {
            "diagnostic": ["diagnosis", "symptoms", "criteria", "assessment"],
            "treatment": ["treatment", "intervention", "therapy", "plan"],
            "risk": ["risk", "safety", "harm", "danger", "crisis"],
            "differential": ["differential", "rule out", "consider", "versus"],
        }

        indicator_counts = {key: 0 for key in clinical_indicators}

        for message in conversation.messages:
            content_lower = message.content.lower()
            for category, keywords in clinical_indicators.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        indicator_counts[category] += 1

        # Score based on breadth and depth of clinical reasoning
        categories_present = sum(1 for count in indicator_counts.values() if count > 0)
        total_indicators = sum(indicator_counts.values())

        if total_indicators == 0:
            return 0.1  # Minimal clinical reasoning

        # Combine breadth (categories) and depth (total indicators)
        breadth_score = categories_present / len(clinical_indicators)
        depth_score = min(total_indicators / 10.0, 1.0)

        return (breadth_score * 0.6) + (depth_score * 0.4)

    def _score_therapeutic_technique(self, conversation: Conversation) -> float:
        """
        Score therapeutic technique sophistication based on:
        - Specific therapeutic modalities (CBT, DBT, etc.)
        - Advanced techniques (reframing, validation, etc.)
        - Therapeutic alliance building

        Args:
            conversation: Conversation to analyze

        Returns:
            Therapeutic technique score (0.0-1.0)
        """
        # Check for therapeutic technique indicators
        techniques = {
            "cbt": ["cognitive", "thought", "belief", "reframe"],
            "dbt": ["mindfulness", "distress tolerance", "emotion regulation"],
            "psychodynamic": ["unconscious", "defense", "transference"],
            "humanistic": ["empathy", "validation", "unconditional"],
            "advanced": ["paradox", "metaphor", "socratic", "exposure"],
        }

        technique_counts = {key: 0 for key in techniques}

        for message in conversation.messages:
            if message.role != "assistant":
                continue  # Only score therapist responses

            content_lower = message.content.lower()
            for modality, keywords in techniques.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        technique_counts[modality] += 1

        # Score based on variety and sophistication
        modalities_used = sum(1 for count in technique_counts.values() if count > 0)
        total_techniques = sum(technique_counts.values())

        if total_techniques == 0:
            return 0.2  # Basic supportive therapy

        # Advanced techniques get bonus weight
        advanced_bonus = min(technique_counts["advanced"] / 5.0, 0.3)

        variety_score = modalities_used / len(techniques)
        depth_score = min(total_techniques / 8.0, 1.0)

        return min((variety_score * 0.5) + (depth_score * 0.5) + advanced_bonus, 1.0)

    def _score_crisis_level(self, conversation: Conversation) -> float:
        """
        Score crisis intervention requirements based on:
        - Suicidal ideation
        - Self-harm
        - Immediate danger
        - Acute distress

        Args:
            conversation: Conversation to analyze

        Returns:
            Crisis level score (0.0-1.0)
        """
        # Crisis indicators by severity
        crisis_keywords = {
            "severe": ["suicide", "kill myself", "end it all", "not worth living"],
            "moderate": ["self-harm", "hurt myself", "can't go on", "give up"],
            "mild": ["crisis", "emergency", "can't cope", "breaking down"],
        }

        severe_count = 0
        moderate_count = 0
        mild_count = 0

        for message in conversation.messages:
            content_lower = message.content.lower()
            for phrase in crisis_keywords["severe"]:
                if phrase in content_lower:
                    severe_count += 1
            for phrase in crisis_keywords["moderate"]:
                if phrase in content_lower:
                    moderate_count += 1
            for phrase in crisis_keywords["mild"]:
                if phrase in content_lower:
                    mild_count += 1

        # Weight severe indicators heavily
        if severe_count > 0:
            return 1.0  # Maximum crisis complexity
        elif moderate_count > 0:
            return 0.7 + min(moderate_count * 0.1, 0.3)
        elif mild_count > 0:
            return 0.4 + min(mild_count * 0.1, 0.3)
        else:
            return 0.1  # No crisis indicators

    def _score_cultural_sensitivity(self, conversation: Conversation) -> float:
        """
        Score cultural sensitivity requirements based on:
        - Cultural references
        - Language considerations
        - Identity factors
        - Diversity awareness

        Args:
            conversation: Conversation to analyze

        Returns:
            Cultural sensitivity score (0.0-1.0)
        """
        # Cultural sensitivity indicators
        cultural_keywords = [
            "culture",
            "cultural",
            "religion",
            "religious",
            "spiritual",
            "race",
            "racial",
            "ethnic",
            "ethnicity",
            "identity",
            "lgbtq",
            "transgender",
            "gender",
            "sexuality",
            "immigrant",
            "refugee",
            "language",
            "tradition",
            "discrimination",
            "bias",
            "prejudice",
            "minority",
        ]

        cultural_count = 0

        for message in conversation.messages:
            content_lower = message.content.lower()
            for keyword in cultural_keywords:
                if keyword in content_lower:
                    cultural_count += 1

        if cultural_count == 0:
            return 0.1  # Minimal cultural considerations

        # Score based on frequency of cultural references
        return min(0.3 + (cultural_count * 0.1), 1.0)

    def _get_complexity_level(self, overall_complexity: float) -> str:
        """
        Determine complexity level category.

        Args:
            overall_complexity: Overall complexity score

        Returns:
            Complexity level string
        """
        for level, (min_score, max_score) in self.COMPLEXITY_LEVELS.items():
            if min_score <= overall_complexity < max_score:
                return level

        return "expert"  # Fallback for 1.0

    def score_conversations(
        self, conversations: List[Conversation]
    ) -> List[Dict[str, float]]:
        """
        Score multiple conversations.

        Args:
            conversations: List of conversations to score

        Returns:
            List of complexity score dictionaries
        """
        scores = []
        for conversation in conversations:
            try:
                score = self.score_conversation(conversation)
                scores.append(score)
            except Exception as e:
                logger.error(
                    f"Error scoring conversation {conversation.conversation_id}: {e}",
                    exc_info=True,
                )
                # Return minimal complexity on error
                scores.append(
                    {
                        "emotional_depth": 0.0,
                        "clinical_reasoning": 0.0,
                        "therapeutic_technique": 0.0,
                        "crisis_level": 0.0,
                        "cultural_sensitivity": 0.0,
                        "overall_complexity": 0.0,
                        "complexity_level": "beginner",
                    }
                )

        return scores

    def get_complexity_distribution(
        self, conversations: List[Conversation]
    ) -> Dict[str, int]:
        """
        Get distribution of conversations across complexity levels.

        Args:
            conversations: List of conversations to analyze

        Returns:
            Dictionary mapping complexity levels to counts
        """
        scores = self.score_conversations(conversations)

        distribution = {level: 0 for level in self.COMPLEXITY_LEVELS}
        for score in scores:
            level = score["complexity_level"]
            distribution[level] += 1

        return distribution
