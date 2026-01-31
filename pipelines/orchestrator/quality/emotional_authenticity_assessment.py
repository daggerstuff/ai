"""
Emotional authenticity assessment system for conversation quality evaluation.

This module provides tools to assess the emotional authenticity and appropriateness
of conversations, particularly important for mental health and therapeutic contexts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from conversation_schema import Conversation, Message

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EmotionalAuthenticityMetrics:
    """Metrics for emotional authenticity assessment."""
    overall_score: float
    emotional_consistency_score: float
    empathy_expression_score: float
    emotional_vocabulary_score: float
    response_appropriateness_score: float
    emotional_progression_score: float
    authenticity_indicators_score: float
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    quality_level: str = ""


class EmotionalAuthenticityAssessor:
    """
    Comprehensive emotional authenticity assessment system.

    Evaluates conversations across multiple dimensions of emotional authenticity:
    1. Emotional consistency - emotions align with context
    2. Empathy expression - appropriate empathetic responses
    3. Emotional vocabulary - rich and appropriate emotional language
    4. Response appropriateness - emotionally appropriate responses
    5. Emotional progression - natural emotional flow
    6. Authenticity indicators - markers of genuine emotional expression
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the assessor with configuration."""
        self.config = config or {}

        # Default weights for different assessment dimensions
        self.weights = self.config.get("weights", {
            "emotional_consistency": 0.20,
            "empathy_expression": 0.18,
            "emotional_vocabulary": 0.16,
            "response_appropriateness": 0.16,
            "emotional_progression": 0.15,
            "authenticity_indicators": 0.15
        })

        # Quality thresholds
        self.thresholds = self.config.get("thresholds", {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40
        })

        # Emotional vocabulary sets
        self._initialize_emotional_vocabulary()

        # Empathy indicators
        self._initialize_empathy_indicators()

        # Authenticity markers
        self._initialize_authenticity_markers()

    def _initialize_emotional_vocabulary(self):
        """Initialize emotional vocabulary sets for assessment."""
        self.positive_emotions = {
            "happy", "joy", "excited", "grateful", "content", "pleased", "satisfied",
            "hopeful", "optimistic", "confident", "proud", "relieved", "calm", "peaceful",
            "love", "affection", "warmth", "compassion", "empathy", "understanding"
        }

        self.negative_emotions = {
            "sad", "angry", "frustrated", "anxious", "worried", "scared", "afraid",
            "depressed", "lonely", "hurt", "disappointed", "overwhelmed", "stressed",
            "guilty", "ashamed", "embarrassed", "confused", "lost", "hopeless",
            "irritated", "annoyed", "upset", "distressed", "troubled", "concerned"
        }

        self.complex_emotions = {
            "ambivalent", "conflicted", "bittersweet", "nostalgic", "melancholy",
            "apprehensive", "cautious", "skeptical", "uncertain", "mixed",
            "complicated", "nuanced", "layered", "multifaceted"
        }

        self.all_emotions = self.positive_emotions | self.negative_emotions | self.complex_emotions

    def _initialize_empathy_indicators(self):
        """Initialize empathy expression indicators."""
        self.empathy_phrases = {
            "i understand", "i hear you", "that sounds", "i can imagine", "i feel for you",
            "that must be", "i'm sorry you're", "it sounds like", "i can see",
            "that's understandable", "i appreciate", "thank you for sharing",
            "i'm here for you", "you're not alone", "that takes courage",
            "i validate", "your feelings are valid", "that makes sense"
        }

        self.empathy_words = {
            "understand", "hear", "feel", "imagine", "appreciate", "validate",
            "acknowledge", "recognize", "respect", "honor", "support"
        }

    def _initialize_authenticity_markers(self):
        """Initialize authenticity markers and indicators."""
        self.authenticity_positive = {
            "personal_disclosure": ["i feel", "i think", "i believe", "in my experience", "personally"],
            "vulnerability": ["i struggle", "i'm not sure", "i don't know", "i'm learning", "i've been there"],
            "specificity": ["specifically", "for example", "in particular", "such as", "like when"],
            "emotional_nuance": ["somewhat", "a bit", "kind of", "sort of", "partly", "mixed feelings"]
        }

        self.authenticity_negative = {
            "generic_responses": ["ok", "yes", "no", "sure", "alright", "fine"],
            "dismissive": ["just", "simply", "only", "merely", "whatever", "anyway"],
            "overly_clinical": ["diagnosis", "symptoms", "treatment", "disorder", "pathology"],
            "robotic": ["please note", "it is important", "you should", "it is recommended"]
        }

    def assess_emotional_authenticity(self, conversation: Conversation) -> EmotionalAuthenticityMetrics:
        """
        Assess the emotional authenticity of a conversation.

        Args:
            conversation: The conversation to assess

        Returns:
            EmotionalAuthenticityMetrics with detailed assessment results
        """
        logger.info(f"Assessing emotional authenticity for conversation {conversation.id}")

        if len(conversation.messages) < 2:
            return EmotionalAuthenticityMetrics(
                overall_score=0.0,
                emotional_consistency_score=0.0,
                empathy_expression_score=0.0,
                emotional_vocabulary_score=0.0,
                response_appropriateness_score=0.0,
                emotional_progression_score=0.0,
                authenticity_indicators_score=0.0,
                issues=["Insufficient messages for emotional authenticity assessment"],
                details={"conversation_length": len(conversation.messages)},
                quality_level="very_poor"
            )

        # Assess different dimensions
        emotional_consistency = self._assess_emotional_consistency(conversation.messages)
        empathy_expression = self._assess_empathy_expression(conversation.messages)
        emotional_vocabulary = self._assess_emotional_vocabulary(conversation.messages)
        response_appropriateness = self._assess_response_appropriateness(conversation.messages)
        emotional_progression = self._assess_emotional_progression(conversation.messages)
        authenticity_indicators = self._assess_authenticity_indicators(conversation.messages)

        # Calculate weighted overall score
        overall_score = (
            emotional_consistency["score"] * self.weights["emotional_consistency"] +
            empathy_expression["score"] * self.weights["empathy_expression"] +
            emotional_vocabulary["score"] * self.weights["emotional_vocabulary"] +
            response_appropriateness["score"] * self.weights["response_appropriateness"] +
            emotional_progression["score"] * self.weights["emotional_progression"] +
            authenticity_indicators["score"] * self.weights["authenticity_indicators"]
        )

        # Compile all issues
        all_issues = []
        all_issues.extend(emotional_consistency.get("issues", []))
        all_issues.extend(empathy_expression.get("issues", []))
        all_issues.extend(emotional_vocabulary.get("issues", []))
        all_issues.extend(response_appropriateness.get("issues", []))
        all_issues.extend(emotional_progression.get("issues", []))
        all_issues.extend(authenticity_indicators.get("issues", []))

        # Compile detailed results
        details = {
            "conversation_length": len(conversation.messages),
            "unique_roles": len({msg.role for msg in conversation.messages}),
            "emotional_consistency_details": emotional_consistency.get("details", {}),
            "empathy_expression_details": empathy_expression.get("details", {}),
            "emotional_vocabulary_details": emotional_vocabulary.get("details", {}),
            "response_appropriateness_details": response_appropriateness.get("details", {}),
            "emotional_progression_details": emotional_progression.get("details", {}),
            "authenticity_indicators_details": authenticity_indicators.get("details", {}),
            "quality_level": self._determine_quality_level(overall_score)
        }

        return EmotionalAuthenticityMetrics(
            overall_score=overall_score,
            emotional_consistency_score=emotional_consistency["score"],
            empathy_expression_score=empathy_expression["score"],
            emotional_vocabulary_score=emotional_vocabulary["score"],
            response_appropriateness_score=response_appropriateness["score"],
            emotional_progression_score=emotional_progression["score"],
            authenticity_indicators_score=authenticity_indicators["score"],
            issues=all_issues,
            details=details,
            quality_level=self._determine_quality_level(overall_score)
        )

    def _assess_emotional_consistency(self, messages: list[Message]) -> dict[str, Any]:
        """Assess emotional consistency throughout the conversation."""
        score = 1.0
        issues = []
        details = {}

        # Track emotional context and consistency
        emotional_context = []
        inconsistencies = 0

        for i, message in enumerate(messages):
            content_lower = message.content.lower()

            # Detect emotional expressions
            emotions_found = []
            for emotion in self.all_emotions:
                if emotion in content_lower:
                    emotions_found.append(emotion)

            emotional_context.append({
                "turn": i,
                "role": message.role,
                "emotions": emotions_found,
                "content_length": len(message.content)
            })

            # Check for emotional inconsistencies
            if i > 0 and emotions_found:
                prev_emotions = emotional_context[i-1]["emotions"]

                # Look for jarring emotional shifts without context
                if prev_emotions and emotions_found:
                    prev_positive = any(e in self.positive_emotions for e in prev_emotions)
                    curr_positive = any(e in self.positive_emotions for e in emotions_found)

                    if prev_positive != curr_positive and len(message.content) < 50:
                        inconsistencies += 1
                        issues.append(f"Abrupt emotional shift at turn {i} without sufficient context")

        # Penalize inconsistencies
        if inconsistencies > 0:
            score -= inconsistencies * 0.15

        # Check for emotional acknowledgment in responses
        acknowledgment_count = 0
        for i in range(1, len(messages)):
            if messages[i].role != messages[i-1].role:  # Different speakers
                prev_content = messages[i-1].content.lower()
                curr_content = messages[i].content.lower()

                # Check if current message acknowledges previous emotional content
                if any(emotion in prev_content for emotion in self.all_emotions):
                    if any(phrase in curr_content for phrase in self.empathy_phrases):
                        acknowledgment_count += 1

        # Reward emotional acknowledgment
        if len(messages) > 2:
            acknowledgment_ratio = acknowledgment_count / (len(messages) // 2)
            if acknowledgment_ratio < 0.3:
                issues.append("Low emotional acknowledgment in responses")
                score -= 0.1

        score = max(0.0, min(1.0, score))

        details.update({
            "emotional_inconsistencies": inconsistencies,
            "emotional_acknowledgment_count": acknowledgment_count,
            "emotional_context_turns": len([ctx for ctx in emotional_context if ctx["emotions"]])
        })

        return {"score": score, "issues": issues, "details": details}

    def _assess_empathy_expression(self, messages: list[Message]) -> dict[str, Any]:
        """Assess empathy expression in responses."""
        score = 1.0
        issues = []
        details = {}

        empathy_expressions = 0
        total_response_opportunities = 0

        for i in range(1, len(messages)):
            if messages[i].role != messages[i-1].role:  # Response to different speaker
                prev_content = messages[i-1].content.lower()
                curr_content = messages[i].content.lower()

                # Check if previous message contains emotional content
                has_emotional_content = any(emotion in prev_content for emotion in self.all_emotions)
                has_distress_indicators = any(word in prev_content for word in
                                            ["help", "problem", "difficult", "struggle", "hard", "worried", "scared"])

                if has_emotional_content or has_distress_indicators:
                    total_response_opportunities += 1

                    # Check for empathy expressions
                    empathy_found = False
                    for phrase in self.empathy_phrases:
                        if phrase in curr_content:
                            empathy_expressions += 1
                            empathy_found = True
                            break

                    if not empathy_found:
                        # Check for empathy words
                        for word in self.empathy_words:
                            if word in curr_content:
                                empathy_expressions += 1
                                empathy_found = True
                                break

                    # Penalize lack of empathy in emotional contexts
                    if not empathy_found and (has_emotional_content or has_distress_indicators):
                        if len(curr_content) < 20:  # Very short response to emotional content
                            issues.append(f"Lack of empathy in response to emotional content at turn {i}")

        # Calculate empathy ratio
        if total_response_opportunities > 0:
            empathy_ratio = empathy_expressions / total_response_opportunities
            if empathy_ratio == 0.0:
                issues.append("No empathy expression in emotional contexts")
                score -= 0.4
            elif empathy_ratio < 0.3:
                issues.append("Very low empathy expression in emotional contexts")
                score -= 0.3
            elif empathy_ratio < 0.6:
                issues.append("Low empathy expression in emotional contexts")
                score -= 0.2
        else:
            # No emotional contexts found
            issues.append("No clear emotional contexts for empathy assessment")
            score = 0.8  # Neutral score

        score = max(0.0, min(1.0, score))

        details.update({
            "empathy_expressions": empathy_expressions,
            "response_opportunities": total_response_opportunities,
            "empathy_ratio": empathy_expressions / max(1, total_response_opportunities)
        })

        return {"score": score, "issues": issues, "details": details}

    def _assess_emotional_vocabulary(self, messages: list[Message]) -> dict[str, Any]:
        """Assess the richness and appropriateness of emotional vocabulary."""
        score = 1.0
        issues = []
        details = {}

        total_words = 0
        emotional_words = 0
        unique_emotions = set()

        for message in messages:
            words = message.content.lower().split()
            total_words += len(words)

            for word in words:
                if word in self.all_emotions:
                    emotional_words += 1
                    unique_emotions.add(word)

        # Calculate emotional vocabulary richness
        if total_words > 0:
            emotional_density = emotional_words / total_words
            if emotional_density < 0.02:  # Less than 2% emotional words
                issues.append("Low emotional vocabulary density")
                score -= 0.15

        # Check for emotional vocabulary diversity
        if len(unique_emotions) < 3 and len(messages) > 4:
            issues.append("Limited emotional vocabulary diversity")
            score -= 0.1

        # Reward complex emotional expressions
        complex_emotion_count = sum(1 for emotion in unique_emotions if emotion in self.complex_emotions)
        if complex_emotion_count > 0:
            score += 0.05  # Small bonus for emotional complexity

        score = max(0.0, min(1.0, score))

        details.update({
            "total_words": total_words,
            "emotional_words": emotional_words,
            "unique_emotions": len(unique_emotions),
            "emotional_density": emotional_words / max(1, total_words),
            "complex_emotions": complex_emotion_count
        })

        return {"score": score, "issues": issues, "details": details}

    def _assess_response_appropriateness(self, messages: list[Message]) -> dict[str, Any]:
        """Assess emotional appropriateness of responses."""
        score = 1.0
        issues = []
        details = {}

        inappropriate_responses = 0
        total_responses = 0

        for i in range(1, len(messages)):
            if messages[i].role != messages[i-1].role:  # Response to different speaker
                prev_content = messages[i-1].content.lower()
                curr_content = messages[i].content.lower()
                total_responses += 1

                # Check for inappropriate responses to emotional content
                has_negative_emotion = any(emotion in prev_content for emotion in self.negative_emotions)
                has_positive_emotion = any(emotion in prev_content for emotion in self.positive_emotions)

                # Inappropriate positive response to negative emotion
                if has_negative_emotion and any(word in curr_content for word in ["great", "awesome", "fantastic", "wonderful", "congratulations"]):
                    inappropriate_responses += 1
                    issues.append(f"Inappropriate positive response to negative emotion at turn {i}")

                # Check for clearly inappropriate celebratory responses to negative situations
                negative_situations = ["lost", "losing", "fired", "died", "death", "sick", "illness", "devastated", "heartbroken"]
                positive_responses = ["great", "congratulations", "awesome", "fantastic", "wonderful", "amazing", "excellent"]

                if any(situation in prev_content for situation in negative_situations):
                    if any(response in curr_content for response in positive_responses):
                        inappropriate_responses += 1
                        issues.append(f"Inappropriate celebratory response to negative situation at turn {i}")

                # Dismissive response to emotional content
                if (has_negative_emotion or has_positive_emotion):
                    if any(phrase in curr_content for phrase in ["get over it", "move on", "just relax", "calm down"]):
                        inappropriate_responses += 1
                        issues.append(f"Dismissive response to emotional content at turn {i}")

                # Generic response to detailed emotional sharing
                if len(prev_content) > 50 and (any(emotion in prev_content for emotion in self.all_emotions) or
                                               any(word in prev_content for word in ["help", "struggling", "problem", "difficult"])):
                    if curr_content.strip().lower() in ["ok", "ok.", "i see", "yes", "no", "sure", "sure."]:
                        inappropriate_responses += 1
                        issues.append(f"Generic response to detailed emotional sharing at turn {i}")

                # Very short responses to substantial emotional content
                if len(prev_content) > 80 and len(curr_content.strip()) < 10:
                    if any(emotion in prev_content for emotion in self.all_emotions):
                        inappropriate_responses += 1
                        issues.append(f"Inadequately brief response to emotional content at turn {i}")

        # Calculate appropriateness ratio
        if total_responses > 0:
            appropriateness_ratio = 1 - (inappropriate_responses / total_responses)
            score = appropriateness_ratio

        score = max(0.0, min(1.0, score))

        details.update({
            "inappropriate_responses": inappropriate_responses,
            "total_responses": total_responses,
            "appropriateness_ratio": 1 - (inappropriate_responses / max(1, total_responses))
        })

        return {"score": score, "issues": issues, "details": details}

    def _assess_emotional_progression(self, messages: list[Message]) -> dict[str, Any]:
        """Assess natural emotional progression throughout conversation."""
        score = 1.0
        issues = []
        details = {}

        emotional_trajectory = []
        abrupt_shifts = 0

        for i, message in enumerate(messages):
            content_lower = message.content.lower()

            # Categorize emotional tone
            positive_count = sum(1 for emotion in self.positive_emotions if emotion in content_lower)
            negative_count = sum(1 for emotion in self.negative_emotions if emotion in content_lower)

            if positive_count > negative_count:
                tone = "positive"
            elif negative_count > positive_count:
                tone = "negative"
            else:
                tone = "neutral"

            emotional_trajectory.append(tone)

            # Check for abrupt emotional shifts
            if i > 0 and emotional_trajectory[i] != emotional_trajectory[i-1]:
                if emotional_trajectory[i-1] != "neutral" and emotional_trajectory[i] != "neutral":
                    # Direct shift from positive to negative or vice versa
                    if len(message.content) < 30:  # Without sufficient explanation
                        abrupt_shifts += 1
                        issues.append(f"Abrupt emotional shift at turn {i} without context")

        # Penalize abrupt shifts
        if abrupt_shifts > 0:
            score -= abrupt_shifts * 0.1

        # Check for emotional resolution patterns
        if len(emotional_trajectory) > 4:
            # Look for positive progression (negative -> neutral -> positive)
            has_progression = False
            for i in range(2, len(emotional_trajectory)):
                if (emotional_trajectory[i-2] == "negative" and
                    emotional_trajectory[i-1] == "neutral" and
                    emotional_trajectory[i] == "positive"):
                    has_progression = True
                    break

            if has_progression:
                score += 0.05  # Small bonus for positive emotional progression

        score = max(0.0, min(1.0, score))

        details.update({
            "emotional_trajectory": emotional_trajectory,
            "abrupt_shifts": abrupt_shifts,
            "trajectory_length": len(emotional_trajectory)
        })

        return {"score": score, "issues": issues, "details": details}

    def _assess_authenticity_indicators(self, messages: list[Message]) -> dict[str, Any]:
        """Assess markers of authentic vs artificial emotional expression."""
        score = 1.0
        issues = []
        details = {}

        positive_indicators = 0
        negative_indicators = 0

        for message in messages:
            content_lower = message.content.lower()

            # Count positive authenticity indicators
            for _category, phrases in self.authenticity_positive.items():
                for phrase in phrases:
                    if phrase in content_lower:
                        positive_indicators += 1
                        break  # Count each category once per message

            # Count negative authenticity indicators
            for _category, phrases in self.authenticity_negative.items():
                for phrase in phrases:
                    if phrase in content_lower:
                        negative_indicators += 1
                        break  # Count each category once per message

        # Calculate authenticity ratio
        total_indicators = positive_indicators + negative_indicators
        if total_indicators > 0:
            authenticity_ratio = positive_indicators / total_indicators
            score = authenticity_ratio
        else:
            # No clear indicators found
            score = 0.7  # Neutral score
            issues.append("No clear authenticity indicators found")

        # Penalize excessive negative indicators
        if negative_indicators > positive_indicators * 2:
            issues.append("High frequency of inauthentic language patterns")
            score -= 0.2

        score = max(0.0, min(1.0, score))

        details.update({
            "positive_indicators": positive_indicators,
            "negative_indicators": negative_indicators,
            "authenticity_ratio": positive_indicators / max(1, total_indicators)
        })

        return {"score": score, "issues": issues, "details": details}

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        if score >= self.thresholds["good"]:
            return "good"
        if score >= self.thresholds["acceptable"]:
            return "acceptable"
        if score >= self.thresholds["poor"]:
            return "poor"
        return "very_poor"


# Backward compatibility function
def assess_emotional_authenticity(conversation: Conversation, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Backward compatibility function for emotional authenticity assessment.

    Args:
        conversation: The conversation to assess
        config: Optional configuration dictionary

    Returns:
        Dictionary with 'score', 'issues', and 'details' keys
    """
    assessor = EmotionalAuthenticityAssessor(config)
    metrics = assessor.assess_emotional_authenticity(conversation)

    return {
        "score": metrics.overall_score,
        "issues": metrics.issues,
        "details": metrics.details
    }
