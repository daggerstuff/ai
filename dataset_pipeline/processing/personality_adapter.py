#!/usr/bin/env python3
"""
Personality-Aware Conversation Adaptation System for Task 6.11
Adapts conversation style and therapeutic approach based on Big Five personality traits.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersonalityProfile:
    """Simplified personality profile for adaptation."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    confidence_score: float


class PersonalityExtractor:
    """Simplified personality extractor for adaptation purposes."""

    def extract_personality(self, conversation: dict[str, Any]) -> PersonalityProfile:
        """Extract personality profile from conversation."""
        content = self._extract_content(conversation)

        # Simple keyword-based personality detection
        scores = {
            "openness": self._analyze_openness(content),
            "conscientiousness": self._analyze_conscientiousness(content),
            "extraversion": self._analyze_extraversion(content),
            "agreeableness": self._analyze_agreeableness(content),
            "neuroticism": self._analyze_neuroticism(content)
        }

        confidence = min(0.8, len(content.split()) / 100)  # Simple confidence based on content length

        return PersonalityProfile(
            openness=scores["openness"],
            conscientiousness=scores["conscientiousness"],
            extraversion=scores["extraversion"],
            agreeableness=scores["agreeableness"],
            neuroticism=scores["neuroticism"],
            confidence_score=confidence
        )

    def _extract_content(self, conversation: dict[str, Any]) -> str:
        """Extract content from conversation."""
        content = ""
        if "turns" in conversation:
            for turn in conversation["turns"]:
                if isinstance(turn, dict) and "content" in turn:
                    content += turn["content"] + " "
        elif "content" in conversation:
            content = conversation["content"]
        return content.strip()

    def _analyze_openness(self, content: str) -> float:
        """Analyze openness to experience."""
        openness_indicators = [
            "creative", "imaginative", "curious", "artistic", "innovative",
            "explore", "new ideas", "different", "unique", "abstract"
        ]

        score = sum(1 for indicator in openness_indicators if indicator in content.lower())
        return min(1.0, score / 5)  # Normalize to 0-1

    def _analyze_conscientiousness(self, content: str) -> float:
        """Analyze conscientiousness."""
        conscientiousness_indicators = [
            "organized", "planned", "schedule", "disciplined", "responsible",
            "goal", "achievement", "systematic", "careful", "thorough"
        ]

        score = sum(1 for indicator in conscientiousness_indicators if indicator in content.lower())
        return min(1.0, score / 5)

    def _analyze_extraversion(self, content: str) -> float:
        """Analyze extraversion."""
        extraversion_indicators = [
            "social", "outgoing", "people", "party", "energetic",
            "talkative", "assertive", "active", "enthusiastic", "group"
        ]

        score = sum(1 for indicator in extraversion_indicators if indicator in content.lower())
        return min(1.0, score / 5)

    def _analyze_agreeableness(self, content: str) -> float:
        """Analyze agreeableness."""
        agreeableness_indicators = [
            "helpful", "kind", "cooperative", "trusting", "empathetic",
            "caring", "supportive", "understanding", "compassionate", "gentle"
        ]

        score = sum(1 for indicator in agreeableness_indicators if indicator in content.lower())
        return min(1.0, score / 5)

    def _analyze_neuroticism(self, content: str) -> float:
        """Analyze neuroticism."""
        neuroticism_indicators = [
            "anxious", "worried", "stressed", "nervous", "emotional",
            "moody", "unstable", "sensitive", "tense", "overwhelmed"
        ]

        score = sum(1 for indicator in neuroticism_indicators if indicator in content.lower())
        return min(1.0, score / 5)


class CommunicationStyle(Enum):
    """Communication styles based on personality traits."""
    DIRECT = "direct"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    STRUCTURED = "structured"
    FLEXIBLE = "flexible"
    COLLABORATIVE = "collaborative"
    AUTHORITATIVE = "authoritative"


class TherapeuticApproach(Enum):
    """Therapeutic approaches matched to personality types."""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    HUMANISTIC = "humanistic_therapy"
    PSYCHODYNAMIC = "psychodynamic_therapy"
    SOLUTION_FOCUSED = "solution_focused_therapy"
    MINDFULNESS = "mindfulness_based_therapy"
    BEHAVIORAL = "behavioral_therapy"
    INTERPERSONAL = "interpersonal_therapy"


@dataclass
class PersonalityAdaptation:
    """Personality-based adaptation recommendations."""
    personality_profile: PersonalityProfile
    communication_style: CommunicationStyle
    therapeutic_approach: TherapeuticApproach
    conversation_pace: str  # "slow", "moderate", "fast"
    detail_level: str  # "high", "moderate", "low"
    emotional_support_level: str  # "high", "moderate", "low"
    structure_preference: str  # "high", "moderate", "low"
    feedback_style: str  # "direct", "gentle", "collaborative"
    motivation_approach: str  # "achievement", "affiliation", "autonomy"
    adaptation_confidence: float
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AdaptedResponse:
    """Response adapted for specific personality."""
    original_response: str
    adapted_response: str
    adaptation_type: str
    personality_factors: list[str]
    confidence_score: float
    adaptation_notes: str


class PersonalityAdapter:
    """
    Adapts conversation style and therapeutic approach based on personality traits.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the personality adapter."""
        self.personality_extractor = PersonalityExtractor()
        self.adaptation_rules = self._load_adaptation_rules(config_path)
        self.communication_patterns = self._load_communication_patterns()
        self.therapeutic_mappings = self._load_therapeutic_mappings()

        logger.info("PersonalityAdapter initialized successfully")

    def _load_adaptation_rules(self, config_path: str | None = None) -> dict[str, Any]:
        """Load personality adaptation rules."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        # Default adaptation rules based on Big Five traits
        return {
            "openness": {
                "high": {
                    "communication_style": "analytical",
                    "detail_level": "high",
                    "therapeutic_approach": "psychodynamic",
                    "conversation_pace": "moderate",
                    "structure_preference": "low"
                },
                "low": {
                    "communication_style": "structured",
                    "detail_level": "moderate",
                    "therapeutic_approach": "cbt",
                    "conversation_pace": "slow",
                    "structure_preference": "high"
                }
            },
            "conscientiousness": {
                "high": {
                    "communication_style": "structured",
                    "feedback_style": "direct",
                    "therapeutic_approach": "cbt",
                    "structure_preference": "high",
                    "motivation_approach": "achievement"
                },
                "low": {
                    "communication_style": "flexible",
                    "feedback_style": "gentle",
                    "therapeutic_approach": "humanistic",
                    "structure_preference": "low",
                    "motivation_approach": "autonomy"
                }
            },
            "extraversion": {
                "high": {
                    "communication_style": "collaborative",
                    "conversation_pace": "fast",
                    "therapeutic_approach": "interpersonal",
                    "emotional_support_level": "moderate",
                    "motivation_approach": "affiliation"
                },
                "low": {
                    "communication_style": "supportive",
                    "conversation_pace": "slow",
                    "therapeutic_approach": "mindfulness",
                    "emotional_support_level": "high",
                    "motivation_approach": "autonomy"
                }
            },
            "agreeableness": {
                "high": {
                    "communication_style": "empathetic",
                    "feedback_style": "gentle",
                    "therapeutic_approach": "humanistic",
                    "emotional_support_level": "high",
                    "motivation_approach": "affiliation"
                },
                "low": {
                    "communication_style": "direct",
                    "feedback_style": "direct",
                    "therapeutic_approach": "cbt",
                    "emotional_support_level": "moderate",
                    "motivation_approach": "achievement"
                }
            },
            "neuroticism": {
                "high": {
                    "communication_style": "supportive",
                    "feedback_style": "gentle",
                    "therapeutic_approach": "dbt",
                    "emotional_support_level": "high",
                    "conversation_pace": "slow"
                },
                "low": {
                    "communication_style": "direct",
                    "feedback_style": "direct",
                    "therapeutic_approach": "solution_focused",
                    "emotional_support_level": "moderate",
                    "conversation_pace": "moderate"
                }
            }
        }

    def _load_communication_patterns(self) -> dict[str, dict[str, list[str]]]:
        """Load communication patterns for different styles."""
        return {
            "direct": {
                "openings": [
                    "Let's focus on the main issue:",
                    "The key point here is:",
                    "What we need to address is:"
                ],
                "transitions": [
                    "Moving on to the next point:",
                    "Now let's consider:",
                    "The next step is:"
                ],
                "closings": [
                    "To summarize:",
                    "The main takeaway is:",
                    "Here's what we've established:"
                ]
            },
            "supportive": {
                "openings": [
                    "I understand this might be difficult, and",
                    "It's completely normal to feel this way, let's explore",
                    "I'm here to support you as we discuss"
                ],
                "transitions": [
                    "When you're ready, we can also look at",
                    "Another aspect we might consider is",
                    "If it feels right, let's also explore"
                ],
                "closings": [
                    "Remember, you're not alone in this",
                    "You've shown great courage in sharing this",
                    "Take your time processing what we've discussed"
                ]
            },
            "analytical": {
                "openings": [
                    "Let's examine the patterns in",
                    "If we analyze the situation, we can see",
                    "Breaking this down systematically:"
                ],
                "transitions": [
                    "This connects to another pattern:",
                    "Looking at this from another angle:",
                    "The data suggests that:"
                ],
                "closings": [
                    "Based on our analysis:",
                    "The evidence points to:",
                    "Logically, this leads us to:"
                ]
            },
            "empathetic": {
                "openings": [
                    "I can really hear the pain in what you're sharing",
                    "It sounds like you're carrying a lot right now",
                    "I can sense how important this is to you"
                ],
                "transitions": [
                    "I'm also hearing that",
                    "It seems like there's also",
                    "I'm wondering if you're also feeling"
                ],
                "closings": [
                    "Your feelings are completely valid",
                    "Thank you for trusting me with this",
                    "I'm honored that you've shared this with me"
                ]
            }
        }

    def _load_therapeutic_mappings(self) -> dict[str, dict[str, Any]]:
        """Load therapeutic approach mappings."""
        return {
            "cbt": {
                "focus": "thoughts_behaviors",
                "techniques": ["thought_challenging", "behavioral_experiments", "homework_assignments"],
                "language_style": "structured_problem_solving",
                "session_structure": "agenda_based"
            },
            "dbt": {
                "focus": "emotion_regulation",
                "techniques": ["mindfulness", "distress_tolerance", "interpersonal_effectiveness"],
                "language_style": "validation_and_skills",
                "session_structure": "skills_focused"
            },
            "humanistic": {
                "focus": "self_actualization",
                "techniques": ["active_listening", "unconditional_positive_regard", "empathic_reflection"],
                "language_style": "person_centered",
                "session_structure": "client_led"
            },
            "psychodynamic": {
                "focus": "unconscious_patterns",
                "techniques": ["interpretation", "transference_analysis", "insight_development"],
                "language_style": "exploratory_reflective",
                "session_structure": "process_oriented"
            }
        }

    def analyze_personality_for_adaptation(self, conversation: dict[str, Any]) -> PersonalityAdaptation:
        """Analyze personality and generate adaptation recommendations."""
        try:
            # Extract personality profile
            personality_profile = self.personality_extractor.extract_personality(conversation)

            # Determine adaptation based on personality traits
            adaptation = self._generate_adaptation(personality_profile)

            logger.info(f"Generated personality adaptation with {adaptation.adaptation_confidence:.2f} confidence")
            return adaptation

        except Exception as e:
            logger.error(f"Error analyzing personality for adaptation: {e}")
            # Return default adaptation
            return self._get_default_adaptation()

    def _generate_adaptation(self, personality_profile: PersonalityProfile) -> PersonalityAdaptation:
        """Generate adaptation based on personality profile."""
        # Determine dominant traits
        traits = {
            "openness": personality_profile.openness,
            "conscientiousness": personality_profile.conscientiousness,
            "extraversion": personality_profile.extraversion,
            "agreeableness": personality_profile.agreeableness,
            "neuroticism": personality_profile.neuroticism
        }

        # Find most influential traits
        dominant_trait = max(traits, key=traits.get)
        trait_level = "high" if traits[dominant_trait] > 0.6 else "low"

        # Get adaptation rules for dominant trait
        rules = self.adaptation_rules.get(dominant_trait, {}).get(trait_level, {})

        # Generate adaptation
        adaptation = PersonalityAdaptation(
            personality_profile=personality_profile,
            communication_style=CommunicationStyle(rules.get("communication_style", "supportive")),
            therapeutic_approach=TherapeuticApproach(rules.get("therapeutic_approach", "humanistic")),
            conversation_pace=rules.get("conversation_pace", "moderate"),
            detail_level=rules.get("detail_level", "moderate"),
            emotional_support_level=rules.get("emotional_support_level", "moderate"),
            structure_preference=rules.get("structure_preference", "moderate"),
            feedback_style=rules.get("feedback_style", "gentle"),
            motivation_approach=rules.get("motivation_approach", "autonomy"),
            adaptation_confidence=personality_profile.confidence_score
        )

        # Generate specific recommendations
        adaptation.recommendations = self._generate_recommendations(adaptation)

        return adaptation

    def _generate_recommendations(self, adaptation: PersonalityAdaptation) -> list[str]:
        """Generate specific recommendations based on adaptation."""
        recommendations = []

        # Communication style recommendations
        if adaptation.communication_style == CommunicationStyle.DIRECT:
            recommendations.append("Use clear, concise language and get straight to the point")
            recommendations.append("Provide specific, actionable advice")
        elif adaptation.communication_style == CommunicationStyle.SUPPORTIVE:
            recommendations.append("Use warm, encouraging language")
            recommendations.append("Validate emotions before offering solutions")
        elif adaptation.communication_style == CommunicationStyle.ANALYTICAL:
            recommendations.append("Provide detailed explanations and evidence")
            recommendations.append("Use logical frameworks and systematic approaches")

        # Therapeutic approach recommendations
        if adaptation.therapeutic_approach == TherapeuticApproach.CBT:
            recommendations.append("Focus on identifying and challenging negative thought patterns")
            recommendations.append("Suggest practical homework assignments")
        elif adaptation.therapeutic_approach == TherapeuticApproach.DBT:
            recommendations.append("Emphasize emotion regulation and distress tolerance skills")
            recommendations.append("Validate intense emotions while teaching coping strategies")
        elif adaptation.therapeutic_approach == TherapeuticApproach.HUMANISTIC:
            recommendations.append("Focus on the client's inherent capacity for growth")
            recommendations.append("Use reflective listening and avoid giving direct advice")

        # Pace and structure recommendations
        if adaptation.conversation_pace == "slow":
            recommendations.append("Allow plenty of time for processing and reflection")
            recommendations.append("Check in frequently to ensure understanding")
        elif adaptation.structure_preference == "high":
            recommendations.append("Provide clear agendas and structured sessions")
            recommendations.append("Use step-by-step approaches and clear goals")

        return recommendations

    def adapt_response(self, original_response: str, adaptation: PersonalityAdaptation) -> AdaptedResponse:
        """Adapt a response based on personality adaptation."""
        try:
            adapted_response = self._apply_adaptation_rules(original_response, adaptation)

            return AdaptedResponse(
                original_response=original_response,
                adapted_response=adapted_response,
                adaptation_type=adaptation.communication_style.value,
                personality_factors=[
                    f"Communication: {adaptation.communication_style.value}",
                    f"Approach: {adaptation.therapeutic_approach.value}",
                    f"Pace: {adaptation.conversation_pace}",
                    f"Support: {adaptation.emotional_support_level}"
                ],
                confidence_score=adaptation.adaptation_confidence,
                adaptation_notes=f"Adapted for {adaptation.communication_style.value} communication style"
            )

        except Exception as e:
            logger.error(f"Error adapting response: {e}")
            return AdaptedResponse(
                original_response=original_response,
                adapted_response=original_response,
                adaptation_type="none",
                personality_factors=[],
                confidence_score=0.0,
                adaptation_notes="No adaptation applied due to error"
            )

    def _apply_adaptation_rules(self, response: str, adaptation: PersonalityAdaptation) -> str:
        """Apply adaptation rules to modify response."""
        adapted_response = response

        # Get communication patterns for the style
        self.communication_patterns.get(adaptation.communication_style.value, {})

        # Modify based on communication style
        if adaptation.communication_style == CommunicationStyle.DIRECT:
            adapted_response = self._make_more_direct(adapted_response)
        elif adaptation.communication_style == CommunicationStyle.SUPPORTIVE:
            adapted_response = self._make_more_supportive(adapted_response)
        elif adaptation.communication_style == CommunicationStyle.ANALYTICAL:
            adapted_response = self._make_more_analytical(adapted_response)
        elif adaptation.communication_style == CommunicationStyle.EMPATHETIC:
            adapted_response = self._make_more_empathetic(adapted_response)

        # Adjust for emotional support level
        if adaptation.emotional_support_level == "high":
            adapted_response = self._increase_emotional_support(adapted_response)

        # Adjust for detail level
        if adaptation.detail_level == "high":
            adapted_response = self._add_more_detail(adapted_response)
        elif adaptation.detail_level == "low":
            adapted_response = self._simplify_response(adapted_response)

        return adapted_response

    def _make_more_direct(self, response: str) -> str:
        """Make response more direct and concise."""
        # Remove hedging language
        hedging_patterns = [
            r"I think maybe", r"perhaps", r"it might be that",
            r"you could possibly", r"it seems like maybe"
        ]

        for pattern in hedging_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Add direct language
        if not response.startswith(("Let's", "You should", "The key", "What you need")):
            response = "Let's focus on this: " + response

        return response.strip()

    def _make_more_supportive(self, response: str) -> str:
        """Make response more supportive and validating."""
        supportive_prefixes = [
            "I understand this is difficult, and ",
            "It's completely normal to feel this way. ",
            "You're showing great strength by addressing this. "
        ]

        # Add supportive prefix if not already present
        if not any(phrase in response.lower() for phrase in ["understand", "normal", "strength"]):
            response = supportive_prefixes[0] + response.lower()

        return response

    def _make_more_analytical(self, response: str) -> str:
        """Make response more analytical and detailed."""
        if not response.startswith(("Let's examine", "If we analyze", "Looking at this systematically")):
            response = "Let's examine this systematically: " + response

        # Add analytical language
        response = response.replace("because", "due to the fact that")
        return response.replace("so", "therefore")


    def _make_more_empathetic(self, response: str) -> str:
        """Make response more empathetic and emotionally attuned."""
        empathetic_phrases = [
            "I can really hear the pain in what you're sharing. ",
            "It sounds like you're carrying a lot right now. ",
            "I can sense how important this is to you. "
        ]

        if not any(phrase in response.lower() for phrase in ["hear", "sense", "feel"]):
            response = empathetic_phrases[0] + response

        return response

    def _increase_emotional_support(self, response: str) -> str:
        """Increase emotional support in response."""
        if not response.endswith(("You're not alone in this.", "I'm here to support you.", "You're doing great.")):
            response += " Remember, you're not alone in this."

        return response

    def _add_more_detail(self, response: str) -> str:
        """Add more detail to response."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP to add relevant details
        if len(response.split()) < 20:
            response += " Let me explain this in more detail to help you understand the full picture."

        return response

    def _simplify_response(self, response: str) -> str:
        """Simplify response for lower detail preference."""
        # Remove complex sentences and jargon
        sentences = response.split(". ")
        simplified_sentences = []

        for sentence in sentences:
            if len(sentence.split()) <= 15:  # Keep shorter sentences
                simplified_sentences.append(sentence)

        return ". ".join(simplified_sentences)

    def _get_default_adaptation(self) -> PersonalityAdaptation:
        """Get default adaptation when personality analysis fails."""
        default_profile = PersonalityProfile(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5, confidence_score=0.5
        )

        return PersonalityAdaptation(
            personality_profile=default_profile,
            communication_style=CommunicationStyle.SUPPORTIVE,
            therapeutic_approach=TherapeuticApproach.HUMANISTIC,
            conversation_pace="moderate",
            detail_level="moderate",
            emotional_support_level="moderate",
            structure_preference="moderate",
            feedback_style="gentle",
            motivation_approach="autonomy",
            adaptation_confidence=0.5,
            recommendations=["Use supportive, person-centered approach"]
        )

    def get_adaptation_summary(self, adaptation: PersonalityAdaptation) -> dict[str, Any]:
        """Get summary of adaptation for reporting."""
        return {
            "personality_scores": {
                "openness": adaptation.personality_profile.openness,
                "conscientiousness": adaptation.personality_profile.conscientiousness,
                "extraversion": adaptation.personality_profile.extraversion,
                "agreeableness": adaptation.personality_profile.agreeableness,
                "neuroticism": adaptation.personality_profile.neuroticism
            },
            "adaptation_settings": {
                "communication_style": adaptation.communication_style.value,
                "therapeutic_approach": adaptation.therapeutic_approach.value,
                "conversation_pace": adaptation.conversation_pace,
                "detail_level": adaptation.detail_level,
                "emotional_support_level": adaptation.emotional_support_level,
                "structure_preference": adaptation.structure_preference,
                "feedback_style": adaptation.feedback_style,
                "motivation_approach": adaptation.motivation_approach
            },
            "confidence": adaptation.adaptation_confidence,
            "recommendations": adaptation.recommendations
        }


def main():
    """Test the personality adapter."""
    adapter = PersonalityAdapter()

    # Test conversation
    test_conversation = {
        "conversation_id": "test_001",
        "turns": [
            {
                "speaker": "user",
                "content": "I've been feeling really overwhelmed lately. I like to have everything planned out and organized, but work has been so chaotic. I prefer to work alone and think things through carefully before making decisions."
            },
            {
                "speaker": "assistant",
                "content": "I understand you're feeling overwhelmed. It sounds like the lack of structure at work is really challenging for you."
            }
        ]
    }

    # Analyze personality and generate adaptation
    adaptation = adapter.analyze_personality_for_adaptation(test_conversation)


    for _i, _rec in enumerate(adaptation.recommendations, 1):
        pass

    # Test response adaptation
    original_response = "Maybe you could try to organize your tasks better. It might help to make a schedule."
    adapter.adapt_response(original_response, adaptation)



if __name__ == "__main__":
    main()
