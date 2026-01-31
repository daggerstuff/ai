"""
Category-specific standardization strategies for different data types.
Provides specialized processing for various conversation categories and domains.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation
from logger import get_logger
from standardizer import from_input_output_pair, from_simple_message_list


class DataCategory(Enum):
    """Categories of conversation data."""

    MENTAL_HEALTH = "mental_health"
    PSYCHOLOGY = "psychology"
    VOICE_TRAINING = "voice_training"
    REASONING = "reasoning"
    PERSONALITY = "personality"
    CLINICAL = "clinical"
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    GENERAL_CHAT = "general_chat"
    CRISIS_INTERVENTION = "crisis_intervention"
    ASSESSMENT = "assessment"
    UNKNOWN = "unknown"


@dataclass
class CategoryConfig:
    """Configuration for category-specific processing."""

    name: str
    description: str
    quality_thresholds: dict[str, float] = field(default_factory=dict)
    preprocessing_rules: list[str] = field(default_factory=list)
    validation_rules: list[str] = field(default_factory=list)
    metadata_requirements: list[str] = field(default_factory=list)


@dataclass
class StandardizationContext:
    """Context information for standardization."""

    category: DataCategory
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_requirements: dict[str, float] = field(default_factory=dict)


class CategoryStandardizer(ABC):
    """Abstract base class for category-specific standardizers."""

    def __init__(self, category: DataCategory, config: CategoryConfig | None = None):
        """
        Initialize category standardizer.

        Args:
            category: Data category this standardizer handles
            config: Optional configuration
        """
        self.category = category
        self.config = config or CategoryConfig(name=category.value, description="")
        self.logger = get_logger(f"{__name__}.{category.value}")

    @abstractmethod
    def can_handle(
        self, data: Any, context: StandardizationContext | None = None
    ) -> bool:
        """Check if this standardizer can handle the given data."""

    @abstractmethod
    def standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Standardize data according to category-specific rules."""

    def preprocess(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Any:
        """Apply category-specific preprocessing."""
        return data

    def validate(
        self,
        conversation: Conversation,
        context: StandardizationContext | None = None,
    ) -> dict[str, Any]:
        """Validate conversation according to category rules."""
        return {"valid": True, "issues": []}


class MentalHealthStandardizer(CategoryStandardizer):
    """Standardizer for mental health conversations."""

    def __init__(self):
        config = CategoryConfig(
            name="mental_health",
            description="Mental health counseling conversations",
            quality_thresholds={
                "min_empathy_score": 0.7,
                "min_therapeutic_appropriateness": 0.8,
                "max_crisis_indicators": 0.1,
            },
            preprocessing_rules=[
                "anonymize_personal_info",
                "normalize_emotional_language",
                "detect_crisis_indicators",
            ],
            validation_rules=[
                "check_therapeutic_boundaries",
                "validate_empathy_presence",
                "ensure_professional_tone",
            ],
        )
        super().__init__(DataCategory.MENTAL_HEALTH, config)

    def can_handle(
        self, data: Any, context: StandardizationContext | None = None
    ) -> bool:
        """Check if data contains mental health conversation patterns."""
        if isinstance(data, dict):
            # Check for mental health keywords
            text_content = self._extract_text_content(data)
            mental_health_keywords = [
                "therapy",
                "counseling",
                "depression",
                "anxiety",
                "stress",
                "mental health",
                "emotional",
                "feelings",
                "cope",
                "support",
            ]

            return any(
                keyword in text_content.lower() for keyword in mental_health_keywords
            )

        return False

    def standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Standardize mental health conversation data."""
        # Preprocess data
        processed_data = self.preprocess(data, context)

        # Convert to standard format
        if isinstance(processed_data, dict):
            if "messages" in processed_data:
                conversation = from_simple_message_list(
                    processed_data["messages"],
                    source=context.source if context else None,
                )
            elif "input" in processed_data and "output" in processed_data:
                conversation = from_input_output_pair(
                    processed_data["input"],
                    processed_data["output"],
                    source=context.source if context else None,
                )
            else:
                raise ValueError("Unable to standardize mental health data format")
        else:
            raise ValueError("Mental health data must be a dictionary")

        # Add category-specific metadata
        if not conversation.meta:
            conversation.meta = {}
        conversation.meta["category"] = self.category.value
        conversation.meta["therapeutic_context"] = True

        return conversation

    def preprocess(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Any:
        """Apply mental health specific preprocessing."""
        if isinstance(data, dict):
            processed = data.copy()

            # Anonymize personal information
            processed = self._anonymize_personal_info(processed)

            # Normalize emotional language
            processed = self._normalize_emotional_language(processed)

            # Detect and flag crisis indicators
            return self._detect_crisis_indicators(processed)


        return data

    def validate(
        self,
        conversation: Conversation,
        context: StandardizationContext | None = None,
    ) -> dict[str, Any]:
        """Validate mental health conversation."""
        issues = []

        # Check for therapeutic appropriateness
        if not self._has_therapeutic_tone(conversation):
            issues.append("Lacks therapeutic tone")

        # Check for empathy indicators
        if not self._has_empathy_indicators(conversation):
            issues.append("Lacks empathy indicators")

        # Check for crisis handling
        if self._has_crisis_content(conversation) and not self._has_crisis_response(
            conversation
        ):
            issues.append("Crisis content without appropriate response")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "category_specific": {
                "therapeutic_tone": self._has_therapeutic_tone(conversation),
                "empathy_present": self._has_empathy_indicators(conversation),
                "crisis_handled": not self._has_crisis_content(conversation)
                or self._has_crisis_response(conversation),
            },
        }

    def _extract_text_content(self, data: dict[str, Any]) -> str:
        """Extract all text content from data."""
        text_parts = []

        if "messages" in data:
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    text_parts.append(msg["content"])
        elif "input" in data:
            text_parts.append(str(data["input"]))
        elif "output" in data:
            text_parts.append(str(data["output"]))

        return " ".join(text_parts)

    def _anonymize_personal_info(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove or anonymize personal information."""

        # Simple anonymization - replace names, emails, phone numbers
        def anonymize_text(text: str) -> str:
            # Replace email addresses
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
            )
            # Replace phone numbers
            text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE]", text)
            # Replace potential names (capitalized words)
            return re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "[NAME]", text)

        processed = data.copy()

        if "messages" in processed:
            for msg in processed["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = anonymize_text(msg["content"])

        return processed

    def _normalize_emotional_language(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize emotional language for consistency."""
        emotion_mappings = {
            "feeling down": "feeling sad",
            "bummed out": "feeling sad",
            "over the moon": "feeling happy",
            "freaking out": "feeling anxious",
        }

        def normalize_text(text: str) -> str:
            for original, normalized in emotion_mappings.items():
                text = text.replace(original, normalized)
            return text

        processed = data.copy()

        if "messages" in processed:
            for msg in processed["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = normalize_text(msg["content"])

        return processed

    def _detect_crisis_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """Detect and flag crisis indicators."""
        crisis_keywords = [
            "suicide",
            "kill myself",
            "end it all",
            "hurt myself",
            "self-harm",
            "cutting",
            "overdose",
            "can't go on",
        ]

        text_content = self._extract_text_content(data).lower()
        has_crisis = any(keyword in text_content for keyword in crisis_keywords)

        processed = data.copy()
        if not isinstance(processed.get("meta"), dict):
            processed["meta"] = {}
        processed["meta"]["crisis_indicators"] = has_crisis

        return processed

    def _has_therapeutic_tone(self, conversation: Conversation) -> bool:
        """Check if conversation has therapeutic tone."""
        therapeutic_indicators = [
            "how are you feeling",
            "tell me more",
            "that sounds difficult",
            "i understand",
            "you're not alone",
            "let's explore",
        ]

        for message in conversation.messages:
            if message.role == "assistant":
                content_lower = message.content.lower()
                if any(
                    indicator in content_lower for indicator in therapeutic_indicators
                ):
                    return True

        return False

    def _has_empathy_indicators(self, conversation: Conversation) -> bool:
        """Check for empathy indicators in responses."""
        empathy_phrases = [
            "i can imagine",
            "that must be",
            "i hear you",
            "it sounds like",
            "i understand",
            "that's really hard",
            "you're going through",
        ]

        for message in conversation.messages:
            if message.role == "assistant":
                content_lower = message.content.lower()
                if any(phrase in content_lower for phrase in empathy_phrases):
                    return True

        return False

    def _has_crisis_content(self, conversation: Conversation) -> bool:
        """Check if conversation contains crisis content."""
        crisis_keywords = [
            "suicide",
            "kill myself",
            "end it all",
            "hurt myself",
            "self-harm",
        ]

        for message in conversation.messages:
            content_lower = message.content.lower()
            if any(keyword in content_lower for keyword in crisis_keywords):
                return True

        return False

    def _has_crisis_response(self, conversation: Conversation) -> bool:
        """Check if there's appropriate crisis response."""
        crisis_responses = [
            "crisis hotline",
            "emergency services",
            "immediate help",
            "please reach out",
            "professional help",
            "safety plan",
        ]

        for message in conversation.messages:
            if message.role == "assistant":
                content_lower = message.content.lower()
                if any(response in content_lower for response in crisis_responses):
                    return True

        return False


class PsychologyStandardizer(CategoryStandardizer):
    """Standardizer for psychology knowledge conversations."""

    def __init__(self):
        config = CategoryConfig(
            name="psychology",
            description="Psychology knowledge and educational content",
            quality_thresholds={
                "min_accuracy_score": 0.9,
                "min_educational_value": 0.8,
            },
            preprocessing_rules=[
                "validate_psychological_concepts",
                "standardize_terminology",
            ],
            validation_rules=[
                "check_factual_accuracy",
                "validate_citations",
                "ensure_educational_structure",
            ],
        )
        super().__init__(DataCategory.PSYCHOLOGY, config)

    def can_handle(
        self, data: Any, context: StandardizationContext | None = None
    ) -> bool:
        """Check if data contains psychology content."""
        if isinstance(data, dict):
            text_content = self._extract_text_content(data)
            psychology_keywords = [
                "psychology",
                "cognitive",
                "behavioral",
                "dsm",
                "diagnosis",
                "personality",
                "attachment",
                "therapy",
                "psychodynamic",
            ]

            return any(
                keyword in text_content.lower() for keyword in psychology_keywords
            )

        return False

    def standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Standardize psychology conversation data."""
        processed_data = self.preprocess(data, context)

        # Convert to standard format
        if isinstance(processed_data, dict):
            if "messages" in processed_data:
                conversation = from_simple_message_list(
                    processed_data["messages"],
                    source=context.source if context else None,
                )
            elif "instruction" in processed_data and "output" in processed_data:
                # Handle psychology instruction format
                conversation = from_input_output_pair(
                    processed_data["instruction"],
                    processed_data["output"],
                    source=context.source if context else None,
                )
            else:
                raise ValueError("Unable to standardize psychology data format")
        else:
            raise ValueError("Psychology data must be a dictionary")

        # Add category-specific metadata
        if not conversation.meta:
            conversation.meta = {}
        conversation.meta["category"] = self.category.value
        conversation.meta["educational_content"] = True

        return conversation

    def preprocess(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Any:
        """Apply psychology-specific preprocessing."""
        if isinstance(data, dict):
            processed = data.copy()

            # Standardize psychological terminology
            processed = self._standardize_terminology(processed)

            # Validate psychological concepts
            return self._validate_concepts(processed)


        return data

    def _extract_text_content(self, data: dict[str, Any]) -> str:
        """Extract all text content from data."""
        text_parts = []

        for key in ["messages", "instruction", "input", "output", "content"]:
            if key in data:
                if key == "messages" and isinstance(data[key], list):
                    for msg in data[key]:
                        if isinstance(msg, dict) and "content" in msg:
                            text_parts.append(msg["content"])
                else:
                    text_parts.append(str(data[key]))

        return " ".join(text_parts)

    def _standardize_terminology(self, data: dict[str, Any]) -> dict[str, Any]:
        """Standardize psychological terminology."""
        terminology_mappings = {
            "CBT": "Cognitive Behavioral Therapy",
            "DBT": "Dialectical Behavior Therapy",
            "PTSD": "Post-Traumatic Stress Disorder",
            "OCD": "Obsessive-Compulsive Disorder",
        }

        def standardize_text(text: str) -> str:
            for abbrev, full_term in terminology_mappings.items():
                text = text.replace(abbrev, full_term)
            return text

        processed = data.copy()

        if "messages" in processed:
            for msg in processed["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = standardize_text(msg["content"])

        return processed

    def _validate_concepts(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate psychological concepts mentioned."""
        # This would typically involve checking against a knowledge base
        # For now, just flag if certain concepts are mentioned

        text_content = self._extract_text_content(data).lower()

        psychological_concepts = [
            "classical conditioning",
            "operant conditioning",
            "attachment theory",
            "cognitive dissonance",
            "defense mechanisms",
            "transference",
        ]

        mentioned_concepts = [
            concept for concept in psychological_concepts if concept in text_content
        ]

        processed = data.copy()
        if not isinstance(processed.get("meta"), dict):
            processed["meta"] = {}
        processed["meta"]["psychological_concepts"] = mentioned_concepts

        return processed


class VoiceTrainingStandardizer(CategoryStandardizer):
    """Standardizer for voice training conversations."""

    def __init__(self):
        config = CategoryConfig(
            name="voice_training",
            description="Voice-derived personality training data",
            quality_thresholds={
                "min_authenticity_score": 0.8,
                "min_personality_consistency": 0.9,
            },
            preprocessing_rules=[
                "normalize_speech_patterns",
                "extract_personality_markers",
            ],
            validation_rules=["check_personality_consistency", "validate_authenticity"],
        )
        super().__init__(DataCategory.VOICE_TRAINING, config)

    def can_handle(
        self, data: Any, context: StandardizationContext | None = None
    ) -> bool:
        """Check if data is voice-derived."""
        if context and context.metadata:
            return (
                context.metadata.get("source_type") == "voice"
                or context.metadata.get("transcribed") is True
            )

        # Check for voice-related indicators in data
        if isinstance(data, dict):
            return "transcription" in data or "audio_source" in data

        return False

    def standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Standardize voice training data."""
        processed_data = self.preprocess(data, context)

        # Convert to standard format
        if isinstance(processed_data, dict):
            if "messages" in processed_data:
                conversation = from_simple_message_list(
                    processed_data["messages"],
                    source=context.source if context else None,
                )
            else:
                raise ValueError("Voice training data must contain messages")
        else:
            raise ValueError("Voice training data must be a dictionary")

        # Add category-specific metadata
        if not conversation.meta:
            conversation.meta = {}
        conversation.meta["category"] = self.category.value
        conversation.meta["voice_derived"] = True
        conversation.meta["authenticity_source"] = "human_voice"

        return conversation

    def preprocess(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Any:
        """Apply voice training specific preprocessing."""
        if isinstance(data, dict):
            processed = data.copy()

            # Normalize speech patterns
            processed = self._normalize_speech_patterns(processed)

            # Extract personality markers
            return self._extract_personality_markers(processed)


        return data

    def _normalize_speech_patterns(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize speech patterns from transcription."""

        def normalize_speech(text: str) -> str:
            # Remove filler words
            fillers = ["um", "uh", "like", "you know", "so"]
            for filler in fillers:
                text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)

            # Clean up extra spaces
            return re.sub(r"\s+", " ", text).strip()


        processed = data.copy()

        if "messages" in processed:
            for msg in processed["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = normalize_speech(msg["content"])

        return processed

    def _extract_personality_markers(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract personality markers from voice data."""
        # This would typically involve more sophisticated analysis
        # For now, just identify some basic patterns

        text_content = self._extract_text_content(data).lower()

        personality_indicators = {
            "empathy": ["i understand", "that must be hard", "i feel for you"],
            "optimism": ["things will get better", "positive", "hopeful"],
            "analytical": ["let me think", "analyze", "consider"],
            "supportive": ["you can do it", "i believe in you", "support"],
        }

        detected_traits = {}
        for trait, indicators in personality_indicators.items():
            detected_traits[trait] = any(
                indicator in text_content for indicator in indicators
            )

        processed = data.copy()
        if not isinstance(processed.get("meta"), dict):
            processed["meta"] = {}
        processed["meta"]["personality_traits"] = detected_traits

        return processed

    def _extract_text_content(self, data: dict[str, Any]) -> str:
        """Extract all text content from data."""
        text_parts = []

        if "messages" in data:
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    text_parts.append(msg["content"])

        return " ".join(text_parts)


class CategoryStandardizerRegistry:
    """Registry for category-specific standardizers."""

    def __init__(self):
        """Initialize registry with built-in standardizers."""
        self.standardizers: list[CategoryStandardizer] = []
        self.logger = get_logger(__name__)

        # Register built-in standardizers
        self.register(MentalHealthStandardizer())
        self.register(PsychologyStandardizer())
        self.register(VoiceTrainingStandardizer())

        self.logger.info(
            f"Initialized registry with {len(self.standardizers)} standardizers"
        )

    def register(self, standardizer: CategoryStandardizer) -> None:
        """Register a category standardizer."""
        self.standardizers.append(standardizer)
        self.logger.info(
            f"Registered standardizer for category: {standardizer.category.value}"
        )

    def get_standardizer(
        self, data: Any, context: StandardizationContext | None = None
    ) -> CategoryStandardizer | None:
        """Get the most appropriate standardizer for the data."""
        for standardizer in self.standardizers:
            if standardizer.can_handle(data, context):
                return standardizer

        return None

    def standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Standardize data using the most appropriate standardizer."""
        standardizer = self.get_standardizer(data, context)

        if standardizer:
            return standardizer.standardize(data, context)
        # Fallback to generic standardization
        self.logger.warning(
            "No specific standardizer found, using generic approach"
        )
        return self._generic_standardize(data, context)

    def _generic_standardize(
        self, data: Any, context: StandardizationContext | None = None
    ) -> Conversation:
        """Generic standardization fallback."""
        if isinstance(data, dict):
            if "messages" in data:
                return from_simple_message_list(
                    data["messages"], source=context.source if context else None
                )
            if "input" in data and "output" in data:
                return from_input_output_pair(
                    data["input"],
                    data["output"],
                    source=context.source if context else None,
                )

        raise ValueError("Unable to standardize data with generic approach")
