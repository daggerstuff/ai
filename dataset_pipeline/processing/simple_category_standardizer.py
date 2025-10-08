"""
Simple Category Standardizer

Concrete implementation of category-specific standardization for different data types.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger
from standardizer import from_input_output_pair, from_simple_message_list

logger = get_logger(__name__)


class DataCategory(Enum):
    """Categories of conversation data."""

    MENTAL_HEALTH = "mental_health"
    PSYCHOLOGY = "psychology"
    VOICE_TRAINING = "voice_training"
    REASONING = "reasoning"
    PERSONALITY = "personality"
    GENERAL = "general"


@dataclass
class CategoryConfig:
    """Configuration for category-specific processing."""

    category: DataCategory
    quality_threshold: float = 0.7
    min_message_length: int = 10
    max_message_length: int = 2000
    required_fields: list[str] = field(default_factory=list)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimpleCategoryStandardizer:
    """Simple concrete implementation of category standardizer."""

    def __init__(
        self,
        category: DataCategory = DataCategory.GENERAL,
        config: CategoryConfig | None = None,
    ):
        self.category = category
        self.config = config or CategoryConfig(category=category)
        self.logger = get_logger(__name__)

        # Category-specific patterns and rules
        self.category_patterns = self._get_category_patterns()

        logger.info(f"SimpleCategoryStandardizer initialized for {category.value}")

    def can_handle(self, data: dict[str, Any]) -> bool:
        """Check if this standardizer can handle the given data."""
        # Simple heuristic based on content keywords
        content = self._extract_text_content(data)
        patterns = self.category_patterns.get(self.category, [])

        # Check if any category-specific patterns match
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)

    def standardize(self, data: dict[str, Any]) -> Conversation:
        """Standardize data to conversation format."""
        # Apply category-specific preprocessing
        processed_data = self._preprocess_data(data)

        # Convert to standard conversation format
        if "messages" in processed_data:
            conversation = from_simple_message_list(
                processed_data["messages"],
                conversation_id=processed_data.get("id"),
                source=self.config.source,
            )
        elif "input" in processed_data and "output" in processed_data:
            conversation = from_input_output_pair(
                processed_data["input"],
                processed_data["output"],
                conversation_id=processed_data.get("id"),
                source=self.config.source,
            )
        else:
            # Fallback: treat as single message
            conversation = from_simple_message_list(
                [{"role": "user", "content": str(processed_data)}],
                source=self.config.source,
            )

        # Apply category-specific post-processing
        return self._postprocess_conversation(conversation)


    def _get_category_patterns(self) -> dict[DataCategory, list[str]]:
        """Get category-specific patterns for content detection."""
        return {
            DataCategory.MENTAL_HEALTH: [
                r"\b(?:therapy|counseling|depression|anxiety|mental health|psychological)\b",
                r"\b(?:therapist|counselor|psychiatrist|psychologist)\b",
                r"\b(?:feeling|emotions|mood|stress|trauma|ptsd)\b",
            ],
            DataCategory.PSYCHOLOGY: [
                r"\b(?:psychology|psychological|cognitive|behavioral)\b",
                r"\b(?:personality|trait|behavior|cognition)\b",
                r"\b(?:research|study|experiment|analysis)\b",
            ],
            DataCategory.VOICE_TRAINING: [
                r"\b(?:voice|speech|audio|pronunciation|accent)\b",
                r"\b(?:speaking|talking|conversation|dialogue)\b",
                r"\b(?:training|practice|exercise|drill)\b",
            ],
            DataCategory.REASONING: [
                r"\b(?:reasoning|logic|problem|solution|analysis)\b",
                r"\b(?:think|thought|consider|conclude|deduce)\b",
                r"\b(?:step|process|method|approach|strategy)\b",
            ],
            DataCategory.PERSONALITY: [
                r"\b(?:personality|character|trait|temperament)\b",
                r"\b(?:introvert|extrovert|openness|conscientiousness)\b",
                r"\b(?:behavior|attitude|preference|style)\b",
            ],
            DataCategory.GENERAL: [
                r"\b(?:conversation|chat|talk|discuss)\b",
                r"\b(?:question|answer|response|reply)\b",
            ],
        }

    def _extract_text_content(self, data: dict[str, Any]) -> str:
        """Extract text content from data for pattern matching."""
        content_parts = []

        # Extract from various possible fields
        for field in ["content", "text", "message", "input", "output"]:
            if field in data:
                content_parts.append(str(data[field]))

        # Extract from messages array
        if "messages" in data and isinstance(data["messages"], list):
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    content_parts.append(str(msg["content"]))

        return " ".join(content_parts)

    def _preprocess_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply category-specific preprocessing."""
        processed = data.copy()

        if self.category == DataCategory.MENTAL_HEALTH:
            processed = self._preprocess_mental_health(processed)
        elif self.category == DataCategory.PSYCHOLOGY:
            processed = self._preprocess_psychology(processed)
        elif self.category == DataCategory.VOICE_TRAINING:
            processed = self._preprocess_voice_training(processed)
        elif self.category == DataCategory.REASONING:
            processed = self._preprocess_reasoning(processed)
        elif self.category == DataCategory.PERSONALITY:
            processed = self._preprocess_personality(processed)

        return processed

    def _preprocess_mental_health(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess mental health data."""
        # Add mental health specific metadata
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["category"] = "mental_health"
        data["metadata"]["requires_sensitivity"] = True

        # Clean sensitive information (placeholder)
        # In practice, this would implement proper anonymization

        return data

    def _preprocess_psychology(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess psychology data."""
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["category"] = "psychology"
        data["metadata"]["research_context"] = True

        return data

    def _preprocess_voice_training(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess voice training data."""
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["category"] = "voice_training"
        data["metadata"]["audio_context"] = True

        return data

    def _preprocess_reasoning(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess reasoning data."""
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["category"] = "reasoning"
        data["metadata"]["logical_structure"] = True

        return data

    def _preprocess_personality(self, data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess personality data."""
        if "metadata" not in data:
            data["metadata"] = {}

        data["metadata"]["category"] = "personality"
        data["metadata"]["trait_analysis"] = True

        return data

    def _postprocess_conversation(self, conversation: Conversation) -> Conversation:
        """Apply category-specific post-processing to conversation."""
        # Add category-specific metadata to conversation
        if not conversation.metadata:
            conversation.metadata = {}

        conversation.metadata["category"] = self.category.value
        conversation.metadata["standardizer"] = "SimpleCategoryStandardizer"
        conversation.metadata["processed_at"] = str(datetime.now())

        # Apply category-specific message processing
        for message in conversation.messages:
            message = self._postprocess_message(message)

        return conversation

    def _postprocess_message(self, message: Message) -> Message:
        """Apply category-specific post-processing to individual messages."""
        # Length validation
        if len(message.content) < self.config.min_message_length:
            # Pad short messages or mark for filtering
            if not message.metadata:
                message.metadata = {}
            message.metadata["short_message"] = True

        if len(message.content) > self.config.max_message_length:
            # Truncate or mark long messages
            if not message.metadata:
                message.metadata = {}
            message.metadata["long_message"] = True
            # Could truncate here: message.content = message.content[:self.config.max_message_length]

        return message

    def get_category_info(self) -> dict[str, Any]:
        """Get information about this category standardizer."""
        return {
            "category": self.category.value,
            "config": {
                "quality_threshold": self.config.quality_threshold,
                "min_message_length": self.config.min_message_length,
                "max_message_length": self.config.max_message_length,
                "required_fields": self.config.required_fields,
            },
            "patterns_count": len(self.category_patterns.get(self.category, [])),
            "source": self.config.source,
        }


class CategoryStandardizerRegistry:
    """Registry for managing multiple category standardizers."""

    def __init__(self):
        self.standardizers: list[SimpleCategoryStandardizer] = []
        self.logger = get_logger(__name__)

        # Initialize with default standardizers
        self._initialize_default_standardizers()

        logger.info("CategoryStandardizerRegistry initialized")

    def _initialize_default_standardizers(self):
        """Initialize default standardizers for all categories."""
        for category in DataCategory:
            standardizer = SimpleCategoryStandardizer(category)
            self.standardizers.append(standardizer)

    def get_standardizer(
        self, data: dict[str, Any]
    ) -> SimpleCategoryStandardizer | None:
        """Get the best standardizer for given data."""
        for standardizer in self.standardizers:
            if standardizer.can_handle(data):
                return standardizer

        # Return general standardizer as fallback
        return self.get_standardizer_by_category(DataCategory.GENERAL)

    def get_standardizer_by_category(
        self, category: DataCategory
    ) -> SimpleCategoryStandardizer | None:
        """Get standardizer by category."""
        for standardizer in self.standardizers:
            if standardizer.category == category:
                return standardizer
        return None

    def add_standardizer(self, standardizer: SimpleCategoryStandardizer):
        """Add a custom standardizer."""
        self.standardizers.append(standardizer)
        logger.info(f"Added standardizer for category: {standardizer.category.value}")

    def list_categories(self) -> list[str]:
        """List all available categories."""
        return [s.category.value for s in self.standardizers]


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Create registry
    registry = CategoryStandardizerRegistry()

    # Test data
    test_data = {
        "id": "test_1",
        "messages": [
            {
                "role": "user",
                "content": "I'm feeling anxious about my therapy session tomorrow.",
            },
            {
                "role": "assistant",
                "content": "It's normal to feel anxious about therapy. Can you tell me more about what's making you feel this way?",
            },
        ],
    }

    # Get appropriate standardizer
    standardizer = registry.get_standardizer(test_data)

    # Standardize data
    conversation = standardizer.standardize(test_data)

    # Get category info
    info = standardizer.get_category_info()
