"""
Dataset categorization system for training allocation.
Categorizes datasets for appropriate training data allocation and organization.
"""

from enum import Enum

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


class DatasetCategory(Enum):
    """Dataset categories for training allocation."""
    PSYCHOLOGY_KNOWLEDGE = "psychology_knowledge"
    VOICE_TRAINING = "voice_training"
    MENTAL_HEALTH = "mental_health"
    REASONING_COT = "reasoning_cot"
    PERSONALITY_BALANCING = "personality_balancing"
    CRISIS_INTERVENTION = "crisis_intervention"
    EDGE_CASES = "edge_cases"


class DatasetCategorizationSystem:
    """
    Categorizes datasets for training allocation.

    Organizes conversations into appropriate categories for
    balanced therapeutic AI training.
    """

    def __init__(self):
        """Initialize the dataset categorization system."""
        self.logger = get_logger(__name__)

        self.category_patterns = {
            DatasetCategory.PSYCHOLOGY_KNOWLEDGE: [
                "dsm", "diagnosis", "psychology", "therapeutic", "clinical"
            ],
            DatasetCategory.VOICE_TRAINING: [
                "voice", "audio", "personality", "authentic", "transcription"
            ],
            DatasetCategory.MENTAL_HEALTH: [
                "mental health", "counseling", "therapy", "depression", "anxiety"
            ],
            DatasetCategory.REASONING_COT: [
                "reasoning", "chain of thought", "cot", "thinking", "analysis"
            ],
            DatasetCategory.CRISIS_INTERVENTION: [
                "crisis", "emergency", "suicide", "danger", "urgent"
            ]
        }

        self.logger.info("DatasetCategorizationSystem initialized")

    def categorize_conversations(self, conversations: list[Conversation]) -> dict[DatasetCategory, list[Conversation]]:
        """Categorize conversations into dataset categories."""
        categorized = {category: [] for category in DatasetCategory}

        for conv in conversations:
            category = self._determine_category(conv)
            categorized[category].append(conv)

        self.logger.info(f"Categorized {len(conversations)} conversations")
        return categorized

    def _determine_category(self, conversation: Conversation) -> DatasetCategory:
        """Determine the category for a single conversation."""
        # Check metadata first
        if "voice_derived" in conversation.metadata:
            return DatasetCategory.VOICE_TRAINING
        if "crisis_intervention" in conversation.metadata:
            return DatasetCategory.CRISIS_INTERVENTION
        if "reasoning_type" in conversation.metadata:
            return DatasetCategory.REASONING_COT

        # Check tags
        for tag in conversation.tags:
            if "psychology" in tag:
                return DatasetCategory.PSYCHOLOGY_KNOWLEDGE
            if "crisis" in tag:
                return DatasetCategory.CRISIS_INTERVENTION
            if "reasoning" in tag:
                return DatasetCategory.REASONING_COT

        # Default to mental health
        return DatasetCategory.MENTAL_HEALTH


def validate_dataset_categorization_system():
    """Validate the DatasetCategorizationSystem functionality."""
    try:
        system = DatasetCategorizationSystem()
        assert hasattr(system, "categorize_conversations")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_dataset_categorization_system():
        pass
    else:
        pass
