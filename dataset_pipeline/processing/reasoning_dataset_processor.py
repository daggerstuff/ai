"""
Reasoning dataset processor for chain-of-thought integration.
Processes CoT reasoning datasets for therapeutic AI training.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningPattern:
    """Represents a chain-of-thought reasoning pattern."""
    pattern_id: str
    reasoning_type: str  # clinical, diagnostic, therapeutic, etc.
    steps: list[str]
    complexity_level: str  # basic, intermediate, advanced
    therapeutic_domain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedReasoningData:
    """Result of reasoning dataset processing."""
    conversations: list[Conversation]
    reasoning_patterns: list[ReasoningPattern]
    quality_metrics: dict[str, float]
    processing_stats: dict[str, Any]


class ReasoningDatasetProcessor:
    """
    Processes chain-of-thought reasoning datasets for therapeutic training.

    Handles CoT datasets including clinical diagnosis, neurodivergent interactions,
    emotional reasoning, and other therapeutic reasoning patterns.
    """

    def __init__(self):
        """Initialize the reasoning dataset processor."""
        self.logger = get_logger(__name__)

        # Reasoning type mappings
        self.reasoning_types = {
            "clinical_diagnosis": "Clinical Diagnostic Reasoning",
            "neurodivergent": "Neurodivergent vs Neurotypical Interactions",
            "emotional": "Emotional Intelligence & Relationship Therapy",
            "mens_health": "Gender-Specific Therapeutic Reasoning",
            "legal_ethical": "Legal/Ethical Reasoning in Therapy",
            "philosophical": "Existential/Philosophical Therapy",
            "medical_psychology": "Medical Psychology Reasoning",
            "temporal": "Time-Based Therapeutic Planning",
            "scientific": "Evidence-Based Practice Reasoning",
            "cultural": "Culturally-Sensitive Therapeutic Approaches"
        }

        # Quality thresholds
        self.quality_thresholds = {
            "reasoning_coherence": 0.7,
            "therapeutic_relevance": 0.8,
            "step_clarity": 0.75,
            "clinical_accuracy": 0.85
        }

        self.logger.info("ReasoningDatasetProcessor initialized")

    def process_cot_dataset(self, dataset_path: str, reasoning_type: str) -> ProcessedReasoningData:
        """
        Process a chain-of-thought reasoning dataset.

        Args:
            dataset_path: Path to the CoT dataset
            reasoning_type: Type of reasoning (clinical_diagnosis, neurodivergent, etc.)

        Returns:
            ProcessedReasoningData with conversations and patterns
        """
        self.logger.info(f"Processing CoT dataset: {dataset_path} (type: {reasoning_type})")

        try:
            # Load dataset
            with open(dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            conversations = []
            reasoning_patterns = []
            quality_scores = []

            for item in raw_data:
                # Extract reasoning pattern
                pattern = self._extract_reasoning_pattern(item, reasoning_type)
                if pattern:
                    reasoning_patterns.append(pattern)

                # Convert to conversation format
                conversation = self._convert_to_conversation(item, reasoning_type)
                if conversation:
                    # Assess quality
                    quality_score = self._assess_reasoning_quality(conversation, pattern)

                    if quality_score >= self.quality_thresholds.get("therapeutic_relevance", 0.8):
                        conversations.append(conversation)
                        quality_scores.append(quality_score)

            # Calculate metrics
            quality_metrics = {
                "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "total_processed": len(raw_data),
                "conversations_created": len(conversations),
                "patterns_extracted": len(reasoning_patterns),
                "acceptance_rate": len(conversations) / len(raw_data) if raw_data else 0
            }

            processing_stats = {
                "reasoning_type": reasoning_type,
                "dataset_path": dataset_path,
                "processed_at": datetime.now().isoformat(),
                "quality_thresholds": self.quality_thresholds
            }

            self.logger.info(f"Processed {len(conversations)} conversations from {len(raw_data)} items")

            return ProcessedReasoningData(
                conversations=conversations,
                reasoning_patterns=reasoning_patterns,
                quality_metrics=quality_metrics,
                processing_stats=processing_stats
            )

        except Exception as e:
            self.logger.error(f"Error processing CoT dataset {dataset_path}: {e}")
            raise

    def _extract_reasoning_pattern(self, item: dict[str, Any], reasoning_type: str) -> ReasoningPattern | None:
        """Extract reasoning pattern from dataset item."""
        try:
            # Look for reasoning steps in various formats
            steps = []

            if "reasoning_steps" in item:
                steps = item["reasoning_steps"]
            elif "chain_of_thought" in item:
                steps = item["chain_of_thought"].split("\n") if isinstance(item["chain_of_thought"], str) else item["chain_of_thought"]
            elif "steps" in item:
                steps = item["steps"]
            elif "thinking" in item:
                steps = [item["thinking"]]

            if not steps:
                return None

            # Determine complexity based on number of steps and content
            complexity = "basic"
            if len(steps) > 5:
                complexity = "intermediate"
            if len(steps) > 10 or any(len(step) > 200 for step in steps):
                complexity = "advanced"

            return ReasoningPattern(
                pattern_id=f"{reasoning_type}_{hash(str(steps)) % 10000}",
                reasoning_type=reasoning_type,
                steps=steps,
                complexity_level=complexity,
                therapeutic_domain=self.reasoning_types.get(reasoning_type, reasoning_type),
                metadata={
                    "original_item_keys": list(item.keys()),
                    "step_count": len(steps),
                    "avg_step_length": sum(len(str(step)) for step in steps) / len(steps)
                }
            )

        except Exception as e:
            self.logger.warning(f"Could not extract reasoning pattern: {e}")
            return None

    def _convert_to_conversation(self, item: dict[str, Any], reasoning_type: str) -> Conversation | None:
        """Convert dataset item to conversation format."""
        try:
            messages = []

            # Extract input/output or question/answer pairs
            if "input" in item and "output" in item:
                messages.append(Message(
                    role="user",
                    content=str(item["input"]),
                    timestamp=datetime.now()
                ))
                messages.append(Message(
                    role="assistant",
                    content=str(item["output"]),
                    timestamp=datetime.now()
                ))
            elif "question" in item and "answer" in item:
                messages.append(Message(
                    role="user",
                    content=str(item["question"]),
                    timestamp=datetime.now()
                ))
                messages.append(Message(
                    role="assistant",
                    content=str(item["answer"]),
                    timestamp=datetime.now()
                ))
            elif "prompt" in item and "response" in item:
                messages.append(Message(
                    role="user",
                    content=str(item["prompt"]),
                    timestamp=datetime.now()
                ))
                messages.append(Message(
                    role="assistant",
                    content=str(item["response"]),
                    timestamp=datetime.now()
                ))
            else:
                return None

            if not messages:
                return None

            conversation_id = f"reasoning_{reasoning_type}_{hash(str(item)) % 100000}"

            return Conversation(
                id=conversation_id,
                messages=messages,
                title=f"Reasoning: {reasoning_type}",
                metadata={
                    "reasoning_type": reasoning_type,
                    "source": "cot_dataset",
                    "therapeutic_domain": self.reasoning_types.get(reasoning_type, reasoning_type),
                    "original_keys": list(item.keys())
                },
                tags=["reasoning", "cot", reasoning_type]
            )

        except Exception as e:
            self.logger.warning(f"Could not convert item to conversation: {e}")
            return None

    def _assess_reasoning_quality(self, conversation: Conversation, pattern: ReasoningPattern | None) -> float:
        """Assess the quality of reasoning in a conversation."""
        try:
            scores = []

            # Check conversation coherence
            if len(conversation.messages) >= 2:
                scores.append(0.8)  # Basic coherence for having Q&A structure

            # Check reasoning pattern quality
            if pattern:
                if len(pattern.steps) >= 3:
                    scores.append(0.85)  # Good reasoning depth
                if pattern.complexity_level in ["intermediate", "advanced"]:
                    scores.append(0.9)  # Complex reasoning

            # Check therapeutic relevance
            therapeutic_keywords = [
                "therapy", "therapeutic", "clinical", "diagnosis", "treatment",
                "mental health", "psychology", "counseling", "intervention"
            ]

            content = " ".join([msg.content.lower() for msg in conversation.messages])
            if any(keyword in content for keyword in therapeutic_keywords):
                scores.append(0.9)

            # Check content quality
            avg_length = sum(len(msg.content) for msg in conversation.messages) / len(conversation.messages)
            if avg_length > 50:  # Substantial content
                scores.append(0.8)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            self.logger.warning(f"Could not assess reasoning quality: {e}")
            return 0.5

    def process_multiple_datasets(self, dataset_configs: list[dict[str, str]]) -> dict[str, ProcessedReasoningData]:
        """
        Process multiple CoT datasets.

        Args:
            dataset_configs: List of dicts with 'path' and 'reasoning_type' keys

        Returns:
            Dict mapping reasoning_type to ProcessedReasoningData
        """
        results = {}

        for config in dataset_configs:
            dataset_path = config["path"]
            reasoning_type = config["reasoning_type"]

            try:
                result = self.process_cot_dataset(dataset_path, reasoning_type)
                results[reasoning_type] = result
                self.logger.info(f"Successfully processed {reasoning_type} dataset")
            except Exception as e:
                self.logger.error(f"Failed to process {reasoning_type} dataset: {e}")

        return results

    def get_reasoning_statistics(self, processed_data: ProcessedReasoningData) -> dict[str, Any]:
        """Get detailed statistics about processed reasoning data."""
        conversations = processed_data.conversations
        patterns = processed_data.reasoning_patterns

        if not conversations:
            return {"error": "No conversations to analyze"}

        # Conversation statistics
        total_messages = sum(len(conv.messages) for conv in conversations)
        avg_messages_per_conv = total_messages / len(conversations)

        # Pattern statistics
        complexity_counts = {}
        for pattern in patterns:
            complexity = pattern.complexity_level
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Content statistics
        total_chars = sum(sum(len(msg.content) for msg in conv.messages) for conv in conversations)
        avg_chars_per_conv = total_chars / len(conversations)

        return {
            "conversation_count": len(conversations),
            "pattern_count": len(patterns),
            "total_messages": total_messages,
            "avg_messages_per_conversation": round(avg_messages_per_conv, 2),
            "complexity_distribution": complexity_counts,
            "avg_characters_per_conversation": round(avg_chars_per_conv, 2),
            "quality_metrics": processed_data.quality_metrics,
            "reasoning_type": processed_data.processing_stats.get("reasoning_type", "unknown")
        }


def validate_reasoning_processor():
    """Validate the ReasoningDatasetProcessor functionality."""
    try:
        processor = ReasoningDatasetProcessor()

        # Test basic functionality
        assert hasattr(processor, "process_cot_dataset")
        assert hasattr(processor, "reasoning_types")
        assert len(processor.reasoning_types) > 0

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # Run validation
    if validate_reasoning_processor():
        pass
    else:
        pass
