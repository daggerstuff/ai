"""
Mental Health Dataset Integration System for Task 5.0
Integrates external mental health conversations (20% of dataset strategy) from multiple sources.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data_loader import load_hf_dataset
from emotional_authenticity_assessment import EmotionalAuthenticityAssessor
from logger import get_logger
from quality_filter import QualityFilter
from standardizer import from_input_output_pair, from_simple_message_list
from therapeutic_accuracy_assessment import TherapeuticAccuracyAssessor

logger = get_logger("dataset_pipeline.mental_health_integrator")


@dataclass
class MentalHealthDatasetConfig:
    """Configuration for mental health dataset integration."""

    name: str
    hf_path: str
    target_conversations: int
    priority: int
    quality_threshold: float = 0.7
    therapeutic_accuracy_threshold: float = 0.75
    emotional_authenticity_threshold: float = 0.7


class MentalHealthIntegrator:
    """
    Integrates external mental health conversations from multiple HuggingFace datasets.
    Handles format standardization, quality assessment, and therapeutic validation.
    """

    def __init__(self, output_dir: str = "data/processed/mental_health"):
        self.output_dir = output_dir
        self.quality_filter = QualityFilter()
        self.emotional_assessor = EmotionalAuthenticityAssessor()
        self.therapeutic_assessor = TherapeuticAccuracyAssessor()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Mental health dataset configurations
        self.dataset_configs = [
            MentalHealthDatasetConfig(
                name="mental_health_counseling",
                hf_path="Amod/mental_health_counseling_conversations",
                target_conversations=15000,
                priority=1,
                quality_threshold=0.75,
                therapeutic_accuracy_threshold=0.8,
            ),
            MentalHealthDatasetConfig(
                name="psych8k",
                hf_path="EmoCareAI/Psych8k",
                target_conversations=8000,
                priority=1,
                quality_threshold=0.7,
                therapeutic_accuracy_threshold=0.75,
            ),
            MentalHealthDatasetConfig(
                name="psychology_10k",
                hf_path="samhog/psychology-10k",
                target_conversations=10000,
                priority=1,
                quality_threshold=0.7,
                therapeutic_accuracy_threshold=0.75,
            ),
            MentalHealthDatasetConfig(
                name="addiction_counseling",
                hf_path="wesley7137/formatted_annotated_addiction_counseling_csv_SFT",
                target_conversations=5000,
                priority=2,
                quality_threshold=0.75,
                therapeutic_accuracy_threshold=0.8,
            ),
        ]

        self.integration_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "therapeutic_filtered": 0,
            "emotional_filtered": 0,
            "datasets_processed": 0,
            "processing_errors": 0,
        }

    def integrate_all_datasets(self) -> dict[str, Any]:
        """
        Integrate all configured mental health datasets.

        Returns:
            Dict containing integration results and statistics
        """
        logger.info("Starting mental health dataset integration")
        start_time = datetime.now()

        integrated_conversations = []
        dataset_results = {}

        # Process datasets by priority
        sorted_configs = sorted(self.dataset_configs, key=lambda x: x.priority)

        for config in sorted_configs:
            logger.info(f"Processing dataset: {config.name} ({config.hf_path})")

            try:
                conversations = self._process_single_dataset(config)
                integrated_conversations.extend(conversations)

                dataset_results[config.name] = {
                    "conversations_processed": len(conversations),
                    "target": config.target_conversations,
                    "success": True,
                    "quality_threshold": config.quality_threshold,
                }

                self.integration_stats["datasets_processed"] += 1
                logger.info(
                    f"Successfully processed {len(conversations)} conversations from {config.name}"
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {config.name}: {e}")
                dataset_results[config.name] = {
                    "conversations_processed": 0,
                    "target": config.target_conversations,
                    "success": False,
                    "error": str(e),
                }
                self.integration_stats["processing_errors"] += 1

        # Save integrated conversations
        output_path = os.path.join(
            self.output_dir, "integrated_mental_health_conversations.jsonl"
        )
        self._save_conversations(integrated_conversations, output_path)

        # Generate integration report
        processing_time = (datetime.now() - start_time).total_seconds()
        integration_report = self._generate_integration_report(
            dataset_results, integrated_conversations, processing_time
        )

        # Save integration report
        report_path = os.path.join(
            self.output_dir, "mental_health_integration_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(integration_report, f, indent=2)

        logger.info(
            f"Mental health integration completed. Processed {len(integrated_conversations)} conversations"
        )
        return integration_report

    def _process_single_dataset(
        self, config: MentalHealthDatasetConfig
    ) -> list[dict[str, Any]]:
        """
        Process a single mental health dataset.

        Args:
            config: Dataset configuration

        Returns:
            List of processed and validated conversations
        """
        # Load dataset from HuggingFace
        dataset = load_hf_dataset(config.hf_path)
        if dataset is None:
            raise ValueError(f"Failed to load dataset: {config.hf_path}")

        processed_conversations = []

        # Convert to list for processing
        data_items = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)

        logger.info(f"Processing {len(data_items)} items from {config.name}")

        for idx, item in enumerate(data_items):
            try:
                # Standardize conversation format
                standardized = self._standardize_conversation(item, config)
                if not standardized:
                    continue

                # Quality assessment
                if not self._assess_quality(standardized, config):
                    self.integration_stats["quality_filtered"] += 1
                    continue

                # Therapeutic accuracy assessment
                if not self._assess_therapeutic_accuracy(standardized, config):
                    self.integration_stats["therapeutic_filtered"] += 1
                    continue

                # Emotional authenticity assessment
                if not self._assess_emotional_authenticity(standardized, config):
                    self.integration_stats["emotional_filtered"] += 1
                    continue

                # Add metadata
                standardized["metadata"].update(
                    {
                        "source_dataset": config.name,
                        "hf_path": config.hf_path,
                        "processing_timestamp": datetime.now().isoformat(),
                        "original_index": idx,
                    }
                )

                processed_conversations.append(standardized)
                self.integration_stats["total_accepted"] += 1

                # Limit to target if specified
                if len(processed_conversations) >= config.target_conversations:
                    logger.info(
                        f"Reached target of {config.target_conversations} conversations for {config.name}"
                    )
                    break

            except Exception as e:
                logger.warning(f"Error processing item {idx} from {config.name}: {e}")
                self.integration_stats["processing_errors"] += 1
                continue

            self.integration_stats["total_processed"] += 1

        return processed_conversations

    def _standardize_conversation(
        self, item: dict[str, Any], config: MentalHealthDatasetConfig
    ) -> dict[str, Any] | None:
        """
        Standardize conversation format based on dataset structure.

        Args:
            item: Raw data item from dataset
            config: Dataset configuration

        Returns:
            Standardized conversation or None if conversion fails
        """
        try:
            # Detect format and convert to standard format
            conversation = self._detect_and_convert_format(item, config.name)

            # Convert to dict format for compatibility
            return {
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ],
                "metadata": {
                    "category": "mental_health",
                    "subcategory": self._determine_subcategory(item, config),
                    "therapeutic_approach": self._determine_therapeutic_approach(item),
                    "emotional_intensity": self._estimate_emotional_intensity(item),
                },
            }


        except Exception as e:
            logger.warning(
                f"Failed to standardize conversation from {config.name}: {e}"
            )
            return None

    def _detect_and_convert_format(self, item: dict[str, Any], source: str):
        """Detect item format and convert to Conversation object."""
        # Try different format patterns
        if "input" in item and "output" in item:
            # Input/output format
            return from_input_output_pair(
                item["input"],
                item["output"],
                input_role="client",
                output_role="therapist",
                source=source,
            )
        if "conversations" in item:
            # HuggingFace conversations format
            conversations = item["conversations"]
            if isinstance(conversations, list) and len(conversations) > 0:
                return from_simple_message_list(conversations, source=source)
        elif "messages" in item:
            # Direct messages format
            return from_simple_message_list(item["messages"], source=source)
        elif "text" in item:
            # Single text format - treat as response
            return from_input_output_pair(
                "",
                item["text"],
                input_role="client",
                output_role="therapist",
                source=source,
            )
        else:
            # Try to extract any text content
            content = str(item)
            return from_input_output_pair(
                "", content, input_role="client", output_role="therapist", source=source
            )
        return None

    def _determine_subcategory(
        self, item: dict[str, Any], config: MentalHealthDatasetConfig
    ) -> str:
        """Determine mental health subcategory based on content and dataset."""
        if "addiction" in config.name.lower():
            return "addiction_counseling"
        if "psych" in config.name.lower():
            return "psychology_expertise"
        if "counseling" in config.name.lower():
            return "general_counseling"
        return "mental_health_general"

    def _determine_therapeutic_approach(self, item: dict[str, Any]) -> str:
        """Determine therapeutic approach from conversation content."""
        # Simple heuristic-based approach detection
        content = str(item).lower()

        if any(
            term in content for term in ["cognitive", "thought", "thinking", "belief"]
        ):
            return "cognitive_behavioral"
        if any(
            term in content for term in ["feeling", "emotion", "mindful", "present"]
        ):
            return "mindfulness_based"
        if any(
            term in content for term in ["behavior", "action", "activity", "goal"]
        ):
            return "behavioral"
        return "integrative"

    def _estimate_emotional_intensity(self, item: dict[str, Any]) -> float:
        """Estimate emotional intensity of conversation (0.0 to 1.0)."""
        content = str(item).lower()

        # High intensity indicators
        high_intensity_terms = [
            "crisis",
            "suicide",
            "emergency",
            "severe",
            "panic",
            "trauma",
        ]
        medium_intensity_terms = [
            "anxious",
            "depressed",
            "worried",
            "stressed",
            "upset",
        ]

        high_count = sum(1 for term in high_intensity_terms if term in content)
        medium_count = sum(1 for term in medium_intensity_terms if term in content)

        if high_count > 0:
            return min(0.8 + (high_count * 0.1), 1.0)
        if medium_count > 0:
            return min(0.5 + (medium_count * 0.1), 0.8)
        return 0.3

    def _assess_quality(
        self, conversation: dict[str, Any], config: MentalHealthDatasetConfig
    ) -> bool:
        """Assess conversation quality using quality filter."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            filter_result = self.quality_filter.filter_conversation(conv_obj)
            return filter_result.overall_score >= config.quality_threshold
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return False

    def _assess_therapeutic_accuracy(
        self, conversation: dict[str, Any], config: MentalHealthDatasetConfig
    ) -> bool:
        """Assess therapeutic accuracy of conversation."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            accuracy_metrics = self.therapeutic_assessor.assess_therapeutic_accuracy(
                conv_obj
            )
            return (
                accuracy_metrics.overall_score >= config.therapeutic_accuracy_threshold
            )
        except Exception as e:
            logger.warning(f"Therapeutic accuracy assessment failed: {e}")
            return False

    def _assess_emotional_authenticity(
        self, conversation: dict[str, Any], config: MentalHealthDatasetConfig
    ) -> bool:
        """Assess emotional authenticity of conversation."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            authenticity_metrics = (
                self.emotional_assessor.assess_emotional_authenticity(conv_obj)
            )
            return (
                authenticity_metrics.overall_score
                >= config.emotional_authenticity_threshold
            )
        except Exception as e:
            logger.warning(f"Emotional authenticity assessment failed: {e}")
            return False

    def _dict_to_conversation(self, conversation_dict: dict[str, Any]):
        """Convert conversation dict to Conversation object for assessments."""
        from conversation_schema import Conversation, Message

        messages = []
        for msg_dict in conversation_dict.get("conversation", []):
            messages.append(
                Message(
                    role=msg_dict["role"],
                    content=msg_dict["content"],
                    timestamp=None,
                    meta={},
                )
            )

        return Conversation(
            id=None,
            messages=messages,
            context=conversation_dict.get("metadata", {}),
            source="mental_health_integration",
            created_at=None,
            meta={},
        )

    def _save_conversations(
        self, conversations: list[dict[str, Any]], output_path: str
    ) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} conversations to {output_path}")

    def _generate_integration_report(
        self,
        dataset_results: dict[str, Any],
        conversations: list[dict[str, Any]],
        processing_time: float,
    ) -> dict[str, Any]:
        """Generate comprehensive integration report."""
        return {
            "integration_summary": {
                "total_conversations": len(conversations),
                "target_conversations": 20000,  # 20% of 100K target
                "completion_percentage": (len(conversations) / 20000) * 100,
                "processing_time_seconds": processing_time,
                "datasets_processed": self.integration_stats["datasets_processed"],
                "processing_errors": self.integration_stats["processing_errors"],
            },
            "quality_metrics": {
                "total_processed": self.integration_stats["total_processed"],
                "total_accepted": self.integration_stats["total_accepted"],
                "acceptance_rate": (
                    self.integration_stats["total_accepted"]
                    / max(self.integration_stats["total_processed"], 1)
                )
                * 100,
                "quality_filtered": self.integration_stats["quality_filtered"],
                "therapeutic_filtered": self.integration_stats["therapeutic_filtered"],
                "emotional_filtered": self.integration_stats["emotional_filtered"],
            },
            "dataset_results": dataset_results,
            "conversation_categories": self._analyze_conversation_categories(
                conversations
            ),
            "therapeutic_approaches": self._analyze_therapeutic_approaches(
                conversations
            ),
            "emotional_intensity_distribution": self._analyze_emotional_intensity(
                conversations
            ),
            "timestamp": datetime.now().isoformat(),
            "task": "5.1-5.5: Mental Health Conversations Integration",
        }

    def _analyze_conversation_categories(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze distribution of conversation categories."""
        categories = {}
        for conv in conversations:
            category = conv.get("metadata", {}).get("subcategory", "unknown")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _analyze_therapeutic_approaches(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze distribution of therapeutic approaches."""
        approaches = {}
        for conv in conversations:
            approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
            approaches[approach] = approaches.get(approach, 0) + 1
        return approaches

    def _analyze_emotional_intensity(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze emotional intensity distribution."""
        intensity_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            intensity = conv.get("metadata", {}).get("emotional_intensity", 0.0)
            if intensity < 0.4:
                intensity_ranges["low"] += 1
            elif intensity < 0.7:
                intensity_ranges["medium"] += 1
            else:
                intensity_ranges["high"] += 1
        return intensity_ranges
