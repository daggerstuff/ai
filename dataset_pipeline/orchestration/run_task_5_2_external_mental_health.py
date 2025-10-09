#!/usr/bin/env python3
"""
Task 5.2: Integrate External Mental Health Datasets
Processes HuggingFace mental health datasets to supplement the consolidated dataset.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.dataset_pipeline.data_loader import load_hf_dataset
from ai.dataset_pipeline.logger import get_logger
from ai.dataset_pipeline.standardizer import from_input_output_pair, from_simple_message_list

logger = get_logger("task_5_2_external_mental_health")


class ExternalMentalHealthProcessor:
    """
    Processor for external HuggingFace mental health datasets.
    Handles multiple dataset formats and integrates them into the pipeline.
    """

    def __init__(self, output_dir: str, target_conversations: int = 5000):
        self.output_dir = output_dir
        self.target_conversations = target_conversations

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Dataset configurations
        self.dataset_configs = [
            {
                "name": "mental_health_counseling",
                "hf_path": "Amod/mental_health_counseling_conversations",
                "target": 2000,
                "priority": 1
            },
            {
                "name": "psych8k",
                "hf_path": "EmoCareAI/Psych8k",
                "target": 1500,
                "priority": 2
            },
            {
                "name": "psychology_10k",
                "hf_path": "samhog/psychology-10k",
                "target": 1000,
                "priority": 3
            },
            {
                "name": "addiction_counseling",
                "hf_path": "wesley7137/formatted_annotated_addiction_counseling_csv_SFT",
                "target": 500,
                "priority": 4
            }
        ]

        self.stats = {
            "datasets_processed": 0,
            "total_processed": 0,
            "total_accepted": 0,
            "format_errors": 0,
            "quality_filtered": 0,
            "processing_errors": 0
        }

    def process_all_datasets(self) -> dict:
        """Process all external mental health datasets."""
        logger.info("Starting external mental health dataset processing")
        start_time = datetime.now()

        all_conversations = []
        dataset_results = {}

        # Process datasets by priority
        sorted_configs = sorted(self.dataset_configs, key=lambda x: x["priority"])

        for config in sorted_configs:
            logger.info(f"Processing dataset: {config['name']} ({config['hf_path']})")

            try:
                conversations = self._process_single_dataset(config)
                all_conversations.extend(conversations)

                dataset_results[config["name"]] = {
                    "conversations_processed": len(conversations),
                    "target": config["target"],
                    "success": True,
                    "hf_path": config["hf_path"]
                }

                self.stats["datasets_processed"] += 1
                logger.info(f"Successfully processed {len(conversations)} conversations from {config['name']}")

                # Check if we've reached overall target
                if len(all_conversations) >= self.target_conversations:
                    logger.info(f"Reached overall target of {self.target_conversations} conversations")
                    break

            except Exception as e:
                logger.error(f"Failed to process dataset {config['name']}: {e}")
                dataset_results[config["name"]] = {
                    "conversations_processed": 0,
                    "target": config["target"],
                    "success": False,
                    "error": str(e),
                    "hf_path": config["hf_path"]
                }
                self.stats["processing_errors"] += 1

        # Save processed conversations
        output_path = os.path.join(self.output_dir, "external_mental_health_conversations.jsonl")
        self._save_conversations(all_conversations, output_path)

        # Generate report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(dataset_results, all_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_2_processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"External processing completed. Processed {len(all_conversations)} conversations")
        return report

    def _process_single_dataset(self, config: dict) -> list:
        """Process a single external dataset."""
        try:
            # Load dataset from HuggingFace
            dataset = load_hf_dataset(config["hf_path"])
            if dataset is None:
                raise ValueError(f"Failed to load dataset: {config['hf_path']}")

            processed_conversations = []

            # Convert to list for processing
            data_items = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)

            logger.info(f"Processing {len(data_items)} items from {config['name']}")

            for idx, item in enumerate(data_items):
                try:
                    # Process conversation
                    processed = self._process_conversation(item, config, idx)
                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                        # Check if we've reached target for this dataset
                        if len(processed_conversations) >= config["target"]:
                            logger.info(f"Reached target of {config['target']} conversations for {config['name']}")
                            break

                    self.stats["total_processed"] += 1

                except Exception as e:
                    logger.warning(f"Error processing item {idx} from {config['name']}: {e}")
                    self.stats["processing_errors"] += 1
                    continue

            return processed_conversations

        except Exception as e:
            logger.error(f"Failed to process dataset {config['name']}: {e}")
            raise

    def _process_conversation(self, item: dict, config: dict, idx: int) -> dict:
        """Process a single conversation with format detection."""
        try:
            # Detect and convert format
            conversation = self._detect_and_convert_format(item, config["name"])
            if not conversation:
                self.stats["format_errors"] += 1
                return None

            # Basic quality checks
            if len(conversation.messages) < 2:
                self.stats["quality_filtered"] += 1
                return None

            # Check content length
            total_content = " ".join([msg.content for msg in conversation.messages])
            if len(total_content) < 20 or len(total_content) > 3000:
                self.stats["quality_filtered"] += 1
                return None

            # Create standardized format
            return {
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ],
                "metadata": {
                    "category": "mental_health",
                    "subcategory": self._determine_subcategory(config["name"]),
                    "therapeutic_approach": self._determine_therapeutic_approach(total_content),
                    "emotional_intensity": self._estimate_emotional_intensity(total_content),
                    "source_dataset": config["name"],
                    "hf_path": config["hf_path"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_index": idx,
                    "conversation_length": len(conversation.messages),
                    "total_content_length": len(total_content)
                }
            }


        except Exception as e:
            logger.warning(f"Failed to process conversation at index {idx}: {e}")
            return None

    def _detect_and_convert_format(self, item: dict, source: str):
        """Detect item format and convert to Conversation object."""
        try:
            # Try different format patterns
            if "input" in item and "output" in item:
                # Input/output format
                return from_input_output_pair(
                    item["input"],
                    item["output"],
                    input_role="client",
                    output_role="therapist",
                    source=source
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
                    source=source
                )
            elif "question" in item and "answer" in item:
                # Question/answer format
                return from_input_output_pair(
                    item["question"],
                    item["answer"],
                    input_role="client",
                    output_role="therapist",
                    source=source
                )
            else:
                # Try to extract any text content
                content = str(item)
                if len(content) > 50:  # Only if substantial content
                    return from_input_output_pair(
                        "",
                        content,
                        input_role="client",
                        output_role="therapist",
                        source=source
                    )
                return None

        except Exception as e:
            logger.warning(f"Format detection failed for {source}: {e}")
            return None

    def _determine_subcategory(self, dataset_name: str) -> str:
        """Determine subcategory based on dataset name."""
        if "addiction" in dataset_name.lower():
            return "addiction_counseling"
        if "psych" in dataset_name.lower():
            return "psychology_expertise"
        if "counseling" in dataset_name.lower():
            return "general_counseling"
        return "mental_health_general"

    def _determine_therapeutic_approach(self, content: str) -> str:
        """Determine therapeutic approach from content."""
        content_lower = content.lower()

        if any(term in content_lower for term in ["cognitive", "thought", "thinking", "belief", "cbt"]):
            return "cognitive_behavioral"
        if any(term in content_lower for term in ["mindful", "present", "awareness", "meditation"]):
            return "mindfulness_based"
        if any(term in content_lower for term in ["behavior", "action", "activity", "goal", "exposure"]):
            return "behavioral"
        if any(term in content_lower for term in ["emotion", "feeling", "empathy", "validation"]):
            return "emotion_focused"
        if any(term in content_lower for term in ["relationship", "family", "communication", "interpersonal"]):
            return "interpersonal"
        return "integrative"

    def _estimate_emotional_intensity(self, content: str) -> float:
        """Estimate emotional intensity of content (0.0 to 1.0)."""
        content_lower = content.lower()

        high_intensity_terms = ["crisis", "suicide", "emergency", "severe", "panic", "trauma", "abuse"]
        medium_intensity_terms = ["anxious", "depressed", "worried", "stressed", "upset", "hurt", "angry"]
        low_intensity_terms = ["concerned", "uncertain", "confused", "tired", "frustrated"]

        high_count = sum(1 for term in high_intensity_terms if term in content_lower)
        medium_count = sum(1 for term in medium_intensity_terms if term in content_lower)
        low_count = sum(1 for term in low_intensity_terms if term in content_lower)

        if high_count > 0:
            return min(0.8 + (high_count * 0.1), 1.0)
        if medium_count > 0:
            return min(0.5 + (medium_count * 0.05), 0.8)
        if low_count > 0:
            return min(0.3 + (low_count * 0.05), 0.5)
        return 0.4

    def _save_conversations(self, conversations: list, output_path: str) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} external conversations to {output_path}")

    def _generate_report(self, dataset_results: dict, conversations: list, processing_time: float) -> dict:
        """Generate processing report."""
        return {
            "task": "5.2: Integrate external mental health datasets from HuggingFace",
            "processing_summary": {
                "total_conversations": len(conversations),
                "target_conversations": self.target_conversations,
                "completion_percentage": (len(conversations) / self.target_conversations) * 100,
                "processing_time_seconds": processing_time,
                "datasets_processed": self.stats["datasets_processed"],
                "average_processing_rate": len(conversations) / processing_time if processing_time > 0 else 0
            },
            "quality_metrics": {
                "total_processed": self.stats["total_processed"],
                "total_accepted": self.stats["total_accepted"],
                "acceptance_rate": (self.stats["total_accepted"] / max(self.stats["total_processed"], 1)) * 100,
                "quality_filtered": self.stats["quality_filtered"],
                "format_errors": self.stats["format_errors"],
                "processing_errors": self.stats["processing_errors"]
            },
            "dataset_results": dataset_results,
            "conversation_analysis": {
                "subcategories": self._analyze_subcategories(conversations),
                "therapeutic_approaches": self._analyze_therapeutic_approaches(conversations),
                "emotional_intensity_distribution": self._analyze_emotional_intensity(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_subcategories(self, conversations: list) -> dict:
        """Analyze distribution of subcategories."""
        subcategories = {}
        for conv in conversations:
            subcategory = conv.get("metadata", {}).get("subcategory", "unknown")
            subcategories[subcategory] = subcategories.get(subcategory, 0) + 1
        return subcategories

    def _analyze_therapeutic_approaches(self, conversations: list) -> dict:
        """Analyze distribution of therapeutic approaches."""
        approaches = {}
        for conv in conversations:
            approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
            approaches[approach] = approaches.get(approach, 0) + 1
        return approaches

    def _analyze_emotional_intensity(self, conversations: list) -> dict:
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


def main():
    """Main execution function for Task 5.2."""

    # Configuration
    output_dir = "data/processed/mental_health"
    target_conversations = 5000  # Additional conversations to supplement consolidated dataset


    try:
        # Create processor
        processor = ExternalMentalHealthProcessor(output_dir, target_conversations)

        # Process datasets
        result = processor.process_all_datasets()

        # Print results


        for _dataset_name, result_data in result["dataset_results"].items():
            "✅" if result_data["success"] else "❌"

        result["conversation_analysis"]["subcategories"]

        result["conversation_analysis"]["therapeutic_approaches"]

        result["conversation_analysis"]["emotional_intensity_distribution"]

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
