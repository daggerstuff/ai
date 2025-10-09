#!/usr/bin/env python3
"""
Task 5.1: Process Existing Consolidated Mental Health Dataset
Processes the 86MB merged_mental_health_dataset.jsonl file (43,683 conversations)
to extract 20,000 high-quality therapeutic conversations (20% of 100K target).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.dataset_pipeline.logger import get_logger
from ai.dataset_pipeline.standardizer import from_input_output_pair

logger = get_logger("task_5_1_consolidated_processing")


class SimpleConsolidatedProcessor:
    """
    Simplified processor for the consolidated mental health dataset.
    Focuses on basic quality filtering and format standardization.
    """

    def __init__(self, input_file: str, output_dir: str, target_conversations: int = 20000):
        self.input_file = input_file
        self.output_dir = output_dir
        self.target_conversations = target_conversations

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self.stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "format_errors": 0,
            "quality_filtered": 0,
            "processing_errors": 0
        }

    def process_dataset(self) -> dict:
        """Process the consolidated dataset with basic quality filtering."""
        logger.info(f"Starting consolidated dataset processing from {self.input_file}")
        start_time = datetime.now()

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        processed_conversations = []

        try:
            with open(self.input_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        item = json.loads(line.strip())

                        # Process conversation
                        processed = self._process_conversation(item, line_num)
                        if processed:
                            processed_conversations.append(processed)
                            self.stats["total_accepted"] += 1

                            # Check if we've reached target
                            if len(processed_conversations) >= self.target_conversations:
                                logger.info(f"Reached target of {self.target_conversations} conversations")
                                break

                        self.stats["total_processed"] += 1

                        # Progress logging
                        if line_num % 5000 == 0:
                            logger.info(f"Processed {line_num} lines, accepted {len(processed_conversations)} conversations")

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        self.stats["format_errors"] += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        self.stats["processing_errors"] += 1
                        continue

        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            raise

        # Save processed conversations
        output_path = os.path.join(self.output_dir, "consolidated_mental_health_conversations.jsonl")
        self._save_conversations(processed_conversations, output_path)

        # Generate report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(processed_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_1_processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Processing completed. Processed {len(processed_conversations)} conversations")
        return report

    def _process_conversation(self, item: dict, line_num: int) -> dict:
        """Process a single conversation with basic quality checks."""
        try:
            # Extract prompt and response
            prompt = item.get("prompt", "")
            response = item.get("response", "")

            if not prompt or not response:
                return None

            # Basic quality checks
            if len(prompt) < 10 or len(response) < 20:
                self.stats["quality_filtered"] += 1
                return None

            if len(response) > 2000:  # Too long responses
                self.stats["quality_filtered"] += 1
                return None

            # Extract client content (remove system instructions)
            client_content = self._extract_client_content(prompt)
            if len(client_content) < 5:
                self.stats["quality_filtered"] += 1
                return None

            # Convert to standard format
            conversation = from_input_output_pair(
                client_content,
                response,
                input_role="client",
                output_role="therapist",
                source="consolidated_mental_health"
            )

            # Create standardized format
            return {
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ],
                "metadata": {
                    "category": "mental_health",
                    "subcategory": "consolidated_counseling",
                    "therapeutic_approach": self._determine_therapeutic_approach(prompt, response),
                    "emotional_intensity": self._estimate_emotional_intensity(prompt, response),
                    "source_dataset": "consolidated_mental_health",
                    "source_file": self.input_file,
                    "original_line_number": line_num,
                    "processing_timestamp": datetime.now().isoformat(),
                    "conversation_length": len(conversation.messages),
                    "client_content_length": len(client_content),
                    "therapist_response_length": len(response)
                }
            }


        except Exception as e:
            logger.warning(f"Failed to process conversation at line {line_num}: {e}")
            return None

    def _extract_client_content(self, prompt: str) -> str:
        """Extract client content from prompt, removing system instructions."""
        # Common system instruction patterns
        system_patterns = [
            "You are a helpful mental health counselling assistant",
            "please answer the mental health questions based on the patient's description",
            "The assistant gives helpful, comprehensive, and appropriate answers",
            "You are a mental health counselor",
            "You are a therapist"
        ]

        client_content = prompt
        for pattern in system_patterns:
            if pattern in client_content:
                parts = client_content.split(pattern)
                if len(parts) > 1:
                    client_content = parts[-1].strip()
                    client_content = client_content.lstrip(". ").strip()

        return client_content

    def _determine_therapeutic_approach(self, prompt: str, response: str) -> str:
        """Determine therapeutic approach from conversation content."""
        content = (prompt + " " + response).lower()

        if any(term in content for term in ["cognitive", "thought", "thinking", "belief", "cbt"]):
            return "cognitive_behavioral"
        if any(term in content for term in ["mindful", "present", "awareness", "meditation"]):
            return "mindfulness_based"
        if any(term in content for term in ["behavior", "action", "activity", "goal", "exposure"]):
            return "behavioral"
        if any(term in content for term in ["emotion", "feeling", "empathy", "validation"]):
            return "emotion_focused"
        if any(term in content for term in ["relationship", "family", "communication", "interpersonal"]):
            return "interpersonal"
        return "integrative"

    def _estimate_emotional_intensity(self, prompt: str, response: str) -> float:
        """Estimate emotional intensity of conversation (0.0 to 1.0)."""
        content = (prompt + " " + response).lower()

        high_intensity_terms = ["crisis", "suicide", "emergency", "severe", "panic", "trauma", "abuse"]
        medium_intensity_terms = ["anxious", "depressed", "worried", "stressed", "upset", "hurt", "angry"]
        low_intensity_terms = ["concerned", "uncertain", "confused", "tired", "frustrated"]

        high_count = sum(1 for term in high_intensity_terms if term in content)
        medium_count = sum(1 for term in medium_intensity_terms if term in content)
        low_count = sum(1 for term in low_intensity_terms if term in content)

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

        logger.info(f"Saved {len(conversations)} conversations to {output_path}")

    def _generate_report(self, conversations: list, processing_time: float) -> dict:
        """Generate processing report."""
        return {
            "task": "5.1: Process existing consolidated mental health dataset (86MB JSONL)",
            "processing_summary": {
                "source_file": self.input_file,
                "total_conversations": len(conversations),
                "target_conversations": self.target_conversations,
                "completion_percentage": (len(conversations) / self.target_conversations) * 100,
                "processing_time_seconds": processing_time,
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
            "conversation_analysis": {
                "therapeutic_approaches": self._analyze_therapeutic_approaches(conversations),
                "emotional_intensity_distribution": self._analyze_emotional_intensity(conversations),
                "conversation_length_stats": self._analyze_conversation_lengths(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

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

    def _analyze_conversation_lengths(self, conversations: list) -> dict:
        """Analyze conversation length statistics."""
        lengths = [conv.get("metadata", {}).get("conversation_length", 0) for conv in conversations]
        if not lengths:
            return {"min": 0, "max": 0, "average": 0}

        return {
            "min": min(lengths),
            "max": max(lengths),
            "average": sum(lengths) / len(lengths)
        }


def main():
    """Main execution function for Task 5.1."""

    # Configuration
    input_file = "ai/merged_mental_health_dataset.jsonl"
    output_dir = "data/processed/mental_health"
    target_conversations = 20000  # 20% of 100K target

    # Check if input file exists
    if not os.path.exists(input_file):
        return False

    # Get file info
    os.path.getsize(input_file) / (1024 * 1024)  # MB

    try:
        # Create processor
        processor = SimpleConsolidatedProcessor(input_file, output_dir, target_conversations)

        # Process dataset
        result = processor.process_dataset()

        # Print results


        result["conversation_analysis"]["therapeutic_approaches"]

        result["conversation_analysis"]["emotional_intensity_distribution"]

        result["conversation_analysis"]["conversation_length_stats"]

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
