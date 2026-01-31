"""
Simple Mental Health Dataset Processor

Simplified version of mental health processing that works independently
without complex dependencies.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of mental health processing."""

    processed_conversations: int
    quality_filtered: int
    safety_filtered: int
    total_size: int
    processing_time: float
    timestamp: str


class SimpleMentalHealthProcessor:
    """Simplified mental health dataset processor."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Processing configuration
        self.config = {
            "min_quality_score": 0.6,
            "safety_keywords": ["harmful", "dangerous", "suicide", "self-harm"],
            "therapeutic_keywords": [
                "therapy",
                "counseling",
                "support",
                "help",
                "understand",
            ],
            "min_conversation_length": 2,
            "max_conversation_length": 50,
        }

        logger.info("SimpleMentalHealthProcessor initialized")

    def process_dataset(self, input_path: str, output_path: str) -> ProcessingResult:
        """Process mental health dataset."""
        start_time = datetime.now()

        processed_conversations = 0
        quality_filtered = 0
        safety_filtered = 0
        total_size = 0

        processed_data = []

        try:
            # Load input data
            with open(input_path, encoding="utf-8") as f:
                if input_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)

            # Process each conversation
            for item in data:
                if self._is_valid_conversation(item):
                    if self._passes_safety_check(item):
                        if self._passes_quality_check(item):
                            processed_item = self._process_conversation(item)
                            processed_data.append(processed_item)
                            processed_conversations += 1
                        else:
                            quality_filtered += 1
                    else:
                        safety_filtered += 1

            # Save processed data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            total_size = os.path.getsize(output_path)
            processing_time = (datetime.now() - start_time).total_seconds()

            result = ProcessingResult(
                processed_conversations=processed_conversations,
                quality_filtered=quality_filtered,
                safety_filtered=safety_filtered,
                total_size=total_size,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(f"Processed {processed_conversations} conversations")
            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _is_valid_conversation(self, item: dict[str, Any]) -> bool:
        """Check if item is a valid conversation."""
        if not isinstance(item, dict):
            return False

        # Check for messages or content
        if "messages" in item:
            messages = item["messages"]
            if (
                not isinstance(messages, list)
                or len(messages) < self.config["min_conversation_length"]
            ):
                return False

            # Check message structure
            return all(not (not isinstance(msg, dict) or "content" not in msg) for msg in messages)

        return bool("content" in item or "text" in item)

    def _passes_safety_check(self, item: dict[str, Any]) -> bool:
        """Check if conversation passes safety requirements."""
        content = self._extract_content(item)

        # Check for safety keywords
        content_lower = content.lower()
        return all(keyword not in content_lower for keyword in self.config["safety_keywords"])

    def _passes_quality_check(self, item: dict[str, Any]) -> bool:
        """Check if conversation passes quality requirements."""
        content = self._extract_content(item)

        # Basic quality checks
        if len(content.strip()) < 10:
            return False

        # Check for therapeutic relevance
        content_lower = content.lower()
        therapeutic_score = sum(
            1
            for keyword in self.config["therapeutic_keywords"]
            if keyword in content_lower
        )

        # Simple quality scoring
        quality_score = min(1.0, therapeutic_score * 0.2 + 0.4)

        return quality_score >= self.config["min_quality_score"]

    def _process_conversation(self, item: dict[str, Any]) -> dict[str, Any]:
        """Process and standardize conversation."""
        processed = {
            "id": item.get("id", f"conv_{hash(str(item)) % 1000000}"),
            "timestamp": datetime.now().isoformat(),
            "source": "mental_health_processor",
            "metadata": {
                "processed": True,
                "quality_checked": True,
                "safety_checked": True,
            },
        }

        # Handle different input formats
        if "messages" in item:
            processed["messages"] = item["messages"]
        elif "content" in item:
            processed["messages"] = [{"role": "user", "content": item["content"]}]
        elif "text" in item:
            processed["messages"] = [{"role": "user", "content": item["text"]}]

        # Add original metadata if present
        if "metadata" in item:
            processed["metadata"].update(item["metadata"])

        return processed

    def _extract_content(self, item: dict[str, Any]) -> str:
        """Extract text content from conversation."""
        content_parts = []

        if "messages" in item:
            for msg in item["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    content_parts.append(msg["content"])
        elif "content" in item:
            content_parts.append(item["content"])
        elif "text" in item:
            content_parts.append(item["text"])

        return " ".join(content_parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "processor_type": "SimpleMentalHealthProcessor",
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    processor = SimpleMentalHealthProcessor()

    # Example data
    test_data = [
        {
            "id": "test_1",
            "messages": [
                {
                    "role": "user",
                    "content": "I am feeling anxious about my therapy session.",
                },
                {
                    "role": "assistant",
                    "content": "It is normal to feel anxious. Therapy can help you work through these feelings.",
                },
            ],
        },
        {"id": "test_2", "content": "I need support with my mental health challenges."},
    ]

    # Save test data
    with open("test_input.json", "w") as f:
        json.dump(test_data, f)

    # Process
    try:
        result = processor.process_dataset("test_input.json", "test_output.json")

        # Clean up
        os.remove("test_input.json")
        os.remove("test_output.json")

    except Exception:
        pass
