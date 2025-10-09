#!/usr/bin/env python3
"""
Fix neuro_qa_SFT_Trainer processing issue

The previous processor failed because it tried to use regex on a list instead of string.
This fixes the parsing and completes Task 5.2.2.3 properly.
"""

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class NeuroQAProcessor:
    """Fixed neuro QA SFT processor."""

    def __init__(self, datasets_path: str = "/home/vivi/pixelated/ai/datasets",
                 output_path: str = "/home/vivi/pixelated/ai/data/processed/professional_complete_integration"):
        self.datasets_path = Path(datasets_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def process_neuro_qa_fixed(self) -> dict[str, Any]:
        """Process neuro_qa_SFT_Trainer with fixed parsing."""
        logger.info("🔧 Fixing neuro_qa_SFT_Trainer processing...")

        dataset_file = self.datasets_path / "neuro_qa_SFT_Trainer/train.json"

        if not dataset_file.exists():
            raise FileNotFoundError(f"Neuro QA SFT dataset not found: {dataset_file}")

        # Load the data
        with open(dataset_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"📋 Raw data type: {type(raw_data)}, length: {len(raw_data)}")

        # Check the structure
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first_item = raw_data[0]
            logger.info(f"📋 First item keys: {first_item.keys() if isinstance(first_item, dict) else 'Not a dict'}")

            if "text" in first_item:
                concatenated_text = first_item["text"]
                logger.info(f"📋 Text length: {len(concatenated_text)} characters")

                # Parse the concatenated text properly
                conversations = self._parse_concatenated_text_fixed(concatenated_text)
                logger.info(f"📋 Parsed {len(conversations)} conversations")

                # Convert to standard format
                processed_conversations = []
                for i, conv_data in enumerate(conversations):
                    try:
                        conversation = self._convert_to_standard_format(conv_data, i)
                        if conversation:
                            processed_conversations.append(conversation)
                    except Exception as e:
                        logger.error(f"Error converting conversation {i}: {e}")
                        continue

                # Remove duplicates
                deduplicated = self._remove_duplicates(processed_conversations)

                # Save processed data
                output_file = self.output_path / "neuro_qa_sft_complete_fixed.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for conv in deduplicated:
                        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

                result = {
                    "dataset_name": "neuro_qa_SFT_Trainer_Fixed",
                    "total_parsed": len(conversations),
                    "total_processed": len(deduplicated),
                    "duplicates_removed": len(processed_conversations) - len(deduplicated),
                    "output_file": str(output_file)
                }

                logger.info(f"✅ neuro_qa_SFT fixed: {len(deduplicated)} conversations processed")
                return result

        raise ValueError("Could not parse neuro_qa_SFT_Trainer data structure")

    def _parse_concatenated_text_fixed(self, text: str) -> list[list[dict[str, Any]]]:
        """Fixed parsing of concatenated text."""
        conversations = []

        # Split by JSON object boundaries
        # Look for patterns like {"role": "user", "content": "..."}
        json_objects = []

        # Find all complete JSON objects in the text
        brace_count = 0
        current_obj = ""

        for char in text:
            current_obj += char
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        if "role" in obj and "content" in obj:
                            json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    current_obj = ""

        # Group into conversations (user-assistant pairs)
        current_conversation = []
        for obj in json_objects:
            if obj["role"] == "user":
                if current_conversation:
                    conversations.append(current_conversation)
                    current_conversation = []
                current_conversation.append(obj)
            elif obj["role"] == "assistant" and current_conversation:
                current_conversation.append(obj)

        # Add the last conversation
        if current_conversation:
            conversations.append(current_conversation)

        return conversations

    def _convert_to_standard_format(self, conversation_data: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
        """Convert to standard conversation format."""
        try:
            if len(conversation_data) >= 2:
                conversations = []
                for i in range(0, len(conversation_data), 2):
                    if i + 1 < len(conversation_data):
                        user_msg = conversation_data[i]
                        assistant_msg = conversation_data[i + 1]

                        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                            conversations.append({"human": user_msg["content"]})
                            conversations.append({"assistant": assistant_msg["content"]})

                if conversations:
                    return {
                        "id": f"neuroqa_fixed_{index:06d}",
                        "conversations": conversations,
                        "source": "neuro-qa-sft-fixed",
                        "metadata": {
                            "dataset": "neuro_qa_SFT_Trainer",
                            "format": "technical_qa_fixed",
                            "professional_type": "neurological_consultation",
                            "clinical_domain": "neurology"
                        }
                    }
            return None

        except Exception as e:
            logger.error(f"Error converting format: {e}")
            return None

    def _remove_duplicates(self, conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicates based on content hash."""
        seen_hashes = set()
        deduplicated = []

        for conv in conversations:
            conv_text = ""
            for turn in conv.get("conversations", []):
                conv_text += turn.get("human", "") + turn.get("assistant", "")

            content_hash = hash(conv_text)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(conv)

        return deduplicated


if __name__ == "__main__":
    processor = NeuroQAProcessor()


    with contextlib.suppress(Exception):
        result = processor.process_neuro_qa_fixed()


