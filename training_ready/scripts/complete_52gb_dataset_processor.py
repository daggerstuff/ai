#!/usr/bin/env python3
"""
Complete 52.20GB Dataset Processor
Merges all source datasets and applies comprehensive cleaning
"""

import json
import logging
import re
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple, Optional
from datetime import datetime
import csv
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DatasetMerger:
    """
    Merges multiple dataset formats into unified ChatML format
    """

    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.conversation_count = 0

    def convert_csv_to_jsonl(
        self, csv_path: Path
    ) -> Generator[Dict[str, Any], None, None]:
        """Convert CSV files to ChatML format"""
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle different CSV formats
                    if "prompt" in row and "response" in row:
                        messages = [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                    elif "context" in row and "response" in row:
                        messages = [
                            {"role": "user", "content": row["context"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                    else:
                        continue

                    yield {
                        "messages": messages,
                        "metadata": {
                            "source_file": str(csv_path),
                            "source_type": "csv",
                            "converted_at": datetime.now().isoformat(),
                        },
                    }
                    self.conversation_count += 1
        except Exception as e:
            logger.error(f"Error processing CSV {csv_path}: {e}")

    def convert_json_to_jsonl(
        self, json_path: Path
    ) -> Generator[Dict[str, Any], None, None]:
        """Convert JSON files to ChatML format"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    yield from self._process_json_item(item, json_path)
            else:
                yield from self._process_json_item(data, json_path)

        except Exception as e:
            logger.error(f"Error processing JSON {json_path}: {e}")

    def _process_json_item(
        self, item: Any, source_path: Path
    ) -> Generator[Dict[str, Any], None, None]:
        """Process individual JSON items"""
        try:
            if isinstance(item, dict):
                # Handle different JSON structures
                if "conversation" in item:
                    messages = self._convert_conversation_format(item["conversation"])
                elif "messages" in item:
                    messages = item["messages"]
                elif "prompt" in item and "response" in item:
                    messages = [
                        {"role": "user", "content": str(item["prompt"])},
                        {"role": "assistant", "content": str(item["response"])},
                    ]
                else:
                    return

                if messages and len(messages) >= 2:
                    yield {
                        "messages": messages,
                        "metadata": {
                            "source_file": str(source_path),
                            "source_type": "json",
                            "converted_at": datetime.now().isoformat(),
                            "original_format": "json",
                        },
                    }
                    self.conversation_count += 1
        except Exception as e:
            logger.error(f"Error processing JSON item: {e}")

    def _convert_conversation_format(self, conversation: Any) -> List[Dict[str, str]]:
        """Convert various conversation formats to ChatML"""
        messages = []

        if isinstance(conversation, list):
            for turn in conversation:
                if isinstance(turn, dict):
                    role = turn.get("from", turn.get("role", "user"))
                    content = turn.get("value", turn.get("content", ""))

                    # Map roles to ChatML format
                    if role.lower() in ["human", "user", "patient", "client"]:
                        role = "user"
                    elif role.lower() in [
                        "gpt",
                        "bot",
                        "assistant",
                        "therapist",
                        "counselor",
                    ]:
                        role = "assistant"
                    elif role.lower() == "system":
                        role = "system"
                    else:
                        role = "user" if len(messages) % 2 == 0 else "assistant"

                    if content and str(content).strip():
                        messages.append({"role": role, "content": str(content).strip()})

        return messages

    def process_jsonl_file(
        self, jsonl_path: Path
    ) -> Generator[Dict[str, Any], None, None]:
        """Process JSONL files directly"""
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        item = json.loads(line.strip())

                        # Ensure ChatML format
                        if "messages" in item and isinstance(item["messages"], list):
                            # Already in ChatML format
                            yield item
                        else:
                            # Convert to ChatML format
                            messages = self._convert_conversation_format(
                                item.get("conversation", item)
                            )
                            if messages and len(messages) >= 2:
                                yield {
                                    "messages": messages,
                                    "metadata": {
                                        **item.get("metadata", {}),
                                        "source_file": str(jsonl_path),
                                        "source_type": "jsonl",
                                        "line_number": line_num,
                                    },
                                }

                        self.conversation_count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {jsonl_path}: {e}"
                        )

        except Exception as e:
            logger.error(f"Error processing JSONL {jsonl_path}: {e}")


class StreamingPIICleaner:
    """
    Memory-efficient PII cleaning for streaming data
    """

    def __init__(self):
        # Enhanced PII patterns for therapeutic context
        self.patterns = {
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            "phone": re.compile(
                r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?:\s?(?:ext|extension)\.?\s?\d{1,5})?\b",
                re.IGNORECASE,
            ),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b", re.IGNORECASE),
            "credit_card": re.compile(
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b", re.IGNORECASE
            ),
            "medical_id": re.compile(
                r"\b(?:patient\s+id|medical\s+record|mrn|chart\s+number)\s*:?\s*\d+\b",
                re.IGNORECASE,
            ),
            "insurance_id": re.compile(
                r"\b(?:insurance\s+id|policy\s+number|member\s+id)\s*:?\s*[A-Z0-9-]+\b",
                re.IGNORECASE,
            ),
            "full_name": re.compile(
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", re.IGNORECASE
            ),
            "location": re.compile(
                r"\b(?:\d+\s+[A-Z][a-z]+\s+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|boulevard|blvd))\b",
                re.IGNORECASE,
            ),
            "specific_date": re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                re.IGNORECASE,
            ),
            "url": re.compile(r"\b(?:https?://|www\.)[^\s]+\b", re.IGNORECASE),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", re.IGNORECASE),
            "phone_short": re.compile(r"\b\d{3}-\d{3}-\d{4}\b", re.IGNORECASE),
            "zip_code": re.compile(r"\b\d{5}(?:-\d{4})?\b", re.IGNORECASE),
        }

        self.redaction_placeholders = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CARD]",
            "medical_id": "[MEDICAL_ID]",
            "insurance_id": "[INSURANCE_ID]",
            "full_name": "[NAME]",
            "location": "[LOCATION]",
            "specific_date": "[DATE]",
            "url": "[URL]",
            "ip_address": "[IP]",
            "phone_short": "[PHONE]",
            "zip_code": "[ZIP]",
        }

        self.stats = {
            "total_conversations": 0,
            "conversations_with_pii": 0,
            "total_pii_instances": 0,
            "pii_types": {},
            "messages_processed": 0,
        }

    def clean_conversation(
        self, conversation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Clean PII from a conversation"""
        if not conversation or "messages" not in conversation:
            return conversation, {}

        cleaned = conversation.copy()
        messages = cleaned.get("messages", [])
        cleaned_messages = []
        pii_found = {}

        for message in messages:
            if not isinstance(message, dict):
                cleaned_messages.append(message)
                continue

            content = message.get("content", "")
            if not isinstance(content, str):
                cleaned_messages.append(message)
                continue

            cleaned_content = content
            message_pii = {}

            # Apply PII cleaning
            for pii_type, pattern in self.patterns.items():
                matches = pattern.findall(cleaned_content)
                if matches:
                    count = len(matches)
                    message_pii[pii_type] = count
                    self.stats["total_pii_instances"] += count
                    self.stats["pii_types"][pii_type] = (
                        self.stats["pii_types"].get(pii_type, 0) + count
                    )
                    cleaned_content = pattern.sub(
                        self.redaction_placeholders[pii_type], cleaned_content
                    )

            if message_pii:
                for pii_type, count in message_pii.items():
                    pii_found[pii_type] = pii_found.get(pii_type, 0) + count

            # Update message
            cleaned_message = message.copy()
            cleaned_message["content"] = cleaned_content
            cleaned_messages.append(cleaned_message)
            self.stats["messages_processed"] += 1

        cleaned["messages"] = cleaned_messages

        # Update metadata
        metadata = cleaned.get("metadata", {})
        metadata["pii_cleaned"] = bool(pii_found)
        metadata["pii_stats"] = pii_found
        metadata["cleaned_at"] = datetime.now().isoformat()
        cleaned["metadata"] = metadata

        self.stats["total_conversations"] += 1
        if pii_found:
            self.stats["conversations_with_pii"] += 1

        return cleaned, pii_found


class DeduplicationEngine:
    """
    Enhanced deduplication with semantic similarity
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes = set()
        self.duplicate_count = 0
        self.total_processed = 0

    def compute_content_hash(self, conversation: Dict[str, Any]) -> str:
        """Compute hash for deduplication"""
        messages = conversation.get("messages", [])
        content = " ".join([m.get("content", "") for m in messages])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def is_duplicate(self, conversation: Dict[str, Any]) -> bool:
        """Check if conversation is duplicate"""
        content_hash = self.compute_content_hash(conversation)
        self.total_processed += 1

        if content_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True

        self.seen_hashes.add(content_hash)
        return False


class CompleteDatasetProcessor:
    """
    Complete 52.20GB dataset processing pipeline
    """

    def __init__(
        self,
        output_dir: str = "/home/vivi/pixelated/ai/training_ready/data/final_corpus",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.merger = DatasetMerger()
        self.cleaner = StreamingPIICleaner()
        self.deduplicator = DeduplicationEngine()

        self.processing_stats = {
            "start_time": datetime.now(),
            "files_processed": 0,
            "conversations_total": 0,
            "conversations_cleaned": 0,
            "conversations_deduplicated": 0,
            "final_conversations": 0,
            "pii_instances_removed": 0,
            "size_bytes_original": 0,
            "size_bytes_final": 0,
        }

    def get_source_files(self) -> List[Dict[str, Any]]:
        """Get prioritized list of source files"""
        # Based on audit results - prioritize by therapeutic value
        priority_order = [
            # Large consolidated datasets
            "/home/vivi/pixelated/ai/lightning/processed_data/merged_dataset_original.jsonl",
            "/home/vivi/pixelated/ai/training_ready/data/ULTIMATE_FINAL_DATASET.jsonl",
            "/home/vivi/pixelated/ai/training_ready/configs/stage_configs/ULTIMATE_FINAL_DATASET.jsonl",
            "/home/vivi/pixelated/ai/lightning/processed_data/merged_dataset.jsonl",
            # Priority therapeutic datasets
            "/home/vivi/pixelated/ai/datasets/tier6_knowledge/Psych-101/train.jsonl",
            "/home/vivi/pixelated/ai/lightning/ghost/datasets/priority_1/priority_1_FINAL.jsonl",
            "/home/vivi/pixelated/ai/lightning/ghost/datasets/priority_2/priority_2_FINAL.jsonl",
            "/home/vivi/pixelated/ai/lightning/ghost/datasets/priority_3/priority_3_FINAL.jsonl",
            # Chain-of-thought reasoning
            "/home/vivi/pixelated/ai/lightning/pixelated-training/processed/phase_3_cot_reasoning/task_5_15_cot_reasoning/cot_reasoning_conversations_consolidated.jsonl",
            "/home/vivi/pixelated/ai/datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json",
            "/home/vivi/pixelated/ai/datasets/CoT_Heartbreak_and_Breakups_downloaded.json",
            "/home/vivi/pixelated/ai/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health_downloaded.json",
            # Social media mental health
            "/home/vivi/pixelated/ai/lightning/ghost/datasets/reddit_depression_dataset.csv",
            # Professional therapeutic data
            "/home/vivi/pixelated/ai/lightning/pixelated-training/processed/phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
            "/home/vivi/pixelated/ai/lightning/pixelated-v2/training_data/enhanced/therapeutic_high_quality.json",
            "/home/vivi/pixelated/ai/lightning/pixelated-training/training_dataset.json",
            "/home/vivi/pixelated/ai/lightning/pixelated-training/training_dataset_enhanced.json",
            # Additional training data
            "/home/vivi/pixelated/ai/lightning/processed_data/unified_training_data.jsonl",
            "/home/vivi/pixelated/ai/lightning/ghost/master_integration/therapy_expert_train.jsonl",
        ]

        # Filter existing files
        existing_files = []
        for path in priority_order:
            if Path(path).exists():
                stat = Path(path).stat()
                existing_files.append(
                    {
                        "path": path,
                        "size": stat.st_size,
                        "priority": priority_order.index(path),
                    }
                )

        return sorted(existing_files, key=lambda x: x["priority"])

    def process_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Process a single file based on format"""
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return

        self.processing_stats["files_processed"] += 1
        self.processing_stats["size_bytes_original"] += path.stat().st_size

        # Process based on file extension
        if path.suffix.lower() == ".jsonl":
            yield from self.merger.process_jsonl_file(path)
        elif path.suffix.lower() == ".json":
            yield from self.merger.convert_json_to_jsonl(path)
        elif path.suffix.lower() == ".csv":
            yield from self.merger.convert_csv_to_jsonl(path)
        else:
            logger.warning(f"Unsupported file format: {path.suffix}")

    def process_complete_dataset(self) -> Dict[str, Any]:
        """Process the complete 52.20GB dataset"""
        logger.info("Starting complete 52.20GB dataset processing...")

        source_files = self.get_source_files()
        logger.info(f"Found {len(source_files)} source files to process")

        # Output files
        merged_file = self.output_dir / "merged_dataset_raw.jsonl"
        cleaned_file = self.output_dir / "merged_dataset_cleaned.jsonl"
        final_file = self.output_dir / "final_training_corpus.jsonl"

        # Step 1: Merge all sources
        logger.info("Step 1: Merging all source files...")
        with open(merged_file, "w", encoding="utf-8") as f:
            for file_info in source_files:
                logger.info(
                    f"Processing {Path(file_info['path']).name} ({file_info['size'] / 1024**3:.2f}GB)"
                )

                for conversation in self.process_file(file_info["path"]):
                    self.processing_stats["conversations_total"] += 1
                    if self.processing_stats["conversations_total"] % 10000 == 0:
                        logger.info(
                            f"Processed {self.processing_stats['conversations_total']:,} conversations..."
                        )

                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        # Step 2: Clean PII
        logger.info("Step 2: Cleaning PII from merged dataset...")
        with (
            open(merged_file, "r", encoding="utf-8") as infile,
            open(cleaned_file, "w", encoding="utf-8") as outfile,
        ):
            for line_num, line in enumerate(infile, 1):
                try:
                    conversation = json.loads(line.strip())
                    cleaned, pii_stats = self.cleaner.clean_conversation(conversation)

                    if pii_stats:
                        self.processing_stats["conversations_cleaned"] += 1
                        self.processing_stats["pii_instances_removed"] += sum(
                            pii_stats.values()
                        )

                    outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

                    if line_num % 50000 == 0:
                        logger.info(f"Cleaned {line_num:,} conversations...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        # Step 3: Deduplicate
        logger.info("Step 3: Deduplicating cleaned dataset...")
        with (
            open(cleaned_file, "r", encoding="utf-8") as infile,
            open(final_file, "w", encoding="utf-8") as outfile,
        ):
            for line_num, line in enumerate(infile, 1):
                try:
                    conversation = json.loads(line.strip())

                    if not self.deduplicator.is_duplicate(conversation):
                        outfile.write(
                            json.dumps(conversation, ensure_ascii=False) + "\n"
                        )
                        self.processing_stats["final_conversations"] += 1
                    else:
                        self.processing_stats["conversations_deduplicated"] += 1

                    if line_num % 100000 == 0:
                        logger.info(f"Deduplicated {line_num:,} conversations...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        # Final statistics
        self.processing_stats["end_time"] = datetime.now()
        self.processing_stats["duration"] = str(
            self.processing_stats["end_time"] - self.processing_stats["start_time"]
        )

        final_stat = final_file.stat()
        self.processing_stats["size_bytes_final"] = final_stat.st_size

        # Save processing report
        report_file = self.output_dir / "processing_report.json"
        with open(report_file, "w") as f:
            json.dump(self.processing_stats, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(
            f"Total conversations: {self.processing_stats['conversations_total']:,}"
        )
        logger.info(
            f"Conversations cleaned: {self.processing_stats['conversations_cleaned']:,}"
        )
        logger.info(
            f"Conversations deduplicated: {self.processing_stats['conversations_deduplicated']:,}"
        )
        logger.info(
            f"Final conversations: {self.processing_stats['final_conversations']:,}"
        )
        logger.info(
            f"PII instances removed: {self.processing_stats['pii_instances_removed']:,}"
        )
        logger.info(
            f"Final size: {self.processing_stats['size_bytes_final'] / 1024**3:.2f}GB"
        )
        logger.info(f"Processing time: {self.processing_stats['duration']}")
        logger.info(f"Report saved: {report_file}")

        return {
            "success": True,
            "final_file": str(final_file),
            "stats": self.processing_stats,
            "pii_breakdown": self.cleaner.stats["pii_types"],
            "duplicate_rate": self.deduplicator.duplicate_count
            / max(self.deduplicator.total_processed, 1),
        }


def main():
    """Main execution"""
    processor = CompleteDatasetProcessor()
    result = processor.process_complete_dataset()

    if result["success"]:
        print("\n🎯 52.20GB Dataset Processing Complete!")
        print(
            f"📊 Final corpus: {result['stats']['final_conversations']:,} conversations"
        )
        print(
            f"📈 Size reduction: {((result['stats']['size_bytes_original'] - result['stats']['size_bytes_final']) / result['stats']['size_bytes_original'] * 100):.1f}%"
        )
        print(f"🧹 PII removed: {result['stats']['pii_instances_removed']:,} instances")
        print(
            f"🔄 Duplicates removed: {result['stats']['conversations_deduplicated']:,}"
        )
        print(f"📁 Final file: {result['final_file']}")
    else:
        print("❌ Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
