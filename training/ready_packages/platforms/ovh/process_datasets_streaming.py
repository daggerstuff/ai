#!/usr/bin/env python3
"""
Memory-Efficient Dataset Processing Pipeline for ChatML Format
Pixelated Empathy - Wayfarer-2-12B Fine-Tuning

STREAMING VERSION - Designed for low-memory servers (4GB RAM)
- Processes files one at a time
- Writes output incrementally
- Clears memory between datasets
- Uses generators where possible

Target Format (ChatML/ShareGPT):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    python3 process_datasets_streaming.py --base-dir ~/datasets/consolidated

Author: Pixelated Empathy AI Team
"""

import json
import csv
import os
import re
import sys
import gc
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Iterator, Any
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# System prompts for different training stages
SYSTEM_PROMPTS = {
    "foundation": """You are a compassionate and skilled mental health counselor.
Your role is to provide supportive, empathetic responses while maintaining professional boundaries.
Listen actively, validate emotions, and help clients explore their feelings and develop coping strategies.""",

    "reasoning": """You are a clinical mental health expert who thinks through problems systematically.
When responding, consider the psychological context, potential underlying issues, and evidence-based approaches.
Demonstrate your clinical reasoning process while remaining warm and supportive.""",

    "voice": """You are a therapeutic guide in the style of Tim Fletcher, combining trauma-informed insights
with accessible, educational explanations. Use clear analogies, validate experiences,
and help people understand the 'why' behind their patterns while offering practical pathways forward."""
}


class StreamingDatasetProcessor:
    """Memory-efficient processor using streaming and generators."""

    def __init__(self, base_dir: str = "~/datasets/consolidated"):
        self.base_dir = Path(base_dir).expanduser()
        self.output_dir = self.base_dir / "chatml_formatted"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = defaultdict(lambda: {"total": 0, "valid": 0, "skipped": 0})
        self.seen_hashes = set()
        self.hash_file = self.output_dir / ".seen_hashes"

        # Load existing hashes if resuming
        if self.hash_file.exists():
            with open(self.hash_file, 'r') as f:
                self.seen_hashes = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.seen_hashes)} existing hashes for deduplication")

    def hash_conversation(self, messages: list[dict]) -> str:
        """Create hash for deduplication."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def is_duplicate(self, messages: list[dict]) -> bool:
        """Check if conversation is a duplicate."""
        h = self.hash_conversation(messages)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def save_hash(self, h: str):
        """Append hash to file for persistence."""
        with open(self.hash_file, 'a') as f:
            f.write(h + '\n')

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def validate_conversation(self, messages: list[dict], min_turns: int = 2) -> bool:
        """Validate a conversation has minimum content."""
        if len(messages) < min_turns:
            return False
        user_content = sum(1 for m in messages if m.get("role") == "user" and len(m.get("content", "").strip()) > 10)
        assistant_content = sum(1 for m in messages if m.get("role") == "assistant" and len(m.get("content", "").strip()) > 10)
        return user_content >= 1 and assistant_content >= 1

    # =========== STREAMING GENERATORS ===========

    def stream_json_array(self, filepath: Path) -> Iterator[dict]:
        """Stream items from a JSON array file without loading entire file."""
        logger.info(f"  Streaming: {filepath.name}")

        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip initial whitespace and opening bracket
            char = f.read(1)
            while char and char in ' \n\t\r':
                char = f.read(1)

            if char != '[':
                # Not an array, try loading as single object or JSONL
                f.seek(0)
                content = f.read()
                if content.strip().startswith('{'):
                    # Single object or JSONL
                    for line in content.strip().split('\n'):
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except:
                                pass
                return

            # Parse array items one at a time
            buffer = ""
            depth = 0
            in_string = False
            escape = False

            for char in iter(lambda: f.read(1), ''):
                if escape:
                    buffer += char
                    escape = False
                    continue

                if char == '\\' and in_string:
                    buffer += char
                    escape = True
                    continue

                if char == '"':
                    in_string = not in_string
                    buffer += char
                    continue

                if in_string:
                    buffer += char
                    continue

                if char == '{':
                    depth += 1
                    buffer += char
                elif char == '}':
                    depth -= 1
                    buffer += char
                    if depth == 0 and buffer.strip():
                        try:
                            yield json.loads(buffer.strip())
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON object: {e}")
                        buffer = ""
                elif char == ',' and depth == 0:
                    buffer = ""
                elif char == ']' and depth == 0:
                    break
                elif depth > 0:
                    buffer += char

    def stream_jsonl(self, filepath: Path) -> Iterator[dict]:
        """Stream items from a JSONL file."""
        logger.info(f"  Streaming JSONL: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    def stream_csv(self, filepath: Path) -> Iterator[dict]:
        """Stream rows from a CSV file."""
        logger.info(f"  Streaming CSV: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    # =========== DATASET PROCESSORS ===========

    def process_mental_health_counseling(self, output_file) -> int:
        """Process mental_health_counseling_conversations."""
        logger.info("Processing: mental_health_counseling_conversations")
        count = 0

        data_file = self.base_dir / "professional/mental_health_counseling_conversations/combined_dataset.json"
        if not data_file.exists():
            logger.warning(f"File not found: {data_file}")
            return 0

        for item in self.stream_json_array(data_file):
            self.stats["mental_health_counseling"]["total"] += 1

            context = self.clean_text(item.get("Context", ""))
            response = self.clean_text(item.get("Response", ""))

            if not context or not response:
                self.stats["mental_health_counseling"]["skipped"] += 1
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                {"role": "user", "content": context},
                {"role": "assistant", "content": response}
            ]

            if not self.is_duplicate(messages):
                output_file.write(json.dumps({"messages": messages, "source": "mental_health_counseling"}, ensure_ascii=False) + '\n')
                self.stats["mental_health_counseling"]["valid"] += 1
                count += 1
            else:
                self.stats["mental_health_counseling"]["skipped"] += 1

        gc.collect()
        return count

    def process_therapist_sft(self, output_file) -> int:
        """Process therapist-sft-format CSV."""
        logger.info("Processing: therapist-sft-format")
        count = 0

        data_file = self.base_dir / "professional/therapist-sft-format/train.csv"
        if not data_file.exists():
            logger.warning(f"File not found: {data_file}")
            return 0

        for row in self.stream_csv(data_file):
            self.stats["therapist_sft"]["total"] += 1

            text = row.get("text", "")
            if not text:
                continue

            messages = [{"role": "system", "content": SYSTEM_PROMPTS["foundation"]}]

            pattern = r'(human:|gpt:)'
            parts = re.split(pattern, text, flags=re.IGNORECASE)

            current_role = None
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.lower() == "human:":
                    current_role = "user"
                elif part.lower() == "gpt:":
                    current_role = "assistant"
                elif current_role and part:
                    content = self.clean_text(part)
                    if content:
                        messages.append({"role": current_role, "content": content})

            if self.validate_conversation(messages) and not self.is_duplicate(messages):
                output_file.write(json.dumps({"messages": messages, "source": "therapist_sft"}, ensure_ascii=False) + '\n')
                self.stats["therapist_sft"]["valid"] += 1
                count += 1
            else:
                self.stats["therapist_sft"]["skipped"] += 1

        gc.collect()
        return count

    def process_cot_dataset(self, dataset_name: str, output_file) -> int:
        """Process a single CoT reasoning dataset."""
        logger.info(f"Processing COT: {dataset_name}")
        count = 0

        dataset_dir = self.base_dir / "cot_reasoning" / dataset_name
        if not dataset_dir.exists():
            return 0

        json_files = [f for f in dataset_dir.glob("*.json") if not f.name.startswith('.') and 'readme' not in f.name.lower()]
        if not json_files:
            json_files = [f for f in dataset_dir.rglob("*.json") if not f.name.startswith('.') and 'readme' not in f.name.lower()]

        for json_file in json_files:
            for item in self.stream_json_array(json_file):
                self.stats[f"cot_{dataset_name}"]["total"] += 1

                answer = self.clean_text(item.get("answer", ""))
                if not answer:
                    self.stats[f"cot_{dataset_name}"]["skipped"] += 1
                    continue

                metadata = item.get("metadata", {})
                reasoning = self.clean_text(metadata.get("reasoning", ""))
                question = self.clean_text(item.get("question", item.get("prompt", "Can you help me understand this situation?")))

                if reasoning:
                    response = f"Let me think through this carefully.\n\n**Reasoning:**\n{reasoning}\n\n**Response:**\n{answer}"
                else:
                    response = answer

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPTS["reasoning"]},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]

                if self.validate_conversation(messages) and not self.is_duplicate(messages):
                    output_file.write(json.dumps({
                        "messages": messages,
                        "source": f"cot_{dataset_name}"
                    }, ensure_ascii=False) + '\n')
                    self.stats[f"cot_{dataset_name}"]["valid"] += 1
                    count += 1
                else:
                    self.stats[f"cot_{dataset_name}"]["skipped"] += 1

            gc.collect()

        return count

    def process_soulchat(self, output_file) -> int:
        """Process SoulChat2.0."""
        logger.info("Processing: SoulChat2.0")
        count = 0

        soulchat_dir = self.base_dir / "professional/SoulChat2.0/PsyDTCorpus"
        if not soulchat_dir.exists():
            return 0

        for json_file in soulchat_dir.glob("*.json"):
            for item in self.stream_json_array(json_file):
                self.stats["soulchat"]["total"] += 1

                messages_raw = item.get("messages", item.get("conversations", []))
                if not messages_raw:
                    continue

                messages = [{"role": "system", "content": SYSTEM_PROMPTS["foundation"]}]
                for msg in messages_raw:
                    role = msg.get("role", "").lower()
                    content = self.clean_text(msg.get("content", ""))

                    if role in ["user", "human"]:
                        role = "user"
                    elif role in ["assistant", "gpt", "bot"]:
                        role = "assistant"
                    else:
                        continue

                    if content:
                        messages.append({"role": role, "content": content})

                if self.validate_conversation(messages) and not self.is_duplicate(messages):
                    output_file.write(json.dumps({"messages": messages, "source": "soulchat"}, ensure_ascii=False) + '\n')
                    self.stats["soulchat"]["valid"] += 1
                    count += 1
                else:
                    self.stats["soulchat"]["skipped"] += 1

            gc.collect()

        return count

    def process_counsel_chat(self, output_file) -> int:
        """Process counsel-chat."""
        logger.info("Processing: counsel-chat")
        count = 0

        counsel_dir = self.base_dir / "professional/counsel-chat"
        if not counsel_dir.exists():
            return 0

        for json_file in counsel_dir.rglob("*.json"):
            for item in self.stream_json_array(json_file):
                self.stats["counsel_chat"]["total"] += 1

                question = self.clean_text(item.get("questionText", item.get("question", item.get("Context", ""))))
                answer = self.clean_text(item.get("answerText", item.get("answer", item.get("Response", ""))))

                if not question or not answer:
                    self.stats["counsel_chat"]["skipped"] += 1
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]

                if not self.is_duplicate(messages):
                    output_file.write(json.dumps({"messages": messages, "source": "counsel_chat"}, ensure_ascii=False) + '\n')
                    self.stats["counsel_chat"]["valid"] += 1
                    count += 1
                else:
                    self.stats["counsel_chat"]["skipped"] += 1

            gc.collect()

        return count

    # =========== MAIN PIPELINE ===========

    def run_staged_pipeline(self):
        """Run the pipeline creating staged output files."""
        logger.info("=" * 60)
        logger.info("STARTING STREAMING DATASET PROCESSING PIPELINE")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Stage 1: Foundation datasets
        logger.info("\n📚 STAGE 1: Processing Foundation Datasets")
        stage1_dir = self.output_dir / "stage1_foundation"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        with open(stage1_dir / "train.jsonl", 'w', encoding='utf-8') as f:
            total_foundation = 0
            total_foundation += self.process_mental_health_counseling(f)
            total_foundation += self.process_therapist_sft(f)
            total_foundation += self.process_soulchat(f)
            total_foundation += self.process_counsel_chat(f)

        logger.info(f"  Stage 1 Total: {total_foundation} conversations")

        # Stage 2: Reasoning datasets
        logger.info("\n🧠 STAGE 2: Processing Reasoning Datasets")
        stage2_dir = self.output_dir / "stage2_reasoning"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        cot_dir = self.base_dir / "cot_reasoning"
        total_reasoning = 0

        with open(stage2_dir / "train.jsonl", 'w', encoding='utf-8') as f:
            if cot_dir.exists():
                for dataset_dir in sorted(cot_dir.iterdir()):
                    if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                        count = self.process_cot_dataset(dataset_dir.name, f)
                        total_reasoning += count
                        logger.info(f"    {dataset_dir.name}: {count}")

        logger.info(f"  Stage 2 Total: {total_reasoning} conversations")

        # Generate final report
        self.generate_report(start_time, total_foundation, total_reasoning)

        logger.info("\n✅ STAGING COMPLETE!")
        logger.info(f"Output directory: {self.output_dir}")

    def generate_report(self, start_time: datetime, foundation: int, reasoning: int):
        """Generate processing statistics report."""
        end_time = datetime.now()
        duration = end_time - start_time

        report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "output_directory": str(self.output_dir),
            "stage_counts": {
                "foundation": foundation,
                "reasoning": reasoning,
                "total": foundation + reasoning
            },
            "datasets": dict(self.stats),
            "unique_hashes": len(self.seen_hashes)
        }

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)

        for name, stats in sorted(self.stats.items()):
            logger.info(f"{name}: {stats['valid']}/{stats['total']} valid")

        logger.info("-" * 60)
        logger.info(f"STAGE 1 (Foundation): {foundation:,}")
        logger.info(f"STAGE 2 (Reasoning): {reasoning:,}")
        logger.info(f"TOTAL: {foundation + reasoning:,}")
        logger.info(f"UNIQUE: {len(self.seen_hashes):,}")
        logger.info(f"DURATION: {duration}")

        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nReport: {report_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream-process datasets for ChatML training")
    parser.add_argument("--base-dir", default="~/datasets/consolidated",
                       help="Base directory for consolidated datasets")

    args = parser.parse_args()

    processor = StreamingDatasetProcessor(base_dir=args.base_dir)
    processor.run_staged_pipeline()


if __name__ == "__main__":
    main()

