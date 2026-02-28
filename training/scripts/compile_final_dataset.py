#!/usr/bin/env python3
"""
Final Dataset Compilation Script - Production Grade
Compiles all staged datasets from S3 into a single canonical training set.
Format: ChatML (messages: [{role, content}])
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add parents to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from ai.core.utils.s3_dataset_loader import S3DatasetLoader
except ImportError:
    try:
        from utils.s3_dataset_loader import S3DatasetLoader
    except ImportError:
        S3DatasetLoader = None

# --- CONFIGURATION & CONSTANTS ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DatasetCompiler")

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
}

SHARD_SIZE = 5000
TRAIN_VAL_TEST_SPLIT = (0.90, 0.05, 0.05)

DEFAULT_SYSTEM_PROMPT = (
    "You are a therapeutic AI assistant. Respond with empathy and clinical accuracy."
)


class DatasetCompiler:
    def __init__(self, bucket: str = "pixel-data"):
        if S3DatasetLoader is None:
            raise ImportError("Could not find S3DatasetLoader.")
        self.loader = S3DatasetLoader(bucket=bucket)
        self.seen_hashes: Set[str] = set()
        self.stats = defaultdict(int)
        self.shards: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
        self.shard_counters = {"train": 0, "val": 0, "test": 0}
        self.output_dir = Path("./compiled_dataset")
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.output_dir / "CHECKPOINT.json"
        self.processed_files = self._load_checkpoint()
        self.file_contributions = {}

    def normalize_record(self, record: Dict[str, Any], family: str) -> Optional[Dict]:
        """Convert various formats into standardized ChatML messages.

        Orchestrates the normalization pipeline by delegating to focused helper methods.
        """
        messages = self._extract_messages_from_record(record)
        if not messages:
            return None

        clean_messages = self._build_message_chain(messages)
        if len(clean_messages) <= 1:
            return None

        self._scrub_pii(clean_messages)

        record_hash = self._compute_content_hash(clean_messages)
        if record_hash is None:
            return None

        if record_hash in self.seen_hashes:
            self.stats["duplicates"] += 1
            return None

        self.seen_hashes.add(record_hash)

        return {
            "messages": clean_messages,
            "metadata": {
                "hash": record_hash,
                "family": family,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

    def _extract_messages_from_record(
        self, record: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract messages from various input formats.
        Supports: ChatML, Alpaca, Q&A, Dialog, and Conversation formats.
        """
        # 1. ChatML / messages format
        msg_key = next(
            (k for k in ["messages", "turns", "dialogs"] if k in record), None
        )
        if msg_key and isinstance(record[msg_key], list):
            return self._extract_conversation_format(record[msg_key], record)

        # 2. Alpaca / Instruction-Response / Q&A / Prompt-Response
        q_keys = ["question", "instruction", "prompt"]
        a_keys = ["answer", "response", "output"]

        q_key = next((k for k in q_keys if k in record), None)
        a_key = next((k for k in a_keys if k in record), None)

        if q_key and a_key:
            return self._extract_qa_format_v2(record, q_key, a_key)

        # 3. Dialog key (simple string content)
        if "dialog" in record and isinstance(record["dialog"], str):
            return self._extract_dialog_format(record)

        # 4. Standard 'conversation' list
        if "conversation" in record and isinstance(record["conversation"], list):
            return self._extract_conversation_format(record["conversation"], record)

        return []

    def _extract_qa_format_v2(
        self, record: Dict[str, Any], q_key: str, a_key: str
    ) -> List[Dict[str, str]]:
        """Enhanced Q&A extractor that supports reasoning chains."""
        reasoning = (
            record.get("reasoning")
            or record.get("reasoning_chain")
            or (
                record.get("metadata", {})
                if isinstance(record.get("metadata"), dict)
                else {}
            ).get("reasoning")
        )

        content = str(record[a_key])
        if reasoning:
            content = f"<thought>\n{reasoning}\n</thought>\n{content}"

        return [
            {"role": "user", "content": str(record[q_key])},
            {"role": "assistant", "content": content},
        ]

    def _extract_dialog_format(self, record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert simple dialog string to messages."""
        return [
            {"role": "user", "content": "Help me with my mental health issue."},
            {"role": "assistant", "content": record["dialog"]},
        ]

    def _extract_conversation_format(
        self, conversation: List[Dict], record: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """Convert conversation list to messages with reasoning support."""
        reasoning = None
        if record:
            reasoning = record.get("reasoning_chain") or (
                record.get("metadata", {})
                if isinstance(record.get("metadata"), dict)
                else {}
            ).get("reasoning")

        messages = []
        for turn in conversation:
            role = self._extract_turn_role(turn)
            content = self._extract_turn_content(turn)
            if role and content:
                messages.append(
                    {"role": self._normalize_role(role), "content": str(content)}
                )

        if reasoning and messages:
            assistant_msg = next(
                (m for m in messages if m["role"] == "assistant"), None
            )
            if assistant_msg and "<thought>" not in assistant_msg["content"]:
                assistant_msg["content"] = (
                    f"<thought>\n{reasoning}\n</thought>\n{assistant_msg['content']}"
                )
        return messages

    def _extract_turn_role(self, turn: Dict) -> Optional[str]:
        """Extract role from a conversation turn using various key names."""
        return next(
            (turn[key] for key in ("role", "from", "speaker") if turn.get(key)), None
        )

    def _extract_turn_content(self, turn: Dict) -> Optional[str]:
        """Extract content from a conversation turn using various key names."""
        content_keys = ("content", "text", "message", "dialog")
        return next(
            (turn[key] for key in content_keys if turn.get(key)),
            None,
        )

    def _build_message_chain(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build clean message chain with system prompt and merged consecutive roles."""
        clean_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()

            if not content or role == "system":
                continue

            # Merge consecutive same-role messages
            if clean_messages and clean_messages[-1]["role"] == role:
                clean_messages[-1]["content"] += "\n" + content
            else:
                clean_messages.append({"role": role, "content": content})

        return clean_messages

    def _scrub_pii(self, messages: List[Dict[str, str]]) -> None:
        """Redact PII from message content in-place."""
        for msg in messages:
            content = msg["content"]
            for label, pattern in PII_PATTERNS.items():
                content = pattern.sub(f"[REDACTED_{label.upper()}]", content)
            msg["content"] = content

    def _compute_content_hash(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Compute deduplication hash from non-system message content."""
        if full_content := "".join(
            m["content"] for m in messages if m["role"] != "system"
        ).strip():
            return hashlib.md5(full_content.encode("utf-8")).hexdigest()
        else:
            return None

    def _normalize_role(self, role: str) -> str:
        """Normalize speaker/role to standard 'user', 'assistant', or 'system'."""
        role = role.lower()
        if role in ["human", "user", "client", "patient"]:
            return "user"
        elif role in ["assistant", "bot", "gpt", "therapist"]:
            return "assistant"
        elif role == "system":
            return "system"
        else:
            return "user"

    def assign_split(self, record_hash: str) -> str:
        # 90/5/5
        val = int(record_hash[-2:], 16) / 256.0
        if val < 0.05:
            return "val"
        elif val < 0.10:
            return "test"
        else:
            return "train"

    def write_shard(self, split: str):
        if not self.shards[split]:
            return
        shard_num = self.shard_counters[split]
        filename = self.output_dir / f"{split}_shard_{shard_num:03}.jsonl"
        with open(filename, "w", encoding="utf-8") as f:
            for rec in self.shards[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Wrote shard: {filename} ({len(self.shards[split])} records)")
        self.shard_counters[split] += 1
        self.shards[split] = []

    def _get_records_from_s3_file(self, s3_path: str):
        """Load records from an S3 file (JSON or JSONL) as an iterable."""
        if not s3_path.endswith(".json"):
            # Yield from the generator directly to keep memory low
            yield from self.loader.stream_jsonl(s3_path)
            return

        data = self.loader.load_json(s3_path)
        if isinstance(data, list):
            yield from data
            return

        yield from next(
            (
                v
                for v in data.values()
                if isinstance(v, list) and v and isinstance(v[0], dict)
            ),
            [],
        )

    def _load_checkpoint(self) -> set:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return self._parse_checkpoint_file(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return set()

    def _parse_checkpoint_file(self, f):
        data = json.load(f)
        self.shard_counters = data.get("shard_counters", self.shard_counters)
        self.stats = defaultdict(int, data.get("stats", self.stats))
        self.file_contributions = data.get("file_contributions", {})
        processed_count = len(data.get("files", []))
        logger.info(
            f"Loaded checkpoint: {processed_count} files already processed. "
            f"Resuming from shard {self.shard_counters}"
        )
        return set(data.get("files", []))

    def _save_checkpoint(self, s3_path: str):
        self.processed_files.add(s3_path)
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(
                    {
                        "files": list(self.processed_files),
                        "file_contributions": self.file_contributions,
                        "shard_counters": self.shard_counters,
                        "stats": dict(self.stats),
                        "last_updated": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def process_s3_prefix(self, prefix: str, family: str, dry_run: bool = False):
        logger.info(f"Processing family '{family}' at prefix '{prefix}'")
        try:
            datasets = self.loader.list_datasets(prefix=prefix)
        except Exception as e:
            logger.error(f"Failed to list S3 prefix {prefix}: {e}")
            return

        logger.info(f"Found {len(datasets)} files in S3.")

        for s3_path in datasets:
            if s3_path in self.processed_files and not dry_run:
                logger.debug(f"Skipping already processed file: {s3_path}")
                continue

            count = 0
            try:
                records = self._get_records_from_s3_file(s3_path)
                for record in records:
                    if normalized := self.normalize_record(record, family):
                        split = self.assign_split(normalized["metadata"]["hash"])
                        self.shards[split].append(normalized)
                        self.stats[f"{split}_total"] += 1
                        count += 1
                        if len(self.shards[split]) >= SHARD_SIZE:
                            self.write_shard(split)
                    if dry_run and count >= 50:
                        break

                self.stats["files_processed"] += 1
                self.file_contributions[s3_path] = count
                logger.info(f"  Processed {s3_path}: {count} valid records.")
                self._save_checkpoint(s3_path)
                # In dry-run, we only process the first file of each prefix/family
                if dry_run:
                    break
            except Exception as e:
                logger.error(f"  Error processing {s3_path}: {e}")

    def process_local_files(self, directory: Path, family: str, dry_run: bool = False):
        logger.info(f"Processing local files for '{family}' in '{directory}'")
        if not directory.exists():
            logger.warning(f"  Local directory {directory} does not exist.")
            return

        for path in directory.glob("*.jsonl"):
            count = 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if normalized := self.normalize_record(record, family):
                            split = self.assign_split(normalized["metadata"]["hash"])
                            self.shards[split].append(normalized)
                            self.stats[f"{split}_total"] += 1
                            count += 1
                            if len(self.shards[split]) >= SHARD_SIZE:
                                self.write_shard(split)
                        if dry_run and count >= 50:
                            break
                logger.info(f"  Processed local {path.name}: {count} valid records.")
                # Dry run: only process first local file
                if dry_run:
                    break
            except Exception as e:
                logger.error(f"  Error processing local {path}: {e}")

    def finalize(self):
        for split in ["train", "val", "test"]:
            if self.shards[split]:
                self.write_shard(split)

        metadata = {
            "version": "1.1.0",
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stats": dict(self.stats),
            "splits": {
                "train": {
                    "total": self.stats["train_total"],
                    "shards": self.shard_counters["train"],
                },
                "val": {
                    "total": self.stats["val_total"],
                    "shards": self.shard_counters["val"],
                },
                "test": {
                    "total": self.stats["test_total"],
                    "shards": self.shard_counters["test"],
                },
            },
        }
        with open(self.output_dir / "METADATA.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Compilation complete.")
        logger.info(f"Summary: {json.dumps(metadata['stats'], indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Compile final training dataset.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    compiler = DatasetCompiler()

    if not args.local_only:
        # Standard prefixes
        prefixes = [
            ("training/v1/stage1_foundation/", "foundation"),
            ("training/v1/stage2_expertise/", "expertise"),
            ("training/v1/stage3_stress_test/", "stress_test"),
            ("processed_ready/", "consolidated"),
            ("archive/gdrive/processed/phase_3_cot_reasoning/", "reasoning"),
            ("archive/gdrive/processed/phase_4_reddit_mental_health/", "clinical"),
        ]

        for prefix, family in prefixes:
            compiler.process_s3_prefix(prefix, family, dry_run=args.dry_run)

    local_dir = REPO_ROOT / "training/ready_packages/datasets/synthetic"
    compiler.process_local_files(local_dir, "synthetic_pix", dry_run=args.dry_run)

    converted_dir = (
        REPO_ROOT / "training/ready_packages/datasets/cache/training_v3_converted"
    )
    compiler.process_local_files(converted_dir, "v3_converted", dry_run=args.dry_run)

    compiler.finalize()

    if args.upload:
        loader = S3DatasetLoader()
        for path in compiler.output_dir.glob("*"):
            s3_key = f"final_dataset/{path.name}"
            loader.upload_file(path, s3_key)
        logger.info("Upload complete: s3://pixel-data/final_dataset/")


if __name__ == "__main__":
    main()
