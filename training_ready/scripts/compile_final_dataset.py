#!/usr/bin/env python3
"""
Compile Final Training Dataset - Creates manifest + compiled ChatML JSONL export
"""

import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.scripts.enhanced_deduplication import (
    ConversationEntry,
    EnhancedDeduplicator,
    compute_content_hash,
)
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class DatasetShard:
    """Represents a sharded dataset file"""

    shard_id: str
    s3_path: str
    size_bytes: int
    sha256: str
    conversation_count: int
    source_families: list[str]


@dataclass
class SplitInfo:
    """Information about a dataset split"""

    conversations: int
    shards: list[DatasetShard]
    total_tokens_approx: int = 0


@dataclass(frozen=True)
class CompilerConfig:
    routing_config_path: Path
    coverage_report_path: Path
    output_dir: Path
    s3_manifest_path: Path
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    shard_size_mb: int = 1000


class FinalDatasetCompiler:
    """Compiles final training dataset with manifest + compiled export"""

    def __init__(self, config: CompilerConfig):
        self.routing_config_path = config.routing_config_path
        self.coverage_report_path = config.coverage_report_path
        self.output_dir = config.output_dir
        self.s3_manifest_path = config.s3_manifest_path
        self.train_split = config.train_split
        self.val_split = config.val_split
        self.test_split = config.test_split
        self.shard_size_mb = config.shard_size_mb

        self.routing_config: dict[str, Any] = {}
        self.coverage_data: dict[str, Any] = {}
        self.s3_manifest: dict[str, Any] = {}
        self.s3_bucket: str = "pixel-data"
        self.all_conversations: list[dict[str, Any]] = []
        self.deduplicator = EnhancedDeduplicator(similarity_threshold=0.95)

        # Hard holdout families (only in test)
        self.holdout_families = [
            "long_running_therapy",
            "edge_case_crisis",
            "sarcasm",
            "voice_persona",
        ]

    def load_configs(self) -> None:
        """Load routing config and coverage report"""
        with open(self.routing_config_path, encoding="utf-8") as f:
            self.routing_config = json.load(f)

        with open(self.coverage_report_path, encoding="utf-8") as f:
            self.coverage_data = json.load(f)

        with open(self.s3_manifest_path, encoding="utf-8") as f:
            self.s3_manifest = json.load(f)
        bucket = self.s3_manifest.get("bucket")
        if isinstance(bucket, str) and bucket:
            self.s3_bucket = bucket

        endpoint = self.s3_manifest.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("s3_manifest.json missing endpoint")
        self.s3_endpoint = endpoint
        self.s3_loader = S3DatasetLoader(bucket=self.s3_bucket, endpoint_url=self.s3_endpoint)

    def _resolve_s3_uri(self, s3_path: str) -> str:
        if s3_path.startswith("s3://"):
            return s3_path
        return f"s3://{self.s3_bucket}/{s3_path.lstrip('/')}"

    @staticmethod
    def _messages_are_chatml(messages: list[Any]) -> bool:
        return all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        )

    @staticmethod
    def _coerce_role(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        v = value.strip().lower()
        if v in {"system"}:
            return "system"
        if v in {"user", "human", "client", "patient"}:
            return "user"
        if v in {"assistant", "bot", "gpt", "therapist", "counselor"}:
            return "assistant"
        return None

    def _conversation_to_messages(self, record: dict[str, Any]) -> list[dict[str, str]] | None:
        """
        Convert common `conversation` formats into ChatML `messages`.

        Many of our S3 JSONLs are shaped as:
        { conversation: [ {from/role/speaker, content/text}, ... ], metadata: {...} }
        """
        conv = record.get("conversation")
        if not isinstance(conv, list) or not conv:
            return None

        messages: list[dict[str, str]] = []
        next_role = "user"

        for turn in conv:
            if isinstance(turn, str):
                content = turn.strip()
                if not content:
                    continue
                role = next_role
            elif isinstance(turn, dict):
                content_val = (
                    turn.get("content")
                    or turn.get("text")
                    or turn.get("message")
                    or turn.get("utterance")
                )
                if not isinstance(content_val, str):
                    continue
                content = content_val.strip()
                if not content:
                    continue

                role = (
                    self._coerce_role(turn.get("role"))
                    or self._coerce_role(turn.get("from"))
                    or self._coerce_role(turn.get("speaker"))
                    or self._coerce_role(turn.get("author"))
                    or next_role
                )
            else:
                continue

            # Skip extra system turns; we can inject a single system prompt if desired.
            if role == "system":
                continue

            messages.append({"role": role, "content": content})
            next_role = "assistant" if role == "user" else "user"

        if not messages:
            return None

        # Optional: ensure there's a system message at the top.
        if all(m.get("role") != "system" for m in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a therapeutic AI assistant. Respond with empathy and practical support.",
                },
            )

        return messages

    def _normalize_chatml_record(
        self, record: Any, *, seen_hashes: set[str]
    ) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None

        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not self._messages_are_chatml(messages)
        ):
            # Attempt conversion from `conversation` format.
            converted = self._conversation_to_messages(record)
            if converted is None:
                return None
            messages = converted
            record["messages"] = messages

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            record["metadata"] = metadata

        content_hash = metadata.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            content_hash = compute_content_hash(messages)
            metadata["content_hash"] = content_hash

        if content_hash in seen_hashes:
            return None

        seen_hashes.add(content_hash)
        return record

    def _load_chatml_jsonl_from_s3(self, *, family_name: str, s3_uri: str) -> list[dict[str, Any]]:
        # Retry streaming reads (OVH S3 can occasionally drop large streams).
        max_attempts = 4
        seen_hashes: set[str] = set()

        for attempt in range(1, max_attempts + 1):
            try:
                out: list[dict[str, Any]] = []
                raw_count = 0
                normalized_count = 0
                skipped_count = 0

                logger.info("Streaming from %s (attempt %s/%s)", s3_uri, attempt, max_attempts)
                for rec in self.s3_loader.stream_jsonl(s3_uri):
                    raw_count += 1
                    normalized = self._normalize_chatml_record(rec, seen_hashes=seen_hashes)
                    if normalized is not None:
                        normalized_count += 1
                        out.append(normalized)
                    else:
                        skipped_count += 1

                logger.info(
                    "Loaded %s/%s conversations from %s (skipped %s duplicates/invalid)",
                    normalized_count,
                    raw_count,
                    s3_uri,
                    skipped_count,
                )
                return out
            except FileNotFoundError as e:
                logger.warning("S3 file not found: %s - %s", s3_uri, e)
                return []
            except Exception as e:
                if attempt >= max_attempts:
                    logger.error(
                        "S3 stream error for %s (%s) after %s attempts: %s",
                        family_name,
                        s3_uri,
                        max_attempts,
                        e,
                    )
                    raise
                logger.warning(
                    "S3 stream error for %s (%s) attempt %s/%s: %s",
                    family_name,
                    s3_uri,
                    attempt,
                    max_attempts,
                    e,
                )
                time.sleep(1.5 * attempt)

        return []

    def _load_local_jsonl(self, path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    def collect_local_generated_conversations(self) -> None:
        """
        Add locally generated datasets (missing families) into the compilation inputs.
        """
        generated_dir = self.s3_manifest_path.parent / "generated"
        candidates = [
            generated_dir / "edge_case_synthetic.jsonl",
            generated_dir / "long_running_therapy.jsonl",
            generated_dir / "cptsd_transcripts.jsonl",
        ]

        for path in candidates:
            if not path.exists():
                continue
            convs = self._load_local_jsonl(path)
            for conv in convs:
                conv["metadata"] = conv.get("metadata", {})
                # Ensure content_hash exists
                if not conv["metadata"].get("content_hash"):
                    conv["metadata"]["content_hash"] = compute_content_hash(
                        conv.get("messages", [])
                    )
            self.all_conversations.extend(convs)
            logger.info("Loaded %s conversations from local %s", len(convs), path)

    def load_conversations_from_s3(
        self, family_name: str, s3_paths: list[str]
    ) -> list[dict[str, Any]]:
        """
        Load ChatML JSONL datasets from S3.

        This intentionally supports only ChatML-like records:
        - Each JSONL line is a dict with `messages: [{role, content}, ...]`
        - Optional `metadata`
        """
        if not s3_paths:
            logger.warning("No S3 paths provided for family %s", family_name)
            return []

        logger.info("Loading conversations for %s from %s S3 paths", family_name, len(s3_paths))
        out: list[dict[str, Any]] = []

        for s3_path in s3_paths:
            resolved = self._resolve_s3_uri(s3_path)
            logger.info("Processing S3 path: %s -> %s", s3_path, resolved)
            loaded = self._load_chatml_jsonl_from_s3(family_name=family_name, s3_uri=resolved)
            out.extend(loaded)
            logger.info("Accumulated %s total conversations for %s", len(out), family_name)

        logger.info(
            "Total loaded for %s: %s conversations from %s paths",
            family_name,
            len(out),
            len(s3_paths),
        )
        return out

    def load_conversations_for_family(self, family_name: str) -> list[dict[str, Any]]:
        """
        Prefer known-good ChatML JSONL S3 keys for each family.

        This is what lets us compile from the *real* S3 training outputs after
        encoding fixes, rather than relying on placeholder loader logic.
        """
        # Keys are relative to the bucket in ai/training_ready/data/s3_manifest.json
        family_keys: dict[str, list[str]] = {
            "mental_health_datasets": [
                "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
                "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
                "datasets/training_v2/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
                "datasets/training_v2/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
            ],
            "edge_case_generator": [
                "datasets/consolidated/conversations/edge_case_dialogues.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
            ],
            "edge_case_resulting_chats": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
                "datasets/gdrive/tier3_edge_crisis/crisis_detection_conversations.jsonl",
            ],
            "sarcasm": [
                # NOTE: this is JSON (not JSONL). We'll keep it out of ChatML ingestion for now.
            ],
            "professional_therapeutic": [
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_13_neuro_qa_sft/neuro_qa_sft_conversations.jsonl",
                "datasets/gdrive/processed/phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
            ],
            "priority_datasets": [
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
                "datasets/gdrive/processed/phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl",
            ],
            "cot_reasoning": [
                "datasets/gdrive/processed/phase_3_cot_reasoning/task_5_25_tot_reasoning/tot_reasoning_conversations.jsonl",
            ],
            "safety_guardrails_annihilator": [
                # These are large/raw; ChatML outputs are already captured in phase_4_reddit_mental_health
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
            ],
            "addiction": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
            ],
            "experimental": [
                "datasets/gdrive/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
            ],
        }

        keys = family_keys.get(family_name, [])
        if not keys:
            logger.warning("No S3 keys configured for family: %s", family_name)
            return []
        logger.info("Loading family %s from %s S3 keys", family_name, len(keys))
        return self.load_conversations_from_s3(family_name, keys)

    def collect_all_conversations(self) -> None:
        """Collect all conversations from all dataset families"""
        logger.info("Collecting conversations from all dataset families...")

        # Always include locally generated missing families (edge_case_synthetic, long_running_therapy, cptsd)
        self.collect_local_generated_conversations()

        # Also include known-good S3 ChatML exports (post encoding-fix).
        # These will massively increase coverage and fix distribution ratios.
        for family_name in (
            "mental_health_datasets",
            "professional_therapeutic",
            "priority_datasets",
            "cot_reasoning",
            "edge_case_generator",
            "edge_case_resulting_chats",
            "safety_guardrails_annihilator",
        ):
            convs = self.load_conversations_for_family(family_name)
            for conv in convs:
                conv["metadata"] = conv.get("metadata", {})
                conv["metadata"]["source_family"] = family_name
                conv["metadata"]["source_key"] = conv["metadata"].get(
                    "source_key",
                    f"s3://{self.s3_bucket}/{conv.get('metadata', {}).get('source_key', '')}",
                )
                if not conv["metadata"].get("content_hash"):
                    conv["metadata"]["content_hash"] = compute_content_hash(
                        conv.get("messages", [])
                    )
            self.all_conversations.extend(convs)
            logger.info("Loaded %s conversations from S3 family %s", len(convs), family_name)

        families = self.routing_config.get("families", {})

        for family_name, family_config in families.items():
            if family_config.get("status") != "available":
                logger.warning(f"Skipping {family_name} - status: {family_config.get('status')}")
                continue

            s3_path = family_config.get("s3_path")
            if not s3_path:
                continue

            # Load conversations (placeholder - implement S3 loading)
            conversations = self.load_conversations_from_s3(family_name, [s3_path])

            # Add metadata
            for conv in conversations:
                conv["metadata"] = conv.get("metadata", {})
                conv["metadata"]["source_family"] = family_name
                conv["metadata"]["source_key"] = s3_path
                conv["metadata"]["content_hash"] = compute_content_hash(conv.get("messages", []))

            self.all_conversations.extend(conversations)
            logger.info(f"Loaded {len(conversations)} conversations from {family_name}")

    def assign_splits(self) -> None:
        """Assign conversations to train/val/test splits"""
        logger.info("Assigning conversations to splits...")

        # Separate holdout families
        holdout_conversations = []
        regular_conversations = []

        for conv in self.all_conversations:
            source_family = conv.get("metadata", {}).get("source_family", "")
            if source_family in self.holdout_families:
                holdout_conversations.append(conv)
            else:
                regular_conversations.append(conv)

        # Holdout families go to test only
        for conv in holdout_conversations:
            conv["metadata"]["split"] = "test"

        # Regular conversations: train/val/test split
        random.shuffle(regular_conversations)
        total_regular = len(regular_conversations)

        train_count = int(total_regular * self.train_split)
        val_count = int(total_regular * self.val_split)
        test_count = total_regular - train_count - val_count

        for i, conv in enumerate(regular_conversations):
            if i < train_count:
                conv["metadata"]["split"] = "train"
            elif i < train_count + val_count:
                conv["metadata"]["split"] = "val"
            else:
                conv["metadata"]["split"] = "test"

        logger.info(
            f"Split assignment: train={train_count}, val={val_count}, test={test_count + len(holdout_conversations)}"
        )

    def deduplicate(self) -> None:
        """Deduplicate conversations"""
        logger.info("Deduplicating conversations...")

        # Convert to ConversationEntry format
        for conv in self.all_conversations:
            entry = ConversationEntry(
                messages=conv.get("messages", []),
                source_family=conv.get("metadata", {}).get("source_family", ""),
                source_key=conv.get("metadata", {}).get("source_key", ""),
                content_hash=conv.get("metadata", {}).get("content_hash", ""),
                split=conv.get("metadata", {}).get("split"),
                metadata=conv.get("metadata", {}),
            )
            self.deduplicator.add_conversation(entry)

        # Find and remove duplicates
        exact_dups = self.deduplicator.find_exact_duplicates()
        near_dups = self.deduplicator.find_near_duplicates()

        logger.info(
            f"Found {len(exact_dups)} exact duplicate groups, {len(near_dups)} near-duplicate pairs"
        )

        # Deduplicate
        deduplicated_entries = self.deduplicator.deduplicate(strategy="keep_first")

        # Convert back to conversation format
        self.all_conversations = [
            {"messages": entry.messages, "metadata": entry.metadata}
            for entry in deduplicated_entries
        ]

        logger.info(f"After deduplication: {len(self.all_conversations)} conversations")

    def check_split_leakage(self) -> dict[str, Any]:
        """Check for split leakage violations"""
        logger.info("Checking for split leakage violations...")

        violations = self.deduplicator.check_split_leakage(self.holdout_families)

        if violations["exact_duplicate_leakage"]:
            logger.warning(
                f"Found {len(violations['exact_duplicate_leakage'])} exact duplicate leakage violations"
            )

        if violations["near_duplicate_leakage"]:
            logger.warning(
                f"Found {len(violations['near_duplicate_leakage'])} near-duplicate leakage violations"
            )

        if violations["holdout_family_leakage"]:
            logger.error(
                f"Found {len(violations['holdout_family_leakage'])} holdout family leakage violations"
            )

        return violations

    def create_shards(
        self, conversations: list[dict[str, Any]], split_name: str
    ) -> list[DatasetShard]:
        """Create sharded files for a split"""
        shards = []
        shard_size_bytes = self.shard_size_mb * 1024 * 1024

        current_shard: list[dict[str, Any]] = []
        current_size = 0
        shard_num = 0

        for conv in conversations:
            conv_json = json.dumps(conv, ensure_ascii=False) + "\n"
            conv_size = len(conv_json.encode("utf-8"))

            if current_size + conv_size > shard_size_bytes and current_shard:
                # Save current shard
                shard_id = f"{split_name}_{shard_num:03d}"
                shard_path = self.output_dir / "shards" / f"{shard_id}.jsonl"
                shard_path.parent.mkdir(parents=True, exist_ok=True)

                with open(shard_path, "w", encoding="utf-8") as f:
                    for c in current_shard:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")

                # Compute hash
                with open(shard_path, "rb") as f:
                    shard_hash = hashlib.sha256(f.read()).hexdigest()

                shard_size = shard_path.stat().st_size
                source_families = list(
                    {c.get("metadata", {}).get("source_family", "") for c in current_shard}
                )

                shard = DatasetShard(
                    shard_id=shard_id,
                    s3_path=f"s3://{self.s3_bucket}/final_dataset/{split_name}/{shard_id}.jsonl",
                    size_bytes=shard_size,
                    sha256=f"sha256:{shard_hash}",
                    conversation_count=len(current_shard),
                    source_families=source_families,
                )
                shards.append(shard)

                # Reset for next shard
                current_shard = []
                current_size = 0
                shard_num += 1

            current_shard.append(conv)
            current_size += conv_size

        # Save final shard
        if current_shard:
            self._extracted_from_create_shards_55(split_name, shard_num, current_shard, shards)
        return shards

    # TODO Rename this here and in `create_shards`
    def _extracted_from_create_shards_55(self, split_name, shard_num, current_shard, shards):
        shard_id = f"{split_name}_{shard_num:03d}"
        shard_path = self.output_dir / "shards" / f"{shard_id}.jsonl"
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(shard_path, "w", encoding="utf-8") as f:
            for c in current_shard:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        with open(shard_path, "rb") as f:
            shard_hash = hashlib.sha256(f.read()).hexdigest()

        shard_size = shard_path.stat().st_size
        source_families = list(
            {c.get("metadata", {}).get("source_family", "") for c in current_shard}
        )

        shard = DatasetShard(
            shard_id=shard_id,
            s3_path=f"s3://{self.s3_bucket}/final_dataset/{split_name}/{shard_id}.jsonl",
            size_bytes=shard_size,
            sha256=f"sha256:{shard_hash}",
            conversation_count=len(current_shard),
            source_families=source_families,
        )
        shards.append(shard)

    def generate_manifest(self) -> dict[str, Any]:
        """Generate dataset manifest"""
        logger.info("Generating dataset manifest...")

        # Group by split
        by_split = defaultdict(list)
        by_family = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

        for conv in self.all_conversations:
            split = conv.get("metadata", {}).get("split", "train")
            family = conv.get("metadata", {}).get("source_family", "unknown")
            by_split[split].append(conv)
            by_family[family][split] += 1

        # Create shards for each split
        splits_info = {}
        for split_name in ["train", "val", "test"]:
            conversations = by_split[split_name]
            shards = self.create_shards(conversations, split_name)

            splits_info[split_name] = {
                "conversations": len(conversations),
                "shards": [
                    {
                        "shard_id": s.shard_id,
                        "s3_path": s.s3_path,
                        "size_bytes": s.size_bytes,
                        "sha256": s.sha256,
                        "conversation_count": s.conversation_count,
                        "source_families": s.source_families,
                    }
                    for s in shards
                ],
            }

        # Build provenance map
        provenance_map = {}
        for conv in self.all_conversations:
            if content_hash := conv.get("metadata", {}).get("content_hash", ""):
                provenance_map[content_hash] = {
                    "source_key": conv.get("metadata", {}).get("source_key", ""),
                    "source_family": conv.get("metadata", {}).get("source_family", ""),
                    "original_format": "jsonl",
                    "processing_steps": ["encoding_fix", "dedup", "chatml_convert"],
                }

        return {
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_conversations": len(self.all_conversations),
            "total_tokens_approx": 0,  # TODO: Compute actual token count
            "splits": splits_info,
            "source_families": {
                family: {"conversations": sum(counts.values()), "splits": counts}
                for family, counts in by_family.items()
            },
            "provenance_map": provenance_map,
            "holdout_families": {
                family: {"test_split_only": True, "rationale": "Hard holdout for evaluation"}
                for family in self.holdout_families
            },
        }

    def create_compiled_export(self) -> Path:
        """Create compiled single-file export"""
        logger.info("Creating compiled export...")

        output_path = self.output_dir / "compiled" / "final_training_dataset.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for conv in self.all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"Compiled export created: {output_path}")
        return output_path

    def compile(self) -> dict[str, Any]:
        """Run complete compilation process"""
        logger.info("Starting final dataset compilation...")

        self.load_configs()
        self.collect_all_conversations()
        self.assign_splits()
        self.deduplicate()

        # Check for leakage
        violations = self.check_split_leakage()
        if violations["holdout_family_leakage"]:
            logger.error("Holdout family leakage detected! Fix before proceeding.")
            return {"error": "Holdout family leakage", "violations": violations}

        # Generate manifest
        manifest = self.generate_manifest()

        # Create compiled export
        compiled_path = self.create_compiled_export()

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Manifest saved: {manifest_path}")

        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "compiled_path": str(compiled_path),
            "total_conversations": len(self.all_conversations),
            "violations": violations,
        }


def main():
    """Main entry point"""
    project_root = Path(__file__).parents[3]

    routing_config_path = (
        project_root / "ai" / "training_ready" / "data" / "dataset_routing_config.json"
    )
    coverage_report_path = (
        project_root / "ai" / "training_ready" / "data" / "dataset_coverage_report.json"
    )
    s3_manifest_path = project_root / "ai" / "training_ready" / "data" / "s3_manifest.json"
    output_dir = project_root / "ai" / "training_ready" / "data" / "final_dataset"

    compiler = FinalDatasetCompiler(
        CompilerConfig(
            routing_config_path=routing_config_path,
            coverage_report_path=coverage_report_path,
            output_dir=output_dir,
            s3_manifest_path=s3_manifest_path,
        )
    )

    result = compiler.compile()

    if "error" in result:
        logger.error(f"Compilation failed: {result['error']}")
        return 1

    logger.info("✅ Final dataset compilation complete!")
    logger.info(f"   Total conversations: {result['total_conversations']}")
    logger.info(f"   Manifest: {result['manifest_path']}")
    logger.info(f"   Compiled export: {result['compiled_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
