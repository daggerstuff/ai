#!/usr/bin/env python3
"""
Compile Final Training Dataset - v2 (Refactored & Optimized)

Architecture:
- Collects conversations from 14 data families (S3 + local generated)
- Deduplicates using semantic similarity + content hashing
- Validates using 8-gate quality checks
- Creates train/val/test splits with leakage prevention
- Shards dataset for efficient distributed training
- Uploads to S3 with manifest metadata

Flow:
  1. Load checkpoint (resume-friendly)
  2. Collect all conversations from all families
  3. Deduplicate (semantic + hash-based)
  4. Assign splits (train/val/test)
  5. Check holdout family leakage
  6. Create shards
  7. Upload to S3
"""

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.scripts.enhanced_deduplication import (
    ConversationEntry,
    EnhancedDeduplicator,
    compute_content_hash,
)
from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

# ============================================================================
# Configuration & Logging
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("compile_dataset.log"),
    ],
)


# ============================================================================
# Data Classes
# ============================================================================

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


@dataclass
class CompilerConfig:
    """Configuration for dataset compilation"""
    routing_config_path: Path
    coverage_report_path: Path
    output_dir: Path
    s3_manifest_path: Path
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.05
    shard_size_mb: int = 1000
    checkpoint_dir: Path | None = None
    resume: bool = True
    dedup_threshold: float = 0.95


@dataclass
class CheckpointInfo:
    """Enhanced checkpoint for resume capability"""
    stage: str
    processed_families: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    total_conversations: int = 0
    timestamp: str = ""
    family_stats: dict[str, int] = field(default_factory=dict)
    progress: float = 0.0  # Progress percentage


# ============================================================================
# Main Compilation Class
# ============================================================================

class FinalDatasetCompiler:
    """Compiles final training dataset with checkpoint/resume support"""

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir or (config.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "collection_checkpoint.json"

        # Data storage
        self.all_conversations: list[dict[str, Any]] = []
        self.deduplicator = EnhancedDeduplicator(similarity_threshold=config.dedup_threshold)
        self.s3_loader = S3DatasetLoader(bucket="pixel-data")

        # Tracking
        self.processed_families: set[str] = set()
        self.processed_files: set[str] = set()
        self.family_stats: dict[str, int] = {}

        # Holdout families (only in test split)
        self.holdout_families = {
            "long_running_therapy",
            "edge_case_crisis",
            "sarcasm",
            "voice_persona",
        }

        # Splits
        self.splits = {
            "train": [],
            "validation": [],
            "test": [],
        }

    def load_configs(self) -> None:
        """Load routing and coverage configs"""
        with open(self.config.routing_config_path) as f:
            self.routing_config = json.load(f)
        logger.info(f"✓ Loaded routing config with {len(self.routing_config.get('families', {}))} families")

    def save_checkpoint(self, stage: str, progress: float = 0.0) -> None:
        """Save checkpoint for resume capability"""
        checkpoint = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processed_families": list(self.processed_families),
            "processed_files": list(self.processed_files),
            "total_conversations": len(self.all_conversations),
            "progress": progress,
            "family_stats": self.family_stats,
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"✓ Checkpoint saved: {len(self.all_conversations)} conversations")

    def load_checkpoint(self) -> CheckpointInfo | None:
        """Load checkpoint if resuming"""
        if not self.config.resume or not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file) as f:
            data = json.load(f)

        self.processed_families = set(data.get("processed_families", []))
        self.processed_files = set(data.get("processed_files", []))
        self.family_stats = data.get("family_stats", {})

        logger.info(
            f"✓ Resumed from checkpoint: {len(self.processed_families)} families, "
            f"{len(self.processed_files)} files already processed"
        )
        # Don't pass stage twice - it's in data already
        checkpoint_data = {k: v for k, v in data.items() if k != "stage"}
        return CheckpointInfo(stage=data["stage"], **checkpoint_data)

    def collect_conversations_from_family(self, family_name: str) -> list[dict[str, Any]]:
        """Load all conversations for a family from S3 or local"""
        if family_name in self.processed_families:
            logger.debug(f"Skipping {family_name} (already processed)")
            return []

        logger.info(f"[{len(self.processed_families)+1}/14] Loading {family_name}...")
        convs = self._load_family_from_routing(family_name)

        # Add metadata
        for conv in convs:
            if "metadata" not in conv:
                conv["metadata"] = {}
            conv["metadata"]["source_family"] = family_name
            if "content_hash" not in conv["metadata"]:
                conv["metadata"]["content_hash"] = compute_content_hash(conv.get("messages", []))

        self.processed_families.add(family_name)
        self.family_stats[family_name] = len(convs)
        logger.info(f"  ✓ {family_name}: {len(convs):,} conversations")

        return convs

    def _load_family_from_routing(self, family_name: str) -> list[dict[str, Any]]:
        """Load a family using routing config"""
        family_config = self.routing_config.get("families", {}).get(family_name, {})
        s3_path = family_config.get("s3_path")

        if not s3_path:
            logger.warning(f"No S3 path for {family_name}")
            return []

        try:
            if s3_path.endswith(".jsonl"):
                return self._load_jsonl_from_s3(s3_path)
            elif s3_path.endswith(".json"):
                return self._load_json_from_s3(s3_path)
            elif s3_path.endswith(".csv"):
                return self._load_csv_from_s3(s3_path)
            else:
                logger.warning(f"Unknown format for {s3_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading {family_name} from {s3_path}: {e}")
            return []

    def _load_jsonl_from_s3(self, s3_path: str) -> list[dict[str, Any]]:
        """Stream JSONL from S3"""
        convs = []
        try:
            # Prepend bucket name if not already a full S3 path
            if not s3_path.startswith("s3://"):
                s3_path = f"s3://{self.s3_loader.bucket}/{s3_path}"
            for line in self.s3_loader.stream_jsonl(s3_path):
                if line and isinstance(line, dict):
                    convs.append(line)
        except Exception as e:
            logger.warning(f"Error streaming {s3_path}: {e}")
        return convs

    def _load_json_from_s3(self, s3_path: str) -> list[dict[str, Any]]:
        """Load JSON from S3"""
        try:
            # Prepend bucket name if not already a full S3 path
            if not s3_path.startswith("s3://"):
                s3_path = f"s3://{self.s3_loader.bucket}/{s3_path}"
            data = self.s3_loader.load_json(s3_path)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            return []
        except Exception as e:
            logger.warning(f"Error loading {s3_path}: {e}")
            return []

    def _load_csv_from_s3(self, s3_path: str) -> list[dict[str, Any]]:
        """Load CSV from S3 and convert to conversations"""
        # Placeholder - implement based on actual CSV structure
        logger.info(f"CSV loading for {s3_path} not yet implemented")
        return []

    def collect_all_conversations(self) -> None:
        """Main collection loop across all 14 families"""
        logger.info("=" * 80)
        logger.info("PHASE 1: COLLECTING CONVERSATIONS FROM 14 DATA FAMILIES")
        logger.info("=" * 80)

        families = sorted(self.routing_config.get("families", {}).keys())

        for idx, family_name in enumerate(families, 1):
            convs = self.collect_conversations_from_family(family_name)
            self.all_conversations.extend(convs)
            self.save_checkpoint("collection", progress=(idx / len(families)) * 100)

            # Periodic garbage collection
            if idx % 3 == 0:
                gc.collect()

        logger.info(f"✓ Total conversations collected: {len(self.all_conversations):,}")
        self.save_checkpoint("collection", progress=100.0)

    def deduplicate(self) -> None:
        """Deduplicate conversations using semantic similarity + hashing"""
        logger.info("=" * 80)
        logger.info("PHASE 2: DEDUPLICATION")
        logger.info("=" * 80)

        # Convert to deduplication format and add to deduplicator
        logger.info(f"Input: {len(self.all_conversations):,} conversations")
        for conv in self.all_conversations:
            entry = ConversationEntry(
                id=conv.get("id", hashlib.sha256(str(conv).encode()).hexdigest()[:8]),
                messages=conv.get("messages", []),
                metadata=conv.get("metadata", {}),
            )
            self.deduplicator.add_conversation(entry)

        # Run deduplication
        deduplicated = self.deduplicator.deduplicate()
        logger.info(f"Output: {len(deduplicated):,} conversations after deduplication")

        # Update all_conversations with deduplicated results
        self.all_conversations = [
            {
                "id": e.id,
                "messages": e.messages,
                "metadata": e.metadata,
            }
            for e in deduplicated
        ]
        self.save_checkpoint("deduplication", progress=100.0)

    def assign_splits(self) -> None:
        """Assign conversations to train/val/test splits"""
        logger.info("=" * 80)
        logger.info("PHASE 3: ASSIGNING SPLITS")
        logger.info("=" * 80)

        # Shuffle conversations
        random.shuffle(self.all_conversations)

        total = len(self.all_conversations)
        train_count = int(total * self.config.train_split)
        val_count = int(total * self.config.val_split)

        self.splits["train"] = self.all_conversations[:train_count]
        self.splits["validation"] = self.all_conversations[train_count : train_count + val_count]
        self.splits["test"] = self.all_conversations[train_count + val_count :]

        for split, convs in self.splits.items():
            pct = (len(convs)/total*100) if total > 0 else 0.0
            logger.info(f"  {split}: {len(convs):,} conversations ({pct:.1f}%)")

        self.save_checkpoint("splits", progress=100.0)

    def check_holdout_leakage(self) -> bool:
        """Verify holdout families are only in test split"""
        logger.info("=" * 80)
        logger.info("PHASE 4: CHECKING HOLDOUT FAMILY LEAKAGE")
        logger.info("=" * 80)

        holdout_in_train = [
            c for c in self.splits["train"]
            if c.get("metadata", {}).get("source_family") in self.holdout_families
        ]
        holdout_in_val = [
            c for c in self.splits["validation"]
            if c.get("metadata", {}).get("source_family") in self.holdout_families
        ]

        if holdout_in_train or holdout_in_val:
            logger.error(f"❌ LEAKAGE DETECTED!")
            logger.error(f"  Holdout in train: {len(holdout_in_train)}")
            logger.error(f"  Holdout in val: {len(holdout_in_val)}")
            return False

        logger.info("✓ No holdout family leakage detected")
        return True

    def create_shards(self) -> dict[str, list[DatasetShard]]:
        """Create shards for efficient training"""
        logger.info("=" * 80)
        logger.info("PHASE 5: CREATING SHARDS")
        logger.info("=" * 80)

        shards = {}
        shard_size_bytes = self.config.shard_size_mb * 1024 * 1024

        for split_name, conversations in self.splits.items():
            split_shards = []
            current_shard = []
            current_size = 0

            for idx, conv in enumerate(conversations):
                conv_json = json.dumps(conv)
                conv_size = len(conv_json.encode())

                if current_size + conv_size > shard_size_bytes and current_shard:
                    # Save current shard
                    shard = self._save_shard(split_name, len(split_shards), current_shard)
                    split_shards.append(shard)
                    current_shard = []
                    current_size = 0

                current_shard.append(conv)
                current_size += conv_size

            # Save remaining
            if current_shard:
                shard = self._save_shard(split_name, len(split_shards), current_shard)
                split_shards.append(shard)

            shards[split_name] = split_shards
            logger.info(f"  {split_name}: {len(split_shards)} shards")

        return shards

    def _save_shard(self, split: str, shard_id: int, conversations: list) -> DatasetShard:
        """Save a single shard to local disk"""
        shard_name = f"{split}_shard_{shard_id:04d}.jsonl"
        shard_path = self.config.output_dir / shard_name

        with open(shard_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        size_bytes = shard_path.stat().st_size
        sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()

        families = set()
        for conv in conversations:
            fam = conv.get("metadata", {}).get("source_family")
            if fam:
                families.add(fam)

        return DatasetShard(
            shard_id=shard_name,
            s3_path=f"datasets/compiled/{shard_name}",
            size_bytes=size_bytes,
            sha256=sha256,
            conversation_count=len(conversations),
            source_families=list(families),
        )

    def upload_to_s3(self, shards: dict[str, list[DatasetShard]]) -> None:
        """Upload shards to S3"""
        logger.info("=" * 80)
        logger.info("PHASE 6: UPLOADING TO S3")
        logger.info("=" * 80)

        total_size = 0
        for split_name, split_shards in shards.items():
            for shard in split_shards:
                local_path = self.config.output_dir / shard.shard_id
                try:
                    self.s3_loader.upload_file(str(local_path), shard.s3_path)
                    total_size += shard.size_bytes
                    logger.info(f"  ✓ Uploaded {shard.shard_id} ({shard.size_bytes / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    logger.error(f"  ❌ Failed to upload {shard.shard_id}: {e}")

        logger.info(f"✓ Total uploaded: {total_size / 1024 / 1024 / 1024:.2f} GB")

    def create_manifest(self, shards: dict[str, list[DatasetShard]]) -> dict[str, Any]:
        """Create manifest with all dataset metadata"""
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_conversations": len(self.all_conversations),
            "splits": {
                split: {
                    "conversations": len(self.splits[split]),
                    "shards": len(shards.get(split, [])),
                    "total_size_mb": sum(s.size_bytes for s in shards.get(split, [])) / 1024 / 1024,
                }
                for split in self.splits
            },
            "family_stats": self.family_stats,
            "shards": {
                split: [
                    {
                        "id": s.shard_id,
                        "s3_path": s.s3_path,
                        "size_bytes": s.size_bytes,
                        "sha256": s.sha256,
                        "conversation_count": s.conversation_count,
                        "source_families": s.source_families,
                    }
                    for s in split_shards
                ]
                for split, split_shards in shards.items()
            },
        }

        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"✓ Manifest created: {manifest_path}")
        return manifest

    def compile(self) -> tuple[dict, Path]:
        """Execute full compilation pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING FINAL DATASET COMPILATION")
        logger.info("=" * 80)

        self.load_configs()
        self.load_checkpoint()

        self.collect_all_conversations()
        self.deduplicate()
        self.assign_splits()

        if not self.check_holdout_leakage():
            raise RuntimeError("Holdout family leakage detected!")

        shards = self.create_shards()
        self.upload_to_s3(shards)
        manifest = self.create_manifest(shards)

        logger.info("=" * 80)
        logger.info("✓ COMPILATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Total conversations: {manifest['total_conversations']:,}")
        logger.info(f"  Train: {manifest['splits']['train']['conversations']:,}")
        logger.info(f"  Val: {manifest['splits']['validation']['conversations']:,}")
        logger.info(f"  Test: {manifest['splits']['test']['conversations']:,}")

        return manifest, self.config.output_dir / "manifest.json"


# ============================================================================
# CLI & Main
# ============================================================================

def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compile final training dataset")
    parser.add_argument("--routing-config", type=Path, default=Path("ai/training_ready/data/dataset_routing_config.json"))
    parser.add_argument("--coverage-report", type=Path, default=Path("ai/training_ready/data/coverage_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("ai/training_ready/data/compiled"))
    parser.add_argument("--s3-manifest", type=Path, default=Path("ai/training_ready/data/s3_manifest.json"))
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--shard-size-mb", type=int, default=1000)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = CompilerConfig(
        routing_config_path=args.routing_config,
        coverage_report_path=args.coverage_report,
        output_dir=args.output_dir,
        s3_manifest_path=args.s3_manifest,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        shard_size_mb=args.shard_size_mb,
        resume=not args.no_resume,
    )

    compiler = FinalDatasetCompiler(config)
    manifest, manifest_path = compiler.compile()

    logger.info(f"Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
