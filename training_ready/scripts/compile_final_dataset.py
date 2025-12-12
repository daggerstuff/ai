#!/usr/bin/env python3
"""
Compile Final Training Dataset - Creates manifest + compiled ChatML JSONL export
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from enhanced_deduplication import ConversationEntry, EnhancedDeduplicator, compute_content_hash

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


class FinalDatasetCompiler:
    """Compiles final training dataset with manifest + compiled export"""

    def __init__(
        self,
        routing_config_path: Path,
        coverage_report_path: Path,
        output_dir: Path,
        train_split: float = 0.9,
        val_split: float = 0.05,
        test_split: float = 0.05,
        shard_size_mb: int = 1000,
    ):
        self.routing_config_path = routing_config_path
        self.coverage_report_path = coverage_report_path
        self.output_dir = output_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shard_size_mb = shard_size_mb

        self.routing_config: dict[str, Any] = {}
        self.coverage_data: dict[str, Any] = {}
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

    def load_conversations_from_s3(
        self, family_name: str, s3_paths: list[str]
    ) -> list[dict[str, Any]]:
        """
        Load conversations from S3 for a dataset family.
        In production, this would use S3DatasetLoader.
        """
        # Placeholder - in production, load from S3
        logger.info(f"Loading conversations for {family_name} from {len(s3_paths)} S3 paths")
        # TODO: Implement actual S3 loading
        return []

    def collect_all_conversations(self) -> None:
        """Collect all conversations from all dataset families"""
        logger.info("Collecting conversations from all dataset families...")

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
                    set(c.get("metadata", {}).get("source_family", "") for c in current_shard)
                )

                shard = DatasetShard(
                    shard_id=shard_id,
                    s3_path=f"s3://pixelated-training-data/final_dataset/{split_name}/{shard_id}.jsonl",
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
                set(c.get("metadata", {}).get("source_family", "") for c in current_shard)
            )

            shard = DatasetShard(
                shard_id=shard_id,
                s3_path=f"s3://pixelated-training-data/final_dataset/{split_name}/{shard_id}.jsonl",
                size_bytes=shard_size,
                sha256=f"sha256:{shard_hash}",
                conversation_count=len(current_shard),
                source_families=source_families,
            )
            shards.append(shard)

        return shards

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
            content_hash = conv.get("metadata", {}).get("content_hash", "")
            if content_hash:
                provenance_map[content_hash] = {
                    "source_key": conv.get("metadata", {}).get("source_key", ""),
                    "source_family": conv.get("metadata", {}).get("source_family", ""),
                    "original_format": "jsonl",
                    "processing_steps": ["encoding_fix", "dedup", "chatml_convert"],
                }

        manifest = {
            "manifest_version": "1.0",
            "generated_at": datetime.now().isoformat(),
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

        return manifest

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
    output_dir = project_root / "ai" / "training_ready" / "data" / "final_dataset"

    compiler = FinalDatasetCompiler(
        routing_config_path=routing_config_path,
        coverage_report_path=coverage_report_path,
        output_dir=output_dir,
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
    exit(main())
