#!/usr/bin/env python3
"""Merge all training datasets into a final combined training file.

This script merges datasets from multiple sources into a single, deduplicated,
shuffled training file ready for model training:

Sources merged:
- Transcripts (all_transcripts_chatml.jsonl)
- CPTSD dataset (cptsd_dataset_from_transcripts.jsonl)
- Long-running therapy sessions (long_running_therapy.jsonl)
- Professional therapeutic datasets (from S3)
- Synthetic Nemotron3 conversations (if available)
- NeMo Data Designer outputs (if available)

Outputs:
- ai/training_ready/data/generated/final_merged_training.jsonl (main output)
- ai/training_ready/data/generated/final_merged_training_stats.json (statistics)
- Optionally splits into train/val/test sets with --split
- Optionally uploads to S3 with --upload-s3

Features:
- Content hash deduplication across all sources
- Configurable train/val/test splits
- Source family tracking and statistics
- Phase/stage tagging for curriculum training
- Memory-efficient streaming for large datasets
- S3 integration for both input and output

Usage Examples:
    # Merge all local generated datasets
    python merge_all_datasets.py

    # Merge with train/val/test split and upload to S3
    python merge_all_datasets.py --split --upload-s3

    # Include S3 datasets in merge
    python merge_all_datasets.py --include-s3-datasets --verbose

    # Merge specific sources only
    python merge_all_datasets.py --source transcripts --source cptsd

    # Custom split ratios
    python merge_all_datasets.py --split --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05

Notes:
- Uses content hashes to deduplicate across all sources
- Shuffles data by default (disable with --no-shuffle)
- Compatible with NeMo Data Designer pipeline
- Tracks provenance for each record

Author: Pixelated Empathy AI Team
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Default local generated datasets
DEFAULT_LOCAL_SOURCES = {
    "transcripts": "ai/training_ready/data/generated/all_transcripts_chatml.jsonl",
    "cptsd": "ai/training_ready/data/generated/cptsd_dataset_from_transcripts.jsonl",
    "long_running": "ai/training_ready/data/generated/long_running_therapy.jsonl",
    "nemotron_synthetic": "ai/training_ready/data/generated/nemotron_synthetic.jsonl",
    "nemo_data_designer": "ai/training_ready/data/generated/nemo_data_designer_output.jsonl",
}

# S3 dataset paths (from MASTER_TRAINING_EPIC)
S3_DATASET_SOURCES = {
    # Stage 1: Foundation Therapeutic Dialogue
    "foundation_amod": "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
    "foundation_helios": "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
    # Stage 2: Clinical Reasoning
    "clinical_expert_qa": "datasets/gdrive/processed/clinical_reasoning/expert_qa.jsonl",
    # Tier 2 Professional datasets
    "soulchat": "datasets/gdrive/tier2_professional/soulchat_2_0_complete_no_limits.jsonl",
    # Phase 1 Priority conversations
    "priority_1": "datasets/gdrive/processed/phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl",
    "priority_2": "datasets/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl",
    "priority_3": "datasets/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
    # Phase 2 Professional datasets
    "counsel_chat": "datasets/gdrive/processed/phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
    "therapist_sft": "datasets/gdrive/processed/phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl",
}

# Stage/phase mappings for curriculum training
STAGE_MAPPINGS = {
    "transcripts": ("stage4_voice_personas", 4),
    "cptsd": ("stage6_specialized_domains", 6),
    "long_running": ("stage5_long_running_therapy", 5),
    "nemotron_synthetic": ("stage1_foundation", 1),
    "nemo_data_designer": ("stage1_foundation", 1),
    "foundation_amod": ("stage1_foundation", 1),
    "foundation_helios": ("stage1_foundation", 1),
    "clinical_expert_qa": ("stage2_clinical_reasoning", 2),
    "soulchat": ("stage1_foundation", 1),
    "priority_1": ("stage1_foundation", 1),
    "priority_2": ("stage1_foundation", 1),
    "priority_3": ("stage1_foundation", 1),
    "counsel_chat": ("stage1_foundation", 1),
    "therapist_sft": ("stage1_foundation", 1),
}

# Progress logging interval
PROGRESS_LOG_INTERVAL = 5000

logger = logging.getLogger(__name__)


def _content_hash(messages: list[dict[str, str]]) -> str:
    """Generate a content hash for deduplication."""
    # Hash the concatenated content of all messages
    content = "".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"


def _iter_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a local JSONL file."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line {line_num} in {path}: {e}")
                    continue
    except OSError as e:
        logger.warning(f"Failed to read {path}: {e}")


def _iter_s3_jsonl(loader: Any, bucket: str, key: str) -> Iterator[dict[str, Any]]:
    """Stream records from an S3 JSONL file."""
    try:
        s3_path = f"s3://{bucket}/{key}"
        return loader.stream_jsonl(s3_path)
    except Exception as e:
        logger.warning(f"Failed to stream S3 key {key}: {e}")
        return iter([])


def _validate_record(record: dict[str, Any]) -> bool:
    """Validate a record has required ChatML structure."""
    if not isinstance(record, dict):
        return False
    
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    
    # Check for at least one user and one assistant message
    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    if "user" not in roles or "assistant" not in roles:
        return False
    
    # Check all messages have content
    for m in messages:
        if not isinstance(m, dict):
            return False
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
    
    return True


def _enrich_metadata(
    record: dict[str, Any],
    source_name: str,
    source_path: str,
) -> dict[str, Any]:
    """Enrich record metadata with merge information."""
    # Get or create metadata
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    
    # Add merge-specific fields
    stage_info = STAGE_MAPPINGS.get(source_name, ("stage1_foundation", 1))
    
    metadata["merge_source"] = source_name
    metadata["merge_source_path"] = source_path
    metadata["merge_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Set stage/phase if not already set
    if "phase" not in metadata:
        metadata["phase"] = stage_info[0]
    if "curriculum_stage" not in metadata:
        metadata["curriculum_stage"] = stage_info[1]
    
    # Ensure split is set
    if "split" not in metadata:
        metadata["split"] = "train"
    
    record["metadata"] = metadata
    return record


def _load_s3_config(manifest_path: Path) -> tuple[str, str]:
    """Load S3 configuration from manifest.
    
    The manifest file can be large (several MB), so we parse it fully
    to extract bucket/endpoint from the top-level keys.
    """
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            bucket = manifest.get("bucket", "pixel-data")
            endpoint = manifest.get("endpoint", "https://s3.us-east-va.io.cloud.ovh.us")
            return bucket, endpoint
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read manifest: {e}")
    
    return "pixel-data", "https://s3.us-east-va.io.cloud.ovh.us"


def _get_s3_loader(bucket: str, endpoint: str) -> Any:
    """Get S3 loader instance."""
    try:
        from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
        return S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)
    except ImportError:
        logger.warning("S3DatasetLoader not available, S3 sources will be skipped")
        return None


def _upload_to_s3(
    local_path: Path,
    s3_key: str,
    bucket: str,
    loader: Any,
) -> bool:
    """Upload a local file to S3."""
    if loader is None:
        logger.warning("S3 loader not available, skipping upload")
        return False
    
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def _write_split_files(
    records: list[dict[str, Any]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Path]:
    """Write records to train/val/test split files."""
    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }
    
    paths = {}
    for split_name, split_records in splits.items():
        # Update split in metadata
        for r in split_records:
            if "metadata" in r:
                r["metadata"]["split"] = split_name
        
        split_path = output_dir / f"final_merged_{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in split_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        paths[split_name] = split_path
        logger.info(f"  {split_name}: {len(split_records):,} records -> {split_path}")
    
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge all training datasets into a final combined training file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Merge local generated datasets
  %(prog)s --split --upload-s3                # Merge, split, and upload to S3
  %(prog)s --include-s3-datasets --verbose    # Include S3 datasets
  %(prog)s --source transcripts --source cptsd  # Merge specific sources
  %(prog)s --split --train-ratio 0.9          # Custom split ratio
""",
    )
    parser.add_argument(
        "--output",
        default="ai/training_ready/data/generated/final_merged_training.jsonl",
        metavar="PATH",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="NAME",
        help="Include specific source by name (repeatable). Names: transcripts, cptsd, long_running, nemotron_synthetic, nemo_data_designer",
    )
    parser.add_argument(
        "--include-s3-datasets",
        action="store_true",
        help="Include S3 datasets in merge (requires S3 access)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split output into train/val/test files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        metavar="RATIO",
        help="Training set ratio (default: 0.9)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="Validation set ratio (default: 0.05)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="Test set ratio (default: 0.05)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle records before writing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        metavar="N",
        help="Maximum records per source (0 = unlimited)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output files to S3 after generation",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="gdrive/processed/merged_training",
        metavar="PREFIX",
        help="S3 prefix for uploaded output",
    )
    parser.add_argument(
        "--manifest",
        default="ai/training_ready/data/s3_manifest.json",
        metavar="PATH",
        help="Path to S3 manifest JSON for bucket/endpoint config",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress logging",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Validate split ratios
    if args.split:
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            parser.error(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s" if args.verbose else "%(message)s",
    )
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parents[3]  # ai/training_ready/scripts -> project root
    output_path = project_root / args.output
    manifest_path = project_root / args.manifest
    
    logger.info("=" * 60)
    logger.info("MERGE ALL TRAINING DATASETS")
    logger.info("=" * 60)
    logger.info(f"Output: {output_path}")
    
    # Determine which local sources to include
    local_sources_to_use = {}
    if args.sources:
        for name in args.sources:
            if name in DEFAULT_LOCAL_SOURCES:
                local_sources_to_use[name] = DEFAULT_LOCAL_SOURCES[name]
            else:
                logger.warning(f"Unknown source: {name}")
    else:
        local_sources_to_use = DEFAULT_LOCAL_SOURCES.copy()
    
    # Load S3 config if needed
    s3_loader = None
    bucket = None
    if args.include_s3_datasets or args.upload_s3:
        bucket, endpoint = _load_s3_config(manifest_path)
        s3_loader = _get_s3_loader(bucket, endpoint)
        if s3_loader:
            logger.info(f"S3 bucket: {bucket}")
            logger.info(f"S3 endpoint: {endpoint}")
    
    # Statistics tracking
    stats: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "output_file": str(output_path),
        "sources_processed": Counter(),
        "sources_kept": Counter(),
        "sources_skipped": Counter(),
        "duplicates_removed": 0,
        "invalid_records": 0,
        "total_records": 0,
        "stage_distribution": Counter(),
    }
    
    # Collect all records with deduplication
    all_records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    last_progress = 0
    
    # Process local sources
    logger.info("")
    logger.info("Processing local sources...")
    for source_name, source_path in local_sources_to_use.items():
        full_path = project_root / source_path
        
        if not full_path.exists():
            logger.info(f"  {source_name}: not found (skipping)")
            stats["sources_skipped"][source_name] = "not_found"
            continue
        
        logger.info(f"  {source_name}: {full_path.name}")
        source_count = 0
        source_kept = 0
        
        for record in _iter_jsonl_file(full_path):
            source_count += 1
            
            # Progress logging
            total_seen = sum(stats["sources_processed"].values()) + source_count
            if total_seen - last_progress >= PROGRESS_LOG_INTERVAL:
                logger.info(f"    Progress: {total_seen:,} seen, {len(all_records):,} kept")
                last_progress = total_seen
            
            # Validate
            if not _validate_record(record):
                stats["invalid_records"] += 1
                continue
            
            # Deduplicate
            messages = record.get("messages", [])
            content_hash = _content_hash(messages)
            if content_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
            seen_hashes.add(content_hash)
            
            # Enrich metadata
            record = _enrich_metadata(record, source_name, str(source_path))
            
            # Track stage
            stage = record.get("metadata", {}).get("phase", "unknown")
            stats["stage_distribution"][stage] += 1
            
            all_records.append(record)
            source_kept += 1
            
            # Check per-source limit
            if args.max_per_source > 0 and source_kept >= args.max_per_source:
                logger.info(f"    Reached limit of {args.max_per_source} for {source_name}")
                break
        
        stats["sources_processed"][source_name] = source_count
        stats["sources_kept"][source_name] = source_kept
        logger.info(f"    ✓ Kept {source_kept:,} / {source_count:,} records")
    
    # Process S3 sources if requested
    if args.include_s3_datasets and s3_loader and bucket:
        logger.info("")
        logger.info("Processing S3 sources...")
        
        for source_name, s3_key in S3_DATASET_SOURCES.items():
            logger.info(f"  {source_name}: s3://{bucket}/{s3_key}")
            source_count = 0
            source_kept = 0
            
            try:
                for record in _iter_s3_jsonl(s3_loader, bucket, s3_key):
                    source_count += 1
                    
                    # Progress logging
                    total_seen = sum(stats["sources_processed"].values()) + source_count
                    if total_seen - last_progress >= PROGRESS_LOG_INTERVAL:
                        logger.info(f"    Progress: {total_seen:,} seen, {len(all_records):,} kept")
                        last_progress = total_seen
                    
                    # Validate
                    if not _validate_record(record):
                        stats["invalid_records"] += 1
                        continue
                    
                    # Deduplicate
                    messages = record.get("messages", [])
                    content_hash = _content_hash(messages)
                    if content_hash in seen_hashes:
                        stats["duplicates_removed"] += 1
                        continue
                    seen_hashes.add(content_hash)
                    
                    # Enrich metadata
                    record = _enrich_metadata(record, source_name, f"s3://{bucket}/{s3_key}")
                    
                    # Track stage
                    stage = record.get("metadata", {}).get("phase", "unknown")
                    stats["stage_distribution"][stage] += 1
                    
                    all_records.append(record)
                    source_kept += 1
                    
                    # Check per-source limit
                    if args.max_per_source > 0 and source_kept >= args.max_per_source:
                        logger.info(f"    Reached limit of {args.max_per_source} for {source_name}")
                        break
                
                stats["sources_processed"][source_name] = source_count
                stats["sources_kept"][source_name] = source_kept
                logger.info(f"    ✓ Kept {source_kept:,} / {source_count:,} records")
                
            except Exception as e:
                logger.warning(f"    ⚠ Failed to process: {e}")
                stats["sources_skipped"][source_name] = str(e)
    
    if not all_records:
        logger.error("No records collected from any source!")
        return 1
    
    stats["total_records"] = len(all_records)
    logger.info("")
    logger.info(f"Total records collected: {len(all_records):,}")
    
    # Shuffle if requested
    if not args.no_shuffle:
        logger.info(f"Shuffling with seed={args.seed}...")
        random.seed(args.seed)
        random.shuffle(all_records)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    if args.split:
        logger.info("")
        logger.info("Writing split files...")
        split_paths = _write_split_files(
            all_records,
            output_path.parent,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        stats["split_files"] = {k: str(v) for k, v in split_paths.items()}
        stats["split_counts"] = {
            "train": int(len(all_records) * args.train_ratio),
            "val": int(len(all_records) * args.val_ratio),
            "test": len(all_records) - int(len(all_records) * args.train_ratio) - int(len(all_records) * args.val_ratio),
        }
    else:
        logger.info(f"Writing merged file to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"✓ Wrote {len(all_records):,} records")
    
    # Finalize stats
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["sources_processed"] = dict(stats["sources_processed"])
    stats["sources_kept"] = dict(stats["sources_kept"])
    stats["sources_skipped"] = dict(stats["sources_skipped"])
    stats["stage_distribution"] = dict(stats["stage_distribution"])
    stats["unique_hashes"] = len(seen_hashes)
    
    # Write stats
    stats_path = output_path.with_name("final_merged_training_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Total records: {len(all_records):,}")
    logger.info(f"  Duplicates removed: {stats['duplicates_removed']:,}")
    logger.info(f"  Invalid records: {stats['invalid_records']:,}")
    logger.info("")
    logger.info("Sources kept:")
    for source, count in sorted(stats["sources_kept"].items(), key=lambda x: -x[1]):
        logger.info(f"  {source}: {count:,}")
    logger.info("")
    logger.info("Stage distribution:")
    for stage, count in sorted(stats["stage_distribution"].items(), key=lambda x: -x[1]):
        logger.info(f"  {stage}: {count:,}")
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"Stats: {stats_path}")
    
    # Upload to S3 if requested
    if args.upload_s3 and s3_loader and bucket:
        logger.info("")
        logger.info("Uploading to S3...")
        
        uploads_success = True
        
        if args.split:
            # Upload split files
            for split_name, split_path in split_paths.items():
                s3_key = f"{args.s3_output_prefix}/final_merged_{split_name}.jsonl"
                if not _upload_to_s3(split_path, s3_key, bucket, s3_loader):
                    uploads_success = False
        else:
            # Upload main file
            s3_key = f"{args.s3_output_prefix}/final_merged_training.jsonl"
            if not _upload_to_s3(output_path, s3_key, bucket, s3_loader):
                uploads_success = False
        
        # Upload stats
        stats_s3_key = f"{args.s3_output_prefix}/final_merged_training_stats.json"
        if not _upload_to_s3(stats_path, stats_s3_key, bucket, s3_loader):
            uploads_success = False
        
        if uploads_success:
            logger.info(f"✓ Uploaded to s3://{bucket}/{args.s3_output_prefix}/")
        else:
            logger.error("Some uploads failed")
            return 1
    
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
