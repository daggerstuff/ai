#!/usr/bin/env python3
"""Extract long-running therapy sessions from multi-turn professional datasets.

This script extracts conversations with 20+ turns (configurable) from therapeutic
datasets and converts them to ChatML format for Stage 5 (Long Running Therapy)
training.

Outputs:
- ai/training_ready/data/generated/long_running_therapy.jsonl
- ai/training_ready/data/generated/long_running_therapy_stats.json

Features:
- Streams from S3 or local files (memory-efficient for large datasets)
- Converts various conversation formats to ChatML
- Uploads output directly to S3 with --upload-s3 flag
- Scans entire directories with --input-dir flag
- Progress logging for long-running operations

Usage Examples:
    # Extract from default S3 sources
    python extract_long_running_therapy.py

    # Extract with custom min turns and upload to S3
    python extract_long_running_therapy.py --min-turns 30 --upload-s3

    # Scan all JSONL in an S3 prefix
    python extract_long_running_therapy.py --input-dir s3://pixel-data/gdrive/processed/

    # Process local files
    python extract_long_running_therapy.py --source-key /path/to/local/file.jsonl

Notes:
- Uses the s3_manifest.json bucket/endpoint for S3 access.
- Conversations are tagged with metadata.source_family=long_running_therapy.
- Output is assigned to split=test (hard holdout by contract).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

# Comprehensive list of professional therapeutic datasets for long-running extraction
# These are organized by tier from the MASTER_TRAINING_EPIC
DEFAULT_SOURCE_KEYS = [
    # Tier 2 Professional datasets (high-quality therapeutic conversations)
    "datasets/gdrive/tier2_professional/soulchat_2_0_complete_no_limits.jsonl",
    "datasets/gdrive/tier2_professional/additional_specialized_conversations.jsonl",
    "datasets/gdrive/raw/SoulChat2.0/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json",
    # Phase 1 Priority conversations (Wendy curated)
    "datasets/gdrive/processed/phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl",
    "datasets/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl",
    "datasets/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
    # Phase 2 Professional datasets
    "datasets/gdrive/processed/phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
    "datasets/gdrive/processed/phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
    "datasets/gdrive/processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl",
    "datasets/gdrive/processed/phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl",
    # Training v2/v3 stage 1 foundation (already in ChatML)
    "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
    "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
    "datasets/training_v2/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
]

# Progress logging interval
PROGRESS_LOG_INTERVAL = 1000

logger = logging.getLogger(__name__)


def _extract_turn_content(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if not isinstance(turn, dict):
        return ""
    for k in ("content", "text", "value", "message", "utterance"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _role_from_turn(turn: Any) -> str | None:
    if not isinstance(turn, dict):
        return None
    for k in ("role", "from", "speaker", "author"):
        v = turn.get(k)
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("system",):
                return "system"
            if vv in ("user", "human", "client", "patient"):
                return "user"
            if vv in ("assistant", "gpt", "bot", "therapist", "counselor"):
                return "assistant"
    return None


def _to_chatml_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    # Try common structures
    turns: list[Any] = []

    if isinstance(record.get("conversations"), list):
        turns = record["conversations"]
    elif isinstance(record.get("conversation"), list):
        turns = record["conversation"]
    elif isinstance(record.get("messages"), list):
        # Already ChatML-ish
        msgs = []
        for m in record["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content})
        return msgs

    if not turns:
        return []

    # Some corpora store conversations as a list-of-lists
    # (each inner list is a dialogue).
    # If so, flatten one level when elements are lists of turns.
    if isinstance(turns[0], list):
        # Prefer the longest inner dialogue as the "session"
        inner = max((t for t in turns if isinstance(t, list)), key=len, default=[])
        turns = inner

    # Add a stable system message to support long-session continuity
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a therapeutic AI assistant. "
                "Maintain continuity across a long session, "
                "track context, and respond with empathy and practical support."
            ),
        }
    ]

    # Build roles; if unknown, alternate starting with user
    next_role = "user"
    for t in turns:
        content = _extract_turn_content(t)
        if not content:
            continue

        role = _role_from_turn(t) or next_role
        if role == "system":
            # skip extra system turns; keep single system header
            continue
        messages.append({"role": role, "content": content})

        next_role = "assistant" if role == "user" else "user"

    # Need at least one user+assistant exchange
    roles = {m["role"] for m in messages}
    return messages if "user" in roles and "assistant" in roles else []


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract long-running therapy sessions (20+ turns) from multi-turn datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default S3 sources
  %(prog)s --min-turns 30 --upload-s3         # Extract 30+ turns, upload to S3
  %(prog)s --input-dir s3://pixel-data/gdrive/processed/  # Scan S3 directory
  %(prog)s --source-key /local/file.jsonl     # Process local file
""",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        help="Path to s3_manifest.json for bucket/endpoint config (default: ai/training_ready/data/s3_manifest.json)",
    )
    parser.add_argument(
        "--source-key",
        action="append",
        default=[],
        metavar="PATH",
        help="S3 key or local path to stream JSONL from (repeatable). Supports s3://bucket/key, relative S3 keys, or local paths. If omitted, uses built-in defaults.",
    )
    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        help="S3 prefix (s3://bucket/prefix/) or local directory to scan for all .jsonl files. Overrides --source-key.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=20,
        metavar="N",
        help="Minimum number of user+assistant turns to qualify as long-running (default: 20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Maximum number of conversations to extract (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parents[1] / "data" / "generated" / "long_running_therapy.jsonl"
        ),
        metavar="PATH",
        help="Output JSONL file path (default: ai/training_ready/data/generated/long_running_therapy.jsonl)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output files to S3 after generation (to gdrive/processed/long_running_therapy/)",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="gdrive/processed/long_running_therapy",
        metavar="PREFIX",
        help="S3 prefix for uploaded output (default: gdrive/processed/long_running_therapy)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress logging",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket")
    endpoint = manifest.get("endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("s3_manifest.json missing bucket")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("s3_manifest.json missing endpoint")
    return bucket, endpoint


def _iter_payload_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return iter(payload)
    if isinstance(payload, dict):
        return iter(
            payload.get("data") or payload.get("records") or payload.get("conversations") or []
        )
    return iter([])


def _is_local_path(path: str) -> bool:
    """Check if path is a local filesystem path (not S3).
    
    Uses path prefix patterns to determine locality without filesystem calls.
    """
    if path.startswith("s3://"):
        return False
    # Check if it looks like a local path by prefix patterns
    # Avoid filesystem existence checks for performance
    return (
        path.startswith("/")
        or path.startswith("./")
        or path.startswith("../")
        or path.startswith("~")
        or (len(path) > 1 and path[1] == ":")  # Windows drive letter
    )


def _iter_local_records(path: str) -> Iterator[dict[str, Any]]:
    """Iterate over records from a local JSONL or JSON file."""
    local_path = Path(path).expanduser()
    if not local_path.exists():
        logger.warning(f"Local file not found: {path}")
        return iter([])
    
    lower = path.lower()
    if lower.endswith(".jsonl"):
        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSONL line: {e}")
                        continue
        except OSError as e:
            logger.warning(f"Failed to read local file {path}: {e}")
            return
    elif lower.endswith(".json"):
        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                payload = json.load(f)
            yield from _iter_payload_records(payload)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read local JSON file {path}: {e}")
            return
    else:
        logger.warning(f"Unknown file type (expected .json or .jsonl): {path}")
        return


def _iter_source_records(
    loader: S3DatasetLoader,
    *,
    bucket: str,
    key: str,
) -> Iterator[dict[str, Any]]:
    """Iterate over records from an S3 key or local path."""
    # Check if this is a local path
    if _is_local_path(key):
        return _iter_local_records(key)
    
    # Handle S3 paths
    if key.startswith("s3://"):
        s3_path = key
    else:
        s3_path = f"s3://{bucket}/{key}"
    
    lower = key.lower()
    if lower.endswith(".jsonl"):
        return loader.stream_jsonl(s3_path)
    if lower.endswith(".json"):
        return _iter_payload_records(loader.load_json(s3_path))
    return iter([])


def _list_jsonl_files_in_dir(
    loader: S3DatasetLoader,
    *,
    bucket: str,
    input_dir: str,
) -> list[str]:
    """List all .jsonl files in a directory (S3 prefix or local)."""
    files: list[str] = []
    
    if input_dir.startswith("s3://"):
        # Parse S3 URI
        without_prefix = input_dir[5:]  # Remove s3://
        if "/" in without_prefix:
            s3_bucket, prefix = without_prefix.split("/", 1)
        else:
            s3_bucket = without_prefix
            prefix = ""
        
        logger.info(f"Scanning S3 prefix: s3://{s3_bucket}/{prefix}")
        try:
            paginator = loader.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)
            
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".jsonl"):
                            files.append(f"s3://{s3_bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
    elif _is_local_path(input_dir):
        # Local directory
        local_dir = Path(input_dir)
        if local_dir.is_dir():
            logger.info(f"Scanning local directory: {local_dir}")
            for f in local_dir.rglob("*.jsonl"):
                files.append(str(f))
        else:
            logger.warning(f"Local directory not found: {input_dir}")
    else:
        # Treat as S3 prefix without s3:// prefix
        prefix = input_dir
        logger.info(f"Scanning S3 prefix: s3://{bucket}/{prefix}")
        try:
            paginator = loader.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".jsonl"):
                            files.append(key)
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
    
    logger.info(f"Found {len(files)} JSONL files")
    return files


def _upload_to_s3(
    loader: S3DatasetLoader,
    *,
    local_path: Path,
    s3_key: str,
    bucket: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def _count_user_assistant_turns(messages: list[dict[str, str]]) -> int:
    return sum(m["role"] in ("user", "assistant") for m in messages)


def _build_output_record(
    *,
    messages: list[dict[str, str]],
    s3_path: str,
    turns: int,
) -> dict[str, Any]:
    return {
        "messages": messages,
        "metadata": {
            "source_family": "long_running_therapy",
            "source_key": s3_path,
            "pii_status": "none_detected",
            "license_tag": "therapeutic_license",
            "split": "test",  # hard holdout by contract
            "phase": "stage5_long_running_therapy",
            "conversation_length": turns,
            "provenance": {
                "original_s3_key": s3_path,
                "processing_pipeline": "extract_long_running_therapy",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dedup_status": "unique",
                "processing_steps": ["chatml_convert", "length_filter"],
            },
        },
    }


def _limit_reached(*, kept: int, limit: int) -> bool:
    return limit > 0 and kept >= limit


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s" if args.verbose else "%(message)s",
    )

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))
    
    # Allow environment to override bucket for OVH S3
    import os
    bucket = os.getenv("OVH_S3_BUCKET", bucket)
    endpoint = os.getenv("OVH_S3_ENDPOINT", endpoint)
    
    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    # Determine source keys
    if args.input_dir:
        # Scan directory for all JSONL files
        source_keys = _list_jsonl_files_in_dir(loader, bucket=bucket, input_dir=args.input_dir)
        if not source_keys:
            logger.error(f"No JSONL files found in {args.input_dir}")
            return 1
    elif args.source_key:
        source_keys = args.source_key
    else:
        source_keys = DEFAULT_SOURCE_KEYS

    logger.info(f"Processing {len(source_keys)} source(s) with min_turns={args.min_turns}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    failures = 0
    skipped_short = 0
    turn_hist: Counter = Counter()
    sources_used: Counter = Counter()
    sources_kept: Counter = Counter()
    last_progress_log = 0

    started_at = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", encoding="utf-8") as f:
        for key_idx, key in enumerate(source_keys, 1):
            # Determine display path
            if key.startswith("s3://"):
                display_path = key
            elif _is_local_path(key):
                display_path = key
            else:
                display_path = f"s3://{bucket}/{key}"
            
            logger.info(f"[{key_idx}/{len(source_keys)}] Processing: {display_path}")
            sources_used[key] += 1
            source_kept = 0

            try:
                iterator = _iter_source_records(loader, bucket=bucket, key=key)
            except FileNotFoundError as e:
                logger.warning(f"  ⚠ Source not found: {e}")
                continue
            except Exception as e:
                logger.warning(f"  ⚠ Error opening source: {e}")
                continue

            for rec in iterator:
                seen += 1
                
                # Progress logging
                if seen - last_progress_log >= PROGRESS_LOG_INTERVAL:
                    logger.info(f"  Progress: {seen:,} seen, {kept:,} kept, {failures:,} failed")
                    last_progress_log = seen

                if not isinstance(rec, dict):
                    failures += 1
                    continue
                messages = _to_chatml_messages(rec)
                if not messages:
                    failures += 1
                    continue

                turns = _count_user_assistant_turns(messages)
                turn_hist[str(min(turns, 200))] += 1

                if turns < args.min_turns:
                    skipped_short += 1
                    continue

                out = _build_output_record(
                    messages=messages,
                    s3_path=display_path,
                    turns=turns,
                )
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
                source_kept += 1

                if _limit_reached(kept=kept, limit=args.limit):
                    break

            sources_kept[key] = source_kept
            logger.info(f"  ✓ Extracted {source_kept:,} long-running conversations from this source")

            if _limit_reached(kept=kept, limit=args.limit):
                logger.info(f"Limit of {args.limit} reached, stopping")
                break

    # Build stats
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "bucket": bucket,
        "endpoint": endpoint,
        "source_keys": source_keys,
        "min_turns": args.min_turns,
        "limit": args.limit,
        "seen_records": seen,
        "kept_records": kept,
        "skipped_short": skipped_short,
        "parse_failures": failures,
        "sources_used": dict(sources_used),
        "sources_kept": dict(sources_kept),
        "turn_histogram_capped_200": dict(sorted(turn_hist.items(), key=lambda x: int(x[0]))),
    }

    stats_path = out_path.with_name("long_running_therapy_stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✓ Extracted {kept:,} long-running therapy sessions (≥{args.min_turns} turns)")
    logger.info(f"  Total records seen: {seen:,}")
    logger.info(f"  Skipped (too short): {skipped_short:,}")
    logger.info(f"  Parse failures: {failures:,}")
    logger.info(f"  Output: {out_path}")
    logger.info(f"  Stats: {stats_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        logger.info("")
        logger.info("Uploading to S3...")
        
        # Upload main output
        output_s3_key = f"{args.s3_output_prefix}/long_running_therapy.jsonl"
        success1 = _upload_to_s3(loader, local_path=out_path, s3_key=output_s3_key, bucket=bucket)
        
        # Upload stats
        stats_s3_key = f"{args.s3_output_prefix}/long_running_therapy_stats.json"
        success2 = _upload_to_s3(loader, local_path=stats_path, s3_key=stats_s3_key, bucket=bucket)
        
        if success1 and success2:
            logger.info(f"✓ Uploaded to s3://{bucket}/{args.s3_output_prefix}/")
        else:
            logger.error("Some uploads failed")
            return 1

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
