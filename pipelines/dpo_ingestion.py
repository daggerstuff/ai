#!/usr/bin/env python3
"""
Stage 5 DPO Dataset Ingestion Pipeline

Downloads and formats preference datasets for DPO training:
- mlx-community/Human-Like-DPO
- flammenai/character-roleplay-DPO
- PJMixers/unalignment_toxic-dpo-v0.2

Output: ai/training_data_consolidated/final/MASTER_STAGE_5.jsonl
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def normalize_schema(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalize different dataset schemas to standard format:
    {prompt: str, chosen: str, rejected: str}
    """
    prompt = None
    chosen = None
    rejected = None

    # Try common field names
    for key in ["prompt", "question", "input", "instruction"]:
        if key in record and record[key]:
            prompt = str(record[key])
            break

    for key in ["chosen", "response_a", "preferred", "accepted", "output"]:
        if key in record and record[key]:
            chosen = str(record[key])
            break

    for key in ["rejected", "response_b", "dispreferred", "rejected_response"]:
        if key in record and record[key]:
            rejected = str(record[key])
            break

    # Handle nested structures
    if not prompt and "messages" in record:
        messages = record["messages"]
        if isinstance(messages, list) and len(messages) >= 2:
            for msg in messages:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                elif msg.get("role") == "assistant" and not chosen:
                    chosen = msg.get("content", "")

    if not chosen and "chosen_response" in record:
        chosen = str(record["chosen_response"])

    if not rejected and "rejected_response" in record:
        rejected = str(record["rejected_response"])

    if not prompt or not chosen or not rejected:
        return None

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def validate_record(record: Dict[str, str]) -> bool:
    """Validate that all fields are non-empty strings."""
    return all(
        isinstance(record.get(key), str) and len(record[key].strip()) > 0 for key in ["prompt", "chosen", "rejected"]
    )


def compute_hash(record: Dict[str, str]) -> str:
    """Compute deterministic hash for deduplication."""
    content = f"{record['prompt']}||{record['chosen']}||{record['rejected']}"
    return hashlib.sha256(content.encode()).hexdigest()


def ingest_dataset(
    dataset_name: str,
    seen_hashes: set,
    output_file: Path,
    stats: Dict[str, int],
) -> None:
    """Ingest a single dataset and append to output file."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 60}")

    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load {dataset_name}: {e}")
        stats["failed_datasets"] += 1
        return

    print(f"Loaded {len(dataset)} records")

    processed = 0
    skipped_invalid = 0
    skipped_duplicate = 0

    with output_file.open("a", encoding="utf-8") as f:
        for record in dataset:
            normalized = normalize_schema(record)

            if not normalized:
                skipped_invalid += 1
                continue

            if not validate_record(normalized):
                skipped_invalid += 1
                continue

            record_hash = compute_hash(normalized)
            if record_hash in seen_hashes:
                skipped_duplicate += 1
                continue

            seen_hashes.add(record_hash)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            processed += 1

            if processed % 1000 == 0:
                print(f"  Processed: {processed:,} records...")

    print(f"✓ Added: {processed:,}")
    print(f"  Skipped (invalid): {skipped_invalid:,}")
    print(f"  Skipped (duplicate): {skipped_duplicate:,}")

    stats["total_records"] += processed
    stats["total_invalid"] += skipped_invalid
    stats["total_duplicate"] += skipped_duplicate


def main():
    """Main ingestion pipeline."""
    print("Stage 5 DPO Dataset Ingestion")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(__file__).parent.parent / "training_data_consolidated" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "MASTER_STAGE_5.jsonl"

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    # Datasets to ingest
    datasets = [
        "mlx-community/Human-Like-DPO",
        "flammenai/character-roleplay-DPO",
        "Dahoas/rm-static",
    ]

    seen_hashes: set = set()
    stats = {
        "total_records": 0,
        "total_invalid": 0,
        "total_duplicate": 0,
        "failed_datasets": 0,
    }

    for dataset_name in datasets:
        ingest_dataset(dataset_name, seen_hashes, output_file, stats)

    # Print summary
    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total records: {stats['total_records']:,}")
    print(f"Invalid records: {stats['total_invalid']:,}")
    print(f"Duplicate records: {stats['total_duplicate']:,}")
    print(f"Failed datasets: {stats['failed_datasets']}")
    print(f"\nOutput: {output_file}")

    if stats["total_records"] == 0:
        print("\nWARNING: No records were ingested!")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
