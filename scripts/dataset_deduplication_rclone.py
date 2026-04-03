#!/usr/bin/env python3
"""
Dataset deduplication using rclone.
Works with DigitalOcean Spaces and identifies duplicate records within and across datasets.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from rclone_dataset_accessor import (
    RcloneDatasetAccessor,
    list_files_in_directory,
    load_jsonl_file,
)


def compute_record_hash(record: Dict[str, Any], key_fields: List[str] = None) -> str:
    """Compute SHA256 hash of a record for deduplication."""
    if key_fields:
        hash_content = json.dumps(
            {k: record.get(k) for k in key_fields if k in record}, sort_keys=True
        )
    else:
        exclude_fields = {"_id", "_hash", "_timestamp", "_source"}
        hash_content = json.dumps(
            {k: v for k, v in record.items() if k not in exclude_fields}, sort_keys=True
        )

    return hashlib.sha256(hash_content.encode("utf-8")).hexdigest()


def find_duplicates_in_dataset(
    records: List[Dict[str, Any]], key_fields: List[str] = None
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Find duplicate records within a dataset."""
    seen_hashes: Dict[str, int] = {}
    duplicate_groups: Dict[str, List[int]] = defaultdict(list)
    deduplicated = []

    for idx, record in enumerate(records):
        record_hash = compute_record_hash(record, key_fields)

        if record_hash in seen_hashes:
            duplicate_groups[record_hash].append(idx)
        else:
            seen_hashes[record_hash] = len(deduplicated)
            deduplicated.append(record)

    return deduplicated, dict(duplicate_groups)


def deduplicate_dataset(
    dataset_name: str, dataset_entry: Dict[str, Any], key_fields: List[str] = None
) -> Dict[str, Any]:
    """Deduplicate a single dataset."""
    s3_path = dataset_entry.get("path", "")
    if not s3_path:
        return {"error": "No path defined"}

    print(f"Deduplicating: {dataset_name}")

    results = {
        "dataset": dataset_name,
        "path": s3_path,
        "original_count": 0,
        "deduplicated_count": 0,
        "duplicates_found": 0,
        "duplicate_groups": 0,
        "deduplication_ratio": 0.0,
    }

    try:
        accessor = RcloneDatasetAccessor(s3_path)
        files = accessor.list_files()

        if not files:
            results["error"] = "No files found"
            return results

        # Load and dedupe each JSONL file
        total_original = 0
        total_deduped = 0
        total_duplicates = 0
        total_groups = 0

        jsonl_files = [f for f in files if f["name"].endswith(".jsonl")]

        for file_info in jsonl_files[:5]:  # Process first 5 files
            records = accessor.load_file(file_info["name"], limit=1000)

            if records:
                deduped, groups = find_duplicates_in_dataset(records, key_fields)

                total_original += len(records)
                total_deduped += len(deduped)
                total_duplicates += len(records) - len(deduped)
                total_groups += len(groups)

        results["original_count"] = total_original
        results["deduplicated_count"] = total_deduped
        results["duplicates_found"] = total_duplicates
        results["duplicate_groups"] = total_groups

        if total_original > 0:
            results["deduplication_ratio"] = round(
                total_duplicates / total_original * 100, 2
            )

    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate datasets using rclone")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of datasets to deduplicate",
    )
    parser.add_argument(
        "--key-fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to use for deduplication",
    )

    args = parser.parse_args()

    key_fields = None
    if args.key_fields:
        key_fields = [f.strip() for f in args.key_fields.split(",")]

    print("=" * 80)
    print("DATASET DEDUPLICATION (RCLONE)")
    print("=" * 80)
    print(f"Registry: {args.registry}")
    print(f"Limit: {args.limit or 'None (all datasets)'}")
    print(f"Key fields: {key_fields or 'All fields'}")
    print()

    with open(args.registry, "r") as f:
        registry = json.load(f)

    # Collect datasets
    datasets = []
    if "datasets" in registry:
        for category_name, category_data in registry["datasets"].items():
            if isinstance(category_data, dict):
                for dataset_name, dataset_entry in category_data.items():
                    if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                        datasets.append(
                            (f"datasets.{category_name}.{dataset_name}", dataset_entry)
                        )

    if args.limit:
        datasets = datasets[: args.limit]

    # Deduplicate each dataset
    all_results = []
    stats = {
        "datasets_checked": 0,
        "total_records": 0,
        "duplicates_found": 0,
        "duplicate_groups": 0,
    }

    for dataset_name, dataset_entry in datasets:
        result = deduplicate_dataset(dataset_name, dataset_entry, key_fields)
        all_results.append(result)

        if "error" not in result:
            stats["datasets_checked"] += 1
            stats["total_records"] += result["original_count"]
            stats["duplicates_found"] += result["duplicates_found"]
            stats["duplicate_groups"] += result["duplicate_groups"]

    # Update registry
    for result in all_results:
        if "error" not in result:
            parts = result["dataset"].split(".")
            if len(parts) == 3:
                cat, subcat, name = parts
                if cat in registry.get("datasets", {}):
                    if subcat in registry["datasets"][cat]:
                        if name in registry["datasets"][cat][subcat]:
                            entry = registry["datasets"][cat][subcat][name]
                            if "quality_metrics" not in entry:
                                entry["quality_metrics"] = {}

                            entry["quality_metrics"]["duplicate_count"] = result[
                                "duplicates_found"
                            ]
                            entry["quality_metrics"]["deduplication_ratio"] = result[
                                "deduplication_ratio"
                            ]

    registry["last_updated"] = datetime.utcnow().isoformat() + "Z"
    with open(args.registry, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("DEDUPLICATION SUMMARY")
    print("=" * 80)
    print(f"Datasets checked: {stats['datasets_checked']}")
    print(f"Total records: {stats['total_records']}")
    print(f"Duplicates found: {stats['duplicates_found']}")
    print(f"Duplicate groups: {stats['duplicate_groups']}")

    # Show per-dataset results
    print("\nPer-Dataset Results:")
    for result in all_results:
        if "error" in result:
            print(f"  {result['dataset']}: ERROR - {result['error']}")
        else:
            print(
                f"  {result['dataset']}: {result['duplicates_found']} duplicates ({result['deduplication_ratio']}%)"
            )


if __name__ == "__main__":
    main()
