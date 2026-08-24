#!/usr/bin/env python3
"""
Simplified dataset validation using rclone.
Works with Hetzner Object Storage and handles directories properly.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rclone_dataset_accessor import (
    RcloneDatasetAccessor,
    calculate_checksum,
    list_files_in_directory,
)


def validate_dataset(dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a single dataset.

    Args:
        dataset_name: Name of dataset
        dataset_entry: Dataset entry from registry

    Returns:
        Validation results
    """
    s3_path = dataset_entry.get("path", "")
    if not s3_path:
        return {"error": "No path defined"}

    results = {
        "dataset": dataset_name,
        "path": s3_path,
        "files_found": 0,
        "total_size_bytes": 0,
        "checksums": {},
        "valid_files": 0,
        "invalid_files": 0,
        "errors": [],
    }

    try:
        # List files in directory
        files = list_files_in_directory(s3_path)

        if not files:
            results["errors"].append("No files found in directory")
            return results

        results["files_found"] = len(files)
        results["total_size_bytes"] = sum(f["size_bytes"] for f in files)

        # Validate first few files
        sample_files = files[:3]  # Check first 3 files

        for file_info in sample_files:
            file_path = f"{s3_path}/{file_info['name']}"

            # Calculate checksum
            checksum = calculate_checksum(file_path, "sha256")

            if checksum:
                results["checksums"][file_info["name"]] = {
                    "sha256": checksum,
                    "valid": True,
                }
                results["valid_files"] += 1
            else:
                results["checksums"][file_info["name"]] = {
                    "sha256": None,
                    "valid": False,
                }
                results["invalid_files"] += 1
                results["errors"].append(f"Failed to validate {file_info['name']}")

        # Sample file content validation
        jsonl_files = [f for f in files if f["name"].endswith(".jsonl")]
        if jsonl_files:
            accessor = RcloneDatasetAccessor(s3_path)
            sample_records = accessor.load_file(jsonl_files[0]["name"], limit=10)

            results["sample_validation"] = {
                "file": jsonl_files[0]["name"],
                "records_loaded": len(sample_records),
                "valid_json": len(sample_records) > 0,
            }

    except Exception as e:
        results["errors"].append(str(e))

    return results


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Validate datasets using rclone")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/configs/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to validate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be validated without doing it",
    )

    args = parser.parse_args()

    # Load registry
    with open(args.registry) as f:
        registry = json.load(f)

    # Collect all datasets
    datasets = []
    if "datasets" in registry:
        for category_name, category_data in registry["datasets"].items():
            if isinstance(category_data, dict):
                for dataset_name, dataset_entry in category_data.items():
                    if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                        datasets.append((f"datasets.{category_name}.{dataset_name}", dataset_entry))

    if args.limit:
        datasets = datasets[: args.limit]

    if args.dry_run:
        for name, entry in datasets:
            pass
        return

    # Validate each dataset
    all_results = []
    stats = {
        "total": len(datasets),
        "successful": 0,
        "failed": 0,
        "total_files": 0,
        "total_size_bytes": 0,
    }

    for dataset_name, dataset_entry in datasets:
        result = validate_dataset(dataset_name, dataset_entry)
        all_results.append(result)

        if result.get("files_found", 0) > 0:
            stats["successful"] += 1
            stats["total_files"] += result["files_found"]
            stats["total_size_bytes"] += result["total_size_bytes"]
        else:
            stats["failed"] += 1

    # Update registry with validation results
    for result in all_results:
        if "error" not in result:
            dataset_name_parts = result["dataset"].split(".")
            if len(dataset_name_parts) == 3:
                cat, subcat, name = dataset_name_parts
                if cat in registry.get("datasets", {}) and subcat in registry["datasets"][cat]:
                    if name in registry["datasets"][cat][subcat]:
                        entry = registry["datasets"][cat][subcat][name]
                        if "validation" not in entry:
                            entry["validation"] = {}

                        entry["validation"]["last_validated"] = datetime.now(UTC).isoformat() + "Z"
                        entry["validation"]["integrity_valid"] = result["valid_files"] > 0
                        entry["validation"]["schema_valid"] = result.get("sample_validation", {}).get(
                            "valid_json", False
                        )

                        if result["checksums"]:
                            entry["validation"]["checksum_sha256"] = next(iter(result["checksums"].values())).get("sha256")

    # Save updated registry
    registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
    with open(args.registry, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Print summary

    # Save detailed report
    report_path = Path("/home/vivi/pixelated/ai/configs/validation_report.json")
    report = {
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "statistics": stats,
        "results": all_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
