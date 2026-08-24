#!/usr/bin/env python3
"""
Dataset sync verification using rclone.
Verifies consistency between S3/DO Spaces and local/backup paths.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import subprocess

from rclone_dataset_accessor import s3_path_to_rclone

_EXPECTED_PARTS = 2
_REGISTRY_PARTS = 3
_MAX_SIZE_DIFF_BYTES = 1024


def run_rclone_ls(remote_path: str) -> list[dict[str, Any]] | None:
    """Run rclone ls and parse output."""
    try:
        result = subprocess.run(["rclone", "ls", remote_path], shell=False, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            return None

        files = []
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == _EXPECTED_PARTS:
                    size, filename = parts
                    files.append({"name": filename, "size_bytes": int(size)})

        return files
    except Exception:
        return None


def _check_s3_status(s3_path: str, results: dict[str, Any]) -> None:
    if s3_path:
        s3_files = run_rclone_ls(s3_path_to_rclone(s3_path))
        if s3_files is not None:
            results["s3_exists"] = True
            results["s3_files"] = len(s3_files)
            results["s3_size_bytes"] = sum(f["size_bytes"] for f in s3_files)
        else:
            results["issues"].append("S3 path not accessible")


def _check_fallback_status(fallback_path: str, s3_path: str, results: dict[str, Any]) -> None:
    if fallback_path and fallback_path != s3_path:
        if fallback_path.startswith("s3://"):
            fallback_files = run_rclone_ls(s3_path_to_rclone(fallback_path))
            if fallback_files is not None:
                results["fallback_exists"] = True
                results["fallback_files"] = len(fallback_files)
                results["fallback_size_bytes"] = sum(f["size_bytes"] for f in fallback_files)
        else:
            local_path = Path(fallback_path)
            if local_path.exists():
                results["fallback_exists"] = True
                if local_path.is_dir():
                    files = list(local_path.rglob("*"))
                    files = [f for f in files if f.is_file()]
                    results["fallback_files"] = len(files)
                    results["fallback_size_bytes"] = sum(f.stat().st_size for f in files)


def verify_dataset_sync(dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Verify sync status for a dataset.

    Checks:
    1. If path exists in S3/DO Spaces
    2. If fallback_path exists
    3. File count and size consistency
    """
    s3_path = dataset_entry.get("path", "")
    fallback_path = dataset_entry.get("fallback_path", "")

    results = {
        "dataset": dataset_name,
        "s3_path": s3_path,
        "fallback_path": fallback_path,
        "s3_exists": False,
        "fallback_exists": False,
        "s3_files": 0,
        "fallback_files": 0,
        "s3_size_bytes": 0,
        "fallback_size_bytes": 0,
        "in_sync": False,
        "issues": [],
    }

    _check_s3_status(s3_path, results)
    _check_fallback_status(fallback_path, s3_path, results)

    # Determine sync status
    if results["s3_exists"]:
        if not fallback_path or results["fallback_exists"]:
            if not fallback_path or (
                results["s3_files"] == results["fallback_files"]
                and abs(results["s3_size_bytes"] - results["fallback_size_bytes"]) < _MAX_SIZE_DIFF_BYTES
            ):
                results["in_sync"] = True
            else:
                results["issues"].append(
                    f"File count or size mismatch: S3({results['s3_files']}) vs Fallback({results['fallback_files']})"
                )
    else:
        results["issues"].append("Primary S3 path not accessible")

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Verify dataset sync using rclone")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/configs/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to verify")
    return parser.parse_args()


def _collect_datasets(registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    datasets = []
    if "datasets" in registry:
        for category_name, category_data in registry["datasets"].items():
            if isinstance(category_data, dict):
                for dataset_name, dataset_entry in category_data.items():
                    if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                        datasets.append((f"datasets.{category_name}.{dataset_name}", dataset_entry))
    return datasets


def _verify_datasets_loop(
    datasets: list[tuple[str, dict[str, Any]]], all_results: list[dict[str, Any]], stats: dict[str, int]
) -> None:
    for dataset_name, dataset_entry in datasets:
        result = verify_dataset_sync(dataset_name, dataset_entry)
        all_results.append(result)

        if result["in_sync"]:
            stats["in_sync"] += 1
        else:
            stats["out_of_sync"] += 1

        if not result["s3_exists"]:
            stats["missing_s3"] += 1
        if result.get("fallback_path") and not result["fallback_exists"]:
            stats["missing_fallback"] += 1


def _update_registry(registry: dict[str, Any], all_results: list[dict[str, Any]]) -> None:
    for result in all_results:
        parts = result["dataset"].split(".")
        if len(parts) == _REGISTRY_PARTS:
            cat, subcat, name = parts
            if (
                cat in registry.get("datasets", {})
                and subcat in registry["datasets"][cat]
                and name in registry["datasets"][cat][subcat]
            ):
                entry = registry["datasets"][cat][subcat][name]
                if "sync_status" not in entry:
                    entry["sync_status"] = {}

                entry["sync_status"]["last_verified"] = datetime.now(UTC).isoformat() + "Z"
                entry["sync_status"]["in_sync"] = result["in_sync"]
                entry["sync_status"]["s3_file_count"] = result["s3_files"]
                entry["sync_status"]["s3_size_bytes"] = result["s3_size_bytes"]


def main():
    args = _parse_args()

    with open(args.registry) as f:
        registry = json.load(f)

    datasets = _collect_datasets(registry)
    if args.limit:
        datasets = datasets[: args.limit]

    all_results = []
    stats = {
        "total": len(datasets),
        "in_sync": 0,
        "out_of_sync": 0,
        "missing_s3": 0,
        "missing_fallback": 0,
    }

    _verify_datasets_loop(datasets, all_results, stats)
    _update_registry(registry, all_results)

    registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
    with open(args.registry, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
