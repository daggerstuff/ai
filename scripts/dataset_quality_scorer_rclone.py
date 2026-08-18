#!/usr/bin/env python3
"""
Dataset quality scorer using rclone.
Scores datasets based on completeness, consistency, and annotation quality.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from rclone_dataset_accessor import (
    RcloneDatasetAccessor,
    list_files_in_directory,
)


def score_completeness(files: list[dict[str, Any]], sample_records: list[dict]) -> float:
    """Score based on file count and record completeness."""
    if not files:
        return 0.0

    score = 50.0  # Base score for having files

    # Bonus for file count
    if len(files) >= 50:
        score += 20
    elif len(files) >= 20:
        score += 15
    elif len(files) >= 10:
        score += 10

    # Bonus for size
    total_size = sum(f["size_bytes"] for f in files)
    if total_size >= 1_000_000_000:  # 1 GB
        score += 15
    elif total_size >= 100_000_000:  # 100 MB
        score += 10
    elif total_size >= 10_000_000:  # 10 MB
        score += 5

    # Check sample records for completeness
    if sample_records:
        required_fields = ["instruction", "input", "output"]
        records_with_all_fields = sum(1 for r in sample_records if all(field in r for field in required_fields))
        completeness_ratio = records_with_all_fields / len(sample_records)
        score += completeness_ratio * 15

    return min(score, 100.0)


def score_consistency(sample_records: list[dict]) -> float:
    """Score based on data consistency."""
    if not sample_records:
        return 0.0

    score = 70.0  # Base score

    # Check for consistent field names
    all_fields = set()
    for record in sample_records:
        all_fields.update(record.keys())

    if len(all_fields) <= 10:  # Reasonable number of fields
        score += 10

    # Check for consistent data types
    field_types = {}
    for record in sample_records:
        for key, value in record.items():
            if key not in field_types:
                field_types[key] = type(value).__name__
            elif field_types[key] != type(value).__name__:
                score -= 2  # Inconsistent types

    return max(score, 0.0)


def score_annotation_quality(sample_records: list[dict]) -> float:
    """Score based on annotation quality."""
    if not sample_records:
        return 0.0

    score = 60.0  # Base score

    # Check for instruction quality
    for record in sample_records:
        instruction = record.get("instruction", "")
        output = record.get("output", "")

        # Length checks
        if len(instruction) > 20:
            score += 1
        if len(output) > 50:
            score += 1

        # Check for placeholder text
        if "TODO" in output or "placeholder" in output.lower():
            score -= 5

        # Check for empty fields
        if not instruction.strip() or not output.strip():
            score -= 10

    return min(max(score, 0.0), 100.0)


def determine_quality_tier(score: float) -> str:
    """Determine quality tier from score."""
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "acceptable"
    return "needs_review"


def score_dataset_quality(dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Score quality for a single dataset.

    Returns dict with completeness, consistency, annotation_quality, and overall scores.
    """
    s3_path = dataset_entry.get("path", "")
    if not s3_path:
        return {"error": "No path defined", "overall_score": 0}

    results = {
        "dataset": dataset_name,
        "path": s3_path,
        "completeness_score": 0.0,
        "consistency_score": 0.0,
        "annotation_quality_score": 0.0,
        "overall_score": 0.0,
        "quality_tier": "needs_review",
        "anomaly_detected": False,
    }

    try:
        # List files
        files = list_files_in_directory(s3_path)
        if not files:
            results["error"] = "No files found"
            return results

        # Load sample records from first JSONL file
        sample_records = []
        jsonl_files = [f for f in files if f["name"].endswith(".jsonl")]

        if jsonl_files:
            accessor = RcloneDatasetAccessor(s3_path)
            sample_records = accessor.load_file(jsonl_files[0]["name"], limit=100)

        # Calculate scores
        results["completeness_score"] = score_completeness(files, sample_records)
        results["consistency_score"] = score_consistency(sample_records)
        results["annotation_quality_score"] = score_annotation_quality(sample_records)

        # Weighted overall score
        results["overall_score"] = (
            results["completeness_score"] * 0.4
            + results["consistency_score"] * 0.3
            + results["annotation_quality_score"] * 0.3
        )

        # Determine tier
        results["quality_tier"] = determine_quality_tier(results["overall_score"])

        # Detect anomalies
        if results["overall_score"] < 60:
            results["anomaly_detected"] = True

    except Exception as e:
        results["error"] = str(e)
        results["anomaly_detected"] = True

    return results


def main():

    parser = argparse.ArgumentParser(description="Score dataset quality using rclone")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to score")

    args = parser.parse_args()

    with open(args.registry) as f:
        registry = json.load(f)

    # Collect datasets
    datasets = []
    if "datasets" in registry:
        for category_name, category_data in registry["datasets"].items():
            if isinstance(category_data, dict):
                for dataset_name, dataset_entry in category_data.items():
                    if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                        datasets.append((f"datasets.{category_name}.{dataset_name}", dataset_entry))

    if args.limit:
        datasets = datasets[: args.limit]

    # Score each dataset
    all_results = []
    stats = {
        "total": len(datasets),
        "anomalies": 0,
        "by_tier": {"excellent": 0, "good": 0, "acceptable": 0, "needs_review": 0},
    }

    for dataset_name, dataset_entry in datasets:
        result = score_dataset_quality(dataset_name, dataset_entry)
        all_results.append(result)

        stats["by_tier"][result["quality_tier"]] += 1
        if result.get("anomaly_detected"):
            stats["anomalies"] += 1

    # Update registry with quality scores
    for result in all_results:
        if "error" not in result:
            parts = result["dataset"].split(".")
            if len(parts) == 3:
                cat, subcat, name = parts
                if cat in registry.get("datasets", {}) and subcat in registry["datasets"][cat]:
                    if name in registry["datasets"][cat][subcat]:
                        entry = registry["datasets"][cat][subcat][name]
                        if "quality_metrics" not in entry:
                            entry["quality_metrics"] = {}

                        entry["quality_metrics"]["completeness_score"] = round(result["completeness_score"], 2)
                        entry["quality_metrics"]["consistency_score"] = round(result["consistency_score"], 2)
                        entry["quality_metrics"]["annotation_quality_score"] = round(
                            result["annotation_quality_score"], 2
                        )
                        entry["quality_metrics"]["overall_score"] = round(result["overall_score"], 2)
                        entry["quality_metrics"]["quality_score"] = round(result["overall_score"], 2)
                        entry["quality_metrics"]["quality_tier"] = result["quality_tier"]
                        entry["quality_metrics"]["anomaly_detected"] = result["anomaly_detected"]
                        entry["quality_metrics"]["data_freshness_days"] = None
                        entry["quality_metrics"]["anomaly_flags"] = []
                        if result["anomaly_detected"]:
                            entry["quality_metrics"]["anomaly_flags"].append("low_quality_score")
                        entry["quality_metrics"]["last_scored"] = datetime.now(UTC).isoformat() + "Z"

    registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
    with open(args.registry, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Print summary
    for _tier, _count in stats["by_tier"].items():
        pass

    # Show datasets needing review
    needs_review = [r for r in all_results if r["quality_tier"] == "needs_review"]
    if needs_review:
        for result in needs_review:
            pass


if __name__ == "__main__":
    main()
