#!/usr/bin/env python3
"""
Analyze Dataset Overlaps - Find duplicate/overlapping content across datasets
Samples random entries from different datasets and compares them for duplicates
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    script_path = Path(__file__).resolve()
    # Go up 3 levels: scripts -> training_ready -> ai -> project_root
    return script_path.parents[3]


def ensure_project_root_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


DEFAULT_S3_BUCKET = "pixel-data"
DEFAULT_SAMPLE_SIZE_PER_FILE = 50
DEFAULT_MAX_FILES_PER_CATEGORY = 5


class ProcessedDatasetFile(TypedDict):
    key: str
    category: str
    size: int


Conversation = dict[str, Any]
Sample = tuple[Conversation, str]
HashMatch = tuple[str, Conversation, str]


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)"""
    return text.lower().strip()


def hash_conversation(conv: Conversation) -> str:
    """Create a hash of a conversation for duplicate detection"""
    # Extract text content - handle different conversation formats
    text_parts = []

    if isinstance(conv, dict):
        # Try common conversation formats
        if "text" in conv:
            text_parts.append(str(conv["text"]))
        elif "conversation" in conv:
            if isinstance(conv["conversation"], list):
                for turn in conv["conversation"]:
                    if isinstance(turn, dict):
                        text_parts.append(str(turn.get("content", turn.get("text", ""))))
                    else:
                        text_parts.append(str(turn))
            else:
                text_parts.append(str(conv["conversation"]))
        elif "messages" in conv:
            text_parts.extend(
                [
                    (
                        str(msg.get("content", msg.get("text", "")))
                        if isinstance(msg, dict)
                        else str(msg)
                    )
                    for msg in conv["messages"]
                ]
            )
        elif "input" in conv and "output" in conv:
            text_parts.extend([str(conv.get("input", "")), str(conv.get("output", ""))])
        else:
            # Fallback: use entire dict as string
            text_parts.append(json.dumps(conv, sort_keys=True))
    else:
        text_parts.append(str(conv))

    # Normalize and hash
    normalized = normalize_text(" ".join(text_parts))
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_text_sample(conv: Conversation, max_length: int = 200) -> str:
    """Extract a text sample from conversation for display"""
    if isinstance(conv, dict):
        if "text" in conv:
            text = str(conv["text"])
        elif "conversation" in conv:
            if isinstance(conv["conversation"], list):
                text = " ".join(
                    str(turn.get("content", turn.get("text", "")))
                    for turn in conv["conversation"]
                    if isinstance(turn, dict)
                )
            else:
                text = str(conv["conversation"])
        elif "input" in conv:
            text = f"{conv.get('input', '')!s} {conv.get('output', '')!s}"
        else:
            text = json.dumps(conv)[:max_length]
    else:
        text = str(conv)[:max_length]

    return f"{text[:max_length]}..." if len(text) > max_length else text


def _normalize_json_entries(data: Any) -> list[dict[str, Any]]:
    """Normalize various JSON structures into a list of dict-like entries."""
    if isinstance(data, dict):
        if "conversations" in data:
            entries = data["conversations"]
        elif "data" in data:
            entries = data["data"]
        else:
            # Single conversation dict
            entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            normalized.append({"value": entry})
    return normalized


def _sample_json_dataset(
    *,
    loader: Any,
    s3_path: str,
    sample_size: int,
) -> list[Sample]:
    data = loader.load_json(s3_path)
    entries = _normalize_json_entries(data)
    if not entries:
        logger.warning("    ⚠️  Unexpected JSON structure, skipping")
        return []

    sampled = random.sample(entries, sample_size) if len(entries) > sample_size else entries
    return [(entry, s3_path) for entry in sampled]


def _sample_jsonl_dataset(
    *,
    loader: Any,
    s3_path: str,
    sample_size: int,
) -> list[Sample]:
    entries: list[Sample] = []
    count = 0

    try:
        for line in loader.stream_jsonl(s3_path):
            if not line:
                continue
            if isinstance(line, dict):
                entries.append((line, s3_path))
            else:
                entries.append(({"value": line}, s3_path))
            count += 1
            # Sample from the first 2x to get randomness
            if count >= sample_size * 2:
                break
    except Exception:
        logger.exception("    ⚠️  Streaming error")

    return random.sample(entries, sample_size) if len(entries) > sample_size else entries


def sample_dataset(loader: Any, s3_path: str, sample_size: int = 100) -> list[Sample]:
    """Sample random entries from a dataset."""
    logger.info("  📊 Sampling %s entries from %s...", sample_size, s3_path)

    try:
        if s3_path.endswith(".json"):
            return _sample_json_dataset(
                loader=loader,
                s3_path=s3_path,
                sample_size=sample_size,
            )
        if s3_path.endswith(".jsonl"):
            return _sample_jsonl_dataset(
                loader=loader,
                s3_path=s3_path,
                sample_size=sample_size,
            )

        logger.warning("    ⚠️  Unknown file format, skipping")
        return []
    except Exception:
        logger.exception("    ❌ Error loading %s", s3_path)
        return []


def find_overlaps(datasets: dict[str, list[Sample]]) -> dict[str, Any]:
    """Find overlaps between datasets"""
    logger.info("🔍 Analyzing for duplicates...")

    @dataclass
    class OverlapSummary:
        count: int = 0
        with_datasets: set[str] = field(default_factory=set)

    # Build hash index: hash -> list of (dataset_name, entry, source_path)
    hash_index: defaultdict[str, list[HashMatch]] = defaultdict(list)

    for dataset_name, entries in datasets.items():
        for entry, source_path in entries:
            conv_hash = hash_conversation(entry)
            hash_index[conv_hash].append((dataset_name, entry, source_path))

    # Find duplicates (hashes with multiple entries)
    duplicates: dict[tuple[str, ...], list[HashMatch]] = {}
    overlap_summary: defaultdict[str, OverlapSummary] = defaultdict(OverlapSummary)

    for matches in hash_index.values():
        if len(matches) > 1:
            # This hash appears in multiple datasets
            datasets_involved = [m[0] for m in matches]
            unique_datasets = set(datasets_involved)

            if len(unique_datasets) > 1:
                # Cross-dataset duplicate
                key = tuple(sorted(unique_datasets))
                duplicates[key] = duplicates.get(key, []) + matches

                for dataset in unique_datasets:
                    overlap_summary[dataset].count += 1
                    overlap_summary[dataset].with_datasets.update(unique_datasets - {dataset})

    return {
        "duplicates": duplicates,
        "overlap_summary": {
            k: {
                "count": v.count,
                "with_datasets": list(v.with_datasets),
            }
            for k, v in overlap_summary.items()
        },
        "total_unique_hashes": len(hash_index),
        "total_duplicate_hashes": len(duplicates),
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    logger.info("📋 Loading manifest from %s...", manifest_path)
    with open(manifest_path) as f:
        return json.load(f)


def extract_processed_dataset_files(
    manifest: dict[str, Any],
) -> list[ProcessedDatasetFile]:
    """Extract processed dataset objects (JSON/JSONL only) from manifest."""
    processed_files: list[ProcessedDatasetFile] = []
    processed_cats = manifest.get("categories", {}).get("gdrive", {}).get("processed", {})

    logger.info("📂 Finding processed datasets...")
    for category, files_info in processed_cats.items():
        if not (isinstance(files_info, dict) and "objects" in files_info):
            continue
        for obj in files_info["objects"]:
            key = obj["key"]
            key_lower = key.lower()
            is_data_file = key.endswith((".json", ".jsonl"))
            is_report_or_metadata = "report" in key_lower or "metadata" in key_lower
            is_large_enough = obj["size"] > 1000  # At least 1KB

            if is_data_file and (not is_report_or_metadata) and is_large_enough:
                processed_files.append(
                    {
                        "key": key,
                        "category": category,
                        "size": obj["size"],
                    }
                )

    return processed_files


def group_files_by_category(
    processed_files: Sequence[ProcessedDatasetFile],
) -> dict[str, list[ProcessedDatasetFile]]:
    files_by_category: defaultdict[str, list[ProcessedDatasetFile]] = defaultdict(list)
    for file_info in processed_files:
        files_by_category[file_info["category"]].append(file_info)
    return files_by_category


def collect_samples_by_category(
    loader: Any,
    files_by_category: dict[str, list[ProcessedDatasetFile]],
    sample_size_per_file: int,
    max_files_per_category: int,
) -> dict[str, list[Sample]]:
    logger.info("🎲 Sampling datasets...")
    all_samples: dict[str, list[Sample]] = {}

    for category, files in files_by_category.items():
        # Sort by size (largest first) and take top N
        files_sorted = sorted(
            files,
            key=lambda x: x["size"],
            reverse=True,
        )[:max_files_per_category]

        logger.info("  Category: %s (%s files)", category, len(files_sorted))
        category_samples: list[Sample] = []

        for file_info in files_sorted:
            s3_path = f"s3://{loader.bucket}/{file_info['key']}"
            samples = sample_dataset(loader, s3_path, sample_size_per_file)
            category_samples.extend(samples)

            if samples:
                logger.info(
                    "    ✅ %s samples from %s",
                    len(samples),
                    Path(file_info["key"]).name,
                )

        if category_samples:
            all_samples[category] = category_samples
            logger.info(
                "  📊 Total samples from %s: %s",
                category,
                len(category_samples),
            )

    return all_samples


def print_overlap_results(overlap_results: dict[str, Any]) -> None:
    logger.info("%s", "=" * 80)
    logger.info("📊 OVERLAP ANALYSIS RESULTS")
    logger.info("%s", "=" * 80)

    logger.info("📈 Summary:")
    logger.info(
        "  Total unique conversation hashes: %s",
        overlap_results["total_unique_hashes"],
    )
    logger.info(
        "  Duplicate hashes (cross-dataset): %s",
        overlap_results["total_duplicate_hashes"],
    )

    if overlap_results["overlap_summary"]:
        logger.info("🔄 Dataset Overlaps:")
        for dataset, info in sorted(
            overlap_results["overlap_summary"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            logger.info("  %s:", dataset)
            logger.info("    - %s duplicate entries", info["count"])
            logger.info("    - Overlaps with: %s", ", ".join(info["with_datasets"]))

    # Show example duplicates
    if overlap_results["duplicates"]:
        logger.info("🔍 Example Duplicates (showing first 5):")
        for datasets_tuple, matches in list(overlap_results["duplicates"].items())[:5]:
            logger.info("  Duplicate across: %s", ", ".join(datasets_tuple))
            logger.info("    Found %s instances", len(matches))

            # Show sample text from first match
            if matches:
                sample_entry = matches[0][1]  # Get the entry dict
                sample_text = extract_text_sample(sample_entry, max_length=150)
                logger.info("    Sample text: %s", sample_text)


def build_overlap_report(
    overlap_results: dict[str, Any],
    all_samples: dict[str, list[Sample]],
) -> dict[str, Any]:
    return {
        "analysis_date": datetime.now(tz=timezone.utc).isoformat(),
        "summary": {
            "total_unique_hashes": overlap_results["total_unique_hashes"],
            "total_duplicate_hashes": overlap_results["total_duplicate_hashes"],
            "datasets_analyzed": list(all_samples.keys()),
            "total_samples": sum(len(samples) for samples in all_samples.values()),
        },
        "overlap_summary": overlap_results["overlap_summary"],
        "duplicate_groups": {
            str(k): {
                "datasets": list(k),
                "count": len(v),
                "sample_entry": (extract_text_sample(v[0][1], max_length=200) if v else None),
            }
            for k, v in list(overlap_results["duplicates"].items())[:20]  # Limit to first 20
        },
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """Main analysis function"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = get_project_root()
    ensure_project_root_on_path(project_root)

    # Import after adding project root to sys.path
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader  # noqa: PLC0415

    # Suppress verbose warnings from S3DatasetLoader
    logging.getLogger("ai.training_ready.utils.s3_dataset_loader").setLevel(logging.ERROR)

    logger.info("🔍 Dataset Overlap Analysis")
    logger.info("%s", "=" * 80)

    # Initialize S3 loader (use pixel-data bucket)
    loader = S3DatasetLoader(bucket=DEFAULT_S3_BUCKET)

    # Load manifest to find processed datasets
    manifest_path = project_root / "ai/training_ready/data/s3_manifest.json"
    if not manifest_path.exists():
        logger.error("❌ Manifest not found at %s", manifest_path)
        return

    manifest = load_manifest(manifest_path)

    # Extract processed dataset files (JSON/JSONL only)
    processed_files = extract_processed_dataset_files(manifest)

    logger.info("  Found %s dataset files to analyze", len(processed_files))

    # Group by category for sampling
    files_by_category = group_files_by_category(processed_files)

    logger.info("📊 Categories found: %s", list(files_by_category.keys()))

    # Sample from each category (prioritize larger files)
    all_samples = collect_samples_by_category(
        loader,
        files_by_category,
        sample_size_per_file=DEFAULT_SAMPLE_SIZE_PER_FILE,
        max_files_per_category=DEFAULT_MAX_FILES_PER_CATEGORY,
    )

    if not all_samples:
        logger.error("❌ No samples collected. Check S3 access and file formats.")
        return

    # Find overlaps
    overlap_results = find_overlaps(all_samples)

    # Print results
    print_overlap_results(overlap_results)

    # Save detailed report
    report_path = project_root / "ai/training_ready/data/dataset_overlap_analysis.json"
    report = build_overlap_report(overlap_results, all_samples)
    save_json(report_path, report)

    logger.info("💾 Detailed report saved to: %s", report_path)
    logger.info("✅ Analysis complete!")


if __name__ == "__main__":
    main()
