#!/usr/bin/env python3
"""
Full Deduplication Scan - Complete analysis of all datasets for duplicates
Scans entire datasets (not just samples) to find all duplicate entries
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import defaultdict
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

Conversation = dict[str, Any]
HashedEntry = tuple[str, Conversation, str, int, str]
Match = tuple[str, str, int, Conversation]


class ProcessedDatasetFile(TypedDict):
    key: str
    category: str
    size: int


class DuplicateEntry(TypedDict):
    dataset: str
    source_path: str
    entry_index: int
    entry_preview: str


class DuplicateGroup(TypedDict):
    hash: str
    count: int
    datasets: set[str]
    entries: list[DuplicateEntry]


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)"""
    return text.lower().strip()


def hash_conversation(conv: Conversation) -> str:
    """Create a hash of a conversation for duplicate detection"""
    text_parts: list[str] = []

    if isinstance(conv, dict):
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
                str(msg.get("content", msg.get("text", "")))
                for msg in conv["messages"]
                if isinstance(msg, dict)
            )
        elif "input" in conv and "output" in conv:
            text_parts.extend((str(conv.get("input", "")), str(conv.get("output", ""))))
        else:
            text_parts.append(json.dumps(conv, sort_keys=True))
    else:
        text_parts.append(str(conv))

    normalized = normalize_text(" ".join(text_parts))
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_text_preview(conv: Conversation, max_length: int = 200) -> str:
    """Extract a text preview from conversation for display"""
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


def _normalize_entries(data: Any) -> list[Conversation]:
    if isinstance(data, dict):
        if "conversations" in data:
            entries = data["conversations"]
        elif "data" in data:
            entries = data["data"]
        else:
            entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    normalized: list[Conversation] = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            normalized.append({"value": entry})
    return normalized


def load_all_entries_from_json(loader: object, s3_path: str) -> list[Conversation]:
    """Load all entries from a JSON file"""
    try:
        data = loader.load_json(s3_path)  # type: ignore[attr-defined]
        return _normalize_entries(data)
    except Exception:
        logger.exception("    ❌ Error loading JSON %s", s3_path)
        return []


def load_all_entries_from_jsonl(loader: object, s3_path: str) -> list[Conversation]:
    """Load all entries from a JSONL file (streaming)"""
    entries: list[Conversation] = []

    try:
        for line in loader.stream_jsonl(s3_path):  # type: ignore[attr-defined]
            if not line:
                continue
            if isinstance(line, dict):
                entries.append(line)
            else:
                entries.append({"value": line})
    except Exception:
        logger.exception("    ⚠️  Streaming error for %s", s3_path)

    return entries


def scan_dataset_file(
    loader: object,
    file_info: ProcessedDatasetFile,
    dataset_name: str,
) -> list[tuple[str, Conversation, str, int]]:
    """
    Scan a single dataset file and return list of (hash, entry, source_path, index)

    Returns:
        List of tuples: (hash, entry_dict, source_path, entry_index)
    """
    bucket = loader.bucket  # type: ignore[attr-defined]
    s3_path = f"s3://{bucket}/{file_info['key']}"
    entries: list[Conversation] = []

    logger.info(
        "  📊 Scanning %s (%s)...",
        Path(file_info["key"]).name,
        dataset_name,
    )

    if file_info["key"].endswith(".json"):
        entries = load_all_entries_from_json(loader, s3_path)
    elif file_info["key"].endswith(".jsonl"):
        entries = load_all_entries_from_jsonl(loader, s3_path)
    else:
        logger.warning("   ⚠️  Unknown format: %s", file_info["key"])
        return []

    # Hash all entries
    hashed_entries: list[tuple[str, Conversation, str, int]] = []
    for idx, entry in enumerate(entries):
        conv_hash = hash_conversation(entry)
        hashed_entries.append((conv_hash, entry, s3_path, idx))

    logger.info("   ✅ %s entries", len(hashed_entries))
    return hashed_entries


def group_files_by_category(
    processed_files: list[ProcessedDatasetFile],
) -> dict[str, list[ProcessedDatasetFile]]:
    files_by_category: defaultdict[str, list[ProcessedDatasetFile]] = defaultdict(list)
    for file_info in processed_files:
        files_by_category[file_info["category"]].append(file_info)
    return files_by_category


def scan_all_datasets(
    loader: object,
    files_by_category: dict[str, list[ProcessedDatasetFile]],
) -> tuple[list[HashedEntry], int]:
    logger.info("🔍 Scanning all datasets...")
    all_hashed_entries: list[HashedEntry] = []
    total_entries = 0

    for category, files in sorted(files_by_category.items()):
        logger.info("📁 Category: %s (%s files)", category, len(files))

        for file_info in sorted(files, key=lambda x: x["size"], reverse=True):
            hashed_entries = scan_dataset_file(loader, file_info, category)

            for hash_val, entry, source_path, index in hashed_entries:
                all_hashed_entries.append((hash_val, entry, source_path, index, category))
                total_entries += 1

    return all_hashed_entries, total_entries


def build_duplicate_index(
    all_hashed_entries: list[HashedEntry],
) -> dict[str, DuplicateGroup]:
    """
    Build index of duplicates from all hashed entries.

    Args:
        all_hashed_entries: List of (hash, entry, source_path, index, dataset_name)

    Returns:
        Dict mapping hash to DuplicateGroup (only for hashes with duplicates)
    """
    # Index: hash -> list of (dataset, source_path, index, entry)
    hash_index: defaultdict[str, list[Match]] = defaultdict(list)

    for conv_hash, entry, source_path, index, dataset_name in all_hashed_entries:
        hash_index[conv_hash].append((dataset_name, source_path, index, entry))

    # Build duplicate groups (only hashes that appear multiple times)
    duplicate_groups: dict[str, DuplicateGroup] = {}

    for conv_hash, matches in hash_index.items():
        if len(matches) > 1:
            # Extract unique datasets
            datasets = {m[0] for m in matches}

            # Build entry list
            duplicate_entries: list[DuplicateEntry] = [
                {
                    "dataset": dataset,
                    "source_path": source_path,
                    "entry_index": index,
                    "entry_preview": extract_text_preview(entry, max_length=150),
                }
                for dataset, source_path, index, entry in matches
            ]

            duplicate_groups[conv_hash] = {
                "hash": conv_hash,
                "count": len(matches),
                "datasets": datasets,
                "entries": duplicate_entries,
            }

    return duplicate_groups


def analyze_duplicates(duplicate_groups: dict[str, DuplicateGroup]) -> dict[str, Any]:
    """Analyze duplicate groups and generate statistics"""
    # Count duplicates per dataset pair
    dataset_pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    dataset_total_counts: defaultdict[str, int] = defaultdict(int)

    for group in duplicate_groups.values():
        datasets_list = sorted(group["datasets"])

        # Count for each pair
        for i, dataset1 in enumerate(datasets_list):
            dataset_total_counts[dataset1] += group["count"]
            for dataset2 in datasets_list[i + 1 :]:
                pair = (dataset1, dataset2)
                dataset_pair_counts[pair] += group["count"]

    # Find datasets with most duplicates
    top_duplicate_datasets = sorted(
        dataset_total_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Find dataset pairs with most overlap
    top_overlapping_pairs = sorted(
        dataset_pair_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "total_duplicate_groups": len(duplicate_groups),
        "total_duplicate_entries": sum(g["count"] for g in duplicate_groups.values()),
        "dataset_total_counts": dict(dataset_total_counts),
        "top_duplicate_datasets": top_duplicate_datasets[:20],
        "dataset_pair_counts": {f"{k[0]} ↔ {k[1]}": v for k, v in dataset_pair_counts.items()},
        "top_overlapping_pairs": [(f"{k[0]} ↔ {k[1]}", v) for k, v in top_overlapping_pairs[:20]],
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load S3 manifest JSON file"""
    with open(manifest_path) as f:
        return json.load(f)


def extract_processed_dataset_files(
    manifest: dict[str, Any],
) -> list[ProcessedDatasetFile]:
    """Extract processed dataset objects (JSON/JSONL only) from manifest"""
    processed_files: list[ProcessedDatasetFile] = []
    processed_cats = manifest.get("categories", {}).get("gdrive", {}).get("processed", {})

    for category, files_info in processed_cats.items():
        if isinstance(files_info, dict) and "objects" in files_info:
            for obj in files_info["objects"]:
                key = obj["key"]
                key_lower = key.lower()
                if (
                    key.endswith((".json", ".jsonl"))
                    and "report" not in key_lower
                    and "metadata" not in key_lower
                    and obj["size"] > 1000  # At least 1KB
                ):
                    processed_files.append(
                        {
                            "key": key,
                            "category": category,
                            "size": obj["size"],
                        }
                    )

    return processed_files


def log_results(
    *,
    total_entries: int,
    duplicate_groups: dict[str, DuplicateGroup],
    analysis: dict[str, Any],
) -> None:
    logger.info("%s", "=" * 80)
    logger.info("📊 FULL DEDUPLICATION RESULTS")
    logger.info("%s", "=" * 80)

    unique_conversations = (
        total_entries - analysis["total_duplicate_entries"] + len(duplicate_groups)
    )
    potential_savings = analysis["total_duplicate_entries"] - analysis["total_duplicate_groups"]

    logger.info("📈 Summary:")
    logger.info("  Total entries scanned: %s", f"{total_entries:,}")
    logger.info("  Unique conversations: %s", f"{unique_conversations:,}")
    logger.info(
        "  Duplicate groups: %s",
        f"{analysis['total_duplicate_groups']:,}",
    )
    logger.info(
        "  Total duplicate entries: %s",
        f"{analysis['total_duplicate_entries']:,}",
    )
    logger.info(
        "  Potential space savings: ~%s entries",
        f"{potential_savings:,}",
    )

    if analysis["top_duplicate_datasets"]:
        logger.info("🔴 Datasets with Most Duplicates:")
        for dataset, count in analysis["top_duplicate_datasets"][:10]:
            logger.info("  %s: %s duplicate entries", dataset, f"{count:,}")

    if analysis["top_overlapping_pairs"]:
        logger.info("🔄 Top Overlapping Dataset Pairs:")
        for pair_str, count in analysis["top_overlapping_pairs"][:10]:
            logger.info("  %s: %s duplicates", pair_str, f"{count:,}")


def build_report(
    *,
    total_entries: int,
    duplicate_groups: dict[str, DuplicateGroup],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    unique_conversations = (
        total_entries - analysis["total_duplicate_entries"] + len(duplicate_groups)
    )
    potential_savings = analysis["total_duplicate_entries"] - analysis["total_duplicate_groups"]

    return {
        "scan_date": datetime.now(tz=timezone.utc).isoformat(),
        "summary": {
            "total_entries_scanned": total_entries,
            "unique_conversations": unique_conversations,
            "duplicate_groups": analysis["total_duplicate_groups"],
            "total_duplicate_entries": analysis["total_duplicate_entries"],
            "potential_savings": potential_savings,
        },
        "analysis": analysis,
        "duplicate_groups": {
            hash_val: {
                "count": group["count"],
                "datasets": list(group["datasets"]),
                "entries": group["entries"][:5],  # Limit to first 5 entries per group
            }
            for hash_val, group in list(duplicate_groups.items())[:100]  # Limit to first 100 groups
        },
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    """Main full deduplication scan"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = get_project_root()
    ensure_project_root_on_path(project_root)

    # Import after adding project root to sys.path
    from ai.training_ready.utils.s3_dataset_loader import (  # noqa: PLC0415
        S3DatasetLoader,
    )

    # Suppress verbose warnings
    logging.getLogger("ai.training_ready.utils.s3_dataset_loader").setLevel(logging.ERROR)

    logger.info("🔍 Full Dataset Deduplication Scan")
    logger.info("%s", "=" * 80)
    logger.info("⚠️  This will scan ALL entries in ALL processed datasets")
    logger.info("%s", "=" * 80)

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=DEFAULT_S3_BUCKET)

    # Load manifest
    manifest_path = project_root / "ai/training_ready/data/s3_manifest.json"
    if not manifest_path.exists():
        logger.error("❌ Manifest not found at %s", manifest_path)
        return 1

    manifest = load_manifest(manifest_path)

    # Extract processed dataset files
    processed_files = extract_processed_dataset_files(manifest)
    logger.info("📂 Found %s dataset files to scan", len(processed_files))

    # Group by category for organization
    files_by_category = group_files_by_category(processed_files)

    logger.info("📊 Categories: %s", len(files_by_category))

    # Scan all files
    all_hashed_entries, total_entries = scan_all_datasets(loader, files_by_category)

    logger.info(
        "✅ Scanned %s total entries across %s files",
        total_entries,
        len(processed_files),
    )

    # Build duplicate index
    logger.info("🔍 Building duplicate index...")
    duplicate_groups = build_duplicate_index(all_hashed_entries)

    logger.info("📊 Found %s duplicate groups", len(duplicate_groups))

    # Analyze duplicates
    logger.info("📈 Analyzing duplicates...")
    analysis = analyze_duplicates(duplicate_groups)

    log_results(
        total_entries=total_entries,
        duplicate_groups=duplicate_groups,
        analysis=analysis,
    )

    # Save detailed report
    report_path = project_root / "ai/training_ready/data/full_deduplication_report.json"
    report = build_report(
        total_entries=total_entries,
        duplicate_groups=duplicate_groups,
        analysis=analysis,
    )
    save_json(report_path, report)

    logger.info("💾 Detailed report saved to: %s", report_path)
    logger.info("✅ Full deduplication scan complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
