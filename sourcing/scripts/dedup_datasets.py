#!/usr/bin/env python3
"""Deduplicate records across all dataset JSONL files.

Strategy:
  - Read all top-level JSONL files under ai/data/raw/
  - For each record, hash normalized user+assistant message content
  - System messages excluded from hash (varies by dataset adapter)
  - First occurrence wins; duplicates tracked by source dataset
  - Write deduped combined output + per-dataset stats
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path("ai/data/raw")
OUTPUT_DIR = Path("ai/data/raw/deduped")

# Datasets to process (top-level JSONL only, skip raw/ subdirs, skip caches)
SKIP_DIRS = {".hf_cache", "s3_cache", "s3_ingestion", "hetzner", "deduped"}


def _normalize(text: str) -> str:
    """Normalize text for dedup: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def _content_hash(record: dict) -> str:
    """Hash normalized user+assistant message content."""
    parts: list[str] = []
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        if role == "system":
            continue
        content = msg.get("content", "")
        parts.append(f"{role}:{_normalize(content)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def find_jsonl_files() -> list[tuple[str, Path]]:
    """Find all top-level JSONL files, return (dataset_name, path) pairs."""
    results: list[tuple[str, Path]] = []
    for child in sorted(RAW_DIR.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        for jsonl in child.glob("*.jsonl"):
            if jsonl.stat().st_size == 0:
                continue
            results.append((child.name, jsonl))
    return results


def dedup() -> dict:
    """Run dedup across all datasets. Returns stats dict."""
    files = find_jsonl_files()
    if not files:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen: dict[str, str] = {}  # hash -> first_source
    deduped: list[dict] = []
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "unique": 0, "dups": 0})
    internal_dups: dict[str, int] = defaultdict(int)
    cross_dups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_input = 0
    total_dups = 0

    for dataset_name, jsonl_path in files:
        print(f"  Scanning {dataset_name}...", file=sys.stderr)
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total_input += 1
                stats[dataset_name]["total"] += 1

                h = _content_hash(record)
                if h in seen:
                    total_dups += 1
                    stats[dataset_name]["dups"] += 1
                    first_ds = seen[h]
                    if first_ds == dataset_name:
                        internal_dups[dataset_name] += 1
                    else:
                        cross_dups[dataset_name][first_ds] += 1
                    continue

                seen[h] = dataset_name
                stats[dataset_name]["unique"] += 1
                deduped.append(record)

    # Write combined deduped output
    combined_path = OUTPUT_DIR / "all_deduped.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write per-dataset deduped files
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in deduped:
        by_source[record.get("source", "unknown")].append(record)
    for source, records in by_source.items():
        path = OUTPUT_DIR / f"{source}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write dup report
    dup_report_path = OUTPUT_DIR / "dup_report.txt"
    with dup_report_path.open("w", encoding="utf-8") as f:
        f.write(f"Total input: {total_input}\n")
        f.write(f"Total unique: {len(deduped)}\n")
        f.write(f"Total duplicates: {total_dups}\n")
        f.write(f"Dedup rate: {total_dups / total_input * 100:.2f}%\n\n")
        f.write("Per-dataset breakdown:\n")
        for ds in sorted(stats):
            s = stats[ds]
            pct = s["dups"] / s["total"] * 100 if s["total"] else 0
            f.write(
                f"  {ds:40s} total={s['total']:>8d}  unique={s['unique']:>8d}  dups={s['dups']:>6d}  ({pct:.1f}%)\n"
            )
        f.write(f"\nInternal dups (same dataset):\n")
        for ds in sorted(internal_dups, key=lambda x: -internal_dups[x]):
            if internal_dups[ds] > 0:
                f.write(f"  {ds:40s} {internal_dups[ds]:>8,}\n")

        f.write(f"\nCross-dataset dups (dup_source -> first_seen):\n")
        for ds in sorted(cross_dups):
            for first_ds, count in sorted(cross_dups[ds].items(), key=lambda x: -x[1]):
                if count > 0:
                    f.write(f"  {ds:40s} -> {first_ds:40s} {count:>8,}\n")

    return {
        "total_input": total_input,
        "total_unique": len(deduped),
        "total_dups": total_dups,
        "per_dataset": dict(stats),
        "combined_path": str(combined_path),
        "dup_report_path": str(dup_report_path),
    }


def main() -> None:
    print("Deduplicating across all datasets...", file=sys.stderr)
    stats = dedup()
    print(f"\nDone:")
    print(f"  Input:    {stats['total_input']:>10,} records")
    print(f"  Unique:   {stats['total_unique']:>10,} records")
    print(f"  Dups:     {stats['total_dups']:>10,} records")
    rate = stats["total_dups"] / stats["total_input"] * 100 if stats["total_input"] else 0
    print(f"  Dedup rate: {rate:.2f}%")
    print(f"  Output: {stats['combined_path']}")
    print(f"  Report: {stats['dup_report_path']}")


if __name__ == "__main__":
    main()
