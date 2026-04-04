#!/usr/bin/env python3
"""
DACT-04: Cross-Source Deduplication

Streams all normalized JSONL files, deduplicates across sources using
content-based hashing, and produces a single merged corpus.

Reuses the primary hash from stage_aware_deduplication.py:
    sha256(lowercase(concat(messages.role + messages.content)))

Usage:
    python -m ai.pipelines.orchestrator.cross_source_dedup \
        --input-dir ai/data/normalized \
        --output ai/data/merged/mental_health_dataset.jsonl \
        --quality-threshold 0.7

    # Or from multiple files explicitly:
    python -m ai.pipelines.orchestrator.cross_source_dedup \
        --input ai/data/normalized/a.jsonl ai/data/normalized/b.jsonl \
        --output ai/data/merged/mental_health_dataset.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Bloom filter for memory-efficient dedup on large corpora
try:
    from bitarray import bitarray
    try:
        import mmh3
        HAS_MURMUR = True
    except ImportError:
        HAS_MURMUR = False
    HAS_BLOOM = True
except ImportError:
    HAS_BLOOM = False


# ─── Content Hash (reuses stage_aware_deduplication.py logic) ───────────────

def compute_primary_hash(record: dict[str, Any]) -> str:
    """
    Primary content hash: sha256(lowercase(concat(messages.role + messages.content)))

    This is the same hash used in stage_aware_deduplication.py, ensuring
    consistency across the pipeline.
    """
    messages = record.get("messages", [])
    content_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        content_parts.append(f"{role}{content}")
    full_content = "".join(content_parts).lower()
    return hashlib.sha256(full_content.encode("utf-8")).hexdigest()


# ─── Deduplication Engine ───────────────────────────────────────────────────

class CrossSourceDeduplicator:
    """
    Cross-source deduplication using content hashing.

    Tracks which sources contribute duplicates and reports cross-source
    overlap statistics.
    """

    def __init__(self, *, use_bloom: bool = True, capacity: int = 200000, error_rate: float = 0.001):
        self.use_bloom = use_bloom and HAS_BLOOM
        self.seen_hashes: set[str] = set()
        self.bloom_filter = None
        self.bloom_count = 0

        # Cross-source duplicate tracking: hash -> list of sources
        self.cross_source_dups: dict[str, list[str]] = defaultdict(list)

        # Stats
        self.total_records = 0
        self.unique_records = 0
        self.duplicate_records = 0
        self.source_counts: dict[str, int] = defaultdict(int)
        self.source_deduped: dict[str, int] = defaultdict(int)

        if self.use_bloom:
            self._init_bloom(capacity, error_rate)

    def _init_bloom(self, capacity: int, error_rate: float) -> None:
        import math
        self.bloom_size = int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))
        self.bloom_hashes = int((self.bloom_size / capacity) * math.log(2))
        self.bloom_filter = bitarray(self.bloom_size)
        self.bloom_filter.setall(0)

    def _bloom_hash(self, item: str, seed: int) -> int:
        if HAS_MURMUR:
            return mmh3.hash(item, seed) % self.bloom_size
        return hash(f"{item}{seed}") % self.bloom_size

    def _bloom_contains(self, item: str) -> bool:
        for i in range(self.bloom_hashes):
            idx = self._bloom_hash(item, i)
            if not self.bloom_filter[idx]:
                return False
        return True

    def _bloom_add(self, item: str) -> bool:
        if self._bloom_contains(item):
            return False
        for i in range(self.bloom_hashes):
            idx = self._bloom_hash(item, i)
            self.bloom_filter[idx] = 1
        self.bloom_count += 1
        return True

    def is_duplicate(self, record: dict[str, Any]) -> bool:
        """Check if a record is a duplicate."""
        h = compute_primary_hash(record)
        source = record.get("source", "unknown")
        self.source_counts[source] += 1

        if self.use_bloom:
            is_dup = not self._bloom_add(h)
        else:
            is_dup = h in self.seen_hashes

        if is_dup:
            self.duplicate_records += 1
            self.source_deduped[source] += 1
            self.cross_source_dups[h].append(source)
            return True

        # First time seeing this content
        if not self.use_bloom:
            self.seen_hashes.add(h)
        self.unique_records += 1
        return False

    @property
    def dedup_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.duplicate_records / self.total_records

    def get_cross_source_report(self) -> dict[str, Any]:
        """Generate a report of cross-source duplicates."""
        cross_source = defaultdict(set)
        for h, sources in self.cross_source_dups.items():
            if len(set(sources)) > 1:
                for s in sources:
                    cross_source[s].update(set(sources) - {s})

        return {
            "total_records_seen": self.total_records,
            "unique_records": self.unique_records,
            "duplicate_records": self.duplicate_records,
            "dedup_rate": round(self.dedup_rate, 4),
            "source_counts": dict(self.source_counts),
            "source_deduped": dict(self.source_deduped),
            "cross_source_overlap": {
                source: list(overlaps) for source, overlaps in cross_source.items()
            },
        }


# ─── Streaming Dedup Pipeline ───────────────────────────────────────────────

def stream_jsonl_file(file_path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file."""
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", file_path)


def dedup_files(
    input_paths: list[Path],
    output_path: Path,
    *,
    quality_threshold: float = 0.7,
    use_bloom: bool = True,
) -> dict[str, Any]:
    """
    Deduplicate across multiple JSONL files and write merged output.

    Args:
        input_paths: List of normalized JSONL files to dedup across
        output_path: Output merged JSONL file
        quality_threshold: Minimum quality score to include
        use_bloom: Use bloom filter for memory efficiency

    Returns:
        Deduplication report
    """
    dedup = CrossSourceDeduplicator(use_bloom=use_bloom)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_below_threshold = 0
    quality_scores: list[float] = []

    logger.info("Starting cross-source dedup across %d files", len(input_paths))
    logger.info("Output: %s", output_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_file in input_paths:
            if not input_file.exists():
                logger.warning("Input file not found, skipping: %s", input_file)
                continue

            logger.info("Processing: %s", input_file.name)
            for record in stream_jsonl_file(input_file):
                dedup.total_records += 1

                # Quality filter
                quality = record.get("metadata", {}).get("quality_score", 0.5)
                quality_scores.append(quality)
                if quality < quality_threshold:
                    records_below_threshold += 1
                    continue

                # Dedup check
                if dedup.is_duplicate(record):
                    continue

                # Write unique record
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = dedup.get_cross_source_report()
    report["quality_threshold"] = quality_threshold
    report["records_below_threshold"] = records_below_threshold
    report["quality_stats"] = {
        "mean": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0,
        "min": round(min(quality_scores), 4) if quality_scores else 0,
        "max": round(max(quality_scores), 4) if quality_scores else 0,
    }
    report["output_file"] = str(output_path)

    logger.info(
        "Dedup complete: %d → %d unique (%.1f%% removed, %d below quality threshold)",
        dedup.total_records,
        dedup.unique_records - dedup.source_deduped.get("__below_threshold__", 0),
        dedup.dedup_rate * 100,
        records_below_threshold,
    )

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DACT-04: Cross-Source Deduplication")
    parser.add_argument("--input", nargs="+", type=Path, help="Input JSONL files")
    parser.add_argument("--input-dir", type=Path, help="Directory of JSONL files to dedup across")
    parser.add_argument("--output", type=Path, default=Path("ai/data/merged/mental_health_dataset.jsonl"))
    parser.add_argument("--quality-threshold", type=float, default=0.7, help="Minimum quality score (default: 0.7)")
    parser.add_argument("--no-bloom", action="store_true", help="Disable bloom filter (use set-based dedup)")
    parser.add_argument("--report", type=Path, help="Write dedup report to this file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Gather input files
    input_files: list[Path] = []
    if args.input:
        input_files.extend(args.input)
    if args.input_dir:
        input_files.extend(sorted(args.input_dir.glob("*.jsonl")))
        input_files.extend(sorted(args.input_dir.glob("*.json")))

    if not input_files:
        parser.error("No input files found. Use --input or --input-dir.")

    logger.info("Found %d input files", len(input_files))
    for f in input_files:
        logger.info("  %s", f)

    report = dedup_files(
        input_files,
        args.output,
        quality_threshold=args.quality_threshold,
        use_bloom=not args.no_bloom,
    )

    print("\n=== CROSS-SOURCE DEDUP REPORT ===")
    print(f"Total records seen: {report['total_records_seen']:,}")
    print(f"Unique records written: {report['unique_records']:,}")
    print(f"Duplicate records removed: {report['duplicate_records']:,}")
    print(f"Dedup rate: {report['dedup_rate']:.2%}")
    print(f"Records below quality threshold: {report['records_below_threshold']:,}")
    print(f"Quality mean: {report['quality_stats']['mean']:.3f}")
    print(f"\nPer-source counts:")
    for source, count in sorted(report['source_counts'].items()):
        deduped = report['source_deduped'].get(source, 0)
        print(f"  {source}: {count:,} input, {deduped:,} deduped")

    if report['cross_source_overlap']:
        print(f"\nCross-source overlap:")
        for source, overlaps in report['cross_source_overlap'].items():
            print(f"  {source} overlaps with: {', '.join(sorted(overlaps))}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
