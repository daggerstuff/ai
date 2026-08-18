#!/usr/bin/env python3
"""Build multi-source dataset join + finalized shard for Training Pipeline Improvements.

Combines all available data sources into a unified training dataset with
70/15/15 train/val/test splits, ChatML normalization, deduplication,
provenance tracking, and a comprehensive stats report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("merge_datasets")

# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------
SHA = hashlib.sha256


class Deduplicator:
    """Exact + near-deduplication engine with provenance-aware collision rules."""

    def __init__(self, jaccard_threshold: float = 0.85) -> None:
        # Hash -> list of records
        self.seen_hashes: dict[str, list[dict]] = {}
        # (token_set, source_type) for near-dedup
        self.token_sets: list[tuple[frozenset[str], str, str]] = []
        self.jaccard_threshold = jaccard_threshold

    @staticmethod
    def _content_hash(text: str) -> str:
        return SHA(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_token_set(text: str) -> frozenset[str]:
        return frozenset(text.lower().split())

    @staticmethod
    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _is_near_duplicate(self, text: str, _source_type: str) -> bool:
        token_set = self._compute_token_set(text)
        # Compare against last N items (window) for efficiency
        for seen_tokens, _seen_type, _ in self.token_sets[-10000:]:
            if self._jaccard(token_set, seen_tokens) >= self.jaccard_threshold:
                return True
        return False

    def dedupe(self, records: list[dict]) -> list[dict]:
        """Filter *records* in order, keeping the first occurrence of each unique entry."""
        kept: list[dict] = []
        for r in records:
            text = _extract_text(r)
            ch = self._content_hash(text)
            source = _get_source_type(r)

            # Exact dedup
            if ch in self.seen_hashes:
                continue

            # Near-dedup (skip for edge-cases if needed)
            if self._is_near_duplicate(text, source):
                continue

            self.seen_hashes[ch] = [r]
            self.token_sets.append((self._compute_token_set(text), source, ch))
            kept.append(r)
        return kept


# ---------------------------------------------------------------------------
# Text / format helpers
# ---------------------------------------------------------------------------


def _extract_text(record: dict) -> str:
    """Extract combined text for dedup from a record."""
    if record.get("messages"):
        return " ".join(m.get("content", "") for m in record["messages"] if isinstance(m, dict))
    if record.get("prompt") and record.get("chosen") and record.get("rejected"):
        return record["prompt"] + " " + record["chosen"] + " " + record["rejected"]
    return record.get("instruction", "") + " " + record.get("output", "")


def _get_source_type(record: dict) -> str:
    """Infer source type from record metadata."""
    meta = record.get("metadata", {})
    source = meta.get("source_type", meta.get("source_channel", "unknown"))
    if source != "unknown":
        return source
    prov = record.get("provenance", {})
    if prov:
        return prov.get("source_type", "unknown")
    return "unknown"


def _to_chatml(record: dict) -> dict:
    """Convert a record to ChatML message format if not already."""
    if "messages" in record:
        return record
    instruction = record.get("instruction", "")
    output = record.get("output", "")
    if not instruction and not output:
        if record.get("prompt") and record.get("chosen"):
            return {
                "messages": [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["chosen"]},
                ],
                "metadata": {
                    **record.get("metadata", {}),
                    "pair_type": "dpo_chosen",
                    "source_format": "nightmare_fuel"
                    if record.get("metadata", {}).get("pair_type") == "nightmare_fuel"
                    else "dpo",
                    "provenance": record.get("provenance", {}),
                },
            }
        return record
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "metadata": {
            "source_channel": record.get("source_channel", ""),
            "category": record.get("category", ""),
            "language": record.get("language", ""),
            "provenance": record.get("provenance", {}),
        },
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl_dir(path: Path) -> list[dict]:
    """Load all JSONL files from a directory."""
    records: list[dict] = []
    if not path.exists():
        logger.warning("Path not found: %s", path)
        return records

    for jsonl_file in sorted(path.rglob("*.jsonl")):
        if jsonl_file.name.endswith("report.jsonl"):
            continue
        try:
            file_records = []
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                        file_records.append(record)
                    except json.JSONDecodeError:
                        continue
            logger.info("Loaded %d records from %s", len(file_records), jsonl_file)
            records.extend(file_records)
        except OSError as exc:
            logger.warning("Cannot read %s: %s", jsonl_file, exc)
    return records


def _load_jsonl_file(path: Path) -> list[dict]:
    """Load a single JSONL file."""
    records: list[dict] = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return records
    try:
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        logger.info("Loaded %d records from %s", len(records), path)
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
    return records


# ---------------------------------------------------------------------------
# Provenance assembly
# ---------------------------------------------------------------------------


def _attach_source_provenance(
    record: dict,
    source_dir: str,
    merge_run_id: str,
) -> dict:
    """Attach or enrich provenance metadata for a record."""
    prov = record.get("provenance", {})
    if not prov:
        prov = {
            "source_url": str(source_dir),
            "source_type": _get_source_type(record),
        }
    # Enrich with merge-specific lineage
    prov["merge_run_id"] = merge_run_id
    prov["merged_at"] = datetime.now(UTC).isoformat()
    _transformations = prov.get("transformations", [])
    if isinstance(_transformations, str):
        _transformations = [_transformations]
    transformations: list[str] = list(_transformations)
    if "chatml_normalize" not in transformations:
        transformations.append("chatml_normalize")
    if "deduplicate" not in transformations:
        transformations.append("deduplicate")
    prov["transformations"] = transformations  # type: ignore[assignment]
    return {**record, "provenance": prov}


# ---------------------------------------------------------------------------
# Core merge pipeline
# ---------------------------------------------------------------------------


def _write_splits_and_collect_stats(
    splits: dict[str, list[dict]],
    output_dir: Path,
    _n_total: int,
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    category_counts: dict[str, Counter[str]] = {}
    source_counts: dict[str, Counter[str]] = {}

    for split_name, split_records in splits.items():
        chatml_records = [_to_chatml(r) for r in split_records]
        split_path = output_dir / f"{split_name}_chatml.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in chatml_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info("%s: %d records -> %s", split_name, len(chatml_records), split_path)

        cat_counter: Counter[str] = Counter()
        src_counter: Counter[str] = Counter()
        for r in split_records:
            cat_counter[r.get("category", "unknown")] += 1
            src_counter[_get_source_type(r)] += 1
        category_counts[split_name] = cat_counter
        source_counts[split_name] = src_counter

    return category_counts, source_counts


def _write_stats_json(stats: dict, output_dir: Path) -> None:
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


def _write_manifest_json(manifest: dict, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def run_merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_run_id = f"merge-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    logger.info("Starting merge run: %s", merge_run_id)

    all_records: list[dict] = []
    source_stats: dict[str, int] = {}

    for source_dir in args.source_dirs:
        path = Path(source_dir)
        if path.is_file():
            records = _load_jsonl_file(path)
        elif path.is_dir():
            records = _load_jsonl_dir(path)
        else:
            logger.warning("Source not found: %s", path)
            continue

        logger.info("Loaded %d raw records from %s", len(records), source_dir)
        source_stats[path.name] = len(records)

        # Attach provenance for each record
        for r in records:
            enriched = _attach_source_provenance(r, str(source_dir), merge_run_id)
            all_records.append(enriched)

    total_loaded = len(all_records)
    logger.info("Total records loaded from all sources: %d", total_loaded)

    if not all_records:
        logger.error("No records found — aborting")
        sys.exit(1)

    # ------------------------
    # Deduplication
    # ------------------------
    if not args.skip_dedup:
        dedup = Deduplicator(jaccard_threshold=args.jaccard_threshold)
        all_records = dedup.dedupe(all_records)
        logger.info("After deduplication: %d records kept", len(all_records))
    else:
        logger.info("Skipping deduplication (--skip_dedup)")

    # ------------------------
    # Shuffle + split
    # ------------------------
    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    n_total = len(all_records)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    # Test gets the remainder
    n_test = n_total - n_train - n_val

    train_records = all_records[:n_train]
    val_records = all_records[n_train : n_train + n_val]
    test_records = all_records[n_train + n_val :]

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    # Category / source stats per split
    category_counts: dict[str, Counter[str]] = {}
    source_counts: dict[str, Counter[str]] = {}

    category_counts, source_counts = _write_splits_and_collect_stats(splits, output_dir, n_total)

    # Write stats
    stats = {
        "merge_run_id": merge_run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "jaccard_threshold": args.jaccard_threshold,
        "splits": {
            "train": {"count": n_train, "ratio": round(n_train / n_total, 4)},
            "val": {"count": n_val, "ratio": round(n_val / n_total, 4)},
            "test": {"count": n_test, "ratio": round(n_test / n_total, 4)},
        },
        "total_records": n_total,
        "source_stats": source_stats,
        "categories": {split: dict(counter.most_common()) for split, counter in category_counts.items()},
        "sources": {split: dict(counter.most_common(20)) for split, counter in source_counts.items()},
    }
    _write_stats_json(stats, output_dir)

    # Write manifest
    manifest = {
        "merge_run_id": merge_run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "splits": {name: str(output_dir / f"{name}_chatml.jsonl") for name in ("train", "val", "test")},
        "stats_file": str(output_dir / "stats.json"),
        "total_records": n_total,
    }
    _write_manifest_json(manifest, output_dir)

    logger.info(
        "Merge complete: %d total records written to %s (manifest: %s)",
        n_total,
        output_dir,
        output_dir / "manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multi-source training data into finalized shards.")
    parser.add_argument(
        "--source_dirs",
        nargs="+",
        required=True,
        help="Source directories/files to merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for train/val/test JSONL.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-dedup (default: 0.85).",
    )
    parser.add_argument(
        "--skip_dedup",
        action="store_true",
        help="Skip the deduplication step.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_merge(args)


if __name__ == "__main__":
    main()
