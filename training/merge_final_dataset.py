#!/usr/bin/env python3
"""Build multi-source dataset join + finalized shard for Training Pipeline Improvements.

Combines all available data sources into a unified training dataset with
70/15/15 train/val/test splits, ChatML normalization, and a stats report.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("merge_datasets")


def _extract_text(record: dict) -> str:
    """Extract combined text for dedup from a record."""
    if "messages" in record and record["messages"]:
        return " ".join(
            m.get("content", "") for m in record["messages"] if isinstance(m, dict)
        )
    if record.get("prompt") and record.get("chosen") and record.get("rejected"):
        return record["prompt"] + " " + record["chosen"] + " " + record["rejected"]
    return record.get("instruction", "") + " " + record.get("output", "")


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
                    "source_format": "nightmare_fuel" if record.get("metadata", {}).get("pair_type") == "nightmare_fuel" else "dpo",
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
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
            logger.info("Loaded %d records from %s", len(records), jsonl_file)
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
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
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


def run_merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for source_dir in args.source_dirs:
        path = Path(source_dir)
        if path.is_file():
            all_records.extend(_load_jsonl_file(path))
        elif path.is_dir():
            all_records.extend(_load_jsonl_dir(path))
        else:
            logger.warning("Source not found: %s", path)

    logger.info("Total records loaded: %d", len(all_records))

    if not all_records:
        logger.error("No records found — aborting")
        sys.exit(1)

    # Shuffle for split
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

    # Category stats per split
    category_counts: dict[str, Counter[str]] = {}
    source_counts: dict[str, Counter[str]] = {}

    for split_name, split_records in splits.items():
        # Convert to ChatML and write
        chatml_records = [_to_chatml(r) for r in split_records]
        split_path = output_dir / f"{split_name}_chatml.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in chatml_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info("%s: %d records -> %s", split_name, len(chatml_records), split_path)

        # Stats
        cat_counter: Counter[str] = Counter()
        src_counter: Counter[str] = Counter()
        for r in split_records:
            cat = r.get("category", "unknown")
            cat_counter[cat] += 1
            meta = r.get("metadata", {})
            src = meta.get("source_type", meta.get("source_channel", "unknown"))
            src_counter[src] += 1
        category_counts[split_name] = cat_counter
        source_counts[split_name] = src_counter

    # Write stats
    stats = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "splits": {
            "train": {"count": n_train, "ratio": round(n_train / n_total, 4)},
            "val": {"count": n_val, "ratio": round(n_val / n_total, 4)},
            "test": {"count": n_test, "ratio": round(n_test / n_total, 4)},
        },
        "total_records": n_total,
        "categories": {
            split: dict(counter.most_common())
            for split, counter in category_counts.items()
        },
        "sources": {
            split: dict(counter.most_common(20))
            for split, counter in source_counts.items()
        },
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    logger.info("Merge complete: %d total records written to %s", n_total, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multi-source training data into finalized shards.")
    parser.add_argument("--source_dirs", nargs="+", required=True, help="Source directories/files to merge.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train/val/test JSONL.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Training set ratio (default: 0.70).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15).")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
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
