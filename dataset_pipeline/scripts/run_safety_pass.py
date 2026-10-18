"""
PIX-4240 standalone safety pass driver.

Runs the HackathonSafetyProcessor over local hackathon-sourced records
(ChatML-formatted JSONL or raw extracts) and produces:

  <out_dir>/clear/shard_NNNNN.jsonl    — PII-stripped, no-routed-toxic
  <out_dir>/toxic_review/shard_NNNNN.jsonl — records routed for human review
  <out_dir>/reports/safety_report.json — aggregate counts + sample findings

The script is deterministic, side-effect-free outside --out-dir, and uses no
network or ML model. It's safe to run against a checkout of hackathon data
without modifying the main extract_everything.py pipeline.

Usage from ai/ repo root:

  python -m dataset_pipeline.scripts.run_safety_pass \
      --input-dir /path/to/hackathon/data \
      --out-dir /tmp/pix-4240-safety-out

  python -m dataset_pipeline.scripts.run_safety_pass \
      --input-file /path/to/single.jsonl \
      --out-dir /tmp/pix-4240-safety-out
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dataset_pipeline.processors.chatml_converter import ChatMLConverter
from dataset_pipeline.processors.safety_processors import (
    HackathonSafetyProcessor,
    SafetyReport,
)

# Shard size matches extract_everything.py's SHARD_SIZE for consistency
SHARD_SIZE = 50000

logger = logging.getLogger(__name__)

# Maximum number of per-error detail lines to log before switching to a summary
# count, so a bad input file does not flood the log.
_MAX_LOGGED_ERRORS = 50


def iter_raw_records(input_path: Path) -> Iterator[dict[str, Any]]:
    """
    Yield raw records from a path. Accepts:
      - *.jsonl : one JSON object per line (raw or ChatML)
      - *.json  : a single JSON list/object
      - *.csv   : rows (treated as dict-shaped records)
      - a directory containing any of the above (recursive)
    """
    if input_path.is_dir():
        for child in sorted(input_path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".csv"}:
                yield from iter_raw_records(child)
        return

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield _normalize(raw, input_path, None)
    elif suffix == ".json":
        with input_path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                return
        if isinstance(data, list):
            for item in data:
                yield _normalize(item, input_path, converter)
        elif isinstance(data, dict):
            yield _normalize(data, input_path, converter)
    elif suffix == ".csv":
        # Lightweight CSV handling: read with csv.DictReader, treat each row
        # as a raw record. Real hackathon CSVs vary; this is a best-effort path.
        import csv

        with input_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield {
                    "raw_data": dict(row),
                    "metadata": {
                        "source_family": "mental_health_conversations",
                        "file_key": str(input_path),
                    },
                }


def _normalize(raw: Any, source: Path, converter: ChatMLConverter) -> dict[str, Any]:
    """
    Normalize a raw JSON object to the shape extract_everything.py expects:
    {"raw_data": ..., "metadata": {...}}. If it's already a ChatML record
    (has "messages" key), wrap it so the safety processor can consume either.
    """
    if isinstance(raw, dict) and "messages" in raw:
        existing_meta = raw.get("metadata", {})
        meta = {"source_family": "mental_health_conversations", "file_key": str(source)}
        if isinstance(existing_meta, dict):
            meta.update({k: v for k, v in existing_meta.items() if k not in meta})
        return {
            "raw_data": raw,
            "metadata": meta,
        }
    if isinstance(raw, dict) and "raw_data" in raw:
        return raw
    return {
        "raw_data": raw if isinstance(raw, dict) else {"text": str(raw)},
        "metadata": {
            "source_family": "mental_health_conversations",
            "file_key": str(source),
        },
    }


def run_pass(input_path: Path, out_dir: Path, shard_size: int = SHARD_SIZE) -> dict[str, Any]:
    """
    Execute the safety pass over input. Returns the aggregate report dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_dir = out_dir / "clear"
    toxic_dir = out_dir / "toxic_review"
    report_dir = out_dir / "reports"
    # Clean stale shards from previous runs to prevent contamination
    for d in (clear_dir, toxic_dir, report_dir):
        if d.exists():
            for f in d.glob("shard_*.jsonl"):
                f.unlink()
            for f in d.glob("*.json"):
                f.unlink()
    clear_dir.mkdir(exist_ok=True)
    toxic_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    processor = HackathonSafetyProcessor()
    converter = ChatMLConverter()

    total_raw = 0
    total_clear = 0
    total_toxic = 0
    total_pii_findings = 0
    pii_by_type: dict[str, int] = {}
    tox_by_category: dict[str, int] = {}
    clinical_match_count = 0
    sample_toxic_findings: list[dict[str, Any]] = []

    dropped_convert = 0
    dropped_process = 0
    logged_errors = 0

    clear_shard: list[dict[str, Any]] = []
    toxic_shard: list[dict[str, Any]] = []
    clear_idx = 0
    toxic_idx = 0

    def flush_clear() -> None:
        nonlocal clear_shard, clear_idx
        if not clear_shard:
            return
        path = clear_dir / f"shard_{clear_idx:05d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in clear_shard:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        clear_idx += 1
        clear_shard = []

    def flush_toxic() -> None:
        nonlocal toxic_shard, toxic_idx
        if not toxic_shard:
            return
        path = toxic_dir / f"shard_{toxic_idx:05d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in toxic_shard:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        toxic_idx += 1
        toxic_shard = []

    for raw_record in iter_raw_records(input_path):
        total_raw += 1
        try:
            chatml = converter.convert(raw_record)
        except Exception as exc:
            dropped_convert += 1
            if logged_errors < _MAX_LOGGED_ERRORS:
                logger.warning(
                    "dropping record %d: ChatML conversion failed: %s",
                    total_raw,
                    exc,
                )
                logged_errors += 1
            continue

        try:
            result = processor.process(chatml)
        except Exception as exc:
            dropped_process += 1
            if logged_errors < _MAX_LOGGED_ERRORS:
                logger.warning(
                    "dropping record %d: safety processing failed: %s",
                    total_raw,
                    exc,
                )
                logged_errors += 1
            continue
        report: SafetyReport = result.report
        cleaned = result.cleaned_record

        # Aggregate PII stats
        for pii_type, count in report.pii_counts.items():
            pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + count
            total_pii_findings += count

        # Aggregate toxicity stats
        for cat in report.toxicity_triggered_categories:
            tox_by_category[cat] = tox_by_category.get(cat, 0) + 1
        clinical_match_count += len(report.clinical_matches_summary)

        if len(sample_toxic_findings) < 50 and report.toxicity_findings_summary:
            remaining = 50 - len(sample_toxic_findings)
            sample_toxic_findings.extend(report.toxicity_findings_summary[: min(3, remaining)])

        if report.routed_to_toxic_review:
            total_toxic += 1
            toxic_shard.append(cleaned)
            if len(toxic_shard) >= shard_size:
                flush_toxic()
        else:
            total_clear += 1
            clear_shard.append(cleaned)
            if len(clear_shard) >= shard_size:
                flush_clear()

    flush_clear()
    flush_toxic()

    aggregate = {
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "totals": {
            "raw_records": total_raw,
            "clear_records": total_clear,
            "routed_toxic_review": total_toxic,
            "dropped_convert_errors": dropped_convert,
            "dropped_process_errors": dropped_process,
        },
        "pii": {
            "total_findings": total_pii_findings,
            "by_type": pii_by_type,
        },
        "toxicity": {
            "by_category": tox_by_category,
            "clinical_matches_downgraded": clinical_match_count,
            "sample_findings": sample_toxic_findings,
        },
    }

    report_path = report_dir / "safety_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PIX-4240 heuristic toxicity + PII stripping safety pass for hackathon data.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path, help="Directory of hackathon data.")
    group.add_argument("--input-file", type=Path, help="Single hackathon file (.jsonl/.json/.csv).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (clear/, toxic_review/, reports/ subdirs created).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Records per output shard (default {SHARD_SIZE}).",
    )
    args = parser.parse_args()

    if args.shard_size <= 0:
        parser.error("--shard-size must be a positive integer")

    input_path: Path = args.input_dir or args.input_file
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    report = run_pass(input_path, args.out_dir, shard_size=args.shard_size)
    if report["totals"]["raw_records"] == 0:
        print("ERROR: No records found in input", file=sys.stderr)
        return 1
    print(json.dumps(report["totals"], indent=2))
    print(f"PII findings: {report['pii']['total_findings']}")
    print(f"Toxicity by category: {report['toxicity']['by_category']}")
    print(f"Report written to: {args.out_dir / 'reports' / 'safety_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
