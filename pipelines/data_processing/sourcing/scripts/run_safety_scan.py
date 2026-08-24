"""Batch safety scanner for all prepared dataset JSONL files.

Scans all .jsonl files in a directory (or the default prepared output dir)
through the SafetyProcessor (PII stripping + heuristic toxicity scoring).
Writes cleaned records back with a ``.safe.jsonl`` suffix and a JSON
report to ``<output_dir>/safety_report.json``.

Usage:
    python -m sourcing.scripts.run_safety_scan [--input-dir DIR] [--drop-toxic] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai.pipelines.data_processing.sourcing.processors.safety_processor import SafetyProcessor


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _scan_file(
    path: Path,
    processor: SafetyProcessor,
    dry_run: bool,
) -> dict:
    """Scan a single JSONL file. Returns a per-file report dict."""
    try:
        records = _load_jsonl(path)
    except Exception as e:
        return {"file": str(path), "error": str(e), "records": 0}

    cleaned, report = processor.process_batch(records)

    out_path = path.with_suffix(".safe.jsonl")
    if not dry_run:
        _write_jsonl(out_path, cleaned)

    summary = report.summary()
    summary["file"] = str(path)
    summary["output"] = str(out_path) if not dry_run else None
    summary["input_records"] = len(records)
    summary["output_records"] = len(cleaned)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heuristic toxicity + PII stripping across prepared datasets",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/prepared",
        help="Directory containing .jsonl files to scan (default: data/prepared)",
    )
    parser.add_argument(
        "--drop-toxic",
        action="store_true",
        help="Drop records that exceed the toxicity threshold instead of flagging them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and report without writing output files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}")
        sys.exit(1)

    # Find all .jsonl files (excluding already-processed .safe.jsonl)
    jsonl_files = sorted(
        f for f in input_dir.rglob("*.jsonl")
        if ".safe." not in f.name
    )

    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        sys.exit(0)

    processor = SafetyProcessor(drop_toxic=args.drop_toxic)

    print(f"Scanning {len(jsonl_files)} file(s) in {input_dir}")
    print(f"Mode: {'dry-run' if args.dry_run else 'write'}, drop_toxic={args.drop_toxic}")
    print("-" * 60)

    all_reports: list[dict] = []

    for f in jsonl_files:
        result = _scan_file(f, processor, args.dry_run)
        all_reports.append(result)

        if "error" in result:
            print(f"  ERROR: {result['file']}: {result['error']}")
        else:
            pii = result.get("pii", {})
            tox = result.get("toxicity", {})
            status = "DRY-RUN" if args.dry_run else "OK"
            print(
                f"  {status} {result['file']}: "
                f"{result['input_records']} in -> {result['output_records']} out, "
                f"PII redactions={pii.get('total_redactions', 0)}, "
                f"toxic flagged={tox.get('flagged_records', 0)}, "
                f"edge bypass={result.get('edge_case_bypassed', 0)}"
            )

    print("-" * 60)

    # Aggregate totals
    total_in = sum(r.get("input_records", 0) for r in all_reports)
    total_out = sum(r.get("output_records", 0) for r in all_reports)
    total_pii = sum(
        r.get("pii", {}).get("total_redactions", 0) for r in all_reports
    )
    total_toxic = sum(
        r.get("toxicity", {}).get("flagged_records", 0) for r in all_reports
    )

    print(f"TOTAL: {total_in} records in -> {total_out} out")
    print(f"PII redactions: {total_pii}")
    print(f"Toxic flagged: {total_toxic}")

    # Write report
    if not args.dry_run:
        report_path = input_dir / "safety_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "files_scanned": len(jsonl_files),
                    "total_input_records": total_in,
                    "total_output_records": total_out,
                    "total_pii_redactions": total_pii,
                    "total_toxic_flagged": total_toxic,
                    "per_file": all_reports,
                },
                f,
                indent=2,
            )
        print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
