#!/usr/bin/env python3
"""CLI script to run V7 dataset integrity tests.

Usage:
    uv run python -m dataset_pipeline.scripts.run_integrity_test --input_dir output/v7/
    uv run python -m dataset_pipeline.scripts.run_integrity_test --input_dir output/v7/ --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dataset_pipeline.processors.integrity_test import (
    format_report,
    run_integrity_test,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run integrity tests against V7 consolidated dataset output.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing shard_*.jsonl files or a single .jsonl file.",
    )
    parser.add_argument(
        "--max_tokens_per_message",
        type=int,
        default=4096,
        help="Max estimated tokens per message (default: 4096).",
    )
    parser.add_argument(
        "--max_tokens_per_conversation",
        type=int,
        default=16384,
        help="Max estimated tokens per conversation (default: 16384).",
    )
    parser.add_argument(
        "--max_message_chars",
        type=int,
        default=20000,
        help="Max characters per message content (default: 20000).",
    )
    parser.add_argument(
        "--skip_metadata_check",
        action="store_true",
        default=False,
        help="Skip V7 metadata field checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with code 1 if any warnings are found (not just errors).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to this file (JSON format). If not given, prints to stdout.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    report = run_integrity_test(
        args.input_dir,
        max_tokens_per_message=args.max_tokens_per_message,
        max_tokens_per_conversation=args.max_tokens_per_conversation,
        max_message_chars=args.max_message_chars,
        check_v7_metadata=not args.skip_metadata_check,
    )

    # Print human-readable report
    print(format_report(report))  # noqa: T201

    # Write JSON report if requested
    if args.output:
        output_path = Path(args.output)
        report_json = {
            "total_records": report.total_records,
            "total_violations": report.total_violations,
            "errors": report.errors,
            "warnings": report.warnings,
            "passed": report.passed,
            "violations": [
                {
                    "severity": v.severity,
                    "check": v.check,
                    "message": v.message,
                    "record_index": v.record_index,
                    "shard_file": v.shard_file,
                    "field_path": v.field_path,
                }
                for v in report.violations
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2)
            f.write("\n")
        logger.info("Report written to %s", output_path)

    # Exit code: 0 if passed, 1 if failed (errors, or warnings in strict mode)
    if not report.passed:
        sys.exit(1)
    if args.strict and report.warnings > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
