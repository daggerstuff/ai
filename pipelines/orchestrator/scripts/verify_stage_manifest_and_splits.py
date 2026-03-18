#!/usr/bin/env python3
"""Verify stage drift and split completeness artifacts for integrated training runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify(
    checklist_path: Path,
    stage_manifest_path: Path,
    split_root: Path,
    allow_empty: bool,
) -> list[str]:
    failures: list[str] = []

    checklist = _load_json(checklist_path)
    report = checklist.get("report", {})
    if not isinstance(report, dict):
        report = {}

    stage_targets = report.get("stage_distribution_targets", {})
    if not isinstance(stage_targets, dict):
        stage_targets = {}
    stage_ids = sorted(stage_targets.keys())

    _require(
        bool(stage_ids),
        "No stage_distribution_targets found in checklist report",
        failures,
    )

    drift_ok = bool(checklist.get("stage_drift_within_tolerance", False))
    drift_failures = checklist.get("stage_drift_failures", [])
    if not isinstance(drift_failures, list):
        drift_failures = []

    _require(
        drift_ok,
        "stage_drift_within_tolerance is false",
        failures,
    )
    _require(
        len(drift_failures) == 0,
        f"stage_drift_failures is not empty: {drift_failures}",
        failures,
    )

    split_counts = checklist.get("split_counts", {})
    if not isinstance(split_counts, dict):
        split_counts = {}

    required_split_keys = ["aggregate", *stage_ids]
    for split_key in required_split_keys:
        counts = split_counts.get(split_key)
        _require(
            isinstance(counts, dict),
            f"split_counts missing section: {split_key}",
            failures,
        )
        if not isinstance(counts, dict):
            continue

        train = counts.get("train")
        val = counts.get("val")
        test = counts.get("test")
        _require(
            isinstance(train, int) and train >= 0,
            f"{split_key}.train must be a non-negative integer",
            failures,
        )
        _require(
            isinstance(val, int) and val >= 0,
            f"{split_key}.val must be a non-negative integer",
            failures,
        )
        _require(
            isinstance(test, int) and test >= 0,
            f"{split_key}.test must be a non-negative integer",
            failures,
        )
        if not allow_empty and all(isinstance(v, int) for v in (train, val, test)):
            _require(
                (train + val + test) > 0,
                f"{split_key} split counts are all zero",
                failures,
            )

    # Verify aggregate split files.
    for split_name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        split_file = split_root / split_name
        _require(
            split_file.exists(), f"Missing aggregate split file: {split_file}", failures
        )
        if split_file.exists() and not allow_empty:
            _require(
                _count_jsonl_lines(split_file) > 0,
                f"Aggregate split file is empty: {split_file}",
                failures,
            )

    # Verify per-stage split files.
    for stage in stage_ids:
        stage_dir = split_root / stage
        _require(
            stage_dir.exists(), f"Missing stage split directory: {stage_dir}", failures
        )
        if not stage_dir.exists():
            continue
        for split_name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            split_file = stage_dir / split_name
            _require(
                split_file.exists(), f"Missing stage split file: {split_file}", failures
            )
            if split_file.exists() and not allow_empty:
                _require(
                    _count_jsonl_lines(split_file) > 0,
                    f"Stage split file is empty: {split_file}",
                    failures,
                )

    stage_manifest = _load_json(stage_manifest_path)
    _require(
        isinstance(stage_manifest.get("generated_at"), str)
        and bool(str(stage_manifest.get("generated_at", "")).strip()),
        "MASTER_STAGE_MANIFEST.json missing non-empty generated_at",
        failures,
    )
    stage_manifest_stages = stage_manifest.get("stages", {})
    _require(
        isinstance(stage_manifest_stages, dict),
        "MASTER_STAGE_MANIFEST.json is missing 'stages' object",
        failures,
    )

    if isinstance(stage_manifest_stages, dict):
        for stage in stage_ids:
            stage_entry = stage_manifest_stages.get(stage)
            _require(
                isinstance(stage_entry, dict),
                f"Stage manifest missing entry for {stage}",
                failures,
            )
            if not isinstance(stage_entry, dict):
                continue

            output_path_raw = stage_entry.get("output_path")
            _require(
                isinstance(output_path_raw, str) and output_path_raw.strip(),
                f"Stage manifest missing output_path for {stage}",
                failures,
            )
            if not isinstance(output_path_raw, str) or not output_path_raw.strip():
                continue

            output_path = Path(output_path_raw)
            _require(
                output_path.exists(),
                f"Stage output_path does not exist for {stage}: {output_path}",
                failures,
            )
            declared_samples = stage_entry.get("samples")
            declared_target = stage_entry.get("target")
            declared_available = stage_entry.get("available")
            _require(
                isinstance(declared_samples, int) and declared_samples >= 0,
                f"Stage manifest samples must be non-negative int for {stage}",
                failures,
            )
            _require(
                isinstance(declared_target, int) and declared_target >= 0,
                f"Stage manifest target must be non-negative int for {stage}",
                failures,
            )
            _require(
                isinstance(declared_available, int) and declared_available >= 0,
                f"Stage manifest available must be non-negative int for {stage}",
                failures,
            )
            if output_path.exists() and isinstance(declared_samples, int):
                line_count = _count_jsonl_lines(output_path)
                _require(
                    line_count == declared_samples,
                    f"Stage samples mismatch for {stage}: manifest={declared_samples}, actual={line_count}",
                    failures,
                )
                if not allow_empty:
                    _require(
                        line_count > 0,
                        f"Stage output has zero rows for {stage}: {output_path}",
                        failures,
                    )

    # Verify split line counts match checklist split_counts contract.
    aggregate_counts = split_counts.get("aggregate", {})
    if isinstance(aggregate_counts, dict):
        for split_name, key in (
            ("train.jsonl", "train"),
            ("val.jsonl", "val"),
            ("test.jsonl", "test"),
        ):
            split_file = split_root / split_name
            if split_file.exists() and isinstance(aggregate_counts.get(key), int):
                actual = _count_jsonl_lines(split_file)
                expected = int(aggregate_counts[key])
                _require(
                    actual == expected,
                    f"Aggregate split count mismatch for {key}: checklist={expected}, actual={actual}",
                    failures,
                )

    for stage in stage_ids:
        stage_counts = split_counts.get(stage, {})
        if not isinstance(stage_counts, dict):
            continue
        stage_dir = split_root / stage
        for split_name, key in (
            ("train.jsonl", "train"),
            ("val.jsonl", "val"),
            ("test.jsonl", "test"),
        ):
            split_file = stage_dir / split_name
            if split_file.exists() and isinstance(stage_counts.get(key), int):
                actual = _count_jsonl_lines(split_file)
                expected = int(stage_counts[key])
                _require(
                    actual == expected,
                    f"Stage split count mismatch for {stage}.{key}: checklist={expected}, actual={actual}",
                    failures,
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify integrated training stage manifest and split artifacts"
    )
    parser.add_argument(
        "--checklist",
        default="ai/lightning/training_run_checklist.json",
        help="Path to generated training run checklist JSON",
    )
    parser.add_argument(
        "--stage-manifest",
        default="ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json",
        help="Path to generated stage manifest JSON",
    )
    parser.add_argument(
        "--split-root",
        default="ai/training_data_consolidated/final/splits",
        help="Root directory containing aggregate and per-stage split JSONL files",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero-sized outputs (useful for dry-runs; disabled in CI)",
    )

    args = parser.parse_args()

    failures = verify(
        checklist_path=Path(args.checklist),
        stage_manifest_path=Path(args.stage_manifest),
        split_root=Path(args.split_root),
        allow_empty=args.allow_empty,
    )

    if failures:
        print("Stage manifest/split verification failed:")
        for idx, failure in enumerate(failures, start=1):
            print(f"  {idx}. {failure}")
        return 1

    print("Stage manifest/split verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
