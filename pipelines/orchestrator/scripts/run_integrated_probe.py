#!/usr/bin/env python3
"""Run a non-writing integrated training probe and report stage sufficiency."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.pipelines.orchestrator.orchestration.integrated_training_pipeline import (  # noqa: E402
    IntegratedPipelineConfig,
    IntegratedTrainingPipeline,
)


STANDARD_THERAPEUTIC_MONOLITH = (
    "gdrive:backups/S3-Complete/processed_ready/ULTIMATE_FINAL_DATASET_processed.jsonl"
)
STANDARD_THERAPEUTIC_SPECIALISTS = (
    "gdrive:backups/S3-Complete/datasets/training_v3/"
    "stage2_specialist_addiction/fadodr_mental_health_therapy.jsonl",
    "gdrive:backups/S3-Complete/datasets/training_v3/"
    "stage2_specialist_personality/Kanakmi_mental-disorders.jsonl",
)
SOURCE_NAMES = (
    "edge_cases",
    "pixel_voice",
    "psychology_knowledge",
    "dual_persona",
    "standard_therapeutic",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated pipeline load/balance path without writing output "
            "artifacts or syncing trackers."
        )
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1000,
        help="Probe run assembly target. This is a probe budget, not corpus truth.",
    )
    parser.add_argument(
        "--standard-max-samples",
        type=int,
        default=None,
        help="Optional probe-only cap for standard_therapeutic loading.",
    )
    parser.add_argument(
        "--edge-max-samples",
        type=int,
        default=None,
        help="Optional probe-only cap for edge_cases loading.",
    )
    parser.add_argument(
        "--specialists-first",
        action="store_true",
        help=(
            "Load Stage 2 specialist shards before the monolithic standard source "
            "when a standard max cap is applied."
        ),
    )
    parser.add_argument(
        "--disable-source",
        action="append",
        choices=SOURCE_NAMES,
        default=[],
        help="Disable one source family for the probe. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON probe report.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> IntegratedPipelineConfig:
    config = IntegratedPipelineConfig(
        target_total_samples=args.target_total,
        enable_bias_detection=False,
        enable_quality_validation=False,
    )

    for source_name in args.disable_source:
        getattr(config, source_name).enabled = False

    config.sync.enable_progress_tracking = False
    config.sync.enable_tracker_sync = False
    config.sync.enable_asana_sync = False
    config.sync.enable_beads_sync = False
    config.sync.enable_jira_sync = False
    config.sync.enable_linear_sync = False
    config.sync.enable_dataset_asana_sync = False

    if args.standard_max_samples is not None:
        config.standard_therapeutic.max_samples = args.standard_max_samples
    if args.edge_max_samples is not None:
        config.edge_cases.max_samples = args.edge_max_samples

    if args.specialists_first:
        config.standard_therapeutic.source_path = None
        config.standard_therapeutic.source_paths = (
            *STANDARD_THERAPEUTIC_SPECIALISTS,
            STANDARD_THERAPEUTIC_MONOLITH,
        )

    return config


def _stage_targets(config: IntegratedPipelineConfig) -> dict[str, int]:
    return {
        stage: int(config.target_total_samples * share)
        for stage, share in config.stage_distribution.items()
    }


def _sufficiency(stage_targets: dict[str, int], pre_stage_counts: dict[str, int]) -> dict[str, Any]:
    sufficiency: dict[str, Any] = {}
    for stage, target in stage_targets.items():
        available = pre_stage_counts.get(stage, 0)
        sufficiency[stage] = {
            "target": target,
            "available": available,
            "sufficient": available >= target,
            "deficit": max(target - available, 0),
        }
    return sufficiency


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    pipeline = IntegratedTrainingPipeline(config)

    all_training_data = pipeline.data_ingestion.load_all_sources()
    pre_stage_counts = Counter(
        item.get("metadata", {}).get("stage", "stage1_foundation")
        for item in all_training_data
    )

    balanced_data, stage_segments = (
        pipeline.curriculum_enforcement_service.balance_dataset(all_training_data)
    )

    validation_error = None
    try:
        pipeline.curriculum_enforcement_service.validate_final_stage_balance(
            balanced_data
        )
    except Exception as exc:  # pragma: no cover - surfaced in report for debugging
        validation_error = f"{type(exc).__name__}: {exc}"

    stage_targets = _stage_targets(config)
    report = {
        "probe": {
            "target_total": config.target_total_samples,
            "standard_max_samples": config.standard_therapeutic.max_samples,
            "edge_max_samples": config.edge_cases.max_samples,
            "specialists_first": bool(args.specialists_first),
            "disabled_sources": sorted(args.disable_source),
        },
        "pre_total": len(all_training_data),
        "pre_stage_counts": dict(pre_stage_counts),
        "pre_source_counts": dict(pipeline.stats.samples_by_source),
        "stage_targets": stage_targets,
        "stage_sufficiency": _sufficiency(stage_targets, dict(pre_stage_counts)),
        "balanced_total": len(balanced_data),
        "balanced_stage_counts": dict(
            Counter(
                item.get("metadata", {}).get("stage", "stage1_foundation")
                for item in balanced_data
            )
        ),
        "segment_sizes": {stage: len(records) for stage, records in stage_segments.items()},
        "stage_balance": pipeline.stats.stage_balance,
        "validation_error": validation_error,
        "warnings": list(pipeline.stats.warnings),
    }
    return report


def main() -> int:
    args = _parse_args()
    report = run_probe(args)

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
