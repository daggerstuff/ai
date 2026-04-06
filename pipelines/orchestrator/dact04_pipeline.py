#!/usr/bin/env python3
"""
DACT-04: End-to-End Normalize → Dedup → Merge Pipeline

Orchestrates the full pipeline:
1. Normalize source files from any format → canonical JSONL
2. Cross-source deduplication
3. Merge into single training corpus

Usage:
    # Full pipeline from a directory of mixed-format source files:
    python -m ai.pipelines.orchestrator.dact04_pipeline \
        --sources data/raw_sources/ \
        --output data/merged/mental_health_dataset.jsonl \
        --report data/reports/dact04_report.json

    # Step-by-step:
    python -m ai.pipelines.orchestrator.dact04_pipeline --step normalize --sources data/raw_sources/
    python -m ai.pipelines.orchestrator.dact04_pipeline --step dedup --normalized-dir data/normalized/
    python -m ai.pipelines.orchestrator.dact04_pipeline --step full --sources data/raw_sources/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.cross_source_dedup import dedup_files
from ai.pipelines.orchestrator.normalizers import (
    NormalizationConfig,
    detect_format,
    get_normalizer,
    normalize_file,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    sources_dir: Path | None = None
    normalized_dir: Path = Path("ai/data/normalized")
    output_path: Path = Path("ai/data/merged/mental_health_dataset.jsonl")
    report_path: Path | None = None
    quality_threshold: float = 0.7
    source_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    dry_run: bool = False


def discover_source_files(sources_dir: Path) -> list[Path]:
    """Discover all data files in the sources directory."""
    patterns = ["**/*.json", "**/*.jsonl", "**/*.csv"]
    files = []
    for pattern in patterns:
        files.extend(sources_dir.glob(pattern))
    # Remove duplicates, sort by name
    return sorted(set(files))


def infer_source_config(file_path: Path, overrides: dict[str, dict[str, Any]]) -> NormalizationConfig:
    """Infer normalization config from file path and overrides."""
    stem = file_path.stem
    name = file_path.name

    # Check for explicit overrides
    for key, cfg in overrides.items():
        if key in name or key in stem:
            return NormalizationConfig(**{k: v for k, v in cfg.items() if k in NormalizationConfig.__annotations__})

    # Auto-infer from filename
    source_name = stem
    license_type = "unknown"
    therapeutic_area = "general"
    topic_tags = []
    stage = None

    # Source-based inference
    lower_name = name.lower()
    if "cot" in lower_name or "reasoning" in lower_name:
        therapeutic_area = "reasoning"
        stage = "stage2_therapeutic_expertise"
        topic_tags = ["chain_of_thought", "reasoning"]
    elif "reddit" in lower_name:
        therapeutic_area = "peer_support"
        topic_tags = ["reddit", "community"]
    elif "mental_health" in lower_name or "counseling" in lower_name:
        therapeutic_area = "counseling"
        stage = "stage1_foundation"
        topic_tags = ["counseling", "mental_health"]
    elif "psych" in lower_name:
        therapeutic_area = "psychology"
        stage = "stage1_foundation"
        topic_tags = ["psychology", "education"]
    elif "crisis" in lower_name or "edge" in lower_name:
        therapeutic_area = "crisis"
        stage = "stage3_edge_stress_test"
        topic_tags = ["crisis", "edge_case"]
    elif "voice" in lower_name or "persona" in lower_name:
        therapeutic_area = "voice"
        stage = "stage4_voice_persona"
        topic_tags = ["voice", "persona"]

    return NormalizationConfig(
        source_name=source_name,
        license=license_type,
        therapeutic_area=therapeutic_area,
        topic_tags=topic_tags,
        stage=stage,
    )


def run_normalize_step(config: PipelineConfig) -> dict[str, Any]:
    """Run the normalization step."""
    if not config.sources_dir or not config.sources_dir.exists():
        raise FileNotFoundError(f"Sources directory not found: {config.sources_dir}")

    source_files = discover_source_files(config.sources_dir)
    if not source_files:
        raise ValueError(f"No source files found in {config.sources_dir}")

    config.normalized_dir.mkdir(parents=True, exist_ok=True)

    step_report = {
        "step": "normalize",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_count": len(source_files),
        "files": [],
        "total_input": 0,
        "total_output": 0,
        "total_skipped": 0,
    }

    for src_file in source_files:
        detected_format = detect_format(src_file)
        src_config = infer_source_config(src_file, config.source_overrides)

        output_file = config.normalized_dir / f"{src_file.stem}_normalized.jsonl"

        logger.info("Normalizing: %s (format: %s) → %s", src_file.name, detected_format, output_file.name)

        if config.dry_run:
            logger.info("  [DRY RUN] Would normalize %s", src_file.name)
            step_report["files"].append({
                "input": str(src_file),
                "output": str(output_file),
                "format": detected_format,
                "status": "dry_run",
            })
            continue

        stats = normalize_file(
            src_file,
            output_file,
            format_name=None,  # auto-detect
            config=src_config,
        )

        step_report["files"].append({
            "input": str(src_file),
            "output": str(output_file),
            "format": detected_format,
            "status": "ok",
            "stats": stats,
        })
        step_report["total_input"] += stats["input"]
        step_report["total_output"] += stats["output"]
        step_report["total_skipped"] += stats["skipped"]

    return step_report


def run_dedup_step(config: PipelineConfig) -> dict[str, Any]:
    """Run the cross-source dedup step."""
    normalized_files = sorted(config.normalized_dir.glob("*.jsonl"))
    if not normalized_files:
        raise ValueError(f"No normalized files found in {config.normalized_dir}")

    logger.info("Cross-source dedup across %d files", len(normalized_files))

    if config.dry_run:
        logger.info("[DRY RUN] Would dedup %d files → %s", len(normalized_files), config.output_path)
        return {
            "step": "dedup",
            "status": "dry_run",
            "input_files": len(normalized_files),
            "output_file": str(config.output_path),
        }

    report = dedup_files(
        normalized_files,
        config.output_path,
        quality_threshold=config.quality_threshold,
    )
    report["step"] = "dedup"
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    return report


def run_full_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the full normalize → dedup pipeline."""
    start_time = time.time()
    pipeline_report = {
        "pipeline": "dact-04",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sources_dir": str(config.sources_dir),
            "normalized_dir": str(config.normalized_dir),
            "output_path": str(config.output_path),
            "quality_threshold": config.quality_threshold,
            "dry_run": config.dry_run,
        },
        "steps": [],
    }

    # Step 1: Normalize
    logger.info("=== STEP 1: Normalize ===")
    normalize_report = run_normalize_step(config)
    pipeline_report["steps"].append(normalize_report)

    # Step 2: Dedup
    logger.info("\n=== STEP 2: Cross-Source Dedup ===")
    dedup_report = run_dedup_step(config)
    pipeline_report["steps"].append(dedup_report)

    # Summary
    elapsed = time.time() - start_time
    pipeline_report["elapsed_seconds"] = round(elapsed, 2)
    pipeline_report["status"] = "dry_run" if config.dry_run else "complete"

    logger.info("\n=== PIPELINE COMPLETE (%.1fs) ===", elapsed)
    logger.info("Output: %s", config.output_path)

    return pipeline_report


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DACT-04: Normalize → Dedup → Merge Pipeline")
    parser.add_argument(
        "--step",
        choices=["normalize", "dedup", "full"],
        default="full",
        help="Which step to run (default: full)",
    )
    parser.add_argument("--sources", type=Path, help="Directory of source files (any format)")
    parser.add_argument("--normalized-dir", type=Path, default=Path("ai/data/normalized"))
    parser.add_argument("--output", type=Path, default=Path("ai/data/merged/mental_health_dataset.jsonl"))
    parser.add_argument("--report", type=Path, help="Write pipeline report here")
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    config = PipelineConfig(
        sources_dir=args.sources,
        normalized_dir=args.normalized_dir,
        output_path=args.output,
        report_path=args.report,
        quality_threshold=args.quality_threshold,
        dry_run=args.dry_run,
    )

    if args.step == "normalize":
        report = run_normalize_step(config)
    elif args.step == "dedup":
        report = run_dedup_step(config)
    else:
        report = run_full_pipeline(config)

    # Write report
    if args.report and not config.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
