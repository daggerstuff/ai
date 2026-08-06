#!/usr/bin/env python3
"""Assemble V7 MASTER Dataset — single-command orchestrator (PIX-4232).

Runs four stages in sequence:
  1. consolidate_assets.py --format v7   (hash dedup + Jaccard near-dedup + ChatML)
  2. dedup_normalize.py --semantic_dedup lsh  (MinHash/LSH semantic dedup, optional)
  3. v7_integrity.py                     (token limits, role validity, UTF-8)
  4. s3_atomic_sync.py                   (atomic swap upload to S3, optional)

Each stage must exit 0 or the pipeline aborts. Stage 2 and 4 are skippable.
Stage 1 and 3 always run.

Usage:
    python assemble_v7_master.py --input_dirs ai/training/output/
    python assemble_v7_master.py --input_dirs data/raw/ --skip_dedup --skip_s3
    python assemble_v7_master.py --input_dirs data/ --s3_bucket pixeldata --s3_prefix datasets/v7/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("assemble_v7_master")

SCRIPTS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPTS_DIR.parent
ConsolidateScript = SCRIPTS_DIR / "consolidate_assets.py"
DedupScript = TRAINING_DIR / "dedup_normalize.py"
IntegrityScript = TRAINING_DIR / "v7_integrity.py"
S3Script = SCRIPTS_DIR / "s3_atomic_sync.py"


@dataclass
class StageResult:
    name: str
    exit_code: int
    duration_s: float
    output_lines: list[str] = field(default_factory=list)


def _run_stage(script_path: Path, args: list[str], stage_name: str) -> StageResult:
    """Run a stage script via subprocess. Returns StageResult."""
    start = time.monotonic()
    cmd = [sys.executable, str(script_path), *args]
    logger.info("Stage '%s' starting: %s", stage_name, " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.monotonic() - start
    lines = (result.stdout + result.stderr).strip().splitlines()
    logger.info(
        "Stage '%s' finished: exit=%d duration=%.1fs",
        stage_name,
        result.returncode,
        duration,
    )
    if result.returncode != 0 and lines:
        for line in lines[-10:]:
            logger.error("[%s] %s", stage_name, line)
    return StageResult(
        name=stage_name,
        exit_code=result.returncode,
        duration_s=duration,
        output_lines=lines,
    )


def _find_jsonl(output_dir: Path) -> Path | None:
    """Find the primary JSONL output in a directory.

    Looks for V7_MASTER.jsonl first, then shard_0000.jsonl, then any .jsonl.
    """
    if not output_dir.is_dir():
        return None
    master = output_dir / "V7_MASTER.jsonl"
    if master.exists():
        return master
    shard = output_dir / "shard_0000.jsonl"
    if shard.exists():
        return shard
    jsonls = sorted(output_dir.glob("*.jsonl"))
    return jsonls[0] if jsonls else None


def run_pipeline(
    input_dirs: list[str],
    work_dir: Path,
    *,
    skip_dedup: bool = False,
    skip_integrity: bool = False,
    skip_s3: bool = False,
    s3_bucket: str = "pixeldata",
    s3_prefix: str = "datasets/v7/",
    s3_region: str = "US-EAST-VA",
    s3_dry_run: bool = False,
    shard_size: int = 0,
    jaccard_threshold: float = 0.92,
    lsh_threshold: float = 0.85,
    max_tokens_per_message: int = 8192,
    max_total_tokens: int = 32768,
) -> list[StageResult]:
    """Run the full V7 assembly pipeline. Returns list of StageResult.

    Aborts on first non-zero exit code; returns partial results.
    """
    results: list[StageResult] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    lsh_dir = work_dir / "lsh"

    # Stage 1: consolidate
    consolidate_args = [
        "--format",
        "v7",
        "--input_dirs",
        *input_dirs,
        "--output_dir",
        str(work_dir),
        "--shard_size",
        str(shard_size),
        "--jaccard_threshold",
        str(jaccard_threshold),
    ]
    r1 = _run_stage(ConsolidateScript, consolidate_args, "consolidate")
    results.append(r1)
    if r1.exit_code != 0:
        logger.error("Aborting: consolidate failed (exit %d)", r1.exit_code)
        return results

    # Determine stage 1 output
    stage1_output = _find_jsonl(work_dir)
    if stage1_output is None:
        logger.error("Aborting: no JSONL output from consolidate in %s", work_dir)
        results.append(StageResult("consolidate-check", 1, 0.0, ["No JSONL found"]))
        return results
    logger.info("Consolidate output: %s", stage1_output)

    # Stage 2: LSH semantic dedup (optional)
    final_output = stage1_output
    if not skip_dedup:
        lsh_dir.mkdir(parents=True, exist_ok=True)
        dedup_args = [
            "--input_dirs",
            str(work_dir),
            "--output_dir",
            str(lsh_dir),
            "--semantic_dedup",
            "lsh",
            "--shard_size",
            str(shard_size),
            "--jaccard_threshold",
            str(lsh_threshold),
        ]
        r2 = _run_stage(DedupScript, dedup_args, "dedup-lsh")
        results.append(r2)
        if r2.exit_code != 0:
            logger.warning("LSH dedup failed (exit %d), using consolidate output", r2.exit_code)
        else:
            lsh_output = _find_jsonl(lsh_dir)
            if lsh_output is not None:
                final_output = lsh_output
                logger.info("LSH dedup output: %s", final_output)
            else:
                logger.warning("No LSH output found, using consolidate output")

    logger.info("Final dataset: %s", final_output)

    # Stage 3: integrity validation
    if not skip_integrity:
        integrity_args = [
            str(final_output),
            "--max_tokens_per_message",
            str(max_tokens_per_message),
            "--max_total_tokens",
            str(max_total_tokens),
        ]
        r3 = _run_stage(IntegrityScript, integrity_args, "integrity")
        results.append(r3)
        if r3.exit_code != 0:
            logger.error("Aborting: integrity check failed (exit %d)", r3.exit_code)
            return results

    # Stage 4: S3 atomic swap sync (optional)
    if not skip_s3:
        s3_key = f"{s3_prefix.rstrip('/')}/{final_output.name}"
        s3_args = [
            "--input",
            str(final_output),
            "--s3_key",
            s3_key,
            "--bucket",
            s3_bucket,
            "--region",
            s3_region,
        ]
        if s3_dry_run:
            s3_args.append("--dry_run")
        r4 = _run_stage(S3Script, s3_args, "s3-sync")
        results.append(r4)
        if r4.exit_code != 0:
            logger.error("S3 sync failed (exit %d)", r4.exit_code)
            return results

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble V7 MASTER Dataset — single-command orchestrator (PIX-4232).",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Directories containing JSONL files to consolidate.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for V7 outputs (default: ai/training/output/v7/).",
    )
    parser.add_argument(
        "--skip_dedup",
        action="store_true",
        help="Skip LSH semantic dedup stage.",
    )
    parser.add_argument(
        "--skip_integrity",
        action="store_true",
        help="Skip integrity validation stage.",
    )
    parser.add_argument(
        "--skip_s3",
        action="store_true",
        help="Skip S3 atomic swap sync stage.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Max records per shard (0 = single file, default 0).",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.92,
        help="Jaccard threshold for consolidate near-dedup (default 0.92).",
    )
    parser.add_argument(
        "--lsh_threshold",
        type=float,
        default=0.85,
        help="Jaccard threshold for LSH semantic dedup (default 0.85).",
    )
    parser.add_argument(
        "--max_tokens_per_message",
        type=int,
        default=8192,
        help="Per-message token limit for integrity check (default 8192).",
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=32768,
        help="Total record token limit for integrity check (default 32768).",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default="pixeldata",
        help="S3 bucket (default pixeldata).",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="datasets/v7/",
        help="S3 key prefix (default datasets/v7/).",
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        default="US-EAST-VA",
        help="S3 region (default US-EAST-VA).",
    )
    parser.add_argument(
        "--s3_dry_run",
        action="store_true",
        help="S3 sync dry-run mode (no actual upload).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.DEBUG if (argv and "-v" in argv) or (argv and "--verbose" in argv) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path.cwd() / "ai" / "training" / "output" / "v7"

    logger.info("=== V7 MASTER Assembly Pipeline ===")
    logger.info("Input dirs: %s", args.input_dirs)
    logger.info("Work dir: %s", work_dir)
    logger.info(
        "Stages: consolidate=yes dedup=%s integrity=%s s3=%s",
        "no" if args.skip_dedup else "yes",
        "no" if args.skip_integrity else "yes",
        "no" if args.skip_s3 else "yes",
    )

    results = run_pipeline(
        input_dirs=args.input_dirs,
        work_dir=work_dir,
        skip_dedup=args.skip_dedup,
        skip_integrity=args.skip_integrity,
        skip_s3=args.skip_s3,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_region=args.s3_region,
        s3_dry_run=args.s3_dry_run,
        shard_size=args.shard_size,
        jaccard_threshold=args.jaccard_threshold,
        lsh_threshold=args.lsh_threshold,
        max_tokens_per_message=args.max_tokens_per_message,
        max_total_tokens=args.max_total_tokens,
    )

    logger.info("=== Pipeline Summary ===")
    all_ok = True
    for r in results:
        status = "OK" if r.exit_code == 0 else "FAIL"
        logger.info("  %-16s %s  exit=%d  %.1fs", r.name, status, r.exit_code, r.duration_s)
        if r.exit_code != 0:
            all_ok = False

    if all_ok:
        logger.info("All stages passed.")
        return 0
    logger.error("Pipeline completed with failures.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
