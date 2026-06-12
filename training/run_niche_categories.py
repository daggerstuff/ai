#!/usr/bin/env python3
"""Sequential niche category generator — imports sdg_pipeline directly."""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure we can import sdg_pipeline
sys.path.insert(0, str(Path(__file__).parent))
from sdg_pipeline import run_sdg, build_parser

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ai" / "training" / "data" / "generated" / "niche"
LOG_DIR = PROJECT_ROOT / "ai" / "training" / "logs"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

CATEGORIES = [
    "dissociation",
    "somatic_therapy",
    "attachment_disorders",
    "narcissistic_abuse_recovery",
    "complicated_grief",
    "eating_disorders",
    "ocd_intrusive_thoughts",
    "personality_disorders",
    "neurodivergent_mental_health",
    "cultural_religious_contexts",
]

NEMO_ENDPOINT = os.environ.get("NVIDIA_BASE_URL", "")
if not NEMO_ENDPOINT:
    print("FATAL: NVIDIA_BASE_URL not set in environment")
    sys.exit(1)

NEMO_MODEL = "meta/llama-3.1-8b-instruct"
NEMO_TIMEOUT = 25
TARGET = 500
MAX_ITER = 2500
CLINICAL_VALIDITY = 0.0

parser = build_parser()


def log(message: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def run_one_category(category: str) -> int:
    """Run generation for one category. Returns sample count."""
    out_path = OUTPUT_DIR / f"{category}.jsonl"
    log_path = LOG_DIR / f"niche_{category}_{TIMESTAMP}.log"

    # Redirect stdout/stderr to per-category log
    tee_out = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = tee_out
    sys.stderr = tee_out

    # Configure logging so sdg_pipeline's logging.getLogger().info() calls appear in log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=tee_out,
        force=True,
    )

    try:
        log(f"Starting {category} -> {out_path}")
        start = time.time()

        args = parser.parse_args(
            [
                "--scenario",
                "niche_category",
                "--category",
                category,
                "--target_count",
                str(TARGET),
                "--max_iterations",
                str(MAX_ITER),
                "--nemo_endpoint",
                NEMO_ENDPOINT,
                "--nemo_model",
                NEMO_MODEL,
                "--nemo_timeout",
                str(NEMO_TIMEOUT),
                "--min_clinical_validity",
                str(CLINICAL_VALIDITY),
                "--output_path",
                str(out_path),
            ]
        )
        run_sdg(args)

        elapsed = time.time() - start
        count = 0
        if out_path.exists():
            count = sum(1 for _ in open(out_path))
        log(f"Done {category}: {count} samples in {elapsed:.0f}s")
        return count
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee_out.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"==> Starting niche generation at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"==> Model: {NEMO_MODEL}")
    print(f"==> Timeout: {NEMO_TIMEOUT}s")
    print(f"==> Endpoint: {NEMO_ENDPOINT}")
    print(f"==> Categories: {len(CATEGORIES)}")
    print()

    start_total = time.time()
    results = []

    for i, cat in enumerate(CATEGORIES, 1):
        print(f"\n{'=' * 70}")
        print(f"[{time.strftime('%H:%M:%S')}] [{i}/{len(CATEGORIES)}] Launching: {cat}")
        print(f"{'=' * 70}", flush=True)

        cat_start = time.time()
        count = run_one_category(cat)
        cat_elapsed = time.time() - cat_start

        print(f"[{time.strftime('%H:%M:%S')}] Done: {cat} — {count} samples in {cat_elapsed:.0f}s", flush=True)
        results.append({"category": cat, "samples": count, "elapsed_s": round(cat_elapsed)})

    total_time = time.time() - start_total
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Total time: {total_time:.0f}s = {total_time / 60:.1f} min")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results if r["samples"] >= 500)
    failed = len(results) - passed
    for r in results:
        status = "✅" if r["samples"] >= 500 else "❌"
        print(f"  {status} {r['category']}: {r['samples']} samples ({r['elapsed_s']}s)")
    print(f"  Passed: {passed} / {len(results)}")

    report = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": total_time,
        "total_categories": len(CATEGORIES),
        "passed": passed,
        "failed": failed,
        "model": NEMO_MODEL,
        "timeout_seconds": NEMO_TIMEOUT,
        "results": results,
    }
    report_path = OUTPUT_DIR / "generation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
