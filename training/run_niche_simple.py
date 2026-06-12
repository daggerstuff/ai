#!/usr/bin/env python3
"""Orchestrate sdg_pipeline.py as subprocess for each niche category sequentially."""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE = Path(__file__).parent / "sdg_pipeline.py"
PYTHON = PROJECT_ROOT / "ai" / ".venv" / "bin" / "python3"
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

NEMO_MODEL = "meta/llama-3.1-8b-instruct"
NEMO_TIMEOUT = 25
NEMO_MIN_CALL_INTERVAL = 1.5
TARGET = 500
MAX_ITER = 2500


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    nemo_endpoint = "https://integrate.api.nvidia.com/v1"

    log(f"Starting niche generation — {TIMESTAMP}")
    log(f"Model: {NEMO_MODEL}")
    log(f"Timeout: {NEMO_TIMEOUT}s")
    log(f"Min call interval: {NEMO_MIN_CALL_INTERVAL}s")
    log(f"Pipeline: {PIPELINE}")
    log(f"Categories: {len(CATEGORIES)}")
    log("")

    start_total = time.time()
    results = []

    for i, cat in enumerate(CATEGORIES, 1):
        out_path = OUTPUT_DIR / f"{cat}.jsonl"
        log_path = LOG_DIR / f"niche_{cat}_{TIMESTAMP}.log"

        log(f"[{i}/{len(CATEGORIES)}] {cat}")

        env = {
            "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY", ""),
            "NVIDIA_BASE_URL": nemo_endpoint,
            "NEMO_ENDPOINT": nemo_endpoint,
        }

        cat_start = time.time()
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                [
                    str(PYTHON),
                    "-u",
                    str(PIPELINE),
                    "--scenario",
                    "niche_category",
                    "--category",
                    cat,
                    "--target_count",
                    str(TARGET),
                    "--max_iterations",
                    str(MAX_ITER),
                    "--nemo_endpoint",
                    nemo_endpoint,
                    "--nemo_model",
                    NEMO_MODEL,
                    "--nemo_timeout",
                    str(NEMO_TIMEOUT),
                    "--nemo_min_call_interval",
                    str(NEMO_MIN_CALL_INTERVAL),
                    "--min_clinical_validity",
                    "0.0",
                    "--output_path",
                    str(out_path),
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
            )
        cat_elapsed = time.time() - cat_start

        count = 0
        if out_path.exists():
            count = sum(1 for _ in open(out_path))
        status = "OK" if result.returncode == 0 else f"EXIT={result.returncode}"
        log(f"  {status} — {count} samples in {cat_elapsed:.0f}s")
        results.append(
            {"category": cat, "samples": count, "elapsed_s": round(cat_elapsed), "returncode": result.returncode}
        )

    total_time = time.time() - start_total
    passed = sum(1 for r in results if r["samples"] >= 500)
    log("=" * 60)
    log(f"ALL DONE — {total_time:.0f}s = {total_time / 60:.1f} min")
    for r in results:
        mark = "OK" if r["samples"] >= 500 else "LOW"
        log(f"  [{mark}] {r['category']}: {r['samples']} samples ({r['elapsed_s']}s)")
    log(f"Passed: {passed}/{len(results)}")

    import json

    report = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": total_time,
        "total_categories": len(CATEGORIES),
        "passed": passed,
        "failed": len(results) - passed,
        "model": NEMO_MODEL,
        "timeout_seconds": NEMO_TIMEOUT,
        "min_call_interval_seconds": NEMO_MIN_CALL_INTERVAL,
        "results": results,
    }
    (OUTPUT_DIR / "generation_report.json").write_text(json.dumps(report, indent=2))
    log(f"Report: {OUTPUT_DIR / 'generation_report.json'}")


if __name__ == "__main__":
    main()
