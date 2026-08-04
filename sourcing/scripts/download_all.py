#!/usr/bin/env python3
"""Download and convert all available datasets via adapters.

Usage:
    uv run python -m ai.sourcing.scripts.download_all [--output-dir ai/data/raw] [--dataset NAME]

Options:
    --output-dir DIR    Output directory for converted JSONL (default: ai/data/raw)
    --dataset NAME      Download only one dataset (default: all)
    --list              List available adapters and exit
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Set HF cache env vars BEFORE any HF/datasets imports
_HF_CACHE = str(Path(__file__).resolve().parents[2] / "data" / "raw" / ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(_HF_CACHE) / "hub"))

from ai.sourcing.dataset_adapters.adapter_factory import get_adapter, list_available_adapters

# Import all adapter modules to trigger @register_adapter registration
from ai.sourcing.dataset_adapters import (  # noqa: F401
    esconv_adapter,
    hope_adapter,
    mitags_adapter,
    memo_adapter,
    mit_psychosis_adapter,
    vera_mh_adapter,
    sim_vail_adapter,
    empath_adapter,
    clinical_redteam_adapter,
    psydial_adapter,
    clpsych_adapter,
    erisk_adapter,
    personalitydbench_adapter,
    annomi_adapter,
    ml_bpd_adapter,
    bopd_adapter,
    dmtcorpus_adapter,
    mhsafeeval_adapter,
    crisis_benchmark_adapter,
    daic_woz_adapter,
    bbrd_adapter,
    reddit_mental_nlp_adapter,
    counseling_conversations_adapter,
    reddit_mental_health_posts_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert all datasets via adapters")
    parser.add_argument("--output-dir", default="ai/data/raw", help="Output directory")
    parser.add_argument("--dataset", default=None, help="Download only one dataset")
    parser.add_argument("--list", action="store_true", help="List available adapters and exit")
    args = parser.parse_args()

    if args.list:
        print("Available adapters:")
        for name in sorted(list_available_adapters()):
            print(f"  - {name}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = list_available_adapters()
    if args.dataset:
        adapters = [a for a in adapters if a.lower() == args.dataset.lower()]
        if not adapters:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(list_available_adapters())}")
            sys.exit(1)

    print(f"{'=' * 60}")
    print(f"  Dataset Adapter Pipeline — {len(adapters)} dataset(s)")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for name in sorted(adapters):
        print(f"--- {name} ---")
        try:
            adapter = get_adapter(name, output_dir)
            result = adapter.run()
            if result and result.exists():
                # Count lines
                count = sum(1 for _ in open(result, encoding="utf-8"))
                print(f"  OK: {count} records -> {result}")
                succeeded.append(name)
            else:
                print(f"  SKIP: no output (may need manual data placement)")
                succeeded.append(name)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append((name, str(e)))
            traceback.print_exc()
        print()

    print(f"{'=' * 60}")
    print(f"  Summary: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(n for n, _ in failed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
