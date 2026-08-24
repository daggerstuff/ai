#!/usr/bin/env python3
"""Extract Kaggle therapy/counseling datasets and output to V7 staging directory.

Usage:
    uv run python -m ai.pipelines.data_processing.scripts.extract_kaggle [options]

Options:
    --raw-dir DIR       Directory containing downloaded Kaggle files (default: ai/data/raw/kaggle)
    --output-dir DIR    Output directory for JSONL (default: ai/data/prepared/v7_staging/kaggle_therapy)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ai/ is on sys.path
_ai_root = Path(__file__).resolve().parent.parent.parent
if str(_ai_root) not in sys.path:
    sys.path.insert(0, str(_ai_root))

from ai.pipelines.data_processing.dataset_adapters.kaggle_therapy_adapter import KaggleTherapyAdapter


def main() -> None:
    raw_dir = Path("ai/data/raw/kaggle")
    output_dir = Path("ai/data/prepared/v7_staging")

    # Allow CLI overrides
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--raw-dir" and i + 1 < len(args):
            raw_dir = Path(args[i + 1])
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    if not raw_dir.exists():
        print(f"Error: raw directory {raw_dir} does not exist.")
        sys.exit(1)

    print(f"Extracting Kaggle datasets from {raw_dir}...")
    adapter = KaggleTherapyAdapter(
        dataset_name="kaggle_therapy",
        output_dir=output_dir,
        raw_dir=raw_dir,
    )
    output_path = adapter.run()
    print(f"\nDone. Output: {output_path}")

    # Print stats
    count = 0
    with open(output_path) as f:
        count = sum(1 for _ in f)
    print(f"Total records: {count}")


if __name__ == "__main__":
    main()
