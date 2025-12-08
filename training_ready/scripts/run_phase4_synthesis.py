#!/usr/bin/env python3
"""
Orchestration script for Phase 4: Synthesis & Instruct Tuning.
Aggregates all data, formats it (Alpaca), and splits into Train/Val/Test.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.dataset_pipeline.synthesis.dataset_synthesizer import DatasetSynthesizer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase4_Orchestrator")


def main():
    logger.info("Initializing Phase 4: Data Synthesis...")

    synthesizer = DatasetSynthesizer(
        base_path=str(project_root / "ai" / "training_ready" / "datasets")
    )

    # 1. Synthesize Data (Alpaca Format)
    logger.info("Synthesizing dataset (Alpaca format)...")
    dataset = synthesizer.synthesize_dataset(format_type="alpaca")

    if not dataset:
        logger.error("No data synthesized! Check input directories.")
        return

    # 2. Split Data
    logger.info(f"Splitting {len(dataset)} items into Train/Val/Test...")
    splits = synthesizer.split_dataset(dataset, train_ratio=0.8, val_ratio=0.1)

    # 3. Export
    output_dir = synthesizer.output_path
    for split_name, items in splits.items():
        output_file = output_dir / f"final_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Exported {len(items)} items to {output_file}")

    logger.info("Phase 4 Synthesis Completed Successfully.")


if __name__ == "__main__":
    main()
