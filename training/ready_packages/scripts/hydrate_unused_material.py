#!/usr/bin/env python3
"""Hydrate unused material into the training pipeline.

This script scans the unused_material directory and registers/converts the data
for the Unified Preprocessing Pipeline.
"""

import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
UNUSED_DIR = WORKSPACE_ROOT / "datasets/cache/local/unused_material"
PROCESSED_DIR = WORKSPACE_ROOT / "datasets/cache/local/hydrated"


def hydrate():
    if not UNUSED_DIR.exists():
        logger.warning(f"Unused material directory not found: {UNUSED_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process text files, csv, etc.
    # For now, we move them to hydrated and tag them for discovery
    count = 0
    for file_path in UNUSED_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in (".txt", ".json", ".jsonl", ".csv"):
            target_path = PROCESSED_DIR / file_path.name
            shutil.copy2(file_path, target_path)
            logger.info(f"Hydrated: {file_path.name}")
            count += 1

    logger.info(f"Successfully hydrated {count} files.")


if __name__ == "__main__":
    hydrate()
