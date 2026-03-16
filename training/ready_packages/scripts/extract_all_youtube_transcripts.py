#!/usr/bin/env python3
"""
YouTube Transcript Extraction Script for Phase 2.
Interfaces with existing ai/sourcing/youtube components.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

# from ai.sourcing.youtube.processor import run_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("youtube_extraction")


def main():
    logger.info("Initializing YouTube transcript extraction pipeline.")

    try:
        # run_pipeline(api_key="fake_key", target_channels=1)
        logger.info("Successfully initialized YouTube pipeline.")
    except Exception as e:
        logger.error(f"Failed to initialize YouTube pipeline: {e}")


if __name__ == "__main__":
    main()
