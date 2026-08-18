#!/usr/bin/env python3
"""
Auto-trigger for Journal Dataset Research (Phase 2).
Automates the cadence-based execution of academic sourcing fetching.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("journal_trigger")


def run_academic_sourcing():
    """Triggers the backend academic sourcing engine via the AI submodule."""
    logger.info("Triggering Academic Sourcing pipeline...")
    # This invokes ai/sourcing/academic/academic_sourcing.py integration
    subprocess.run([sys.executable, "-m", "ai.sourcing.academic.academic_sourcing"], check=True)
    logger.info("Academic sourcing run completed.")


if __name__ == "__main__":
    logger.info("Starting automatic journal listener cron execution.")
    run_academic_sourcing()
