#!/usr/bin/env python3
"""Cross-Referencing Validator for Stage 1-5 Training Pipeline.

This validator checks the centralized cache area for data coverage across
all training stages (Foundation, Expertise, Edge Cases, Nuance, Long Running).
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STAGES = {
    "stage1_foundation": ["nemo_synthetic", "hydrated"],
    "stage2_therapeutic_expertise": ["journal_research", "therapeutic_books"],
    "stage3_high_intensity": ["nightmare_fuel", "edge_case_synthetic"],
    "stage4_nuance": ["youtube_transcripts"],
    "stage5_long_running": ["long_running_therapy"],
}

# Find the workspace root (assuming script is in ai/pipelines/orchestrator/)
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
CACHE_BASE = WORKSPACE_ROOT / "ai/training/ready_packages/datasets/cache/local"


class CrossReferenceValidator:
    def __init__(self, cache_root: Path = CACHE_BASE):
        self.cache_root = cache_root

    def validate_coverage(self) -> dict[str, bool]:
        """Validate that each stage has at least one data source with content."""
        report = {}
        logger.info("🔍 Running Cross-Reference Validation...")

        for stage_name, subdirs in STAGES.items():
            has_data = False
            found_paths = []

            for subdir in subdirs:
                target_dir = self.cache_root / subdir
                if target_dir.exists():
                    # Check for any .jsonl or .json files
                    files = list(target_dir.rglob("*.json*"))
                    # Filter out tiny files (likely empty or just headers)
                    real_files = [f for f in files if f.stat().st_size > 100]

                    if real_files:
                        has_data = True
                        found_paths.append(subdir)

            report[stage_name] = has_data
            status = "✅" if has_data else "❌"
            logger.info(
                f"{status} {stage_name}: {', '.join(found_paths) if found_paths else 'MISSING'}"
            )

        return report

    def check_for_gaps(self) -> list[str]:
        """Return a list of stages that are missing data."""
        report = self.validate_coverage()
        gaps = [stage for stage, present in report.items() if not present]
        if not gaps:
            logger.info("✨ All stages have high-fidelity data coverage.")
        else:
            logger.warning(f"⚠️ Gaps detected in: {', '.join(gaps)}")
        return gaps


if __name__ == "__main__":
    validator = CrossReferenceValidator()
    validator.check_for_gaps()
