#!/usr/bin/env python3
"""
Orchestration script for Phase 2: Processing & Validation.
Loads data from Phase 1, runs PII scrubbing, normalization, and quality checks.
Outputs clean data to ai/training_ready/datasets/stage4_cleaning.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.pipelines.orchestrator.processing.data_normalizer import DataNormalizer
from ai.pipelines.orchestrator.processing.pii_scrubber import PIIScrubber
from ai.pipelines.orchestrator.processing.quality_validator import QualityValidator

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase2_Orchestrator")


def load_json_files(directory: Path) -> list[dict]:
    """Loads all JSON files from a directory into a single list."""
    data = []
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return data

    for file_path in directory.rglob("*.json"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                elif isinstance(content, dict):
                    data.append(content)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    return data


def save_json(data: list[dict], output_path: Path, filename: str):
    """Saves data to a JSON file."""
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} records to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")


def main():
    logger.info("Initializing Phase 2 Processing & Validation...")

    # Define Paths
    training_ready_datasets = project_root / "ai" / "training_ready" / "datasets"
    input_path = training_ready_datasets / "stage2_reasoning"

    # We might also want to process stage3_edge if it exists
    input_paths = [
        training_ready_datasets / "stage2_reasoning" / "academic_literature",
        training_ready_datasets / "stage2_reasoning" / "clinical_instruments",
        training_ready_datasets / "stage3_edge" / "crisis_scenarios",
        training_ready_datasets / "stage2_reasoning" / "therapeutic_conversations",
    ]

    output_path = training_ready_datasets / "stage4_cleaning"

    # Initialize Processors
    pii_scrubber = PIIScrubber()
    normalizer = DataNormalizer()
    validator = QualityValidator()

    total_processed = 0

    for path in input_paths:
        if not path.exists():
            continue

        logger.info(f"Processing data from: {path.name}")

        # 1. Load Data
        raw_data = load_json_files(path)
        if not raw_data:
            logger.info(f"No data found in {path.name}. Skipping.")
            continue

        # 2. PII Scrubbing
        scrubbed_data = pii_scrubber.scrub_dataset(raw_data)

        # 3. Normalization
        normalized_data = normalizer.standardize_keys(scrubbed_data)

        # 4. Validation
        valid_data, invalid_data = validator.validate_dataset(normalized_data)

        # 5. Export
        batch_name = f"processed_{path.name}.json"
        save_json(valid_data, output_path, batch_name)

        if invalid_data:
            save_json(invalid_data, output_path / "quarantine", f"invalid_{path.name}.json")

        total_processed += len(valid_data)

    logger.info(f"Phase 2 Processing Completed. Total valid records generated: {total_processed}")


if __name__ == "__main__":
    main()
