"""
Integration test for data augmentation pipeline with real merged dataset
Tests augmentation on actual training data from ai/lightning/processed_data/merged_dataset.jsonl
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_augmentation import AugmentationConfig, DataAugmentationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_augmentation_on_merged_dataset():
    """Test augmentation on actual merged dataset"""
    logger.info("=" * 80)
    logger.info("INTEGRATION TEST: Augmentation on Merged Dataset")
    logger.info("=" * 80)

    dataset_path = Path(__file__).parent.parent.parent / "lightning" / "processed_data" / "merged_dataset.jsonl"

    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return False

    logger.info(f"Loading dataset from {dataset_path}")

    # Load sample records
    records = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample 100 records for testing
                break
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {i + 1}: {e}")

    logger.info(f"Loaded {len(records)} sample records")

    # Create augmentation pipeline with all features enabled
    config = AugmentationConfig(
        augmentation_probability=0.8,
        context_expansion_enabled=True,
        crisis_scenario_generation_enabled=True,
        dialogue_variation_enabled=True,
        semantic_paraphrase_enabled=True,
        demographic_diversity_enabled=True,
        preserve_crisis_keywords=True,
        max_augmentations_per_record=3,
    )

    pipeline = DataAugmentationPipeline(config)

    # Augment dataset
    logger.info("Starting augmentation...")
    augmented_records = pipeline.augment_dataset(records)

    logger.info(f"Augmentation complete:")
    logger.info(f"  Original records: {len(records)}")
    logger.info(f"  Augmented records: {len(augmented_records)}")
    logger.info(f"  Augmentation ratio: {len(augmented_records) / len(records):.2f}x")
    logger.info(f"  Records added: {len(augmented_records) - len(records)}")

    # Analyze augmentation types
    augmentation_types = {}
    for record in augmented_records[len(records):]:  # Skip original records
        aug_type = record.get("metadata", {}).get("augmentation_type", "unknown")
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1

    logger.info("\nAugmentation type distribution:")
    for aug_type, count in sorted(augmentation_types.items()):
        logger.info(f"  {aug_type}: {count}")

    # Verify augmentation quality
    logger.info("\nSample augmented records:")
    for i, record in enumerate(augmented_records[len(records):len(records) + 3]):
        logger.info(f"\n  Record {i + 1}:")
        logger.info(f"    Category: {record.get('category', 'N/A')}")
        logger.info(f"    Scenario type: {record.get('scenario_type', 'N/A')}")
        logger.info(f"    Augmentation type: {record.get('metadata', {}).get('augmentation_type', 'N/A')}")
        if "instructions" in record:
            instructions = record["instructions"][:100] + "..." if len(record["instructions"]) > 100 else record["instructions"]
            logger.info(f"    Instructions: {instructions}")

    return True


def test_augmentation_statistics():
    """Test augmentation statistics and metrics"""
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST: Augmentation Statistics")
    logger.info("=" * 80)

    dataset_path = Path(__file__).parent.parent.parent / "lightning" / "processed_data" / "merged_dataset.jsonl"

    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return False

    # Load records by category
    categories = {}
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 500:  # Sample 500 records
                break
            try:
                record = json.loads(line.strip())
                category = record.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
            except json.JSONDecodeError:
                pass

    logger.info("Dataset category distribution:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {category}: {count}")

    # Test augmentation on different categories
    config = AugmentationConfig(augmentation_probability=1.0)
    pipeline = DataAugmentationPipeline(config)

    logger.info("\nAugmentation impact by category:")
    for category in list(categories.keys())[:3]:  # Test first 3 categories
        records = []
        with open(dataset_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 500:
                    break
                try:
                    record = json.loads(line.strip())
                    if record.get("category") == category:
                        records.append(record)
                        if len(records) >= 10:
                            break
                except json.JSONDecodeError:
                    pass

        if records:
            augmented = pipeline.augment_dataset(records)
            ratio = len(augmented) / len(records)
            logger.info(f"  {category}: {len(records)} → {len(augmented)} ({ratio:.2f}x)")

    return True


def main():
    """Run integration tests"""
    logger.info("\n" + "=" * 80)
    logger.info("TIER 1.3: Data Augmentation Integration Tests")
    logger.info("=" * 80 + "\n")

    tests = [
        test_augmentation_on_merged_dataset,
        test_augmentation_statistics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            failed += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

