"""
Test suite for data augmentation pipeline
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_context_expander():
    """Test context expansion"""
    from data_augmentation import ContextExpander

    logger.info("=" * 80)
    logger.info("TEST: Context Expander")
    logger.info("=" * 80)

    expander = ContextExpander()

    # Test record
    record = {
        "prompt_id": "test_001",
        "category": "general",
        "instructions": "This is a test conversation.",
    }

    expanded = expander.expand(record)

    # Verify expansions
    assert "therapeutic_context" in expanded, "Missing therapeutic_context"
    assert "emotional_context" in expanded, "Missing emotional_context"
    assert "therapeutic_goal" in expanded, "Missing therapeutic_goal"
    assert "instructions" in expanded, "Missing instructions"

    # Verify context was added to instructions
    assert len(expanded["instructions"]) > len(record["instructions"]), "Instructions not expanded"

    logger.info("✓ Context expansion working correctly")
    logger.info(f"  Original instructions length: {len(record['instructions'])}")
    logger.info(f"  Expanded instructions length: {len(expanded['instructions'])}")
    logger.info(f"  Therapeutic context: {expanded['therapeutic_context']}")
    logger.info(f"  Emotional context: {expanded['emotional_context']}")
    logger.info(f"  Therapeutic goal: {expanded['therapeutic_goal']}")

    return True


def test_crisis_scenario_generator():
    """Test crisis scenario generation"""
    from data_augmentation import CrisisScenarioGenerator

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Crisis Scenario Generator")
    logger.info("=" * 80)

    generator = CrisisScenarioGenerator()

    # Test record
    record = {
        "prompt_id": "test_001",
        "category": "general",
        "instructions": "This is a test conversation.",
    }

    crisis = generator.generate(record)

    # Verify crisis fields
    assert "scenario_type" in crisis, "Missing scenario_type"
    assert "difficulty" in crisis, "Missing difficulty"
    assert crisis["category"] == "crisis_scenarios", "Category not set to crisis_scenarios"
    assert "metadata" in crisis, "Missing metadata"
    assert crisis["metadata"]["crisis_type"] in CrisisScenarioGenerator.CRISIS_TYPES, (
        "Invalid crisis type"
    )
    assert crisis["metadata"]["difficulty_level"] in CrisisScenarioGenerator.DIFFICULTY_LEVELS, (
        "Invalid difficulty level"
    )

    logger.info("✓ Crisis scenario generation working correctly")
    logger.info(f"  Crisis type: {crisis['scenario_type']}")
    logger.info(f"  Difficulty: {crisis['difficulty']}")
    logger.info(f"  Prompt ID: {crisis['prompt_id']}")

    return True


def test_dialogue_variation_generator():
    """Test dialogue variation generation"""
    from data_augmentation import DialogueVariationGenerator

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Dialogue Variation Generator")
    logger.info("=" * 80)

    generator = DialogueVariationGenerator()

    # Test record
    record = {
        "prompt_id": "test_001",
        "category": "general",
        "instructions": "This is a test conversation.",
    }

    variation = generator.generate_variation(record)

    # Verify variations
    assert "dialogue_variations" in variation, "Missing dialogue_variations"
    assert len(variation["dialogue_variations"]) > 0, "No dialogue variations generated"

    for variation_type in variation["dialogue_variations"]:
        assert variation_type in DialogueVariationGenerator.DIALOGUE_VARIATIONS, (
            f"Invalid variation type: {variation_type}"
        )

    logger.info("✓ Dialogue variation generation working correctly")
    logger.info(f"  Variations generated: {len(variation['dialogue_variations'])}")
    for var_type, var_text in variation["dialogue_variations"].items():
        logger.info(f"    - {var_type}: {var_text}")

    return True


def test_augmentation_pipeline():
    """Test full augmentation pipeline"""
    from data_augmentation import AugmentationConfig, DataAugmentationPipeline

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Data Augmentation Pipeline")
    logger.info("=" * 80)

    config = AugmentationConfig(augmentation_probability=1.0)  # Always augment for testing
    pipeline = DataAugmentationPipeline(config)

    # Test records
    records = [
        {
            "prompt_id": "test_001",
            "category": "general",
            "instructions": "Test conversation 1",
        },
        {
            "prompt_id": "test_002",
            "category": "general",
            "instructions": "Test conversation 2",
        },
    ]

    augmented = pipeline.augment_dataset(records)

    # Verify augmentation
    assert len(augmented) > len(records), "No augmentation occurred"
    assert len(augmented) >= len(records) * 2, "Expected at least 2x augmentation"

    logger.info("✓ Augmentation pipeline working correctly")
    logger.info(f"  Original records: {len(records)}")
    logger.info(f"  Augmented records: {len(augmented)}")
    logger.info(f"  Augmentation ratio: {len(augmented) / len(records):.2f}x")

    return True


def test_augmentation_jsonl_file():
    """Test augmentation on JSONL file"""
    from data_augmentation import AugmentationConfig, DataAugmentationPipeline

    logger.info("\n" + "=" * 80)
    logger.info("TEST: JSONL File Augmentation")
    logger.info("=" * 80)

    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_file = f.name
        for i in range(5):
            record = {
                "prompt_id": f"test_{i:03d}",
                "category": "general",
                "instructions": f"Test conversation {i}",
            }
            f.write(json.dumps(record) + "\n")

    try:
        # Create output file
        output_file = test_file.replace(".jsonl", "_augmented.jsonl")

        # Run augmentation
        config = AugmentationConfig(augmentation_probability=1.0)
        pipeline = DataAugmentationPipeline(config)
        stats = pipeline.augment_jsonl_file(test_file, output_file, sample_size=5)

        # Verify output
        assert Path(output_file).exists(), "Output file not created"

        output_records = []
        with open(output_file) as f:
            for line in f:
                output_records.append(json.loads(line.strip()))

        assert len(output_records) > 5, "No augmentation in output file"

        logger.info("✓ JSONL file augmentation working correctly")
        logger.info(f"  Input records: {stats['original_records']}")
        logger.info(f"  Output records: {stats['augmented_records']}")
        logger.info(f"  Augmentation ratio: {stats['augmentation_ratio']:.2f}x")

        return True

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_semantic_paraphraser():
    """Test semantic paraphrasing"""
    from data_augmentation import SemanticParaphraser

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Semantic Paraphraser")
    logger.info("=" * 80)

    paraphraser = SemanticParaphraser()

    # Test text
    text = "I understand how you feel. That sounds difficult. Tell me more about your experience."

    # Paraphrase with crisis keyword preservation
    paraphrased = paraphraser.paraphrase_text(text, preserve_crisis_keywords=True)

    assert len(paraphrased) > 0, "Paraphrased text is empty"
    logger.info("✓ Semantic paraphrasing working correctly")
    logger.info(f"  Original: {text}")
    logger.info(f"  Paraphrased: {paraphrased}")

    return True


def test_demographic_diversity_injector():
    """Test demographic diversity injection"""
    from data_augmentation import DemographicDiversityInjector

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Demographic Diversity Injector")
    logger.info("=" * 80)

    injector = DemographicDiversityInjector()

    # Test record
    record = {
        "prompt_id": "test_001",
        "category": "general",
        "instructions": "This is a test conversation.",
    }

    diverse = injector.inject_diversity(record)

    # Verify diversity injection
    assert "metadata" in diverse, "Missing metadata"
    assert "demographic_context" in diverse["metadata"], "Missing demographic_context"
    assert "age_group" in diverse["metadata"]["demographic_context"], "Missing age_group"
    assert "cultural_context" in diverse["metadata"]["demographic_context"], (
        "Missing cultural_context"
    )
    assert "socioeconomic" in diverse["metadata"]["demographic_context"], "Missing socioeconomic"

    logger.info("✓ Demographic diversity injection working correctly")
    logger.info(f"  Age group: {diverse['metadata']['demographic_context']['age_group']}")
    logger.info(
        f"  Cultural context: {diverse['metadata']['demographic_context']['cultural_context']}"
    )
    logger.info(f"  Socioeconomic: {diverse['metadata']['demographic_context']['socioeconomic']}")

    return True


def test_enhanced_augmentation_pipeline():
    """Test enhanced augmentation pipeline with all features"""
    from data_augmentation import AugmentationConfig, DataAugmentationPipeline

    logger.info("\n" + "=" * 80)
    logger.info("TEST: Enhanced Augmentation Pipeline")
    logger.info("=" * 80)

    config = AugmentationConfig(
        augmentation_probability=1.0,
        semantic_paraphrase_enabled=True,
        demographic_diversity_enabled=True,
        max_augmentations_per_record=5,
    )
    pipeline = DataAugmentationPipeline(config)

    # Test record
    record = {
        "prompt_id": "test_001",
        "category": "general",
        "instructions": "This is a test conversation about anxiety.",
    }

    augmented = pipeline.augment_record(record)

    # Verify augmentation
    assert len(augmented) > 1, "No augmentation occurred"
    assert augmented[0] == record, "Original record was modified"

    logger.info("✓ Enhanced augmentation pipeline working correctly")
    logger.info(f"  Original record + {len(augmented) - 1} augmented versions")
    for i, aug in enumerate(augmented[1:], 1):
        aug_type = aug.get("metadata", {}).get("augmentation_type", "unknown")
        logger.info(f"    {i}. {aug_type}")

    return True


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("TIER 1.3: Data Augmentation Pipeline Tests")
    logger.info("=" * 80 + "\n")

    tests = [
        test_context_expander,
        test_crisis_scenario_generator,
        test_dialogue_variation_generator,
        test_augmentation_pipeline,
        test_augmentation_jsonl_file,
        test_semantic_paraphraser,
        test_demographic_diversity_injector,
        test_enhanced_augmentation_pipeline,
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
    logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
