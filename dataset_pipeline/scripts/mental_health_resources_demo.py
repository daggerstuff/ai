#!/usr/bin/env python3
"""
Mental Health Resources Demo Script

Demonstrates usage of all the new mental health dataset and validation components.
Run this script to verify everything is working correctly.

Usage:
    cd /home/vivi/pixelated
    uv run python ai/dataset_pipeline/scripts/mental_health_resources_demo.py
"""

import logging
import sys
from pathlib import Path

# Setup path
script_path = Path(__file__).parent
pipeline_root = script_path.parent.parent
sys.path.insert(0, str(pipeline_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_huggingface_loader():
    """Demo HuggingFace mental health dataset loading."""
    print("\n" + "=" * 60)
    print("1. HuggingFace Mental Health Dataset Loader")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.ingestion.tier_loaders import (
            HuggingFaceMentalHealthLoader,
            HUGGINGFACE_MENTAL_HEALTH_DATASETS,
        )

        print("\nAvailable datasets:")
        for key, config in HUGGINGFACE_MENTAL_HEALTH_DATASETS.items():
            print(f"  - {key}: {config.dataset_id}")
            print(f"    Strategy: {config.conversion_strategy}")

        # Initialize loader (don't actually download yet)
        loader = HuggingFaceMentalHealthLoader(
            datasets_to_load=["mental_health_preprocessed"]  # Smallest dataset
        )
        print(f"\n✓ Loader initialized successfully")
        print(f"  Datasets to load: {loader.datasets_to_load}")

        # Uncomment to actually load data:
        # print("\nLoading dataset (this may take a moment)...")
        # datasets = loader.load_datasets()
        # for name, convs in datasets.items():
        #     print(f"  {name}: {len(convs)} conversations")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_synthetic_distillation():
    """Demo synthetic data distillation pipeline."""
    print("\n" + "=" * 60)
    print("2. Synthetic Data Distillation Pipeline")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.generation.synthetic_data_distillation import (
            SyntheticDataDistillationPipeline,
            SyntheticGenerationConfig,
            DistillationStrategy,
            create_distillation_pipeline,
        )

        print("\nAvailable strategies:")
        for strategy in DistillationStrategy:
            print(f"  - {strategy.value}")

        # Create pipeline (won't generate without LLM)
        config = SyntheticGenerationConfig(
            strategy=DistillationStrategy.MULTI_STEP_PROMPTING,
            model_provider="ollama",
            teacher_model="llama3.2",
        )

        print(f"\n✓ Pipeline configured successfully")
        print(f"  Strategy: {config.strategy.value}")
        print(f"  Provider: {config.model_provider}")
        print(f"  Model: {config.teacher_model}")

        # Note: Actual generation requires Ollama running
        print("\nTo generate data, ensure Ollama is running:")
        print("  ollama serve")
        print("  ollama pull llama3.2")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_empathy_validator():
    """Demo empathy scoring framework."""
    print("\n" + "=" * 60)
    print("3. Empathy Mental Health Validator")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.quality.empathy_mental_health_validator import (
            EmpathyMentalHealthValidator,
            EmpathyLevel,
        )
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

        # Create validator
        validator = EmpathyMentalHealthValidator(min_empathy_threshold=0.5)

        # Test conversation
        test_conv = Conversation(
            conversation_id="demo_001",
            source="demo",
            messages=[
                Message(
                    role="user",
                    content="I've been feeling really anxious lately and I can't sleep."
                ),
                Message(
                    role="assistant",
                    content="I'm so sorry to hear you've been struggling with anxiety. "
                    "That sounds really overwhelming, especially when it's affecting your sleep. "
                    "Can you tell me more about what's been on your mind?"
                ),
            ],
        )

        # Validate
        assessment = validator.validate_conversation(test_conv)

        print("\nTest Conversation Empathy Assessment:")
        print(f"  Overall Score: {assessment.overall_empathy_score}")
        print(f"  Level: {assessment.empathy_level.name}")
        print(f"  Emotional Reactions: {assessment.average_emotional_reactions}")
        print(f"  Interpretations: {assessment.average_interpretations}")
        print(f"  Explorations: {assessment.average_explorations}")
        print(f"  Validation: {assessment.average_validation}")

        if assessment.strengths:
            print(f"\n  Strengths:")
            for s in assessment.strengths:
                print(f"    ✓ {s}")

        if assessment.recommendations:
            print(f"\n  Recommendations:")
            for r in assessment.recommendations:
                print(f"    → {r}")

        print(f"\n✓ Empathy validator working correctly")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_dpo_loader():
    """Demo DPO dataset loader."""
    print("\n" + "=" * 60)
    print("4. DPO (Direct Preference Optimization) Loader")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.ingestion.tier_loaders import (
            DPODatasetLoader,
            DPODatasetType,
            DPO_DATASETS,
        )

        print("\nAvailable DPO datasets:")
        for key, config in DPO_DATASETS.items():
            print(f"  - {key}: {config.dataset_id}")
            print(f"    Type: {config.dataset_type.value}")
            print(f"    Use case: {config.use_case}")

        # Initialize loader
        loader = DPODatasetLoader()

        print(f"\n✓ DPO loader initialized")
        print(f"  Quality threshold: {loader.quality_threshold}")

        # Show dataset types
        print("\nDataset types:")
        for dtype in DPODatasetType:
            datasets = loader.get_datasets_by_type(dtype)
            print(f"  {dtype.value}: {len(datasets)} datasets")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_dpo_training_style():
    """Demo DPO training style configuration."""
    print("\n" + "=" * 60)
    print("5. DPO Training Style Configuration")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.training_styles import (
            TrainingStyleManager,
            TrainingStyle,
            DPOConfig,
        )
        from typing import cast

        manager = TrainingStyleManager()

        # Check DPO is registered
        print("\nTraining styles available:")
        for style in TrainingStyle:
            print(f"  - {style.value}")

        # Create DPO config
        config = cast(DPOConfig, manager.create_config(
            TrainingStyle.DPO,
            name="therapeutic_dpo_demo",
            beta=0.1,
            loss_type="sigmoid",
        ))

        print(f"\n✓ DPO config created")
        print(f"  Name: {config.name}")
        print(f"  Beta: {config.beta}")
        print(f"  Loss type: {config.loss_type}")
        print(f"  Reference free: {config.reference_free}")
        print(f"  Max prompt length: {config.max_prompt_length}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_memo_summarizer():
    """Demo MEMO counseling summarization."""
    print("\n" + "=" * 60)
    print("6. MEMO Counseling Summarization")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.ingestion.memo_counseling_dataset import (
            MEMODatasetLoader,
            CounselingSummarizer,
        )
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

        # Check MEMO access
        loader = MEMODatasetLoader()
        access = loader.check_access()

        print(f"\nMEMO Dataset Status:")
        print(f"  Has Access: {access['has_access']}")
        print(f"  Path: {access['dataset_path']}")

        if not access['has_access']:
            print(f"\n  ⚠ MEMO dataset requires academic access request")
            print(f"    Visit: https://github.com/LCS2-IIITD/MEMO")

        # Demo summarizer (works without MEMO data)
        summarizer = CounselingSummarizer()

        test_conv = Conversation(
            conversation_id="demo_session",
            source="demo",
            messages=[
                Message(role="user", content="I've been having trouble at work."),
                Message(role="assistant", content="Tell me more about what's happening at work."),
                Message(role="user", content="My boss keeps criticizing me and I feel worthless."),
                Message(role="assistant", content="It sounds like the criticism is really affecting your self-worth. That's understandable."),
            ],
        )

        summary = summarizer.summarize_conversation(test_conv)

        print(f"\n✓ Summarizer working (rule-based fallback)")
        print(f"  Summary type: {summary['summary_type']}")
        print(f"  Summary: {summary['summary'][:200]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_safety_validator():
    """Demo safety alignment validator."""
    print("\n" + "=" * 60)
    print("7. Safety Alignment Validator")
    print("=" * 60)

    try:
        from ai.dataset_pipeline.quality.safety_alignment_validator import (
            SafetyAlignmentValidator,
            SafetySeverity,
        )
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message

        # Create validator
        validator = SafetyAlignmentValidator(safety_threshold=0.85)

        print(f"\nSafety rules: {len(validator.rules)}")
        print("Rule categories:")
        categories = {}
        for rule in validator.rules:
            cat = rule.violation_type.value
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in categories.items():
            print(f"  - {cat}: {count} rules")

        # Test safe conversation
        safe_conv = Conversation(
            conversation_id="safe_demo",
            source="demo",
            messages=[
                Message(role="user", content="I've been feeling stressed about work."),
                Message(role="assistant", content="That sounds challenging. Can you tell me more about what's causing the stress?"),
            ],
        )

        safe_assessment = validator.validate_conversation(safe_conv)
        print(f"\nSafe conversation test:")
        print(f"  Is Safe: {safe_assessment.is_safe}")
        print(f"  Score: {safe_assessment.safety_score}")
        print(f"  Violations: {len(safe_assessment.violations)}")

        # Test conversation with crisis content
        crisis_conv = Conversation(
            conversation_id="crisis_demo",
            source="demo",
            messages=[
                Message(role="user", content="I don't want to live anymore."),
                Message(role="assistant", content="I'm deeply concerned about what you just shared. "
                        "If you're having thoughts of ending your life, please call 988 right now. "
                        "You matter and help is available. Can you tell me more about what's going on?"),
            ],
        )

        crisis_assessment = validator.validate_conversation(crisis_conv)
        print(f"\nCrisis content test:")
        print(f"  Is Safe: {crisis_assessment.is_safe}")
        print(f"  Score: {crisis_assessment.safety_score}")
        print(f"  Severity: {crisis_assessment.overall_severity.name}")
        print(f"  Requires Review: {crisis_assessment.requires_human_review}")

        if crisis_assessment.violations:
            print(f"  Violations detected:")
            for v in crisis_assessment.violations[:2]:
                print(f"    - {v.rule_id}: {v.violation_type.value}")

        print(f"\n✓ Safety validator working correctly")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demos."""
    print("=" * 60)
    print("Mental Health Resources Integration Demo")
    print("=" * 60)
    print("\nThis script demonstrates all newly integrated components.")
    print("It runs validation tests without downloading large datasets.")

    results = {
        "HuggingFace Loader": demo_huggingface_loader(),
        "Synthetic Distillation": demo_synthetic_distillation(),
        "Empathy Validator": demo_empathy_validator(),
        "DPO Loader": demo_dpo_loader(),
        "DPO Training Style": demo_dpo_training_style(),
        "MEMO Summarizer": demo_memo_summarizer(),
        "Safety Validator": demo_safety_validator(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for component, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {component}")

    print(f"\n{passed}/{total} components working correctly")

    if passed == total:
        print("\n✅ All components ready!")
        print("\nNext steps:")
        print("  1. Load datasets: python huggingface_mental_health_loader.py")
        print("  2. Generate synthetic data: Start Ollama, then run distillation")
        print("  3. Request MEMO access: https://github.com/LCS2-IIITD/MEMO")
        print("  4. See full guide: ai/dataset_pipeline/MENTAL_HEALTH_RESOURCES_GUIDE.md")
    else:
        print("\n⚠ Some components need attention. Check errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

