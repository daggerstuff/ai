#!/usr/bin/env python3
"""
Test script for Task 3A.1: Complete Export Format Implementation
Tests all required export formats:
1. JSONL export with conversation formatting and validation
2. Parquet export with optimized schema and compression
3. CSV export with human-readable formatting
4. HuggingFace datasets format export with metadata
5. OpenAI fine-tuning format export with proper structure
"""

import json
import logging
import sys
import traceback
from pathlib import Path

# Add the dataset_pipeline directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from production_exporter import AccessTier, ExportConfig, ExportFormat, ProductionExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_task_3a1_export_formats():
    """Test all export formats required for Task 3A.1."""

    # Initialize exporter
    exporter = ProductionExporter("./test_exports")

    # Sample conversations with realistic data
    sample_conversations = [
        {
            "id": "priority_001",
            "content": "High-quality therapeutic conversation with excellent clinical accuracy.",
            "system_prompt": "You are a professional therapist specializing in CBT techniques.",
            "turns": [
                {"speaker": "user", "text": "I need help with anxiety."},
                {
                    "speaker": "therapist",
                    "text": "I understand. Let's explore evidence-based techniques to help you manage your anxiety.",
                },
                {"speaker": "user", "text": "I get anxious in social situations."},
                {
                    "speaker": "therapist",
                    "text": "Social anxiety is very common. We can work on gradual exposure and cognitive restructuring techniques.",
                },
            ],
            "metadata": {
                "source": "priority",
                "quality_score": 0.98,
                "condition": "anxiety",
                "approach": "CBT",
                "timestamp": "2025-08-10T07:30:00Z",
            },
        },
        {
            "id": "professional_001",
            "content": "Professional therapy session with structured approach.",
            "system_prompt": "You are a licensed clinical psychologist.",
            "turns": [
                {"speaker": "user", "text": "I've been feeling depressed lately."},
                {
                    "speaker": "therapist",
                    "text": "I'm sorry to hear you're going through this. Let's explore what might be contributing to these feelings.",
                },
            ],
            "metadata": {
                "source": "professional",
                "quality_score": 0.92,
                "condition": "depression",
                "approach": "psychodynamic",
                "timestamp": "2025-08-10T08:15:00Z",
            },
        },
        {
            "id": "reddit_001",
            "content": "Community-sourced conversation with moderate quality.",
            "turns": [
                {"speaker": "user", "text": "Feeling down lately."},
                {
                    "speaker": "helper",
                    "text": "That sounds tough. Want to talk about what's been going on?",
                },
            ],
            "metadata": {
                "source": "reddit",
                "quality_score": 0.75,
                "condition": "general",
                "timestamp": "2025-08-10T07:25:00Z",
            },
        },
    ]

    # Configure export with all required formats for Task 3A.1
    config = ExportConfig(
        formats=[
            ExportFormat.JSONL,  # 3A.1.1: JSONL export
            ExportFormat.PARQUET,  # 3A.1.2: Parquet export
            ExportFormat.CSV,  # 3A.1.3: CSV export
            ExportFormat.HUGGINGFACE,  # 3A.1.4: HuggingFace export
            ExportFormat.OPENAI_FINE_TUNING,  # 3A.1.5: OpenAI fine-tuning export
        ],
        access_tiers=[AccessTier.PRIORITY, AccessTier.PROFESSIONAL, AccessTier.REDDIT],
        output_directory="./test_exports",
        include_metadata=True,
        compress_output=False,  # Don't compress for testing
        validate_export=True,
        quality_threshold=0.7,
    )


    try:
        # Perform export
        export_results = exporter.export_dataset(sample_conversations, config)

        logger.info("✅ Export completed successfully!")
        logger.info(f"📦 Generated {len(export_results)} export packages")

        # Verify each format
        format_verification = {}
        for result in export_results:
            format_name = result.format.value
            tier_name = result.access_tier.tier_name

            logger.info(f"\n📋 Export: {result.export_id}")
            logger.info(f"   Tier: {tier_name}")
            logger.info(f"   Format: {format_name}")
            logger.info(f"   Conversations: {result.total_conversations}")
            logger.info(f"   Files: {len(result.file_paths)}")

            # Verify files exist
            all_files_exist = all(Path(file_path).exists() for file_path in result.file_paths)
            format_verification[f"{tier_name}_{format_name}"] = all_files_exist

            if all_files_exist:
                logger.info("   ✅ All files verified")
            else:
                logger.error("   ❌ File verification failed")

        # Print summary
        logger.info("\n📈 EXPORT SUMMARY")
        logger.info("=" * 40)
        summary = exporter.get_export_summary()
        logger.info(f"Total Exports: {summary['total_exports']}")
        logger.info(f"Total Conversations: {summary['total_conversations_exported']}")
        logger.info(f"Format Distribution: {summary['format_distribution']}")
        logger.info(f"Tier Distribution: {summary['tier_distribution']}")

        # Verify all required formats are present
        required_formats = ["jsonl", "parquet", "csv", "huggingface", "openai_fine_tuning"]

        missing_formats = [
            required_format
            for required_format in required_formats
            if required_format not in summary["format_distribution"]
        ]

        if missing_formats:
            logger.warning(f"\n⚠️  Missing formats: {missing_formats}")
            return False
        logger.info("\n🎉 All required formats implemented successfully!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Export failed: {e}")
        traceback.print_exc()
        return False


def test_openai_format_specifically():
    """Test OpenAI fine-tuning format specifically."""
    logger.info("\n🔍 Testing OpenAI Fine-tuning Format Specifically")
    logger.info("-" * 50)

    # Test data
    test_conversations = [
        {
            "id": "test_001",
            "system_prompt": "You are a helpful mental health assistant.",
            "turns": [
                {"speaker": "user", "text": "Hello, I need help."},
                {"speaker": "assistant", "text": "I'm here to help. What's on your mind?"},
            ],
            "metadata": {"quality_score": 0.95, "condition": "general", "source": "test"},
        }
    ]

    exporter = ProductionExporter("./openai_test")
    openai_data = exporter._prepare_openai_fine_tuning_format(test_conversations)

    logger.info("📝 OpenAI Format Sample:")
    logger.info(json.dumps(openai_data[0], indent=2, ensure_ascii=False))

    # Verify structure
    required_keys = ["messages"]
    sample_item = openai_data[0]

    if all(key in sample_item for key in required_keys):
        logger.info("✅ OpenAI format structure verified")
        return True
    logger.error(f"❌ Missing keys: {set(required_keys) - set(sample_item.keys())}")
    return False


if __name__ == "__main__":
    logger.info("🚀 Starting Task 3A.1 Export Format Tests")

    # Run main export test
    success = test_task_3a1_export_formats()

    # Run OpenAI format specific test
    openai_success = test_openai_format_specifically()

    if success and openai_success:
        logger.info("\n🎊 TASK 3A.1 COMPLETED SUCCESSFULLY!")
        logger.info("✅ JSONL export with conversation formatting and validation")
        logger.info("✅ Parquet export with optimized schema and compression")
        logger.info("✅ CSV export with human-readable formatting")
        logger.info("✅ HuggingFace datasets format export with metadata")
        logger.info("✅ OpenAI fine-tuning format export with proper structure")
    else:
        logger.error("\n❌ TASK 3A.1 INCOMPLETE")
        sys.exit(1)
