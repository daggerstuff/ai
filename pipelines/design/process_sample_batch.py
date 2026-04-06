#!/usr/bin/env python3
"""
Sample Batch Processing Script for Hybrid Taxonomy Classifier

Process a sample batch of therapeutic conversation records using the hybrid
classifier (keyword + NVIDIA NIM GLM4.7 LLM) to validate the Phase 2 implementation.

Usage:
    cd ai/pipelines/design && uv run python process_sample_batch.py --sample-size 50
"""

import argparse

# Direct imports to avoid NeMo dependency in __init__.py
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

script_dir = Path(__file__).parent

# Import taxonomy_classifier directly
taxonomy_spec = importlib.util.spec_from_file_location(
    "taxonomy_classifier", script_dir / "taxonomy_classifier.py"
)
taxonomy_module = importlib.util.module_from_spec(taxonomy_spec)
taxonomy_spec.loader.exec_module(taxonomy_module)

TherapeuticCategory = taxonomy_module.TherapeuticCategory
CategoryClassification = taxonomy_module.CategoryClassification
TaxonomyClassifier = taxonomy_module.TaxonomyClassifier

# Import llm_classifier without triggering service imports
llm_spec = importlib.util.spec_from_file_location(
    "llm_classifier", script_dir / "llm_classifier.py"
)
llm_module = importlib.util.module_from_spec(llm_spec)

# Inject dependencies into llm_module before loading
llm_module.TherapeuticCategory = TherapeuticCategory
llm_module.CategoryClassification = CategoryClassification
llm_spec.loader.exec_module(llm_module)

LLMClassificationConfig = llm_module.LLMClassificationConfig
LLMTaxonomyClassifier = llm_module.LLMTaxonomyClassifier

# Import hybrid_classifier
hybrid_spec = importlib.util.spec_from_file_location(
    "hybrid_classifier", script_dir / "hybrid_classifier.py"
)
hybrid_module = importlib.util.module_from_spec(hybrid_spec)

# Inject dependencies into hybrid_module before loading
hybrid_module.TherapeuticCategory = TherapeuticCategory
hybrid_module.CategoryClassification = CategoryClassification
hybrid_module.TaxonomyClassifier = TaxonomyClassifier
hybrid_module.LLMTaxonomyClassifier = LLMTaxonomyClassifier
hybrid_module.LLMClassificationConfig = LLMClassificationConfig
hybrid_spec.loader.exec_module(hybrid_module)

HybridTaxonomyClassifier = hybrid_module.HybridTaxonomyClassifier
HybridClassificationStats = hybrid_module.HybridClassificationStats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample therapeutic conversations for testing
SAMPLE_CONVERSATIONS = [
    {
        "messages": [
            {
                "role": "user",
                "content": "I keep having these nightmares about the accident. I can't sleep anymore.",
            },
            {
                "role": "assistant",
                "content": "Those sound like flashbacks. Can you tell me more about what happens in these nightmares?",
            },
        ],
        "metadata": {"source": "test_samples"},
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "My husband and I keep fighting about money. We're at our wit's end.",
            },
            {
                "role": "assistant",
                "content": "Financial stress is one of the most common causes of relationship conflict. Let's explore some strategies.",
            },
        ],
        "metadata": {"source": "test_samples"},
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "I've been feeling really down lately. I don't enjoy my hobbies anymore.",
            },
            {
                "role": "assistant",
                "content": "That sounds like you might be experiencing depression. Have you noticed any other changes?",
            },
        ],
        "metadata": {"source": "test_samples"},
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "I've been thinking about ending it all. I just can't take it anymore.",
            },
            {
                "role": "assistant",
                "content": "I'm deeply concerned about what you're saying. I want you to know that you're not alone and there's help available.",
            },
        ],
        "metadata": {"source": "test_samples"},
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "I've been having trouble concentrating at work since the incident.",
            },
            {
                "role": "assistant",
                "content": "Let's talk about what happened and how it's affecting you day to day.",
            },
        ],
        "metadata": {"source": "test_samples"},
    },
]


def create_test_jsonl(sample_size: int, output_path: Path) -> None:
    """
    Create a test JSONL file with sample conversations.

    Args:
        sample_size: Number of sample records to create
        output_path: Path to write the JSONL file
    """
    logger.info(f"Creating test JSONL with {sample_size} records...")

    # Generate samples by cycling through the sample conversations
    records = []
    for i in range(sample_size):
        sample = SAMPLE_CONVERSATIONS[i % len(SAMPLE_CONVERSATIONS)].copy()
        sample["metadata"] = sample.get("metadata", {}).copy()
        sample["metadata"]["record_id"] = i
        records.append(sample)

    # Write to JSONL
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"✓ Created {len(records)} test records in {output_path}")


def analyze_results(stats, output_path: Path) -> None:
    """
    Analyze and display classification results.

    Args:
        stats: HybridClassificationStats from classification
        output_path: Path to output file for detailed results
    """
    print("\n" + "=" * 80)
    print("📊 SAMPLE BATCH PROCESSING RESULTS")
    print("=" * 80)
    print(f"Total records: {stats.total_records:,}")
    print(
        f"Keyword classified: {stats.keyword_classified:,} "
        f"({stats.keyword_classified / stats.total_records * 100:.1f}%)"
    )
    print(
        f"LLM classified: {stats.llm_classified:,} "
        f"({stats.llm_classified / stats.total_records * 100:.1f}%)"
    )
    print(f"Low confidence: {stats.low_confidence:,}")
    print(f"\nAPI Calls:")
    print(f"  LLM API calls: {stats.llm_api_calls:,}")
    print(f"  Estimated cost: ${stats.estimated_cost:.4f} USD")
    print(f"\nConfidence Averages:")
    print(f"  Keyword: {stats.avg_keyword_confidence:.2%}")
    print(f"  LLM: {stats.avg_llm_confidence:.2%}")
    print(f"  Overall: {stats.avg_overall_confidence:.2%}")
    print(f"\nCategory Distribution:")
    for cat, count in sorted(stats.categories.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / stats.total_records * 100) if stats.total_records > 0 else 0
            print(f"  • {cat}: {count:,} ({pct:.1f}%)")
    print("=" * 80)

    # Read a few classified records to show examples
    print("\n📝 Sample Classified Records:")
    print("-" * 80)

    with open(output_path) as f:
        for i, line in enumerate(f, 1):
            if i > 3:  # Show first 3 records
                break
            record = json.loads(line)
            metadata = record.get("metadata", {})
            print(f"\nRecord {i}:")
            print(f"  Category: {metadata.get('category', 'N/A')}")
            print(f"  Confidence: {metadata.get('category_confidence', 0):.2%}")
            print(f"  Method: {metadata.get('classification_method', 'N/A')}")
            print(f"  Reasoning: {metadata.get('category_reasoning', 'N/A')[:100]}...")
            if "messages" in record and record["messages"]:
                print(f"  Text preview: {record['messages'][0]['content'][:80]}...")

    print("\n" + "=" * 80)


def main():
    """Main entry point for sample batch processing."""
    parser = argparse.ArgumentParser(
        description="Process sample batch with hybrid taxonomy classifier"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of sample records to process (default: 20)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSONL file (if not provided, creates test samples)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_batch_results.jsonl"),
        help="Output JSONL file for classified data",
    )
    parser.add_argument(
        "--keyword-threshold",
        type=float,
        default=0.70,
        help="Keyword confidence threshold (lower = more LLM usage)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="z-ai/glm4.7",
        help="NVIDIA NIM model (default: z-ai/glm4.7)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM fallback (keyword-only mode - NOT RECOMMENDED)",
    )

    args = parser.parse_args()

    # Validate NOT keyword-only mode (user explicitly requested hybrid)
    if args.no_llm:
        logger.warning(
            "⚠️  WARNING: LLM disabled. This is keyword-only mode, which was explicitly NOT requested."
        )
        logger.warning(
            "⚠️  User requested hybrid approach with NVIDIA NIM GLM4.7 LLM fallback."
        )

    # Create test data if no input provided
    if args.input:
        input_path = args.input
    else:
        input_path = Path("test_samples.jsonl")
        create_test_jsonl(args.sample_size, input_path)

    # Initialize hybrid classifier
    llm_config = LLMClassificationConfig(model=args.model)

    classifier = HybridTaxonomyClassifier(
        keyword_confidence_threshold=args.keyword_threshold,
        final_confidence_threshold=0.70,
        llm_config=llm_config,
        enable_llm=not args.no_llm,
    )

    # Process the batch
    logger.info(f"\n🚀 Starting hybrid classification with NVIDIA NIM GLM4.7...")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Keyword threshold: {args.keyword_threshold}")
    logger.info(f"LLM model: {args.model}")
    logger.info(f"LLM enabled: {not args.no_llm}")

    stats = classifier.classify_file(
        input_path=input_path, output_path=args.output, max_records=args.sample_size
    )

    # Analyze and display results
    analyze_results(stats, args.output)

    # Clean up test file if we created it
    if not args.input and input_path.exists():
        input_path.unlink()
        logger.info(f"\n✓ Cleaned up test file: {input_path}")

    # Success message
    print("\n✅ SAMPLE BATCH PROCESSING COMPLETE!")
    print(f"✓ Results saved to: {args.output}")
    print(f"✓ LLM API calls made: {stats.llm_api_calls}")
    print(
        f"✓ Hybrid approach working: YES"
        if stats.llm_classified > 0
        else "⚠️  WARNING: No LLM classifications"
    )
    print()


if __name__ == "__main__":
    main()
