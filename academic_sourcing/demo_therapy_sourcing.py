#!/usr/bin/env python3
"""
Demo script for Therapy Dataset Sourcing

This script demonstrates how to find high-quality therapy conversation datasets
from HuggingFace Hub with specific criteria like 20+ turn conversations.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.academic_sourcing.therapy_dataset_sourcing import (
    TherapyDatasetSourcing,
    find_therapy_datasets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_search():
    """Demo: Basic search for therapy datasets"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Search for Therapy Datasets")
    print("=" * 70)

    sourcing = TherapyDatasetSourcing()

    # Search for datasets
    datasets = sourcing.search_huggingface(
        query="therapy conversation mental health", min_turns=20, limit=20
    )

    print(f"\nFound {len(datasets)} datasets")
    for i, dataset in enumerate(datasets[:5], 1):
        print(f"\n{i}. {dataset.name}")
        print(f"   Turns: {dataset.avg_turns:.0f if dataset.avg_turns else 'unknown'}")
        print(f"   Quality: {dataset.quality_score:.2f}")
        print(f"   Relevance: {dataset.therapeutic_relevance:.2f}")
        print(f"   Downloads: {dataset.downloads}")


def demo_filtered_search():
    """Demo: Search with quality and relevance filters"""
    print("\n" + "=" * 70)
    print("DEMO 2: Filtered Search (High Quality + High Relevance)")
    print("=" * 70)

    sourcing = TherapyDatasetSourcing()

    # Search
    datasets = sourcing.search_huggingface(
        query="therapy counseling dialogue", min_turns=15, limit=30
    )

    # Apply filters
    datasets = sourcing.filter_by_quality(datasets, min_quality=0.6)
    datasets = sourcing.filter_by_therapeutic_relevance(datasets, min_relevance=0.6)
    datasets = sourcing.rank_datasets(datasets)

    print(f"\nFound {len(datasets)} high-quality, relevant datasets")
    for i, dataset in enumerate(datasets[:5], 1):
        turns_info = f"{dataset.avg_turns:.0f}" if dataset.avg_turns else "?"
        print(f"\n{i}. {dataset.name}")
        print(f"   Composite Score: {dataset.quality_score:.2f}")
        print(f"   Turns: {turns_info} | Downloads: {dataset.downloads}")
        print(f"   URL: {dataset.url}")


def demo_full_pipeline():
    """Demo: Full pipeline with report generation"""
    print("\n" + "=" * 70)
    print("DEMO 3: Full Pipeline (20+ Turns, Quality ≥0.5)")
    print("=" * 70)

    # Use convenience function
    datasets = find_therapy_datasets(min_turns=20, min_quality=0.5)

    print(f"\n✅ Pipeline complete! Found {len(datasets)} datasets")
    print(
        "📁 Results saved to: ai/training_ready/datasets/sourced/therapy_datasets.json"
    )


def demo_custom_ranking():
    """Demo: Custom ranking weights"""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Ranking (Prioritize Long Conversations)")
    print("=" * 70)

    sourcing = TherapyDatasetSourcing()

    # Search
    datasets = sourcing.search_huggingface(
        query="mental health conversation", min_turns=10, limit=25
    )

    # Rank with custom weights (prioritize conversation length)
    custom_weights = {
        "quality": 0.2,
        "relevance": 0.2,
        "turns": 0.5,  # Prioritize long conversations
        "popularity": 0.1,
    }

    datasets = sourcing.rank_datasets(datasets, weights=custom_weights)

    print("\nTop 5 datasets (prioritizing conversation length):")
    for i, dataset in enumerate(datasets[:5], 1):
        turns_info = f"{dataset.avg_turns:.0f}" if dataset.avg_turns else "unknown"
        print(f"\n{i}. {dataset.name}")
        print(f"   Turns: {turns_info}")
        print(f"   Score: {dataset.quality_score:.2f}")


def main():
    """Run all demos"""
    print("\n🎯 Therapy Dataset Sourcing - Demo Suite")
    print("=" * 70)

    try:
        demo_basic_search()
        demo_filtered_search()
        demo_full_pipeline()
        demo_custom_ranking()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
