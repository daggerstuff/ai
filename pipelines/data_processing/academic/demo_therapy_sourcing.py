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

from ai.sourcing.academic.therapy_dataset_sourcing import (
    TherapyDatasetSourcing,
    find_therapy_datasets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_search():
    """Demo: Basic search for therapy datasets"""

    sourcing = TherapyDatasetSourcing()

    # Search for datasets
    datasets = sourcing.search_huggingface(query="therapy conversation mental health", min_turns=20, limit=20)

    for _i, _dataset in enumerate(datasets[:5], 1):
        pass


def demo_filtered_search():
    """Demo: Search with quality and relevance filters"""

    sourcing = TherapyDatasetSourcing()

    # Search
    datasets = sourcing.search_huggingface(query="therapy counseling dialogue", min_turns=15, limit=30)

    # Apply filters
    datasets = sourcing.filter_by_quality(datasets, min_quality=0.6)
    datasets = sourcing.filter_by_therapeutic_relevance(datasets, min_relevance=0.6)
    datasets = sourcing.rank_datasets(datasets)

    for _i, _dataset in enumerate(datasets[:5], 1):
        pass


def demo_full_pipeline():
    """Demo: Full pipeline with report generation"""

    # Use convenience function
    find_therapy_datasets(min_turns=20, min_quality=0.5)


def demo_custom_ranking():
    """Demo: Custom ranking weights"""

    sourcing = TherapyDatasetSourcing()

    # Search
    datasets = sourcing.search_huggingface(query="mental health conversation", min_turns=10, limit=25)

    # Rank with custom weights (prioritize conversation length)
    custom_weights = {
        "quality": 0.2,
        "relevance": 0.2,
        "turns": 0.5,  # Prioritize long conversations
        "popularity": 0.1,
    }

    datasets = sourcing.rank_datasets(datasets, weights=custom_weights)

    for _i, _dataset in enumerate(datasets[:5], 1):
        pass


def main():
    """Run all demos"""

    try:
        demo_basic_search()
        demo_filtered_search()
        demo_full_pipeline()
        demo_custom_ranking()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
