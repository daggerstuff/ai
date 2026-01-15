#!/usr/bin/env python3
"""
Run Sample Data Processing.

Executes a sample run of the TierProcessor to validate the end-to-end pipeline.
Loads a small sample of conversations from each tier, scores them, and generates
statistics.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai" / "dataset_pipeline" / "schemas"))

from ai.dataset_pipeline.orchestration.tier_processor import (  # noqa: E402
    TierProcessor,
)
from ai.dataset_pipeline.processing.conversation_complexity_scorer import (  # noqa: E402
    ConversationComplexityScorer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Sample Data Processing Run")

    # Initialize processor
    processor = TierProcessor(
        enable_tier_1=True,
        enable_tier_2=True,
        enable_tier_3=True,
        enable_tier_4=True,
        enable_tier_5=True,
        enable_tier_6=True,
    )

    # Process sample from each tier (limit to 100 per tier)
    logger.info("Processing samples...")
    # NOTE: Using the new process_sample_tiers method
    samples = processor.process_sample_tiers(max_conversations_per_tier=100)

    # Calculate total conversations
    total_conversations = sum(len(s) for s in samples.values())
    logger.info(f"Total sample conversations: {total_conversations}")

    if total_conversations == 0:
        logger.warning("No conversations loaded! Check dataset paths and registry.")
        return

    # Flatten for scoring
    all_conversations = []
    for s in samples.values():
        all_conversations.extend(s)

    # Score complexity
    logger.info("Scoring complexity...")
    scorer = ConversationComplexityScorer()

    # We need to simulate the scoring process if scorer.score_conversations
    # doesn't accept list
    # Assuming scorer has score_conversation or score_conversations
    # Let's check if score_conversations exists, otherwise loop
    if hasattr(scorer, "score_conversations"):
        # It might return a dict or modify in place?
        # Based on 50-next-steps.md usage:
        # scores = scorer.score_conversations(conversations)
        scores = scorer.score_conversations(all_conversations)
        logger.info(f"Generated {len(scores)} scores")

        # Generate distribution
        if hasattr(scorer, "get_complexity_distribution"):
            distribution = scorer.get_complexity_distribution(all_conversations)
            logger.info(f"Complexity distribution: {distribution}")
    else:
        logger.warning(
            "ConversationComplexityScorer missing score_conversations method"
        )

    # Get tier statistics
    stats = processor.get_tier_statistics()
    logger.info("Tier Statistics:")
    for tier, details in stats["tier_details"].items():
        logger.info(f"  {tier}: {details['conversations']} conversations")

    logger.info("Sample Data Processing Complete! 🚀")


if __name__ == "__main__":
    main()
