"""
Tier Processor

Unified system to process all tiers (Tier 1-6) with proper weighting and balancing.
Implements tier-based quality thresholds and tier-weighted sampling strategy.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.dataset_pipeline.ingestion.tier_loaders import (
    Tier1PriorityLoader,
    Tier2ProfessionalLoader,
    Tier3CoTLoader,
    Tier4RedditLoader,
    Tier5ResearchLoader,
    Tier6KnowledgeLoader,
)
from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class TierProcessor:
    """
    Unified processor for Tier 1-6 datasets.

    Provides:
    1. Tier-based quality thresholds
    2. Tier-weighted sampling strategy
    3. Tier distribution optimization
    4. Tier metadata tracking
    """

    # Tier quality thresholds (as defined in plan)
    TIER_QUALITY_THRESHOLDS = {
        1: 0.99,  # Tier 1: 99%
        2: 0.95,  # Tier 2: 95%
        3: 0.90,  # Tier 3: 90%
        4: 0.85,  # Tier 4: 85%
        5: 0.80,  # Tier 5: 80%
        6: 1.0,  # Tier 6: Reference (not scored)
    }

    # Training ratio strategy weights
    TRAINING_RATIO_WEIGHTS = {
        1: 0.40,  # Tier 1: 40% weight
        2: 0.25,  # Tier 2: 25% weight
        3: 0.20,  # Tier 3: 20% weight
        4: 0.10,  # Tier 4: 10% weight
        5: 0.04,  # Tier 5: 4% weight
        6: 0.01,  # Tier 6: 1% weight (reference)
    }

    def __init__(
        self,
        base_path: Optional[Path] = None,
        enable_tier_1: bool = True,
        enable_tier_2: bool = True,
        enable_tier_3: bool = True,
        enable_tier_4: bool = True,
        enable_tier_5: bool = True,
        enable_tier_6: bool = True,
    ):
        """
        Initialize tier processor.

        Args:
            base_path: Optional base path to datasets directory
            enable_tier_1-6: Flags to enable/disable each tier
        """
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")

        # Initialize tier loaders
        self.tier_loaders = {}

        if enable_tier_1:
            self.tier_loaders[1] = Tier1PriorityLoader(
                base_path=self.base_path / "datasets-wendy",
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[1],
            )

        if enable_tier_2:
            self.tier_loaders[2] = Tier2ProfessionalLoader(
                base_path=self.base_path,
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[2],
            )

        if enable_tier_3:
            self.tier_loaders[3] = Tier3CoTLoader(
                base_path=self.base_path,
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[3],
            )

        if enable_tier_4:
            self.tier_loaders[4] = Tier4RedditLoader(
                base_path=self.base_path / "old-datasets",
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[4],
            )

        if enable_tier_5:
            self.tier_loaders[5] = Tier5ResearchLoader(
                base_path=self.base_path,
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[5],
            )

        if enable_tier_6:
            self.tier_loaders[6] = Tier6KnowledgeLoader(
                base_path=self.base_path,
                quality_threshold=self.TIER_QUALITY_THRESHOLDS[6],
            )

        # Processed datasets by tier
        self.processed_datasets: Dict[int, Dict[str, List[Conversation]]] = {}

        logger.info(
            f"Initialized TierProcessor with {len(self.tier_loaders)} tiers enabled"
        )

    def process_all_tiers(self) -> Dict[int, Dict[str, List[Conversation]]]:
        """
        Process all enabled tiers.

        Returns:
            Dictionary mapping tier number to dataset dictionary
        """
        logger.info("Processing all enabled tiers")

        for tier_num, loader in self.tier_loaders.items():
            logger.info(f"Processing Tier {tier_num}")

            try:
                datasets = loader.load_datasets()
                self.processed_datasets[tier_num] = datasets

                total_conversations = sum(len(convs) for convs in datasets.values())
                logger.info(
                    f"Tier {tier_num} complete: {len(datasets)} datasets, "
                    f"{total_conversations} conversations"
                )

            except Exception as e:
                logger.error(
                    f"Error processing Tier {tier_num}: {e}",
                    exc_info=True,
                )
                self.processed_datasets[tier_num] = {}
                continue

        total_all_tiers = sum(
            sum(len(convs) for convs in tier_datasets.values())
            for tier_datasets in self.processed_datasets.values()
        )

        logger.info(
            f"All tiers processing complete: {len(self.processed_datasets)} tiers, "
            f"{total_all_tiers} total conversations"
        )

        return self.processed_datasets

    def process_tier(self, tier_num: int) -> Dict[str, List[Conversation]]:
        """
        Process a specific tier.

        Args:
            tier_num: Tier number (1-6)

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        if tier_num not in self.tier_loaders:
            logger.warning(f"Tier {tier_num} not enabled or not found")
            return {}

        logger.info(f"Processing Tier {tier_num}")

        try:
            loader = self.tier_loaders[tier_num]
            datasets = loader.load_datasets()
            self.processed_datasets[tier_num] = datasets

            total_conversations = sum(len(convs) for convs in datasets.values())
            logger.info(
                f"Tier {tier_num} complete: {len(datasets)} datasets, "
                f"{total_conversations} conversations"
            )

            return datasets

        except Exception as e:
            logger.error(
                f"Error processing Tier {tier_num}: {e}",
                exc_info=True,
            )
            return {}

    def get_tier_quality_threshold(self, tier_num: int) -> float:
        """
        Get quality threshold for a tier.

        Args:
            tier_num: Tier number (1-6)

        Returns:
            Quality threshold (0.0-1.0)
        """
        return self.TIER_QUALITY_THRESHOLDS.get(tier_num, 0.7)

    def get_tier_weight(self, tier_num: int) -> float:
        """
        Get training ratio weight for a tier.

        Args:
            tier_num: Tier number (1-6)

        Returns:
            Weight (0.0-1.0)
        """
        return self.TRAINING_RATIO_WEIGHTS.get(tier_num, 0.0)

    def get_tier_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all processed tiers.

        Returns:
            Dictionary with tier statistics
        """
        stats = {
            "tiers_processed": len(self.processed_datasets),
            "tier_details": {},
            "total_conversations": 0,
        }

        for tier_num, tier_datasets in self.processed_datasets.items():
            tier_total = sum(len(convs) for convs in tier_datasets.values())
            stats["tier_details"][f"tier_{tier_num}"] = {
                "datasets": len(tier_datasets),
                "conversations": tier_total,
                "quality_threshold": self.get_tier_quality_threshold(tier_num),
                "weight": self.get_tier_weight(tier_num),
            }
            stats["total_conversations"] += tier_total

        return stats

    def get_all_conversations(self) -> List[Conversation]:
        """
        Get all conversations from all processed tiers.

        Returns:
            List of all conversations
        """
        all_conversations = []

        for tier_datasets in self.processed_datasets.values():
            for conversations in tier_datasets.values():
                all_conversations.extend(conversations)

        return all_conversations

    def process_sample_tiers(
        self, max_conversations_per_tier: int = 100
    ) -> Dict[int, List[Conversation]]:
        """
        Process a sample from all enabled tiers.

        Args:
            max_conversations_per_tier: Max conversations per tier

        Returns:
            Dictionary mapping tier number to list of sample conversations
        """
        logger.info(
            f"Processing sample tiers (max {max_conversations_per_tier} per tier)"
        )
        samples = {}

        for tier_num, loader in self.tier_loaders.items():
            logger.info(f"Sampling Tier {tier_num}")
            try:
                # Check if loader has load_sample method
                # (it should if inheriting from BaseTierLoader)
                if hasattr(loader, "load_sample"):
                    tier_sample = loader.load_sample(
                        max_conversations=max_conversations_per_tier
                    )
                else:
                    # Fallback
                    datasets = loader.load_datasets()
                    all_convs = []
                    for convs in datasets.values():
                        all_convs.extend(convs)
                    tier_sample = all_convs[:max_conversations_per_tier]

                samples[tier_num] = tier_sample
                # Update processed_datasets for stats
                self.processed_datasets[tier_num] = {"sample": tier_sample}

                logger.info(f"Tier {tier_num} sample: {len(tier_sample)} conversations")

            except Exception as e:
                logger.error(f"Error sampling Tier {tier_num}: {e}", exc_info=True)
                samples[tier_num] = []

        return samples
