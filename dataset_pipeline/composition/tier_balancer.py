"""
Tier Balancer

Balances datasets across tiers according to training ratio strategy.
Ensures proper distribution per tier weights.
"""

import logging
import random
from typing import Dict, List, Optional

from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class TierBalancer:
    """
    Balances datasets across tiers according to training ratio strategy.

    Training Ratio Strategy:
    - Tier 1: 40% weight
    - Tier 2: 25% weight
    - Tier 3: 20% weight
    - Tier 4: 10% weight
    - Tier 5: 4% weight
    - Tier 6: 1% weight (reference)
    """

    # Tier weights for training ratio strategy
    TIER_WEIGHTS = {
        1: 0.40,
        2: 0.25,
        3: 0.20,
        4: 0.10,
        5: 0.04,
        6: 0.01,
    }

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize tier balancer.

        Args:
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)

        logger.info("Initialized TierBalancer")

    def balance_datasets(
        self,
        tier_datasets: Dict[int, List[Conversation]],
        target_total: Optional[int] = None,
    ) -> List[Conversation]:
        """
        Balance conversations across tiers according to training ratio strategy.

        Args:
            tier_datasets: Dictionary mapping tier number to list of conversations
            target_total: Optional target total number of conversations

        Returns:
            Balanced list of conversations
        """
        logger.info("Balancing datasets across tiers")

        # Calculate available conversations per tier
        tier_counts = {
            tier: len(convs) for tier, convs in tier_datasets.items()
        }

        total_available = sum(tier_counts.values())
        logger.info(f"Total available conversations: {total_available}")

        # Determine target counts per tier
        if target_total is None:
            target_total = total_available

        tier_targets = {}
        for tier, weight in self.TIER_WEIGHTS.items():
            if tier in tier_datasets:
                target_count = int(target_total * weight)
                available_count = tier_counts[tier]
                # Don't exceed available
                tier_targets[tier] = min(target_count, available_count)
            else:
                tier_targets[tier] = 0

        logger.info(f"Tier targets: {tier_targets}")

        # Sample conversations from each tier
        balanced_conversations = []

        for tier, target_count in tier_targets.items():
            if tier not in tier_datasets:
                continue

            conversations = tier_datasets[tier]

            if target_count >= len(conversations):
                # Use all conversations
                balanced_conversations.extend(conversations)
                logger.info(
                    f"Tier {tier}: Using all {len(conversations)} conversations"
                )
            else:
                # Sample target_count conversations
                sampled = random.sample(conversations, target_count)
                balanced_conversations.extend(sampled)
                logger.info(
                    f"Tier {tier}: Sampled {target_count} from {len(conversations)} conversations"
                )

        logger.info(
            f"Balanced dataset: {len(balanced_conversations)} total conversations"
        )

        return balanced_conversations

    def get_tier_distribution(
        self, conversations: List[Conversation]
    ) -> Dict[int, int]:
        """
        Get distribution of conversations across tiers.

        Args:
            conversations: List of conversations

        Returns:
            Dictionary mapping tier number to count
        """
        distribution = {}

        for conv in conversations:
            tier = conv.metadata.get("tier", 0)
            distribution[tier] = distribution.get(tier, 0) + 1

        return distribution

    def validate_distribution(
        self, conversations: List[Conversation], tolerance: float = 0.05
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Validate that distribution matches training ratio strategy.

        Args:
            conversations: List of conversations
            tolerance: Tolerance for deviation from target weights

        Returns:
            Tuple of (is_valid, validation_details)
        """
        total = len(conversations)
        if total == 0:
            return False, {"error": "No conversations provided"}

        distribution = self.get_tier_distribution(conversations)

        validation_details = {
            "total": total,
            "distribution": distribution,
            "target_weights": self.TIER_WEIGHTS,
            "actual_weights": {},
            "deviations": {},
            "within_tolerance": {},
        }

        all_valid = True

        for tier, target_weight in self.TIER_WEIGHTS.items():
            actual_count = distribution.get(tier, 0)
            actual_weight = actual_count / total if total > 0 else 0.0

            deviation = abs(actual_weight - target_weight)
            within_tolerance = deviation <= tolerance

            validation_details["actual_weights"][tier] = actual_weight
            validation_details["deviations"][tier] = deviation
            validation_details["within_tolerance"][tier] = within_tolerance

            if not within_tolerance:
                all_valid = False
                logger.warning(
                    f"Tier {tier} distribution out of tolerance: "
                    f"target={target_weight:.2%}, actual={actual_weight:.2%}, "
                    f"deviation={deviation:.2%}"
                )

        return all_valid, validation_details

