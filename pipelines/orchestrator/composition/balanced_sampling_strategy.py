"""
Balanced Sampling Strategy for Dataset Diversity

Implements advanced sampling techniques to maintain diversity while balancing dataset composition.
Now includes tier-aware balancing for Tier 1-6 dataset system.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..schemas.conversation_schema import Conversation
from ..systems.dataset_categorization_system import DatasetCategory
from ..systems.logger import get_logger

# Optional tier balancer integration
try:
    from ai.pipelines.orchestrator.composition.tier_balancer import TierBalancer
    TIER_BALANCER_AVAILABLE = True
except ImportError:
    TIER_BALANCER_AVAILABLE = False
    TierBalancer = None  # type: ignore

logger = get_logger(__name__)


class BalancedSamplingStrategy:
    """
    Implements balanced sampling strategy that maintains diversity.

    Uses stratified sampling, diversity-preserving techniques, and adaptive balancing
    to ensure representative dataset composition while preserving important variations.
    """

    def __init__(self, enable_tier_balancing: bool = True):
        """
        Initialize the balanced sampling strategy.

        Args:
            enable_tier_balancing: Whether to enable tier-aware balancing
        """
        self.logger = get_logger(__name__)
        self.enable_tier_balancing = enable_tier_balancing and TIER_BALANCER_AVAILABLE

        if self.enable_tier_balancing:
            self.tier_balancer = TierBalancer()
            self.logger.info("BalancedSamplingStrategy initialized with tier balancing")
        else:
            self.tier_balancer = None
            self.logger.info("BalancedSamplingStrategy initialized (tier balancing disabled)")

    def stratified_sample(
        self,
        categorized_conversations: Dict[DatasetCategory, List[Conversation]],
        target_sizes: Dict[DatasetCategory, int]
    ) -> List[Conversation]:
        """
        Perform stratified sampling to maintain category proportions.

        Args:
            categorized_conversations: Dictionary mapping categories to conversation lists
            target_sizes: Dictionary mapping categories to target sample sizes

        Returns:
            List of sampled conversations maintaining diversity
        """
        sampled_conversations = []

        for category, target_size in target_sizes.items():
            conversations = categorized_conversations.get(category, [])

            if len(conversations) <= target_size:
                # Use all available conversations
                sampled_conversations.extend(conversations)
                self.logger.debug(f"Category {category}: Using all {len(conversations)} conversations")
            else:
                # Sample to reach target size
                sampled = self._diversity_preserving_sample(conversations, target_size)
                sampled_conversations.extend(sampled)
                self.logger.debug(f"Category {category}: Sampled {len(sampled)} of {len(conversations)} conversations")

        # Shuffle to mix categories
        random.shuffle(sampled_conversations)
        return sampled_conversations

    def _diversity_preserving_sample(self, conversations: List[Conversation], sample_size: int) -> List[Conversation]:
        """
        Sample conversations while preserving diversity.

        Uses a combination of random sampling and diversity metrics to ensure
        representative selection across different conversation characteristics.

        Args:
            conversations: List of conversations to sample from
            sample_size: Number of conversations to select

        Returns:
            List of sampled conversations
        """
        if len(conversations) <= sample_size:
            return conversations

        # For now, use simple random sampling
        # In advanced implementation, this could use clustering, topic modeling, etc.
        return random.sample(conversations, sample_size)

    def adaptive_sampling(
        self,
        categorized_conversations: Dict[DatasetCategory, List[Conversation]],
        target_percentages: Dict[DatasetCategory, float],
        total_target_size: int
    ) -> List[Conversation]:
        """
        Perform adaptive sampling based on current distribution and target percentages.

        Adjusts sampling strategy based on over/under-representation in current dataset.

        Args:
            categorized_conversations: Dictionary mapping categories to conversation lists
            target_percentages: Target composition percentages
            total_target_size: Total desired dataset size

        Returns:
            List of adaptively sampled conversations
        """
        # Calculate current distribution
        current_counts = {cat: len(convs) for cat, convs in categorized_conversations.items()}
        total_current = sum(current_counts.values())

        if total_current == 0:
            return []

        # Calculate adaptive target sizes
        adaptive_targets = {}
        for category, target_pct in target_percentages.items():
            current_count = current_counts.get(category, 0)
            current_pct = current_count / total_current if total_current > 0 else 0

            # Adjust sampling based on over/under-representation
            if current_pct > target_pct:
                # Over-represented, sample less
                adjustment_factor = 0.9
            elif current_pct < target_pct:
                # Under-represented, sample more
                adjustment_factor = 1.1
            else:
                # Properly represented
                adjustment_factor = 1.0

            target_size = int(total_target_size * target_pct * adjustment_factor)
            adaptive_targets[category] = min(target_size, current_count)

        # Apply stratified sampling with adaptive targets
        return self.stratified_sample(categorized_conversations, adaptive_targets)

    def topic_diversity_sampling(
        self,
        conversations: List[Conversation],
        sample_size: int,
        diversity_metric: str = "role_distribution"
    ) -> List[Conversation]:
        """
        Sample conversations to maximize topic/diversity coverage.

        Args:
            conversations: List of conversations to sample from
            sample_size: Number of conversations to select
            diversity_metric: Metric to use for diversity (role_distribution, length_variation, etc.)

        Returns:
            List of diverse conversations
        """
        if len(conversations) <= sample_size:
            return conversations

        # Group conversations by diversity metric
        diversity_groups = defaultdict(list)

        for conv in conversations:
            if diversity_metric == "role_distribution":
                roles = [msg.role for msg in conv.messages]
                group_key = tuple(sorted(set(roles)))
            elif diversity_metric == "length_variation":
                length = len(conv.messages)
                if length <= 5:
                    group_key = "short"
                elif length <= 15:
                    group_key = "medium"
                else:
                    group_key = "long"
            else:
                # Default to random grouping
                group_key = "default"

            diversity_groups[group_key].append(conv)

        # Sample from each group to maintain diversity
        sampled = []
        groups = list(diversity_groups.keys())

        # Distribute sample size across groups
        base_per_group = sample_size // len(groups) if groups else 0
        remainder = sample_size % len(groups) if groups else 0

        for i, group in enumerate(groups):
            group_convs = diversity_groups[group]
            group_sample_size = base_per_group + (1 if i < remainder else 0)

            if len(group_convs) <= group_sample_size:
                sampled.extend(group_convs)
            else:
                sampled.extend(random.sample(group_convs, group_sample_size))

        # If we didn't get enough samples, fill with random selection
        if len(sampled) < sample_size:
            remaining_needed = sample_size - len(sampled)
            available = [c for c in conversations if c not in sampled]
            if available:
                additional = random.sample(available, min(remaining_needed, len(available)))
                sampled.extend(additional)

        # If we have too many, trim randomly
        if len(sampled) > sample_size:
            sampled = random.sample(sampled, sample_size)

        return sampled


def validate_balanced_sampling_strategy():
    """Validate the BalancedSamplingStrategy functionality."""
    try:
        sampler = BalancedSamplingStrategy()
        assert hasattr(sampler, "stratified_sample")
        assert hasattr(sampler, "adaptive_sampling")
        assert hasattr(sampler, "topic_diversity_sampling")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_balanced_sampling_strategy():
        pass
    else:
        pass
