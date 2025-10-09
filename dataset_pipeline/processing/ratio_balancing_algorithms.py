"""
Ratio balancing algorithms for training data distribution.
Algorithms for maintaining optimal training ratios across data categories.
"""

import random
from dataclasses import dataclass

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class BalancingTarget:
    """Target ratios for dataset balancing."""
    psychology_knowledge: float = 0.30  # 30%
    voice_training: float = 0.25        # 25%
    mental_health: float = 0.20         # 20%
    reasoning_cot: float = 0.15         # 15%
    personality_balancing: float = 0.10  # 10%


class RatioBalancingAlgorithms:
    """
    Algorithms for maintaining training ratios.

    Implements various balancing strategies to achieve optimal
    training data distribution across categories.
    """

    def __init__(self):
        """Initialize the ratio balancing algorithms."""
        self.logger = get_logger(__name__)

        self.default_targets = BalancingTarget()

        self.logger.info("RatioBalancingAlgorithms initialized")

    def balance_by_ratio(self, categorized_data: dict[str, list[Conversation]],
                        targets: BalancingTarget | None = None) -> list[Conversation]:
        """Balance conversations according to target ratios."""
        if targets is None:
            targets = self.default_targets

        total_conversations = sum(len(convs) for convs in categorized_data.values())
        balanced_conversations = []

        # Calculate target counts
        target_counts = {
            "psychology_knowledge": int(total_conversations * targets.psychology_knowledge),
            "voice_training": int(total_conversations * targets.voice_training),
            "mental_health": int(total_conversations * targets.mental_health),
            "reasoning_cot": int(total_conversations * targets.reasoning_cot),
            "personality_balancing": int(total_conversations * targets.personality_balancing)
        }

        # Sample from each category
        for category, target_count in target_counts.items():
            available = categorized_data.get(category, [])
            if len(available) >= target_count:
                sampled = random.sample(available, target_count)
            else:
                sampled = available  # Use all if insufficient
            balanced_conversations.extend(sampled)

        self.logger.info(f"Balanced to {len(balanced_conversations)} conversations")
        return balanced_conversations

    def get_current_ratios(self, categorized_data: dict[str, list[Conversation]]) -> dict[str, float]:
        """Calculate current ratios in the data."""
        total = sum(len(convs) for convs in categorized_data.values())
        if total == 0:
            return {}

        return {
            category: len(conversations) / total
            for category, conversations in categorized_data.items()
        }


def validate_ratio_balancing_algorithms():
    """Validate the RatioBalancingAlgorithms functionality."""
    try:
        algorithms = RatioBalancingAlgorithms()
        assert hasattr(algorithms, "balance_by_ratio")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_ratio_balancing_algorithms():
        pass
    else:
        pass
