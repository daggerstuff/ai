"""
6-Tier Dataset Balancer.

Implements the training ratio strategy from TRAINING_DATA.md spec:
- Tier 1: Curated Priority (40%)
- Tier 2: Professional Therapeutic (25%)
- Tier 3: Chain-of-Thought Reasoning (20%)
- Tier 4: Voice/Persona (10%)
- Tier 5: Research & Specialized (4%)
- Tier 6: Knowledge Base (1%)
"""

import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Configuration for a dataset tier."""

    name: str
    quality_threshold: float  # 0-1
    training_weight: float  # 0-1, sum should = 1
    description: str


class TierBalancer:
    """
    Balances datasets across 6 tiers according to training ratio strategy.
    """

    TIERS: dict[str, TierConfig] = {
        "tier1_priority": TierConfig(
            name="Curated Priority",
            quality_threshold=0.99,
            training_weight=0.40,
            description="Gold-standard therapeutic conversations",
        ),
        "tier2_professional": TierConfig(
            name="Professional Therapeutic",
            quality_threshold=0.95,
            training_weight=0.25,
            description="Clinical-grade professional therapy",
        ),
        "tier3_cot_reasoning": TierConfig(
            name="Chain-of-Thought Reasoning",
            quality_threshold=0.90,
            training_weight=0.20,
            description="Advanced therapeutic reasoning patterns",
        ),
        "tier4_voice_persona": TierConfig(
            name="Voice & Persona",
            quality_threshold=0.85,
            training_weight=0.10,
            description="Tim Fletcher, YouTube transcripts, persona data",
        ),
        "tier5_research": TierConfig(
            name="Research & Specialized",
            quality_threshold=0.80,
            training_weight=0.04,
            description="Academic research datasets",
        ),
        "tier6_knowledge": TierConfig(
            name="Knowledge Base",
            quality_threshold=1.0,  # Reference quality
            training_weight=0.01,
            description="DSM-5, psychology concepts, reference",
        ),
    }

    def __init__(self, base_path: Path = Path("ai/training_ready/datasets")):
        self.base_path = base_path
        self.tier_data: dict[str, list[dict]] = {tier: [] for tier in self.TIERS}

    def load_tier(self, tier_name: str) -> list[dict]:
        """Load all data from a tier directory."""
        tier_path = self.base_path / tier_name
        data = []

        if not tier_path.exists():
            logger.warning(f"Tier path not found: {tier_path}")
            return data

        for file_path in tier_path.glob("*.jsonl"):
            with open(file_path) as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for file_path in tier_path.glob("*.json"):
            with open(file_path) as f:
                try:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    else:
                        data.append(content)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(data)} records from {tier_name}")
        return data

    def load_all_tiers(self) -> dict[str, list[dict]]:
        """Load all tier data."""
        for tier_name in self.TIERS:
            self.tier_data[tier_name] = self.load_tier(tier_name)
        return self.tier_data

    def sample_balanced(self, total_samples: int) -> Iterator[tuple[str, dict]]:
        """
        Sample data according to tier weights.

        Yields (tier_name, record) tuples balanced by training weights.
        """
        samples_per_tier = {}

        for tier_name, config in self.TIERS.items():
            target_count = int(total_samples * config.training_weight)
            available = len(self.tier_data.get(tier_name, []))

            if available == 0:
                samples_per_tier[tier_name] = 0
                continue

            # Take min of target and available
            samples_per_tier[tier_name] = min(target_count, available)

        logger.info(f"Sampling distribution: {samples_per_tier}")

        # Yield samples from each tier
        for tier_name, count in samples_per_tier.items():
            tier_samples = self.tier_data.get(tier_name, [])
            if not tier_samples:
                continue

            # Random sample
            selected = random.sample(tier_samples, min(count, len(tier_samples)))
            for record in selected:
                yield (tier_name, record)

    def create_balanced_dataset(self, total_samples: int, output_path: Path) -> dict:
        """
        Create a balanced mixed dataset.

        Returns statistics about the created dataset.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = dict.fromkeys(self.TIERS, 0)

        with open(output_path, "w") as f:
            for tier_name, record in self.sample_balanced(total_samples):
                # Add tier metadata
                record["_tier"] = tier_name
                record["_tier_weight"] = self.TIERS[tier_name].training_weight
                f.write(json.dumps(record) + "\n")
                stats[tier_name] += 1

        total = sum(stats.values())
        logger.info(f"Created balanced dataset with {total} samples at {output_path}")
        logger.info(f"Distribution: {stats}")

        return {
            "output_path": str(output_path),
            "total_samples": total,
            "distribution": stats,
            "weights": {t: c.training_weight for t, c in self.TIERS.items()},
        }

    def get_tier_stats(self) -> dict[str, dict]:
        """Get statistics for each tier."""
        stats = {}
        for tier_name, config in self.TIERS.items():
            data = self.tier_data.get(tier_name, [])
            stats[tier_name] = {
                "name": config.name,
                "count": len(data),
                "target_weight": config.training_weight,
                "quality_threshold": config.quality_threshold,
            }
        return stats


def create_balanced_training_set(total_samples: int = 50000) -> dict:
    """
    Create a balanced training set using the 6-tier system.
    """
    balancer = TierBalancer()
    balancer.load_all_tiers()

    output_path = Path("ai/training_ready/datasets/final_balanced/balanced_train.jsonl")
    return balancer.create_balanced_dataset(total_samples, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = create_balanced_training_set(10000)
    print(json.dumps(result, indent=2))
