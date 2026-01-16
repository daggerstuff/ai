#!/usr/bin/env python3
"""
Run Sample Data Processing.

Executes a sample run of the TierProcessor to validate the end-to-end pipeline.
Loads a small sample of conversations from each tier, scores them, and generates
detailed statistics and reports.
"""

import json
import logging
import sys
from collections import defaultdict
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


class SampleProcessingRunner:
    """Runner for sample data processing validation."""

    def __init__(self, output_dir: Path = None):
        """Initialize runner."""
        self.output_dir = output_dir or Path("ai/dataset_pipeline/analytics/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run(self, max_samples_per_tier: int = 100):
        """Run the sample processing pipeline."""
        logger.info(f"Starting Sample Processing Run (n={max_samples_per_tier})")

        # 1. Initialize Processor
        processor = TierProcessor(
            enable_tier_1=True,
            enable_tier_2=True,
            enable_tier_3=True,
            enable_tier_4=True,
            enable_tier_5=True,
            enable_tier_6=True,
        )

        # 2. Process Samples
        logger.info("loading samples from tiers...")
        samples = processor.process_sample_tiers(
            max_conversations_per_tier=max_samples_per_tier
        )

        # 3. Analyze Results
        total_conversations = sum(len(s) for s in samples.values())
        logger.info(f"Loaded {total_conversations} total conversations")

        if total_conversations == 0:
            logger.error("No conversations loaded! Check configurations.")
            return

        # 4. Flatten and Score
        all_conversations = []
        tier_counts = {}
        for tier, convs in samples.items():
            all_conversations.extend(convs)
            tier_counts[tier] = len(convs)

        logger.info("Scoring complexity...")
        scorer = ConversationComplexityScorer()
        scores = scorer.score_conversations(all_conversations)

        # 5. Generate Stats
        complexity_dist = scorer.get_complexity_distribution(all_conversations)

        # Calculate average scores per dimension
        dim_totals = defaultdict(float)
        valid_scores = 0
        for score in scores:
            for dim in [
                "emotional_depth",
                "clinical_reasoning",
                "therapeutic_technique",
                "crisis_level",
                "cultural_sensitivity",
                "overall_complexity",
            ]:
                if dim in score:
                    dim_totals[dim] += score[dim]
            valid_scores += 1

        avg_scores = {
            dim: total / valid_scores if valid_scores > 0 else 0.0
            for dim, total in dim_totals.items()
        }

        # 6. Compile Report
        self.results = {
            "total_conversations": total_conversations,
            "max_samples_per_tier": max_samples_per_tier,
            "tier_counts": tier_counts,
            "complexity_distribution": complexity_dist,
            "average_scores": avg_scores,
            "samples_processed": True,
        }

        self._save_report()
        self._print_summary()

    def _save_report(self):
        """Save results to JSON."""
        out_path = self.output_dir / "sample_run_report.json"
        with open(out_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Report saved to {out_path}")

    def _print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("SAMPLE PROCESSING VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Conversations: {self.results['total_conversations']}")

        print("\nTier Breakdown:")
        print(f"{'Tier':<6} {'Count':<10} {'Status':<10}")
        print("-" * 30)
        for tier, count in sorted(self.results["tier_counts"].items()):
            status = "✓ OK" if count > 0 else "⚠ EMPTY"
            print(f"{tier:<6} {count:<10} {status:<10}")

        print("\nComplexity Distribution:")
        for level, count in self.results["complexity_distribution"].items():
            print(f"  {level:<15}: {count}")

        print("\nAverage Scores:")
        for dim, score in self.results["average_scores"].items():
            print(f"  {dim:<25}: {score:.3f}")

        print("\n" + "=" * 60)


def main():
    runner = SampleProcessingRunner()
    runner.run(max_samples_per_tier=100)


if __name__ == "__main__":
    main()
