#!/usr/bin/env python3
"""
Example: Using TierProcessor with PipelineOrchestrator

This example demonstrates how to use the integrated tier processing system
with the pipeline orchestrator.
"""

import asyncio
from pathlib import Path

from ai.pipelines.orchestrator.orchestration.pipeline_orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
)
from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root


async def main():
    """Example usage of tier-integrated pipeline orchestrator."""

    # Configure pipeline with tier processing enabled
    config = PipelineConfig(
        output_directory=get_dataset_pipeline_output_root() / "processed",
        quality_threshold=0.7,
        # Tier processing configuration
        enable_tier_processing=True,
        enable_tier_1=True,  # Curated Priority (40% weight)
        enable_tier_2=True,  # Professional Therapeutic (25% weight)
        enable_tier_3=True,  # Chain-of-Thought (20% weight)
        enable_tier_4=True,  # Reddit Archive (10% weight)
        enable_tier_5=True,  # Research & Specialized (4% weight)
        enable_tier_6=True,  # Knowledge Base (1% weight)
        enable_tier_balancing=True,  # Apply tier-weighted balancing
        tier_balancing_target_total=None,  # Use all available (or set specific number)
    )

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config=config)

    print("Starting pipeline execution with tier processing...")
    print("=" * 60)

    # Execute pipeline with tier processing
    result = await orchestrator.execute_pipeline(
        include_huggingface=True,
        include_local=True,
        include_generated=True,
        include_tiers=True,  # Enable tier processing
    )

    # Display results
    print("\n" + "=" * 60)
    print("Pipeline Execution Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Total Datasets: {result.metrics.total_datasets}")
    print(f"Completed Datasets: {result.metrics.completed_datasets}")
    print(f"Failed Datasets: {result.metrics.failed_datasets}")
    print(f"Total Conversations: {result.metrics.total_conversations}")
    print(f"Accepted Conversations: {result.metrics.accepted_conversations}")
    print(f"Rejected Conversations: {result.metrics.rejected_conversations}")

    # Display tier statistics if available
    if orchestrator.tier_processor:
        stats = orchestrator.tier_processor.get_tier_statistics()
        print("\nTier Statistics:")
        print("-" * 60)
        for tier_key, tier_info in stats["tier_details"].items():
            print(f"{tier_key}:")
            print(f"  Datasets: {tier_info['datasets']}")
            print(f"  Conversations: {tier_info['conversations']}")
            print(f"  Quality Threshold: {tier_info['quality_threshold']:.2%}")
            print(f"  Training Weight: {tier_info['weight']:.2%}")

    # Display tier-balanced dataset if available
    if "tier_balanced" in result.datasets:
        balanced_convos = result.datasets["tier_balanced"]
        print(f"\nTier-Balanced Dataset: {len(balanced_convos)} conversations")

        if orchestrator.tier_balancer:
            distribution = orchestrator.tier_balancer.get_tier_distribution(balanced_convos)
            print("Tier Distribution:")
            for tier, count in sorted(distribution.items()):
                pct = (count / len(balanced_convos)) * 100 if balanced_convos else 0
                print(f"  Tier {tier}: {count} conversations ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Pipeline execution complete!")

    return result


if __name__ == "__main__":
    asyncio.run(main())

