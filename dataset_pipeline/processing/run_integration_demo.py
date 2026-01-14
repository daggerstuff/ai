"""
Demonstration Script for Mental Health Dataset Integration (PIX-60 + PIX-31)

Shows how to use the EnhancedMentalHealthIntegrator with Tier 2 professional data.
"""

import logging
from pathlib import Path
from enhanced_mental_health_integrator import (
    EnhancedMentalHealthIntegrator,
    DatasetIntegrationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run mental health dataset integration demo."""
    
    logger.info("=" * 80)
    logger.info("Mental Health Dataset Integration Demo (PIX-60 + PIX-31)")
    logger.info("=" * 80)
    
    # Initialize integrator
    integrator = EnhancedMentalHealthIntegrator(
        output_dir=Path("data/processed/mental_health"),
        base_datasets_path=Path("ai/datasets"),
        enable_tier2=True,  # Enable Tier 2 professional datasets
        enable_progress_bar=True,
    )
    
    # Optionally add additional dataset configurations
    # Example: Adding a custom dataset
    # config = DatasetIntegrationConfig(
    #     name="custom_dataset",
    #     source_path=Path("path/to/custom_data.jsonl"),
    #     format_type="generic_chatml",
    #     tier=3,
    #     quality_threshold=0.7,
    # )
    # integrator.add_dataset_config(config)
    
    # Run integration
    logger.info("\nStarting integration process...")
    
    try:
        report = integrator.integrate_all_datasets(
            include_tier2=True,
            max_conversations_per_dataset=None,  # No limit
        )
        
        logger.info("\n✅ Integration completed successfully!")
        logger.info(f"Report: {report['telemetry']['summary']}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()
