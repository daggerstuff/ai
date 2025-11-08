"""
Example usage scripts for NVIDIA NeMo Data Designer.

These examples demonstrate how to use the NeMo Data Designer service
for various use cases in the Pixelated Empathy platform.
"""

import json
import logging
from pathlib import Path

from ai.data_designer.service import NeMoDataDesignerService
from ai.data_designer.config import DataDesignerConfig
from nemo_microservices.data_designer.essentials import (
    SamplerColumnConfig,
    SamplerType,
    CategorySamplerParams,
    UniformSamplerParams,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def example_therapeutic_dataset():
    """Example: Generate a therapeutic dataset."""
    print("=" * 80)
    print("Example 1: Generating Therapeutic Dataset")
    print("=" * 80)

    try:
        service = NeMoDataDesignerService()
        # Preview API supports up to 10 samples (fast, no job execution)
        # For larger datasets, jobs are required (currently requires Docker registry auth)
        result = service.generate_therapeutic_dataset(num_samples=10)

        print(f"\nGenerated {result['num_samples']} samples")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Columns: {', '.join(result['columns'])}")

        # Save to file
        output_path = Path("data/therapeutic_dataset_example.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nDataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating therapeutic dataset: {e}")
        raise


def example_bias_detection_dataset():
    """Example: Generate a bias detection dataset."""
    print("=" * 80)
    print("Example 2: Generating Bias Detection Dataset")
    print("=" * 80)

    try:
        service = NeMoDataDesignerService()
        result = service.generate_bias_detection_dataset(
            num_samples=500,
            protected_attributes=["gender", "ethnicity", "age_group"],
        )

        print(f"\nGenerated {result['num_samples']} samples")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Protected attributes: {', '.join(result['protected_attributes'])}")

        # Save to file
        output_path = Path("data/bias_detection_dataset_example.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nDataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating bias detection dataset: {e}")
        raise


def example_custom_dataset():
    """Example: Generate a custom dataset."""
    print("=" * 80)
    print("Example 3: Generating Custom Dataset")
    print("=" * 80)

    try:
        service = NeMoDataDesignerService()

        # Define custom columns
        column_configs = [
            SamplerColumnConfig(
                name="patient_id",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=1.0, high=10000.0, decimal_places=0),
            ),
            SamplerColumnConfig(
                name="session_date",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
                ),
            ),
            SamplerColumnConfig(
                name="therapy_type",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=["Individual", "Group", "Couples", "Family"],
                ),
            ),
            SamplerColumnConfig(
                name="session_length_minutes",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=30.0, high=120.0, decimal_places=0),
            ),
            SamplerColumnConfig(
                name="mood_score",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=1.0, high=10.0),
            ),
        ]

        result = service.generate_custom_dataset(
            column_configs=column_configs,
            num_samples=200,
        )

        print(f"\nGenerated {result['num_samples']} samples")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Columns: {', '.join(result['columns'])}")

        # Save to file
        output_path = Path("data/custom_dataset_example.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nDataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating custom dataset: {e}")
        raise


def example_with_custom_config():
    """Example: Use custom configuration."""
    print("=" * 80)
    print("Example 4: Using Custom Configuration")
    print("=" * 80)

    try:
        # Create custom configuration
        config = DataDesignerConfig(
            base_url="https://ai.api.nvidia.com/v1/nemo/dd",
            api_key="your-api-key-here",  # Replace with your actual API key
            timeout=600,  # 10 minutes
            max_retries=5,
            batch_size=500,
        )

        service = NeMoDataDesignerService(config=config)
        result = service.generate_therapeutic_dataset(num_samples=50)

        print(f"\nGenerated {result['num_samples']} samples with custom config")
        print(f"Generation time: {result['generation_time']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with custom config: {e}")
        raise


if __name__ == "__main__":
    print("\nNVIDIA NeMo Data Designer - Example Scripts\n")

    # Run examples
    try:
        example_therapeutic_dataset()
        print("\n")

        example_bias_detection_dataset()
        print("\n")

        example_custom_dataset()
        print("\n")

        # Uncomment to run with custom config
        # example_with_custom_config()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise

