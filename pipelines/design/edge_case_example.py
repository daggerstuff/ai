"""
Example usage of the Edge Case Generator

This script demonstrates how to generate edge case scenarios for therapeutic training.
"""

import logging
import json
from ai.pipelines.design.edge_case_generator import EdgeCaseGenerator, EdgeCaseType
from ai.pipelines.design.edge_case_api import EdgeCaseAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def example_crisis_scenarios():
    """Generate crisis intervention edge cases."""
    print("=" * 80)
    print("Generating Crisis Intervention Edge Cases")
    print("=" * 80)

    generator = EdgeCaseGenerator()

    result = generator.generate_edge_case_dataset(
        edge_case_type=EdgeCaseType.CRISIS,
        num_samples=10,
        difficulty_level="advanced",
    )

    print(f"\n✅ Generated {len(result['data'])} crisis scenarios")
    print(f"Edge case type: {result['edge_case_type']}")
    print(f"Difficulty level: {result['difficulty_level']}")

    # Print first scenario
    if result['data']:
        print("\nFirst scenario:")
        print(json.dumps(result['data'][0], indent=2, default=str))


def example_multi_type_scenarios():
    """Generate multiple types of edge cases."""
    print("\n" + "=" * 80)
    print("Generating Multiple Edge Case Types")
    print("=" * 80)

    generator = EdgeCaseGenerator()

    result = generator.generate_multi_edge_case_dataset(
        edge_case_types=[
            EdgeCaseType.CRISIS,
            EdgeCaseType.CULTURAL_COMPLEXITY,
            EdgeCaseType.BOUNDARY_VIOLATION,
        ],
        num_samples_per_type=5,
        difficulty_level="advanced",
    )

    print(f"\n✅ Generated {result['total_samples']} edge case scenarios")
    print(f"Edge case types: {', '.join(result['edge_case_types'])}")
    print(f"Total samples: {result['total_samples']}")


def example_api_format():
    """Generate edge cases formatted for the scenario API."""
    print("\n" + "=" * 80)
    print("Generating Edge Cases for Scenario API")
    print("=" * 80)

    api = EdgeCaseAPI()

    result = api.generate_scenario(
        edge_case_type="crisis",
        num_samples=5,
        difficulty_level="advanced",
    )

    print(f"\n✅ Generated {len(result['scenarios'])} formatted scenarios")
    print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")

    # Print first scenario
    if result['scenarios']:
        print("\nFirst formatted scenario:")
        print(json.dumps(result['scenarios'][0], indent=2, default=str))


def example_trauma_scenarios():
    """Generate trauma disclosure edge cases."""
    print("\n" + "=" * 80)
    print("Generating Trauma Disclosure Edge Cases")
    print("=" * 80)

    generator = EdgeCaseGenerator()

    result = generator.generate_edge_case_dataset(
        edge_case_type=EdgeCaseType.TRAUMA_DISCLOSURE,
        num_samples=10,
        difficulty_level="advanced",
    )

    print(f"\n✅ Generated {len(result['data'])} trauma disclosure scenarios")

    # Count by trauma type
    trauma_types = {}
    for record in result['data']:
        if isinstance(record, dict):
            trauma_type = record.get('trauma_type', 'unknown')
            trauma_types[trauma_type] = trauma_types.get(trauma_type, 0) + 1

    print("\nTrauma type distribution:")
    for trauma_type, count in trauma_types.items():
        print(f"  {trauma_type}: {count}")


def main():
    """Run all examples."""
    try:
        # Generate crisis scenarios
        example_crisis_scenarios()

        # Generate multiple types
        example_multi_type_scenarios()

        # Generate API-formatted scenarios
        example_api_format()

        # Generate trauma scenarios
        example_trauma_scenarios()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

