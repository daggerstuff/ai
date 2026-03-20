"""
Edge Case Generation Sourcing.

Generates synthetic edge case scenarios for Stage 3 stress testing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class EdgeCaseSource:
    """
    Generate edge case scenarios for stress testing.

    Generates scenarios for:
    - Suicidality crisis
    - Homicidal ideation
    - Psychotic episodes
    - Severe dissociation
    - Substance abuse crisis
    - Trauma flashbacks
    - Borderline crisis
    - Domestic violence situations

    Usage:
        source = EdgeCaseSource()
        for ref in source.fill_gap(1000):
            print(f"Generated: {ref}")
    """

    # Edge case categories with crisis intensity levels
    CATEGORIES = {
        'suicidality': {
            'intensity': 'very_high',
            'scenarios': [
                "Client expresses active suicidal ideation with plan",
                "Client describes specific suicide method and timeline",
                "Client has suicide note and means available",
                "Client reporting command hallucinations to self-harm",
            ]
        },
        'homicidal': {
            'intensity': 'very_high',
            'scenarios': [
                "Client expresses homicidal thoughts toward specific person",
                "Client has plan to harm identified individual",
                "Client reporting command hallucinations to harm others",
            ]
        },
        'psychosis': {
            'intensity': 'high',
            'scenarios': [
                "Client experiencing acute psychotic break",
                "Client reporting paranoid delusions about being persecuted",
                "Client experiencing command hallucinations",
                "Client unable to distinguish reality from delusion",
            ]
        },
        'dissociation': {
            'intensity': 'high',
            'scenarios': [
                "Client experiencing severe dissociative episode",
                "Client reports losing time and finding unfamiliar items",
                "Client switching between distinct identity states",
            ]
        },
        'substance_crisis': {
            'intensity': 'very_high',
            'scenarios': [
                "Client reporting overdose with suicidal intent",
                "Client experiencing substance-induced psychosis",
                "Client in withdrawal with medical complications",
            ]
        },
        'trauma_flashback': {
            'intensity': 'high',
            'scenarios': [
                "Client experiencing severe flashback during session",
                "Client dissociating into past trauma",
                "Client having trauma response to trigger",
            ]
        },
        'borderline_crisis': {
            'intensity': 'high',
            'scenarios': [
                "Client threatening self-harm if therapist leaves",
                "Client in borderline rage episode",
                "Client reporting chronic suicidal crisis",
            ]
        },
        'domestic_violence': {
            'intensity': 'very_high',
            'scenarios': [
                "Client reporting immediate danger from partner",
                "Client minimizing abuse while describing injuries",
                "Client considering returning to abuser",
            ]
        },
    }

    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Initialize edge case source.

        Args:
            output_dir: Directory for generated scenarios
        """
        self.output_dir = Path(output_dir) if output_dir else Path("ai/data/acquired_datasets/edge_cases")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "edge_cases_generated.jsonl"

    def generate_scenario(
        self,
        category: str,
        scenario: str,
        intensity: str = "high"
    ) -> Dict[str, Any]:
        """
        Generate a single edge case scenario.

        Args:
            category: Edge case category
            scenario: Scenario description
            intensity: Crisis intensity level

        Returns:
            Scenario record
        """
        return {
            'prompt': f"Therapist: A client presents with: {scenario}. How do you respond?",
            'response': f"[Generated therapeutic response for {category} crisis]",
            'metadata': {
                'category': category,
                'crisis_intensity': intensity,
                'is_training_edge_case': True,
                'source': 'edge_case_generation',
                'purpose': 'difficult_client',
                'difficulty_level': intensity,
                'expected_challenges': [
                    'crisis_intervention',
                    'safety_assessment',
                    'de-escalation',
                ]
            }
        }

    def generate_batch(
        self,
        category: str,
        count: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate batch of scenarios for a category.

        Args:
            category: Edge case category
            count: Number to generate

        Yields:
            Scenario records
        """
        cat_info = self.CATEGORIES.get(category, {})
        scenarios = cat_info.get('scenarios', ['Crisis scenario'])
        intensity = cat_info.get('intensity', 'high')

        for i in range(count):
            scenario = scenarios[i % len(scenarios)]
            yield self.generate_scenario(category, f"{scenario} (variant {i+1})", intensity)

    def fill_gap(self, gap: int, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Generate edge cases to fill gap.

        Args:
            gap: Number of samples needed

        Yields:
            Generated scenario records
        """
        logger.info(f"EdgeCaseSource generating {gap} samples")

        # Distribute across categories
        categories = list(self.CATEGORIES.keys())
        per_category = max(1, gap // len(categories))

        count = 0
        for category in categories:
            if count >= gap:
                break

            batch_size = min(per_category, gap - count)
            logger.info(f"Generating {batch_size} {category} scenarios")

            for record in self.generate_batch(category, batch_size):
                yield record
                count += 1
                if count >= gap:
                    break

        logger.info(f"Edge case generation complete: {count} samples")

    def discover(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Discover available edge case categories."""
        for category, info in self.CATEGORIES.items():
            yield {
                'category': category,
                'intensity': info['intensity'],
                'scenario_count': len(info['scenarios']),
                'stage': 'stage3_edge_stress_test',
                'quality_profile': 'edge_crisis',
            }

    def save_batch(self, records: Iterator[Dict[str, Any]], filename: str = None):
        """Save generated records to file."""
        output_path = self.output_file if not filename else self.output_dir / filename

        count = 0
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
                count += 1

        logger.info(f"Saved {count} records to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test generation
    source = EdgeCaseSource()

    print("Generating edge case scenarios...")
    count = 0
    for record in source.fill_gap(20):
        print(f"  [{record['metadata']['category']}] {record['prompt'][:60]}...")
        count += 1

    print(f"\nGenerated {count} edge case scenarios")
