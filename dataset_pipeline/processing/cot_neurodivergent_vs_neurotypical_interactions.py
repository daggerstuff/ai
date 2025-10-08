"""
CoT Neurodivergent vs Neurotypical Interactions

Integrates CoT_Neurodivergent_vs_Neurotypical_Interactions dataset for neurodiversity-aware therapeutic approaches.
Specialized processing for understanding and supporting neurodivergent clients.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)

@dataclass
class NeurodivergentInteraction:
    """Neurodivergent vs neurotypical interaction analysis."""
    interaction_id: str
    neurodivergent_perspective: str
    neurotypical_perspective: str
    therapeutic_approach: str
    reasoning_chain: list[str] = field(default_factory=list)
    accommodation_strategies: list[str] = field(default_factory=list)
    interaction_metadata: dict[str, Any] = field(default_factory=dict)

class CoTNeurodivergentVsNeurotypicalInteractions:
    """Processes CoT reasoning for neurodivergent vs neurotypical interactions."""

    def __init__(self, dataset_path: str = "./CoT_Neurodivergent_vs_Neurotypical_Interactions",
                 output_dir: str = "./processed_cot"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Neurodivergent conditions and characteristics
        self.neurodivergent_conditions = {
            "autism": {
                "characteristics": ["sensory sensitivity", "social communication differences", "repetitive behaviors", "special interests"],
                "strengths": ["attention to detail", "pattern recognition", "systematic thinking", "honesty"],
                "challenges": ["social cues", "sensory overload", "change adaptation", "executive function"]
            },
            "adhd": {
                "characteristics": ["inattention", "hyperactivity", "impulsivity", "executive dysfunction"],
                "strengths": ["creativity", "hyperfocus", "energy", "out-of-box thinking"],
                "challenges": ["time management", "organization", "sustained attention", "emotional regulation"]
            },
            "dyslexia": {
                "characteristics": ["reading difficulties", "phonological processing", "working memory"],
                "strengths": ["visual-spatial skills", "creative thinking", "problem-solving", "big picture thinking"],
                "challenges": ["reading fluency", "spelling", "written expression", "processing speed"]
            },
            "tourettes": {
                "characteristics": ["motor tics", "vocal tics", "involuntary movements", "co-occurring conditions"],
                "strengths": ["resilience", "empathy", "creativity", "determination"],
                "challenges": ["tic management", "social stigma", "concentration", "self-esteem"]
            }
        }

        # Therapeutic approaches for neurodivergent clients
        self.therapeutic_approaches = [
            "neurodiversity_affirming_therapy",
            "sensory_integration_techniques",
            "executive_function_coaching",
            "social_skills_training",
            "cognitive_behavioral_therapy_adapted",
            "acceptance_commitment_therapy",
            "mindfulness_based_interventions",
            "strength_based_approaches"
        ]

        # Accommodation strategies
        self.accommodation_strategies = [
            "sensory_accommodations",
            "communication_adaptations",
            "environmental_modifications",
            "scheduling_flexibility",
            "processing_time_adjustments",
            "alternative_assessment_methods",
            "assistive_technology",
            "peer_support_systems"
        ]

        logger.info("CoTNeurodivergentVsNeurotypicalInteractions initialized")

    def process_neurodivergent_interactions(self) -> dict[str, Any]:
        """Process the CoT neurodivergent vs neurotypical interactions dataset."""
        start_time = datetime.now()

        result = {
            "success": False,
            "interactions_processed": 0,
            "reasoning_chains_analyzed": 0,
            "accommodation_strategies_identified": 0,
            "quality_metrics": {},
            "neurodivergent_conditions_covered": [],
            "issues": [],
            "output_path": None
        }

        try:
            # Check if dataset exists, create mock if not
            if not self.dataset_path.exists():
                self._create_mock_neurodivergent_data()
                result["issues"].append("Created mock neurodivergent interactions data for testing")

            # Load interactions
            interactions = self._load_interactions()

            # Process each interaction
            processed_interactions = []
            total_reasoning_chains = 0
            total_accommodations = 0
            conditions_covered = set()

            for interaction_data in interactions:
                processed_interaction = self._process_interaction(interaction_data)
                if processed_interaction:
                    processed_interactions.append(processed_interaction)
                    total_reasoning_chains += len(processed_interaction.reasoning_chain)
                    total_accommodations += len(processed_interaction.accommodation_strategies)

                    # Track conditions covered
                    condition = processed_interaction.interaction_metadata.get("neurodivergent_condition")
                    if condition:
                        conditions_covered.add(condition)

            # Quality assessment
            quality_metrics = self._assess_interaction_quality(processed_interactions)

            # Save processed interactions
            output_path = self._save_processed_interactions(processed_interactions, quality_metrics)

            # Update result
            result.update({
                "success": True,
                "interactions_processed": len(processed_interactions),
                "reasoning_chains_analyzed": total_reasoning_chains,
                "accommodation_strategies_identified": total_accommodations,
                "quality_metrics": quality_metrics,
                "neurodivergent_conditions_covered": list(conditions_covered),
                "output_path": str(output_path),
                "processing_time": (datetime.now() - start_time).total_seconds()
            })

            logger.info(f"Successfully processed neurodivergent interactions: {len(processed_interactions)} interactions")

        except Exception as e:
            result["issues"].append(f"Processing failed: {e!s}")
            logger.error(f"Neurodivergent interactions processing failed: {e}")

        return result

    def _create_mock_neurodivergent_data(self):
        """Create mock neurodivergent vs neurotypical interactions data."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        interactions = []

        # Generate interactions for each neurodivergent condition
        for condition, details in self.neurodivergent_conditions.items():
            for i in range(25):  # 25 interactions per condition
                interaction = {
                    "interaction_id": f"neurodivergent_{condition}_{i:03d}",
                    "scenario": f"Therapeutic session with {condition} client experiencing {details['challenges'][i % len(details['challenges'])]}",
                    "neurodivergent_perspective": f"Client with {condition} experiences {details['challenges'][i % len(details['challenges'])]} which affects their daily functioning and therapeutic engagement.",
                    "neurotypical_perspective": f"Therapist recognizes the need to understand {condition} characteristics and adapt therapeutic approach accordingly.",
                    "reasoning_chain": [
                        f"Identify {condition} characteristics: {', '.join(details['characteristics'])}",
                        f"Recognize strengths: {details['strengths'][i % len(details['strengths'])]}",
                        f"Address challenge: {details['challenges'][i % len(details['challenges'])]}",
                        f"Consider accommodation: {self.accommodation_strategies[i % len(self.accommodation_strategies)]}",
                        f"Apply therapeutic approach: {self.therapeutic_approaches[i % len(self.therapeutic_approaches)]}"
                    ],
                    "therapeutic_approach": self.therapeutic_approaches[i % len(self.therapeutic_approaches)],
                    "accommodation_strategies": [
                        self.accommodation_strategies[i % len(self.accommodation_strategies)],
                        self.accommodation_strategies[(i + 1) % len(self.accommodation_strategies)]
                    ],
                    "metadata": {
                        "neurodivergent_condition": condition,
                        "primary_challenge": details["challenges"][i % len(details["challenges"])],
                        "identified_strength": details["strengths"][i % len(details["strengths"])],
                        "session_phase": ["assessment", "intervention", "maintenance"][i % 3],
                        "complexity_level": ["basic", "intermediate", "advanced"][i % 3]
                    }
                }
                interactions.append(interaction)

        # Save interactions
        with open(self.dataset_path / "neurodivergent_interactions.jsonl", "w") as f:
            for interaction in interactions:
                f.write(json.dumps(interaction) + "\n")

        # Create metadata
        metadata = {
            "dataset_name": "CoT Neurodivergent vs Neurotypical Interactions",
            "description": "Chain-of-thought reasoning for neurodiversity-aware therapeutic approaches",
            "total_interactions": len(interactions),
            "neurodivergent_conditions": list(self.neurodivergent_conditions.keys()),
            "therapeutic_approaches": self.therapeutic_approaches,
            "accommodation_strategies": self.accommodation_strategies,
            "created_at": datetime.now().isoformat()
        }

        with open(self.dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_interactions(self) -> list[dict[str, Any]]:
        """Load interactions from dataset."""
        interactions = []

        data_file = self.dataset_path / "neurodivergent_interactions.jsonl"
        if data_file.exists():
            with open(data_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            interaction = json.loads(line)
                            interactions.append(interaction)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")

        return interactions

    def _process_interaction(self, interaction_data: dict[str, Any]) -> NeurodivergentInteraction | None:
        """Process a single neurodivergent interaction."""
        try:
            interaction_id = interaction_data.get("interaction_id", f"interaction_{hash(str(interaction_data))%10000}")

            return NeurodivergentInteraction(
                interaction_id=interaction_id,
                neurodivergent_perspective=interaction_data.get("neurodivergent_perspective", ""),
                neurotypical_perspective=interaction_data.get("neurotypical_perspective", ""),
                therapeutic_approach=interaction_data.get("therapeutic_approach", ""),
                reasoning_chain=interaction_data.get("reasoning_chain", []),
                accommodation_strategies=interaction_data.get("accommodation_strategies", []),
                interaction_metadata={
                    **interaction_data.get("metadata", {}),
                    "processed_at": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error processing neurodivergent interaction: {e}")
            return None

    def _assess_interaction_quality(self, interactions: list[NeurodivergentInteraction]) -> dict[str, float]:
        """Assess quality of neurodivergent interactions dataset."""
        if not interactions:
            return {"overall_quality": 0.0}

        # Calculate quality metrics
        reasoning_chain_lengths = [len(i.reasoning_chain) for i in interactions]
        accommodation_counts = [len(i.accommodation_strategies) for i in interactions]

        # Condition coverage
        conditions_covered = {
            i.interaction_metadata.get("neurodivergent_condition")
            for i in interactions
            if i.interaction_metadata.get("neurodivergent_condition")
        }
        condition_coverage = len(conditions_covered) / len(self.neurodivergent_conditions)

        # Therapeutic approach diversity
        approaches_used = {i.therapeutic_approach for i in interactions if i.therapeutic_approach}
        approach_diversity = len(approaches_used) / len(self.therapeutic_approaches)

        # Accommodation strategy diversity
        all_accommodations = set()
        for interaction in interactions:
            all_accommodations.update(interaction.accommodation_strategies)
        accommodation_diversity = len(all_accommodations) / len(self.accommodation_strategies)

        return {
            "overall_quality": (condition_coverage + approach_diversity + accommodation_diversity) / 3,
            "condition_coverage": condition_coverage,
            "approach_diversity": approach_diversity,
            "accommodation_diversity": accommodation_diversity,
            "average_reasoning_chain_length": sum(reasoning_chain_lengths) / len(reasoning_chain_lengths) if reasoning_chain_lengths else 0,
            "average_accommodations_per_interaction": sum(accommodation_counts) / len(accommodation_counts) if accommodation_counts else 0,
            "comprehensive_interactions": sum(1 for i in interactions if len(i.reasoning_chain) >= 4 and len(i.accommodation_strategies) >= 2) / len(interactions)
        }


    def _save_processed_interactions(self, interactions: list[NeurodivergentInteraction],
                                   quality_metrics: dict[str, float]) -> Path:
        """Save processed neurodivergent interactions."""
        output_file = self.output_dir / "cot_neurodivergent_vs_neurotypical_interactions_processed.json"

        # Convert to serializable format
        interactions_data = []
        for interaction in interactions:
            interaction_dict = {
                "interaction_id": interaction.interaction_id,
                "neurodivergent_perspective": interaction.neurodivergent_perspective,
                "neurotypical_perspective": interaction.neurotypical_perspective,
                "therapeutic_approach": interaction.therapeutic_approach,
                "reasoning_chain": interaction.reasoning_chain,
                "accommodation_strategies": interaction.accommodation_strategies,
                "interaction_metadata": interaction.interaction_metadata
            }
            interactions_data.append(interaction_dict)

        output_data = {
            "dataset_info": {
                "name": "CoT Neurodivergent vs Neurotypical Interactions",
                "description": "Neurodiversity-aware therapeutic approaches with chain-of-thought reasoning",
                "total_interactions": len(interactions),
                "processed_at": datetime.now().isoformat()
            },
            "quality_metrics": quality_metrics,
            "neurodivergent_conditions": self.neurodivergent_conditions,
            "therapeutic_approaches": self.therapeutic_approaches,
            "accommodation_strategies": self.accommodation_strategies,
            "interactions": interactions_data
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Processed neurodivergent interactions saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = CoTNeurodivergentVsNeurotypicalInteractions()

    # Process neurodivergent interactions
    result = processor.process_neurodivergent_interactions()

    # Show results
    if result["success"]:
        pass
    else:
        pass

