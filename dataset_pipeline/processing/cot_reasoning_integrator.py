"""
Chain-of-Thought Reasoning Integrator

Integrates various CoT reasoning datasets for therapeutic applications:
- CoT_Neurodivergent_vs_Neurotypical_Interactions
- CoT_Reasoning_Mens_Mental_Health
- CoT_Philosophical_Understanding
- CoT_Temporal_Reasoning_Dataset
- CoT-Reasoning_Cultural_Nuances
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CoTExample:
    """Chain-of-Thought reasoning example."""

    example_id: str
    reasoning_type: str
    problem_statement: str
    reasoning_chain: list[str]
    final_conclusion: str
    therapeutic_context: dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0


@dataclass
class CoTDataset:
    """Complete CoT reasoning dataset."""

    dataset_name: str
    reasoning_type: str
    examples: list[CoTExample]
    metadata: dict[str, Any] = field(default_factory=dict)


class CoTReasoningIntegrator:
    """Integrates Chain-of-Thought reasoning datasets for therapeutic applications."""

    def __init__(
        self,
        base_path: str = "./cot_datasets",
        output_dir: str = "./integrated_datasets",
    ):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # CoT dataset configurations
        self.cot_configs = {
            "neurodivergent": {
                "name": "CoT_Neurodivergent_vs_Neurotypical_Interactions",
                "description": "Neurodiversity-aware therapeutic approaches",
                "reasoning_type": "neurodiversity_reasoning",
                "therapeutic_focus": "inclusive_therapy",
                "expected_size": "medium",
            },
            "mens_mental_health": {
                "name": "CoT_Reasoning_Mens_Mental_Health",
                "description": "Gender-specific therapeutic reasoning",
                "reasoning_type": "gender_specific_reasoning",
                "therapeutic_focus": "mens_therapy",
                "expected_size": "medium",
            },
            "philosophical": {
                "name": "CoT_Philosophical_Understanding",
                "description": "33MB, 60K existential/philosophical therapy",
                "reasoning_type": "philosophical_reasoning",
                "therapeutic_focus": "existential_therapy",
                "expected_size": "large",
            },
            "temporal": {
                "name": "CoT_Temporal_Reasoning_Dataset",
                "description": "15MB, 30K time-based therapeutic planning",
                "reasoning_type": "temporal_reasoning",
                "therapeutic_focus": "treatment_planning",
                "expected_size": "medium",
            },
            "cultural": {
                "name": "CoT-Reasoning_Cultural_Nuances",
                "description": "Culturally-sensitive therapeutic approaches",
                "reasoning_type": "cultural_reasoning",
                "therapeutic_focus": "cultural_therapy",
                "expected_size": "medium",
            },
        }

        # Reasoning patterns for each type
        self.reasoning_patterns = {
            "neurodiversity_reasoning": [
                "Consider neurodivergent perspective",
                "Assess sensory processing differences",
                "Evaluate communication preferences",
                "Account for executive function variations",
                "Recognize masking behaviors",
            ],
            "gender_specific_reasoning": [
                "Consider societal gender expectations",
                "Assess masculine identity pressures",
                "Evaluate emotional expression barriers",
                "Account for help-seeking stigma",
                "Recognize vulnerability challenges",
            ],
            "philosophical_reasoning": [
                "Examine existential concerns",
                "Explore meaning and purpose",
                "Consider life's fundamental questions",
                "Assess values and beliefs",
                "Evaluate spiritual dimensions",
            ],
            "temporal_reasoning": [
                "Assess timeline of symptoms",
                "Plan treatment progression",
                "Consider developmental stages",
                "Evaluate progress markers",
                "Project future outcomes",
            ],
            "cultural_reasoning": [
                "Consider cultural background",
                "Assess family dynamics",
                "Evaluate cultural values",
                "Account for language barriers",
                "Recognize cultural stigma",
            ],
        }

        logger.info("CoTReasoningIntegrator initialized")

    def integrate_all_cot_datasets(self) -> dict[str, Any]:
        """Integrate all CoT reasoning datasets."""
        start_time = datetime.now()

        results = {}
        total_examples = 0

        for dataset_key, config in self.cot_configs.items():
            logger.info(f"Integrating {config['name']}...")

            result = self.integrate_cot_dataset(dataset_key)
            results[dataset_key] = result

            if result["success"]:
                total_examples += result["examples_processed"]

        # Create combined summary
        return {
            "integration_type": "Chain-of-Thought Reasoning Datasets",
            "total_datasets": len(self.cot_configs),
            "successful_integrations": sum(1 for r in results.values() if r["success"]),
            "total_examples": total_examples,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "individual_results": results,
        }


    def integrate_cot_dataset(self, dataset_key: str) -> dict[str, Any]:
        """Integrate a specific CoT reasoning dataset."""
        if dataset_key not in self.cot_configs:
            return {"success": False, "issues": [f"Unknown dataset key: {dataset_key}"]}

        config = self.cot_configs[dataset_key]

        result = {
            "success": False,
            "dataset_name": config["name"],
            "examples_processed": 0,
            "quality_metrics": {},
            "issues": [],
            "output_path": None,
        }

        try:
            # Check if dataset exists, create mock if not
            dataset_path = self.base_path / config["name"]
            if not dataset_path.exists():
                self._create_mock_cot_data(dataset_key, config)
                result["issues"].append(f"Created mock data for {config['name']}")

            # Load CoT examples
            cot_examples = self._load_cot_examples(dataset_key, config)

            # Process examples
            processed_examples = []
            for example in cot_examples:
                processed_example = self._process_cot_example(example, config)
                if processed_example:
                    processed_examples.append(processed_example)

            # Create CoT dataset
            cot_dataset = CoTDataset(
                dataset_name=config["name"],
                reasoning_type=config["reasoning_type"],
                examples=processed_examples,
                metadata={
                    "description": config["description"],
                    "therapeutic_focus": config["therapeutic_focus"],
                    "integrated_at": datetime.now().isoformat(),
                },
            )

            # Quality assessment
            quality_metrics = self._assess_cot_quality(cot_dataset)

            # Save dataset
            output_path = self._save_cot_dataset(cot_dataset, quality_metrics)

            # Update result
            result.update(
                {
                    "success": True,
                    "examples_processed": len(processed_examples),
                    "quality_metrics": quality_metrics,
                    "output_path": str(output_path),
                }
            )

            logger.info(
                f"Successfully integrated {config['name']}: {len(processed_examples)} examples"
            )

        except Exception as e:
            result["issues"].append(f"Integration failed: {e!s}")
            logger.error(f"CoT integration failed for {dataset_key}: {e}")

        return result

    def _create_mock_cot_data(self, dataset_key: str, config: dict[str, Any]):
        """Create mock CoT reasoning data."""
        dataset_path = self.base_path / config["name"]
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Determine number of examples based on expected size
        size_mapping = {"small": 50, "medium": 200, "large": 500}
        num_examples = size_mapping.get(config["expected_size"], 100)

        examples = []
        reasoning_patterns = self.reasoning_patterns[config["reasoning_type"]]

        # Generate examples based on reasoning type
        for i in range(num_examples):
            example = self._generate_mock_cot_example(i, config, reasoning_patterns)
            examples.append(example)

        # Save examples
        data_file = dataset_path / "cot_examples.jsonl"
        with open(data_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        # Create metadata
        metadata = {
            "dataset_name": config["name"],
            "description": config["description"],
            "reasoning_type": config["reasoning_type"],
            "therapeutic_focus": config["therapeutic_focus"],
            "total_examples": len(examples),
            "reasoning_patterns": reasoning_patterns,
            "created_at": datetime.now().isoformat(),
        }

        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_mock_cot_example(
        self, index: int, config: dict[str, Any], reasoning_patterns: list[str]
    ) -> dict[str, Any]:
        """Generate mock CoT example based on reasoning type."""
        reasoning_type = config["reasoning_type"]

        if reasoning_type == "neurodiversity_reasoning":
            problem = "A client with autism spectrum disorder is struggling with social anxiety in workplace settings. They report feeling overwhelmed by office noise and unexpected schedule changes."
            reasoning_chain = [
                "Consider neurodivergent perspective: Client may have heightened sensory sensitivity",
                "Assess sensory processing differences: Office environment may be overstimulating",
                "Evaluate communication preferences: Direct, clear communication may be preferred",
                "Account for executive function variations: Schedule changes may be particularly challenging",
                "Recognize masking behaviors: Client may be exhausted from masking autistic traits",
            ]
            conclusion = "Recommend sensory accommodations, structured communication protocols, and validation of neurodivergent experiences while building coping strategies."

        elif reasoning_type == "gender_specific_reasoning":
            problem = "A male client in his 30s is reluctant to discuss emotional struggles, presenting only with work stress and relationship conflicts."
            reasoning_chain = [
                "Consider societal gender expectations: Men often discouraged from emotional expression",
                "Assess masculine identity pressures: May feel vulnerability threatens masculinity",
                "Evaluate emotional expression barriers: Limited emotional vocabulary common",
                "Account for help-seeking stigma: Therapy may feel like admission of weakness",
                "Recognize vulnerability challenges: Need safe space to explore emotions",
            ]
            conclusion = "Use strength-based approach, normalize emotional experiences, and gradually build emotional awareness through practical frameworks."

        elif reasoning_type == "philosophical_reasoning":
            problem = "A client is experiencing existential crisis following major life transition, questioning life's meaning and purpose."
            reasoning_chain = [
                "Examine existential concerns: Client facing fundamental questions about existence",
                "Explore meaning and purpose: Transition has disrupted sense of direction",
                "Consider life's fundamental questions: What makes life worth living?",
                "Assess values and beliefs: Core beliefs may be challenged or evolving",
                "Evaluate spiritual dimensions: May need to explore transcendent meaning",
            ]
            conclusion = "Engage in existential exploration, help client construct personal meaning, and support values clarification process."

        elif reasoning_type == "temporal_reasoning":
            problem = "A client with depression needs comprehensive treatment planning considering symptom progression and recovery timeline."
            reasoning_chain = [
                "Assess timeline of symptoms: Depression developed over 6-month period",
                "Plan treatment progression: Start with stabilization, then skill-building",
                "Consider developmental stages: Client in early career development phase",
                "Evaluate progress markers: Weekly mood tracking and functional improvements",
                "Project future outcomes: Expect gradual improvement over 3-6 months",
            ]
            conclusion = "Implement phased treatment approach with clear milestones and regular progress evaluation."

        else:  # cultural_reasoning
            problem = "A client from collectivist cultural background is struggling with individual therapy approach and family expectations."
            reasoning_chain = [
                "Consider cultural background: Collectivist values may conflict with individual focus",
                "Assess family dynamics: Family involvement may be crucial for success",
                "Evaluate cultural values: Honor and family harmony highly valued",
                "Account for language barriers: May need culturally adapted interventions",
                "Recognize cultural stigma: Mental health treatment may carry cultural shame",
            ]
            conclusion = "Adapt therapy to include family systems perspective and culturally sensitive interventions."

        return {
            "id": f"{reasoning_type}_example_{index:03d}",
            "problem_statement": problem,
            "reasoning_chain": reasoning_chain,
            "conclusion": conclusion,
            "metadata": {
                "reasoning_type": reasoning_type,
                "therapeutic_focus": config["therapeutic_focus"],
                "complexity_level": ["basic", "intermediate", "advanced"][index % 3],
                "example_index": index,
            },
        }

    def _load_cot_examples(
        self, dataset_key: str, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Load CoT examples from dataset."""
        examples = []

        dataset_path = self.base_path / config["name"]
        data_file = dataset_path / "cot_examples.jsonl"

        if data_file.exists():
            with open(data_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            examples.append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Invalid JSON on line {line_num} in {data_file}: {e}"
                            )

        return examples

    def _process_cot_example(
        self, example_data: dict[str, Any], config: dict[str, Any]
    ) -> CoTExample | None:
        """Process and validate a CoT example."""
        try:
            example_id = example_data.get("id", f"cot_{hash(str(example_data))%10000}")
            problem_statement = example_data.get("problem_statement", "")
            reasoning_chain = example_data.get("reasoning_chain", [])
            conclusion = example_data.get("conclusion", "")

            if not all([problem_statement, reasoning_chain, conclusion]):
                logger.warning(f"Incomplete CoT example: {example_id}")
                return None

            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                reasoning_chain, conclusion
            )

            # Extract therapeutic context
            therapeutic_context = {
                **example_data.get("metadata", {}),
                "reasoning_steps": len(reasoning_chain),
                "therapeutic_focus": config["therapeutic_focus"],
                "processed_at": datetime.now().isoformat(),
            }

            return CoTExample(
                example_id=example_id,
                reasoning_type=config["reasoning_type"],
                problem_statement=problem_statement,
                reasoning_chain=reasoning_chain,
                final_conclusion=conclusion,
                therapeutic_context=therapeutic_context,
                complexity_score=complexity_score,
            )

        except Exception as e:
            logger.error(f"Error processing CoT example: {e}")
            return None

    def _calculate_complexity_score(
        self, reasoning_chain: list[str], conclusion: str
    ) -> float:
        """Calculate complexity score for CoT example."""
        score = 0.5  # Base score

        # Number of reasoning steps
        step_count = len(reasoning_chain)
        if step_count >= 5:
            score += 0.2
        elif step_count >= 3:
            score += 0.1

        # Depth of reasoning (longer steps indicate deeper thinking)
        avg_step_length = (
            sum(len(step.split()) for step in reasoning_chain) / len(reasoning_chain)
            if reasoning_chain
            else 0
        )
        if avg_step_length >= 15:
            score += 0.2
        elif avg_step_length >= 10:
            score += 0.1

        # Conclusion quality (length and specificity)
        conclusion_words = len(conclusion.split())
        if conclusion_words >= 20:
            score += 0.1

        return min(1.0, score)

    def _assess_cot_quality(self, cot_dataset: CoTDataset) -> dict[str, float]:
        """Assess quality of CoT dataset."""
        examples = cot_dataset.examples

        if not examples:
            return {"overall_quality": 0.0}

        # Calculate quality metrics
        complexity_scores = [e.complexity_score for e in examples]
        reasoning_step_counts = [len(e.reasoning_chain) for e in examples]

        return {
            "overall_quality": sum(complexity_scores) / len(complexity_scores),
            "average_reasoning_steps": sum(reasoning_step_counts)
            / len(reasoning_step_counts),
            "complexity_variance": sum(
                (c - sum(complexity_scores) / len(complexity_scores)) ** 2
                for c in complexity_scores
            )
            / len(complexity_scores),
            "high_complexity_examples": sum(1 for c in complexity_scores if c >= 0.8)
            / len(complexity_scores),
            "reasoning_depth": sum(1 for steps in reasoning_step_counts if steps >= 5)
            / len(reasoning_step_counts),
        }


    def _save_cot_dataset(
        self, cot_dataset: CoTDataset, quality_metrics: dict[str, float]
    ) -> Path:
        """Save CoT dataset."""
        output_file = self.output_dir / f"{cot_dataset.reasoning_type}_integrated.json"

        # Convert to serializable format
        examples_data = []
        for example in cot_dataset.examples:
            example_dict = {
                "example_id": example.example_id,
                "reasoning_type": example.reasoning_type,
                "problem_statement": example.problem_statement,
                "reasoning_chain": example.reasoning_chain,
                "final_conclusion": example.final_conclusion,
                "therapeutic_context": example.therapeutic_context,
                "complexity_score": example.complexity_score,
            }
            examples_data.append(example_dict)

        output_data = {
            "dataset_info": {
                "name": cot_dataset.dataset_name,
                "reasoning_type": cot_dataset.reasoning_type,
                "total_examples": len(cot_dataset.examples),
                "integrated_at": datetime.now().isoformat(),
            },
            "quality_metrics": quality_metrics,
            "metadata": cot_dataset.metadata,
            "examples": examples_data,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"CoT dataset saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = CoTReasoningIntegrator()

    # Integrate all CoT datasets
    results = integrator.integrate_all_cot_datasets()

    # Show results

    for _dataset_key, result in results["individual_results"].items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        if result["success"]:
            pass
        else:
            pass
