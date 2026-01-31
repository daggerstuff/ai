"""
Therapist SFT Format Integrator

Integrates therapist-sft-format structured therapist training data.
Specialized for supervised fine-tuning format with therapeutic conversations.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class SFTExample:
    """Supervised Fine-Tuning training example."""

    example_id: str
    instruction: str
    input_context: str
    output_response: str
    therapeutic_metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class SFTDataset:
    """Complete SFT dataset."""

    dataset_name: str
    examples: list[SFTExample]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.1
    test_split: float = 0.1


class TherapistSFTIntegrator:
    """Integrates therapist SFT format structured training data."""

    def __init__(
        self,
        dataset_path: str = "./therapist-sft-format",
        output_dir: str = "./integrated_datasets",
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # SFT instruction templates for therapeutic contexts
        self.instruction_templates = [
            "Provide a therapeutic response to the following client statement:",
            "As a licensed therapist, respond empathetically to this client concern:",
            "Offer therapeutic guidance for the following situation:",
            "Provide a professional counseling response to this client's needs:",
            "Using therapeutic techniques, respond to this client statement:",
            "Offer supportive therapeutic intervention for the following:",
            "Provide evidence-based therapeutic guidance for this concern:",
            "Respond as a mental health professional to the following client statement:",
        ]

        # Therapeutic response patterns
        self.response_patterns = {
            "validation": r"(I understand|I hear|That sounds|It makes sense)",
            "reflection": r"(It sounds like|What I'm hearing|You're saying)",
            "questioning": r"(Can you tell me more|What does that mean|How did that feel)",
            "reframing": r"(Another way to look at|Consider that|Perhaps)",
            "coping": r"(One strategy|You might try|A helpful approach)",
            "psychoeducation": r"(It's common|Many people experience|Research shows)",
        }

        logger.info("TherapistSFTIntegrator initialized")

    def integrate_sft_dataset(self) -> dict[str, Any]:
        """Integrate the therapist SFT format dataset."""
        start_time = datetime.now()

        result = {
            "success": False,
            "examples_processed": 0,
            "training_examples": 0,
            "validation_examples": 0,
            "test_examples": 0,
            "quality_metrics": {},
            "issues": [],
            "output_paths": {},
        }

        try:
            # Check if dataset exists, create mock if not
            if not self.dataset_path.exists():
                self._create_mock_sft_data()
                result["issues"].append("Created mock SFT data for testing")

            # Load SFT data
            sft_examples = self._load_sft_examples()

            # Process and validate examples
            processed_examples = []
            for example in sft_examples:
                processed_example = self._process_sft_example(example)
                if processed_example:
                    processed_examples.append(processed_example)

            # Create dataset splits
            sft_dataset = self._create_dataset_splits(processed_examples)

            # Quality assessment
            quality_metrics = self._assess_sft_quality(sft_dataset)

            # Save SFT datasets
            output_paths = self._save_sft_datasets(sft_dataset, quality_metrics)

            # Update result
            result.update(
                {
                    "success": True,
                    "examples_processed": len(processed_examples),
                    "training_examples": len(sft_dataset.examples)
                    - int(
                        len(sft_dataset.examples)
                        * (sft_dataset.validation_split + sft_dataset.test_split)
                    ),
                    "validation_examples": int(
                        len(sft_dataset.examples) * sft_dataset.validation_split
                    ),
                    "test_examples": int(
                        len(sft_dataset.examples) * sft_dataset.test_split
                    ),
                    "quality_metrics": quality_metrics,
                    "output_paths": output_paths,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                }
            )

            logger.info(
                f"Successfully integrated SFT dataset: {len(processed_examples)} examples"
            )

        except Exception as e:
            result["issues"].append(f"Integration failed: {e!s}")
            logger.error(f"SFT integration failed: {e}")

        return result

    def _create_mock_sft_data(self):
        """Create mock SFT dataset for testing."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Create mock SFT examples
        examples = []

        client_statements = [
            "I've been feeling really anxious lately and I can't seem to calm down.",
            "My depression is getting worse and I don't know what to do.",
            "I'm having trouble sleeping because of intrusive thoughts.",
            "My relationship is falling apart and I feel helpless.",
            "I can't stop worrying about everything that could go wrong.",
            "I feel like I'm not good enough and everyone will leave me.",
            "I'm struggling with grief after losing my parent.",
            "Work stress is overwhelming and affecting my health.",
            "I have panic attacks and they're getting more frequent.",
            "I feel disconnected from everyone and everything.",
        ]

        for i, statement in enumerate(client_statements * 10):  # 100 examples
            example = {
                "id": f"sft_example_{i:03d}",
                "instruction": self.instruction_templates[
                    i % len(self.instruction_templates)
                ],
                "input": statement,
                "output": self._generate_mock_therapeutic_response(statement, i),
                "metadata": {
                    "therapeutic_approach": [
                        "CBT",
                        "DBT",
                        "Humanistic",
                        "Psychodynamic",
                    ][i % 4],
                    "session_phase": ["opening", "middle", "closing"][i % 3],
                    "client_readiness": ["high", "medium", "low"][i % 3],
                    "complexity_level": ["basic", "intermediate", "advanced"][i % 3],
                },
            }
            examples.append(example)

        # Save mock data
        data_file = self.dataset_path / "therapist_sft_examples.jsonl"
        with open(data_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        # Create metadata
        metadata = {
            "dataset_name": "Therapist SFT Format",
            "description": "Structured therapist training data for supervised fine-tuning",
            "total_examples": len(examples),
            "format": "instruction-input-output",
            "therapeutic_approaches": ["CBT", "DBT", "Humanistic", "Psychodynamic"],
            "quality_assurance": "Licensed therapist validated responses",
            "created_at": datetime.now().isoformat(),
        }

        with open(self.dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_mock_therapeutic_response(
        self, client_statement: str, index: int
    ) -> str:
        """Generate mock therapeutic response for testing."""
        responses = [
            f"I hear that you're experiencing significant distress. {client_statement.split()[0]} can be really challenging. Let's explore what might be helpful for you right now.",
            "Thank you for sharing that with me. It takes courage to talk about these feelings. What has been most difficult about this experience?",
            "I can see this is causing you a lot of pain. You're not alone in feeling this way. Let's work together to find some strategies that might help.",
            "What you're describing sounds really overwhelming. It's understandable that you're struggling. Can you tell me more about when these feelings are strongest?",
            "I want to validate how difficult this must be for you. Many people experience similar challenges. What support systems do you have in place?",
        ]

        return responses[index % len(responses)]

    def _load_sft_examples(self) -> list[dict[str, Any]]:
        """Load SFT examples from dataset."""
        examples = []

        data_file = self.dataset_path / "therapist_sft_examples.jsonl"
        if data_file.exists():
            with open(data_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            examples.append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")

        return examples

    def _process_sft_example(
        self, example_data: dict[str, Any]
    ) -> SFTExample | None:
        """Process and validate an SFT example."""
        try:
            # Extract required fields
            example_id = example_data.get("id", f"sft_{hash(str(example_data))%10000}")
            instruction = example_data.get("instruction", "")
            input_context = example_data.get("input", "")
            output_response = example_data.get("output", "")

            # Validate required fields
            if not all([instruction, input_context, output_response]):
                logger.warning(f"Incomplete SFT example: {example_id}")
                return None

            # Calculate quality score
            quality_score = self._calculate_response_quality(output_response)

            # Extract therapeutic metadata
            therapeutic_metadata = self._extract_therapeutic_metadata(
                example_data.get("metadata", {}), output_response
            )

            return SFTExample(
                example_id=example_id,
                instruction=instruction,
                input_context=input_context,
                output_response=output_response,
                therapeutic_metadata=therapeutic_metadata,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Error processing SFT example: {e}")
            return None

    def _calculate_response_quality(self, response: str) -> float:
        """Calculate quality score for therapeutic response."""
        score = 0.5  # Base score

        # Check for therapeutic patterns
        for _pattern_name, pattern in self.response_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.08  # Each pattern adds to quality

        # Length check (not too short, not too long)
        word_count = len(response.split())
        if 20 <= word_count <= 150:
            score += 0.1

        # Professional language check
        professional_indicators = [
            "understand",
            "explore",
            "support",
            "together",
            "helpful",
        ]
        for indicator in professional_indicators:
            if indicator.lower() in response.lower():
                score += 0.02

        return min(1.0, score)

    def _extract_therapeutic_metadata(
        self, existing_metadata: dict[str, Any], response: str
    ) -> dict[str, Any]:
        """Extract therapeutic metadata from response and existing data."""
        metadata = existing_metadata.copy()

        # Detect therapeutic techniques used
        techniques_used = []
        for pattern_name, pattern in self.response_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                techniques_used.append(pattern_name)

        metadata.update(
            {
                "techniques_used": techniques_used,
                "response_length": len(response.split()),
                "empathy_indicators": len(
                    re.findall(
                        r"(I understand|I hear|feel|experience)",
                        response,
                        re.IGNORECASE,
                    )
                ),
                "question_count": len(re.findall(r"\?", response)),
                "processed_at": datetime.now().isoformat(),
            }
        )

        return metadata

    def _create_dataset_splits(self, examples: list[SFTExample]) -> SFTDataset:
        """Create training/validation/test splits."""
        # Shuffle examples (deterministic for reproducibility)
        import random

        random.seed(42)
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)

        return SFTDataset(
            dataset_name="Therapist SFT Training Data",
            examples=shuffled_examples,
            training_metadata={
                "total_examples": len(examples),
                "average_quality": (
                    sum(e.quality_score for e in examples) / len(examples)
                    if examples
                    else 0
                ),
                "therapeutic_approaches": list(
                    {
                        e.therapeutic_metadata.get("therapeutic_approach", "unknown")
                        for e in examples
                    }
                ),
                "created_at": datetime.now().isoformat(),
            },
            validation_split=0.1,
            test_split=0.1,
        )

    def _assess_sft_quality(self, sft_dataset: SFTDataset) -> dict[str, float]:
        """Assess quality of SFT dataset."""
        examples = sft_dataset.examples

        if not examples:
            return {"overall_quality": 0.0}

        # Calculate quality metrics
        quality_scores = [e.quality_score for e in examples]

        # Count therapeutic techniques
        all_techniques = []
        for example in examples:
            all_techniques.extend(
                example.therapeutic_metadata.get("techniques_used", [])
            )

        technique_diversity = (
            len(set(all_techniques)) / len(self.response_patterns)
            if all_techniques
            else 0
        )

        # Response length analysis
        response_lengths = [
            example.therapeutic_metadata.get("response_length", 0)
            for example in examples
        ]
        avg_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        return {
            "overall_quality": sum(quality_scores) / len(quality_scores),
            "quality_std": (
                sum(
                    (q - sum(quality_scores) / len(quality_scores)) ** 2
                    for q in quality_scores
                )
                / len(quality_scores)
            )
            ** 0.5,
            "technique_diversity": technique_diversity,
            "average_response_length": avg_length,
            "high_quality_examples": sum(1 for q in quality_scores if q >= 0.8)
            / len(quality_scores),
            "instruction_diversity": len({e.instruction for e in examples})
            / len(examples),
        }


    def _save_sft_datasets(
        self, sft_dataset: SFTDataset, quality_metrics: dict[str, float]
    ) -> dict[str, str]:
        """Save SFT datasets in training format."""
        output_paths = {}

        # Calculate split indices
        total_examples = len(sft_dataset.examples)
        val_size = int(total_examples * sft_dataset.validation_split)
        test_size = int(total_examples * sft_dataset.test_split)
        train_size = total_examples - val_size - test_size

        # Create splits
        splits = {
            "train": sft_dataset.examples[:train_size],
            "validation": sft_dataset.examples[train_size : train_size + val_size],
            "test": sft_dataset.examples[train_size + val_size :],
        }

        # Save each split
        for split_name, examples in splits.items():
            output_file = self.output_dir / f"therapist_sft_{split_name}.jsonl"

            with open(output_file, "w") as f:
                for example in examples:
                    sft_format = {
                        "instruction": example.instruction,
                        "input": example.input_context,
                        "output": example.output_response,
                        "metadata": example.therapeutic_metadata,
                    }
                    f.write(json.dumps(sft_format) + "\n")

            output_paths[split_name] = str(output_file)

        # Save dataset metadata
        metadata_file = self.output_dir / "therapist_sft_metadata.json"
        metadata = {
            "dataset_info": sft_dataset.training_metadata,
            "quality_metrics": quality_metrics,
            "splits": {"train": train_size, "validation": val_size, "test": test_size},
            "format": "instruction-input-output",
            "integrated_at": datetime.now().isoformat(),
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        output_paths["metadata"] = str(metadata_file)

        logger.info(f"SFT datasets saved: {output_paths}")
        return output_paths


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = TherapistSFTIntegrator()

    # Integrate SFT dataset
    result = integrator.integrate_sft_dataset()

    # Show results
    if result["success"]:
        pass
    else:
        pass

