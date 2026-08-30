#!/usr/bin/env python3
"""
Evaluation Suite for Fine-Tuned Memory-Augmented Models

Evaluates fine-tuned models on:
- Memory recall precision and recall
- Context relevance scores
- Reflection quality assessments
- Comparison against base model

Usage:
    python -m ai.training.evaluate_finetuned_model \
        --model-path ./models/fine-tuned \
        --test-data ./data/finetuning/finetuning_test.jsonl \
        --base-model zai-org/glm-5.3-flash
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single example."""

    example_id: str
    example_type: str

    # Memory metrics
    memory_recall_precision: float = 0.0
    memory_recall_recall: float = 0.0
    memory_relevance_score: float = 0.0

    # Generation metrics
    generation_quality: float = 0.0
    context_relevance: float = 0.0
    reflection_quality: float = 0.0

    # Metadata
    predicted_text: str = ""
    target_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "example_type": self.example_type,
            "memory_recall_precision": self.memory_recall_precision,
            "memory_recall_recall": self.memory_recall_recall,
            "memory_relevance_score": self.memory_relevance_score,
            "generation_quality": self.generation_quality,
            "context_relevance": self.context_relevance,
            "reflection_quality": self.reflection_quality,
            "predicted_text": self.predicted_text,
            "target_text": self.target_text,
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""

    model_path: str
    test_examples: int
    evaluated_examples: int

    # Memory metrics
    avg_memory_recall_precision: float = 0.0
    avg_memory_recall_recall: float = 0.0
    avg_memory_relevance: float = 0.0

    # Quality metrics
    avg_generation_quality: float = 0.0
    avg_context_relevance: float = 0.0
    avg_reflection_quality: float = 0.0

    # Overall
    overall_score: float = 0.0

    # Detailed results
    results: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "test_examples": self.test_examples,
            "evaluated_examples": self.evaluated_examples,
            "memory_metrics": {
                "avg_recall_precision": self.avg_memory_recall_precision,
                "avg_recall_recall": self.avg_memory_recall_recall,
                "avg_relevance": self.avg_memory_relevance,
            },
            "quality_metrics": {
                "avg_generation_quality": self.avg_generation_quality,
                "avg_context_relevance": self.avg_context_relevance,
                "avg_reflection_quality": self.avg_reflection_quality,
            },
            "overall_score": self.overall_score,
        }


class ModelEvaluator:
    """Evaluator for fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "zai-org/glm-5.3-flash",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.device = device

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if (model_path / "tokenizer_config.json").exists() else base_model_name
        )
        self.model.eval()

        logger.info(f"Model loaded on device: {device}")

    def evaluate(
        self,
        test_data_path: str | Path,
        max_samples: int | None = None,
    ) -> EvaluationReport:
        """
        Evaluate model on test data.

        Args:
            test_data_path: Path to test data JSONL
            max_samples: Maximum number of samples to evaluate

        Returns:
            Evaluation report
        """
        test_data_path = Path(test_data_path)

        # Load test examples
        test_examples = self._load_test_data(test_data_path, max_samples)

        logger.info(f"Evaluating on {len(test_examples)} test examples")

        # Evaluate each example
        results = []
        for example in tqdm(test_examples, desc="Evaluating"):
            result = self._evaluate_example(example)
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results, len(test_examples))

    def _load_test_data(
        self,
        path: Path,
        max_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load test data from JSONL."""
        examples = []

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse line {i}")

        return examples

    def _evaluate_example(self, example: dict[str, Any]) -> EvaluationResult:
        """Evaluate a single example."""
        example_id = example.get("id", "unknown")
        example_type = example.get("example_type", "standard")
        input_text = example.get("input", "")
        target_text = example.get("target", "")
        memories = example.get("relevant_memories", [])

        # Generate prediction
        predicted_text = self._generate_response(input_text, memories)

        # Compute metrics
        memory_precision, memory_recall = self._compute_memory_metrics(
            predicted_text, memories
        )

        relevance_score = self._compute_relevance_score(
            predicted_text, target_text
        )

        generation_quality = self._compute_generation_quality(
            predicted_text, target_text
        )

        context_relevance = self._compute_context_relevance(
            predicted_text, input_text
        )

        reflection_quality = self._compute_reflection_quality(
            predicted_text, example_type
        )

        return EvaluationResult(
            example_id=example_id,
            example_type=example_type,
            memory_recall_precision=memory_precision,
            memory_recall_recall=memory_recall,
            memory_relevance_score=relevance_score,
            generation_quality=generation_quality,
            context_relevance=context_relevance,
            reflection_quality=reflection_quality,
            predicted_text=predicted_text,
            target_text=target_text,
        )

    def _generate_response(
        self,
        input_text: str,
        memories: list[dict[str, Any]] | None = None,
        max_length: int = 512,
    ) -> str:
        """Generate response from model."""
        # Format input with memory context
        if memories:
            memory_context = "\n".join(
                f"[{m.get('category', 'general')}] {m.get('content', '')}"
                for m in memories[:5]
            )
            full_input = f"Relevant memories:\n{memory_context}\n\n{input_text}"
        else:
            full_input = input_text

        # Tokenize
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def _compute_memory_metrics(
        self,
        predicted_text: str,
        memories: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """Compute memory recall precision and recall."""
        if not memories:
            return 1.0, 1.0

        # Check if predicted text contains memory content
        memory_contents = [m.get("content", "").lower() for m in memories]
        predicted_lower = predicted_text.lower()

        matches = sum(1 for content in memory_contents if content in predicted_lower)

        precision = matches / len(memories) if memories else 1.0
        recall = matches / max(len(memories), 1)

        return precision, recall

    def _compute_relevance_score(
        self,
        predicted_text: str,
        target_text: str,
    ) -> float:
        """Compute relevance score using simple text overlap."""
        if not target_text:
            return 0.0

        predicted_words = set(predicted_text.lower().split())
        target_words = set(target_text.lower().split())

        if not target_words:
            return 0.0

        overlap = len(predicted_words & target_words)
        score = overlap / len(target_words)

        return min(1.0, score)

    def _compute_generation_quality(
        self,
        predicted_text: str,
        target_text: str,
    ) -> float:
        """Compute generation quality score."""
        if not target_text or not predicted_text:
            return 0.0

        # Simple BLEU-like approximation
        predicted_words = predicted_text.lower().split()
        target_words = target_text.lower().split()

        if not predicted_words or not target_words:
            return 0.0

        # Word overlap ratio
        overlap = len(set(predicted_words) & set(target_words))
        quality = overlap / max(len(predicted_words), len(target_words))

        return min(1.0, quality)

    def _compute_context_relevance(
        self,
        predicted_text: str,
        input_text: str,
    ) -> float:
        """Compute context relevance score."""
        if not input_text or not predicted_text:
            return 0.0

        # Check if predicted text is relevant to input
        input_words = set(input_text.lower().split())
        predicted_words = set(predicted_text.lower().split())

        overlap = len(input_words & predicted_words)
        relevance = overlap / max(len(input_words), 1)

        return min(1.0, relevance)

    def _compute_reflection_quality(
        self,
        predicted_text: str,
        example_type: str,
    ) -> float:
        """Compute reflection quality score."""
        if not predicted_text:
            return 0.0

        # Check for reflection indicators
        reflection_indicators = [
            "insight", "reflect", "understand", "learned",
            "pattern", "awareness", "realize", "growth"
        ]

        predicted_lower = predicted_text.lower()
        matches = sum(1 for indicator in reflection_indicators
                     if indicator in predicted_lower)

        quality = matches / len(reflection_indicators)

        # Boost for memory-related example types
        if "memory" in example_type:
            quality = min(1.0, quality * 1.2)

        return quality

    def _aggregate_results(
        self,
        results: list[EvaluationResult],
        total_examples: int,
    ) -> EvaluationReport:
        """Aggregate individual results into report."""
        if not results:
            return EvaluationReport(
                model_path=str(self.model_path),
                test_examples=total_examples,
                evaluated_examples=0,
            )

        # Compute averages
        avg_memory_precision = statistics.mean(
            r.memory_recall_precision for r in results
        )
        avg_memory_recall = statistics.mean(
            r.memory_recall_recall for r in results
        )
        avg_memory_relevance = statistics.mean(
            r.memory_relevance_score for r in results
        )
        avg_generation_quality = statistics.mean(
            r.generation_quality for r in results
        )
        avg_context_relevance = statistics.mean(
            r.context_relevance for r in results
        )
        avg_reflection_quality = statistics.mean(
            r.reflection_quality for r in results
        )

        # Overall score (weighted average)
        overall_score = (
            avg_memory_precision * 0.25 +
            avg_memory_recall * 0.25 +
            avg_generation_quality * 0.25 +
            avg_context_relevance * 0.25
        )

        return EvaluationReport(
            model_path=str(self.model_path),
            test_examples=total_examples,
            evaluated_examples=len(results),
            avg_memory_recall_precision=avg_memory_precision,
            avg_memory_recall_recall=avg_memory_recall,
            avg_memory_relevance=avg_memory_relevance,
            avg_generation_quality=avg_generation_quality,
            avg_context_relevance=avg_context_relevance,
            avg_reflection_quality=avg_reflection_quality,
            overall_score=overall_score,
            results=results,
        )



def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSONL",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="zai-org/glm-5.3-flash",
        help="Base model name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report (JSON)",
    )

    args = parser.parse_args()

    # Evaluate
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )

    report = evaluator.evaluate(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
    )

    # Print summary

    # Save report if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
