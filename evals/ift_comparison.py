#!/usr/bin/env python3
"""
IFT vs Prompt Engineering Comparison Study

Reproduces Mental-LLM's core experiment on Pixelated's evaluation set:
- zero-shot prompt
- few-shot prompt
- instruction fine-tuned (IFT) model

Metrics: accuracy, F1, clinical relevance, hallucination rate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

<<<<<<< HEAD
<<<<<<< HEAD
from training.mental_health_instruction_dataset import MentalHealthTaskType
=======
from ai.training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
from training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)

logger = logging.getLogger(__name__)


TASK_PROMPTS: dict[str, str] = {
    MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value: (
<<<<<<< HEAD
<<<<<<< HEAD
        "Identify the primary mental health symptoms described in the text. Return a comma-separated list of symptoms."
=======
        "Identify the primary mental health symptoms described in the text. "
        "Return a comma-separated list of symptoms."
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
        "Identify the primary mental health symptoms described in the text. Return a comma-separated list of symptoms."
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
    ),
    MentalHealthTaskType.SEVERITY_ESTIMATION.value: (
        "Estimate the severity of the described mental health symptoms on a scale of 1-10. "
        "Respond with only the number."
    ),
    MentalHealthTaskType.RISK_ASSESSMENT.value: (
<<<<<<< HEAD
<<<<<<< HEAD
        "Assess the level of risk described in the text. Respond with one of: none, low, moderate, high, imminent."
    ),
    MentalHealthTaskType.EMPATHY_SCORING.value: (
        "Score the empathy of the therapist response. Provide an overall empathy score from 1-5."
    ),
    MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value: ("Respond as a compassionate, evidence-based therapist."),
=======
        "Assess the level of risk described in the text. "
        "Respond with one of: none, low, moderate, high, imminent."
=======
        "Assess the level of risk described in the text. Respond with one of: none, low, moderate, high, imminent."
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
    ),
    MentalHealthTaskType.EMPATHY_SCORING.value: (
        "Score the empathy of the therapist response. Provide an overall empathy score from 1-5."
    ),
<<<<<<< HEAD
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
    MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value: ("Respond as a compassionate, evidence-based therapist."),
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
}


FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value: [
        {
            "input": "I can't sleep and I feel hopeless.",
            "output": "insomnia, hopelessness",
        },
        {
            "input": "My heart races in social situations.",
            "output": "anxiety, panic, social avoidance",
        },
    ],
    MentalHealthTaskType.SEVERITY_ESTIMATION.value: [
        {"input": "I feel a little down today.", "output": "2"},
        {"input": "I can't function and think about ending my life.", "output": "9"},
    ],
    MentalHealthTaskType.RISK_ASSESSMENT.value: [
        {"input": "I'm having a good day.", "output": "none"},
        {"input": "I have a plan to hurt myself tonight.", "output": "imminent"},
    ],
    MentalHealthTaskType.EMPATHY_SCORING.value: [
        {"input": "That sounds hard.", "output": "3"},
        {"input": "I hear you, and your feelings make sense. What do you need right now?", "output": "5"},
    ],
    MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value: [
        {
            "input": "I'm overwhelmed at work.",
            "output": "It makes sense that you're feeling overwhelmed. One small step is to identify one task you can set aside for now.",
        },
    ],
}


@dataclass
class ComparisonResult:
    """Result for a single model/prompt approach on a task."""

    approach: str
    task_type: str
    accuracy: float | None
    f1: float | None
    precision: float | None
    recall: float | None
    clinical_relevance: float
    hallucination_rate: float
    mean_latency_ms: float
    n: int


@dataclass
class ComparisonStudyReport:
    """Full comparison study report."""

    model_name: str
    results: list[ComparisonResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def best_approach_per_task(self) -> dict[str, str]:
        """Return best approach per task by F1 (or clinical relevance if F1 unavailable)."""
        by_task: dict[str, list[ComparisonResult]] = {}
        for r in self.results:
            by_task.setdefault(r.task_type, []).append(r)

        best: dict[str, str] = {}
        for task, results in by_task.items():
            best[task] = max(
                results,
<<<<<<< HEAD
<<<<<<< HEAD
                key=lambda r: r.f1 if r.f1 is not None else r.clinical_relevance,
=======
                key=lambda r: (r.f1 if r.f1 is not None else r.clinical_relevance),
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
                key=lambda r: r.f1 if r.f1 is not None else r.clinical_relevance,
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
            ).approach
        return best

    def improvement_over_baseline(self, baseline: str = "zero-shot") -> dict[str, dict[str, float]]:
        """Compute IFT improvement over baseline per task."""
        by_task: dict[str, dict[str, ComparisonResult]] = {}
        for r in self.results:
            by_task.setdefault(r.task_type, {})[r.approach] = r

        improvements: dict[str, dict[str, float]] = {}
        for task, approaches in by_task.items():
            if baseline not in approaches or "ift" not in approaches:
                continue
            base = approaches[baseline]
            ift = approaches["ift"]
            improvements[task] = {
                "accuracy_gain": (ift.accuracy or 0.0) - (base.accuracy or 0.0),
                "f1_gain": (ift.f1 or 0.0) - (base.f1 or 0.0),
                "clinical_relevance_gain": ift.clinical_relevance - base.clinical_relevance,
                "hallucination_reduction": base.hallucination_rate - ift.hallucination_rate,
            }
        return improvements


def _normalize_prediction(pred: Any, task_type: str) -> str:
    """Normalize model output for comparison."""
    if pred is None:
        return ""
    text = str(pred).strip().lower()
    if task_type == MentalHealthTaskType.SEVERITY_ESTIMATION.value:
        import re

        match = re.search(r"\d+", text)
        return match.group() if match else ""
    if task_type == MentalHealthTaskType.EMPATHY_SCORING.value:
        import re

        match = re.search(r"\d+", text)
        return match.group() if match else ""
    return text


def _exact_match(pred: str, ref: str) -> bool:
<<<<<<< HEAD
    return pred.strip().lower() == ref.strip().lower()


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
=======
    return pred == ref.lower().strip()


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.lower().strip().split())
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    fp = len(pred_tokens - ref_tokens)
    fn = len(ref_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _clinical_relevance_score(pred: str, ref: str, task_type: str) -> float:
    """Simple lexical overlap proxy for clinical relevance."""
    if task_type == MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value:
        therapeutic_terms = {
            "feel",
            "understand",
            "support",
            "coping",
            "strategy",
            "validate",
            "safe",
            "help",
            "care",
        }
        pred_tokens = set(pred.split())
        overlap = len(pred_tokens & therapeutic_terms)
        return min(1.0, overlap / 3.0)
    return _token_f1(pred, ref)


def _hallucination_score(pred: str, ref: str) -> float:
<<<<<<< HEAD
<<<<<<< HEAD
    """Proxy hallucination: prediction contains tokens not in reference."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().strip().split())
=======
    """Proxy hallucination: prediction contains tokens not in reference for classification tasks."""
    pred_tokens = set(pred.split(","))
    ref_tokens = set(ref.lower().strip().split(","))
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
    """Proxy hallucination: prediction contains tokens not in reference."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().strip().split())
>>>>>>> 6b3e88de (fix(PIX-3911): Phase 3 bug fixes — bias audit parsing, abs disparity, deque log, hallucination scoring, to_chat fields, test imports)
    if not pred_tokens:
        return 0.0
    extra = pred_tokens - ref_tokens
    return len(extra) / len(pred_tokens)


class IFTComparisonStudy:
    """Runs comparison study between prompt engineering and IFT."""

    def __init__(
        self,
        zero_shot_fn: Callable[[str], str],
        few_shot_fn: Callable[[str], str] | None = None,
        ift_fn: Callable[[str], str] | None = None,
    ):
        self.zero_shot_fn = zero_shot_fn
        self.few_shot_fn = few_shot_fn or zero_shot_fn
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
        if ift_fn is None:
            raise ValueError(
                "IFTComparisonStudy requires an explicit ift_fn. "
                "Passing None silently falls back to zero-shot, defeating the comparison."
            )
        self.ift_fn = ift_fn
<<<<<<< HEAD
=======
        self.ift_fn = ift_fn or zero_shot_fn
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)

    def _build_zero_shot_prompt(self, task_type: str, input_text: str) -> str:
        prompt = TASK_PROMPTS.get(task_type, "")
        return f"{prompt}\n\nInput: {input_text}\n\nOutput:"

    def _build_few_shot_prompt(self, task_type: str, input_text: str) -> str:
        prompt = TASK_PROMPTS.get(task_type, "")
        lines = [prompt, ""]
        for ex in FEW_SHOT_EXAMPLES.get(task_type, []):
            lines.append(f"Input: {ex['input']}")
            lines.append(f"Output: {ex['output']}")
            lines.append("")
        lines.append(f"Input: {input_text}")
        lines.append("Output:")
        return "\n".join(lines)

    def evaluate(
        self,
        examples: list[dict[str, Any]],
        model_name: str = "mental-health-model",
    ) -> ComparisonStudyReport:
        """Evaluate all approaches on the provided examples."""
        results: list[ComparisonResult] = []

        for task_type in TASK_PROMPTS:
            task_examples = [ex for ex in examples if ex.get("task_type") == task_type]
            if not task_examples:
                continue

            for approach, fn, prompt_builder in [
                ("zero-shot", self.zero_shot_fn, self._build_zero_shot_prompt),
                ("few-shot", self.few_shot_fn, self._build_few_shot_prompt),
                ("ift", self.ift_fn, self._build_zero_shot_prompt),
            ]:
<<<<<<< HEAD
<<<<<<< HEAD
                metrics = self._evaluate_approach(task_examples, fn, prompt_builder, task_type)
=======
                metrics = self._evaluate_approach(
                    task_examples, fn, prompt_builder, task_type
                )
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
                metrics = self._evaluate_approach(task_examples, fn, prompt_builder, task_type)
>>>>>>> 30f2438c (fix(PIX-3911): critical pipeline fixes - inference wiring, bias audit, evaluation gates)
                results.append(
                    ComparisonResult(
                        approach=approach,
                        task_type=task_type,
                        **metrics,
                    )
                )

        return ComparisonStudyReport(model_name=model_name, results=results)

    def _evaluate_approach(
        self,
        examples: list[dict[str, Any]],
        inference_fn: Callable[[str], str],
        prompt_builder: Callable[[str, str], str],
        task_type: str,
    ) -> dict[str, Any]:
        """Evaluate a single approach on a task."""
        import time

        latencies: list[float] = []
        correct = 0
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        relevances: list[float] = []
        hallucinations: list[float] = []

        for ex in examples:
            input_text = ex.get("input", "")
            reference = ex.get("output", "")
            prompt = prompt_builder(task_type, input_text)

            start = time.perf_counter()
            raw_pred = inference_fn(prompt)
            latencies.append((time.perf_counter() - start) * 1000)

            pred = _normalize_prediction(raw_pred, task_type)
            ref = _normalize_prediction(reference, task_type)

            if _exact_match(pred, ref):
                correct += 1

            f1 = _token_f1(pred, ref)
            f1s.append(f1)

            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            tp = len(pred_tokens & ref_tokens)
            fp = len(pred_tokens - ref_tokens)
            fn = len(ref_tokens - pred_tokens)
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

            relevances.append(_clinical_relevance_score(pred, ref, task_type))
            hallucinations.append(_hallucination_score(pred, ref))

        n = len(examples)
        return {
            "accuracy": correct / n if n else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "clinical_relevance": float(np.mean(relevances)) if relevances else 0.0,
            "hallucination_rate": float(np.mean(hallucinations)) if hallucinations else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "n": n,
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Demo with deterministic dummy inference
    def dummy_inference(prompt: str) -> str:
        if "symptom" in prompt.lower():
            return "insomnia, hopelessness"
        if "severity" in prompt.lower():
            return "7"
        if "risk" in prompt.lower():
            return "moderate"
        if "empathy" in prompt.lower():
            return "4"
        return "I understand that this is difficult for you."

    examples = [
        {
            "task_type": MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value,
            "input": "I can't sleep and I feel hopeless.",
            "output": "insomnia, hopelessness",
        },
        {
            "task_type": MentalHealthTaskType.SEVERITY_ESTIMATION.value,
            "input": "I feel completely unable to cope.",
            "output": "8",
        },
        {
            "task_type": MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value,
            "input": "I'm overwhelmed at work.",
            "output": "It makes sense that you're feeling overwhelmed.",
        },
    ]

    study = IFTComparisonStudy(
        zero_shot_fn=dummy_inference,
        few_shot_fn=dummy_inference,
        ift_fn=dummy_inference,
    )
    report = study.evaluate(examples, model_name="demo")
    print(json.dumps(report.to_dict(), indent=2))
    print("Best per task:", report.best_approach_per_task())
    print("Improvements:", report.improvement_over_baseline())


if __name__ == "__main__":
    main()
