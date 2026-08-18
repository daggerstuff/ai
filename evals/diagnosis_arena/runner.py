"""
DiagnosisArena evaluation runner.

Ties the benchmark, judge, and reporter together: given a benchmark, a
model response producer, and a judge, evaluate every (case, response)
pair, return per-case scores and an aggregate summary.

The model response producer is a callable matching
``(case: ClinicalCase, format: ResponseFormat) -> ModelResponse``
so the runner is agnostic to whether responses come from a real model,
a recorded log, or a synthetic fixture.
"""

from __future__ import annotations

from collections.abc import Callable

from .benchmark import DiagnosisArenaBenchmark
from .error_taxonomy import classify_errors
from .judge import Judge
from .reporter import summarize
from .types import (
    BenchmarkSummary,
    CaseScore,
    ModelResponse,
    ResponseFormat,
)

ResponseProducer = Callable[[object, ResponseFormat], ModelResponse]


def evaluate_case(
    case,
    response: ModelResponse,
    judge: Judge,
) -> CaseScore:
    """Score one (case, response) pair and return a CaseScore."""
    from .types import ClinicalCase  # local import to keep types lazy

    if not isinstance(case, ClinicalCase):
        msg = f"expected ClinicalCase, got {type(case).__name__}"
        raise TypeError(msg)

    judgment = judge.judge(case, response)
    dims = {d.name: d.score for d in judgment.dimensions}
    errors = classify_errors(case, response, judgment.tier)
    return CaseScore(
        case_id=case.case_id,
        difficulty=case.difficulty,
        format=response.format,
        tier=judgment.tier,
        tier_numeric=judgment.tier_numeric,
        dimensions=dims,
        aggregate_dimension_score=judgment.aggregate_dimension_score,
        error_taxonomy=errors,
    )


def run(
    benchmark: DiagnosisArenaBenchmark,
    producer: ResponseProducer,
    judge: Judge,
    *,
    response_format: ResponseFormat = ResponseFormat.OPEN_ENDED,
    model_label: str = "model",
) -> tuple[list[CaseScore], BenchmarkSummary]:
    """
    Evaluate the full benchmark.

    Returns (per-case scores, aggregate summary). The producer is called
    once per case.
    """
    scores: list[CaseScore] = []
    for case in benchmark:
        response = producer(case, response_format)
        scores.append(evaluate_case(case, response, judge))
    summary = summarize(model_label, response_format, scores)
    return scores, summary


__all__ = ["evaluate_case", "run", "ResponseProducer"]
