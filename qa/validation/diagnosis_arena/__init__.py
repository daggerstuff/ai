"""
DiagnosisArena evaluation suite (arXiv 2505.14107).

Public surface::

    from ai.qa.validation.diagnosis_arena import (
        DiagnosisArenaBenchmark,
        HeuristicJudge,
        OpenAIDiagnosisJudge,
        OpenAIBenchmarkPipeline,
        evaluate_case,
        run,
        summarize,
        to_markdown,
        to_json,
        ClinicalCase,
        ModelResponse,
        GeneratedDiagnosis,
        CaseScore,
        BenchmarkSummary,
        EvaluationReport,
        DimensionScore,
        JudgmentResult,
        Difficulty,
        ResponseFormat,
        TierScore,
        DIAGNOSTIC_DIMENSIONS,
        DIMENSION_WEIGHTS,
        BenchmarkArtifactStore,
        classify_errors,
        ERROR_TAXONOMY,
        Leaderboard,
        SystemEvaluation,
        inter_rater_agreement,
        write_leaderboard,
        solve_case_for_system,
    )

Sub-modules::

    ai.qa.validation.diagnosis_arena.benchmark  - case loading + iteration
    ai.qa.validation.diagnosis_arena.judge      - Judge interface + HeuristicJudge + LLMJudge
    ai.qa.validation.diagnosis_arena.openai_judge - GPT-4o-as-judge with 3-way majority vote
    ai.qa.validation.diagnosis_arena.runner     - end-to-end evaluation entry point
    ai.qa.validation.diagnosis_arena.reporter   - summary aggregation + JSON/Markdown reports
    ai.qa.validation.diagnosis_arena.error_taxonomy - diagnostic bias classification
    ai.qa.validation.diagnosis_arena.types      - dataclasses and enums
    ai.qa.validation.diagnosis_arena.pipeline   - multi-system orchestration + leaderboard
    ai.qa.validation.diagnosis_arena.leaderboard - ranking and comparison views
"""

from __future__ import annotations

from .benchmark import DiagnosisArenaBenchmark, case_from_dict
from .error_taxonomy import ERROR_TAXONOMY, ErrorTaxonomy, classify_errors
from .judge import ClinicalDiagnosisJudge, HeuristicJudge, Judge, Judgment, LLMJudge
from .leaderboard import Leaderboard, SystemEvaluation, top_errors, write_leaderboard
from .openai_judge import OpenAIDiagnosisJudge, inter_rater_agreement
from .pipeline import OpenAIBenchmarkPipeline, run_multi_system_benchmark, solve_case_for_system
from .reporter import summarize, to_json, to_markdown, write_report
from .runner import ResponseProducer, evaluate_case, run
from .types import (
    DIAGNOSTIC_DIMENSIONS,
    DIMENSION_WEIGHTS,
    BenchmarkArtifactStore,
    BenchmarkSummary,
    CaseScore,
    ClinicalCase,
    Difficulty,
    DimensionScore,
    EvaluationReport,
    GeneratedDiagnosis,
    JudgmentResult,
    ModelResponse,
    ResponseFormat,
    TierScore,
)

__all__ = [
    "DIAGNOSTIC_DIMENSIONS",
    "DIMENSION_WEIGHTS",
    "ERROR_TAXONOMY",
    "BenchmarkArtifactStore",
    "BenchmarkSummary",
    "CaseScore",
    "ClinicalCase",
    "ClinicalDiagnosisJudge",
    "DiagnosisArenaBenchmark",
    "Difficulty",
    "DimensionScore",
    "ErrorTaxonomy",
    "EvaluationReport",
    "GeneratedDiagnosis",
    "HeuristicJudge",
    "Judge",
    "Judgment",
    "JudgmentResult",
    "LLMJudge",
    "Leaderboard",
    "ModelResponse",
    "OpenAIBenchmarkPipeline",
    "OpenAIDiagnosisJudge",
    "ResponseFormat",
    "ResponseProducer",
    "SystemEvaluation",
    "TierScore",
    "case_from_dict",
    "classify_errors",
    "evaluate_case",
    "inter_rater_agreement",
    "run",
    "run_multi_system_benchmark",
    "solve_case_for_system",
    "summarize",
    "to_json",
    "to_markdown",
    "top_errors",
    "write_leaderboard",
    "write_report",
]
