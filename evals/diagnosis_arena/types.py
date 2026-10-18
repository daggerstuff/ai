"""
DiagnosisArena types.

Core data structures for the diagnostic evaluation suite based on
DiagnosisArena (arXiv 2505.14107). The suite is fully offline: no model
API calls happen inside this package. A pluggable ``Judge`` interface
lets external systems inject GPT-4o-as-judge, an LLM of choice, or a
deterministic mock for unit tests.

Layered scoring mirrors the paper:

- Tier score (3 levels): Identical / Relevant / Irrelevant
- Dimension score (4 axes): Hypothesis, Evidence, Differential, Final
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Difficulty(Enum):
    """Per-case difficulty tiers from the paper."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ResponseFormat(Enum):
    """Open-ended vs MCQ response format."""

    OPEN_ENDED = "open_ended"
    MCQ = "mcq"


class TierScore(Enum):
    """3-tier scoring per the paper."""

    IDENTICAL = "identical"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"

    @property
    def numeric(self) -> float:
        if self is TierScore.IDENTICAL:
            return 1.0
        if self is TierScore.RELEVANT:
            return 0.5
        return 0.0


DIMENSION_WEIGHTS: dict[str, float] = {
    "hypothesis_generation": 0.15,
    "evidence_interpretation": 0.20,
    "differential_diagnosis": 0.30,
    "final_diagnosis": 0.35,
}

DIAGNOSTIC_DIMENSIONS: tuple[str, ...] = tuple(DIMENSION_WEIGHTS.keys())


@dataclass(frozen=True)
class ClinicalCase:
    """A single DiagnosisArena clinical case."""

    case_id: str
    difficulty: Difficulty
    presentation: str
    history: str = ""
    exam: str = ""
    labs: str = ""
    imaging: str = ""
    progression: str = ""
    mcq_options: tuple[str, ...] = ()
    final_diagnosis: str = ""
    differential_diagnoses: tuple[str, ...] = ()
    supporting_evidence: tuple[str, ...] = ()
    key_differentiators: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    """A model's diagnostic response to a ClinicalCase."""

    response_id: str
    case_id: str
    format: ResponseFormat
    hypothesis_list: tuple[str, ...] = ()
    differential_list: tuple[str, ...] = ()
    evidence_cited: tuple[str, ...] = ()
    final_diagnosis: str = ""
    reasoning: str = ""
    mcq_selected: str = ""


@dataclass(frozen=True)
class DimensionScore:
    """Per-dimension scoring in [0, 1]."""

    name: str
    score: float
    rationale: str = ""


@dataclass(frozen=True)
class Judgment:
    """Aggregated judgment for a (case, response) pair."""

    response_id: str
    case_id: str
    tier: TierScore
    dimensions: tuple[DimensionScore, ...] = ()
    judge_model: str = ""
    notes: str = ""

    @property
    def tier_numeric(self) -> float:
        return self.tier.numeric

    @property
    def aggregate_dimension_score(self) -> float:
        """Weighted aggregate of dimension scores per DIMENSION_WEIGHTS."""
        if not self.dimensions:
            return 0.0
        total = 0.0
        for dim in self.dimensions:
            total += dim.score * DIMENSION_WEIGHTS.get(dim.name, 0.0)
        return total


@dataclass(frozen=True)
class CaseScore:
    """Per-case evaluation output."""

    case_id: str
    difficulty: Difficulty
    format: ResponseFormat
    tier: TierScore
    tier_numeric: float
    dimensions: dict[str, float]
    aggregate_dimension_score: float
    error_taxonomy: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate metrics for a model on a benchmark."""

    model_label: str
    format: ResponseFormat
    cases_evaluated: int
    overall_accuracy: float
    by_difficulty: dict[str, float] = field(default_factory=dict)
    by_dimension: dict[str, float] = field(default_factory=dict)
    error_distribution: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""


@dataclass(frozen=True)
class GeneratedDiagnosis:
    """A model's diagnostic response to a ClinicalCase."""

    response_id: str
    case_id: str
    format: ResponseFormat
    hypothesis_list: tuple[str, ...] = ()
    differential_list: tuple[str, ...] = ()
    evidence_cited: tuple[str, ...] = ()
    final_diagnosis: str = ""
    reasoning: str = ""
    mcq_selected: str = ""


@dataclass(frozen=True)
class JudgmentResult:
    """Aggregated judgment for a (case, response) pair, compatible with JSON persistence."""

    response_id: str
    case_id: str
    tier: TierScore
    dimensions: tuple[DimensionScore, ...] = ()
    error_tags: tuple[str, ...] = ()
    latency_ms: float = 0.0
    judge_model: str = ""

    @property
    def tier_numeric(self) -> float:
        return self.tier.numeric

    @property
    def aggregate_dimension_score(self) -> float:
        if not self.dimensions:
            return 0.0
        total = 0.0
        for dim in self.dimensions:
            total += dim.score * DIMENSION_WEIGHTS.get(dim.name, 0.0)
        return total


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics for a model on a benchmark, persisted to JSON/Markdown."""

    model_label: str
    format: ResponseFormat
    total_cases: int
    total_generations: int
    tier_distribution: dict[str, int]
    dimension_stats: dict[str, float]
    overall_accuracy: float
    open_ended_accuracy: float = 0.0
    mcq_accuracy: float = 0.0
    difficulty_breakdown: dict[str, float] = field(default_factory=dict)
    error_taxonomy_counts: dict[str, int] = field(default_factory=dict)
    latency_p95_ms: float = 0.0
    raw_judgments: tuple[JudgmentResult, ...] = ()


@dataclass(frozen=True)
class BenchmarkArtifactStore:
    """On-disk artifact layout for benchmark cases and evaluation reports."""

    root: Path = field(default_factory=lambda: Path("artifacts/diagnosis_arena"))

    def report_path(self) -> Path:
        return self.root / "reports"

    def case_path(self) -> Path:
        return self.root / "cases"

    def write_case(self, case: ClinicalCase) -> Path:
        path = self.case_path() / f"{case.case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "difficulty": case.difficulty.value,
                    "presentation": case.presentation,
                    "history": case.history,
                    "exam": case.exam,
                    "labs": case.labs,
                    "imaging": case.imaging,
                    "progression": case.progression,
                    "mcq_options": list(case.mcq_options),
                    "final_diagnosis": case.final_diagnosis,
                    "differential_diagnoses": list(case.differential_diagnoses),
                    "supporting_evidence": list(case.supporting_evidence),
                    "key_differentiators": list(case.key_differentiators),
                },
                indent=2,
            )
        )
        return path

    def case_manifest(self) -> list[str]:
        return sorted(p.name.removesuffix(".json") for p in self.case_path().glob("*.json"))

    def latest_report(self) -> Path | None:
        reports = sorted(self.report_path().glob("report-*.json"))
        return reports[-1] if reports else None
