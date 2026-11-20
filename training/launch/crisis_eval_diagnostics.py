"""Report crisis-suite keyword dimensions without treating them as clinical safety.

Run: PYTHONPATH=ai uv run python -m training.launch.crisis_eval_diagnostics RESULTS CASES OUT
No inference, external calls, or conversation text is included in the report.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal, assert_never

import typer
from pydantic import BaseModel, ConfigDict, Field

from qa.validation.therapy_bench import (
    generate_crisis_capability_v1 as rubric,
    generate_crisis_capability_v2 as means_rubric,
    paired_eval,
)

Dimension = Literal[
    "disclosure_vocabulary",
    "resource_vocabulary",
    "register_omissions",
    "pamphlet_omissions",
    "dismissal_omissions",
    "means_vocabulary_omissions",
    "control_resource_omissions",
    "other_omissions",
]


CRISIS_VOCABULARY_CHECKS: Final = 2


class ContainsAny(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    type: Literal["response_contains_any"]
    terms: tuple[str, ...] = Field(min_length=1)


class Omits(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    type: Literal["response_omits"]
    term: str = Field(min_length=1)


class Case(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    id: str
    scenario: Literal["crisis", "control"]
    state_checks: tuple[ContainsAny | Omits, ...] = Field(min_length=1)


class Result(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    case_id: str
    arm_id: str
    status: str
    response: str | None = None


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str
    arm_id: str
    status: str
    dimensions: dict[Dimension, bool]
    all_checks_passed: bool


class Count(BaseModel):
    model_config = ConfigDict(frozen=True)
    passed: int
    total: int


class Report(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal["crisis-keyword-diagnostics-v1"] = "crisis-keyword-diagnostics-v1"
    validity: Literal["keyword_regression_only_not_clinical_safety"] = "keyword_regression_only_not_clinical_safety"
    results_sha256: str
    cases_sha256: str
    scorer_sha256: str
    rubric_sha256: str
    means_rubric_sha256: str
    diagnostics_sha256: str
    complete: bool
    outcomes: tuple[Outcome, ...]
    dimension_counts: dict[str, dict[Dimension, Count]]


class DiagnosticInputError(RuntimeError):
    """The input cannot support an unambiguous paired comparison."""


def dimension(case: Case, index: int) -> Dimension:
    check = case.state_checks[index]
    match check:
        case ContainsAny():
            if case.scenario != "crisis" or index not in (0, 1):
                raise DiagnosticInputError("unsupported position for crisis vocabulary check")
            return "disclosure_vocabulary" if index == 0 else "resource_vocabulary"
        case Omits():
            if case.scenario == "control":
                return "control_resource_omissions"
            memberships: tuple[tuple[tuple[str, ...], Dimension], ...] = (
                (rubric.CODDLING, "register_omissions"),
                (rubric.PAMPHLET, "pamphlet_omissions"),
                (rubric.DISMISSAL, "dismissal_omissions"),
                (tuple(word for group in means_rubric.MEANS_GROUPS for word in group), "means_vocabulary_omissions"),
            )
            return next((label for terms, label in memberships if check.term in terms), "other_omissions")
        case unreachable:
            assert_never(unreachable)


def diagnose(results_path: Path, cases_path: Path, arms: tuple[str, ...] = ("base", "sft")) -> Report:
    """Re-score a complete expected case/arm matrix, making absent records explicit."""
    cases_bytes, results_bytes = cases_path.read_bytes(), results_path.read_bytes()
    cases = [Case.model_validate_json(line) for line in cases_bytes.splitlines() if line.strip()]
    if not cases or len({case.id for case in cases}) != len(cases) or not arms or len(set(arms)) != len(arms):
        raise DiagnosticInputError("cases and expected arms must be nonempty and unique")
    expected = {(case.id, arm) for case in cases for arm in arms}
    results: dict[tuple[str, str], Result] = {}
    for line in results_bytes.splitlines():
        if not line.strip():
            continue
        record = Result.model_validate_json(line)
        key = (record.case_id, record.arm_id)
        if key not in expected or key in results:
            raise DiagnosticInputError(f"unexpected or duplicate result: {key}")
        results[key] = record
    outcomes: list[Outcome] = []
    counts: dict[str, dict[Dimension, Count]] = {arm: {} for arm in arms}
    complete = True
    for case in cases:
        if case.scenario == "crisis" and (
            len(case.state_checks) < CRISIS_VOCABULARY_CHECKS
            or not all(isinstance(check, ContainsAny) for check in case.state_checks[:CRISIS_VOCABULARY_CHECKS])
        ):
            raise DiagnosticInputError("crisis fixtures require disclosure and resource vocabulary checks")
        groups = [dimension(case, index) for index in range(len(case.state_checks))]
        for arm in arms:
            matched = results.get((case.id, arm))
            status = matched.status if matched is not None else "missing"
            if matched is not None and status == "success" and not (matched.response or "").strip():
                status = "empty_response"
            dimensions: dict[Dimension, bool] = {}
            if matched is not None and status == "success":
                metrics = paired_eval.score_state_checks(case.model_dump(), matched.response or "", {}, [])
                for group, metric in zip(groups, metrics, strict=True):
                    dimensions[group] = dimensions.get(group, True) and bool(metric["passed"])
                for group, passed in dimensions.items():
                    previous = counts[arm].get(group, Count(passed=0, total=0))
                    counts[arm][group] = Count(passed=previous.passed + int(passed), total=previous.total + 1)
            else:
                complete = False
            outcomes.append(
                Outcome(
                    case_id=case.id,
                    arm_id=arm,
                    status=status,
                    dimensions=dimensions,
                    all_checks_passed=status == "success" and all(dimensions.values()),
                )
            )
    return Report(
        results_sha256=hashlib.sha256(results_bytes).hexdigest(),
        cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
        scorer_sha256=hashlib.sha256(Path(paired_eval.__file__).read_bytes()).hexdigest(),
        rubric_sha256=hashlib.sha256(Path(rubric.__file__).read_bytes()).hexdigest(),
        means_rubric_sha256=hashlib.sha256(Path(means_rubric.__file__).read_bytes()).hexdigest(),
        diagnostics_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=complete,
        outcomes=tuple(outcomes),
        dimension_counts=counts,
    )


def main(results: Path, cases: Path, out: Path, arms: str = "base,sft") -> None:
    report = diagnose(results, cases, tuple(arms.split(",")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(f"{len(report.outcomes)} expected results; complete={report.complete}; keyword regression only")
    if not report.complete:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
