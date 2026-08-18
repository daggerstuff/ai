from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PriorityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GateDecision(StrEnum):
    PASS = "pass"
    BLOCK = "block"
    EXCEPTION = "exception"


SCORE_TIER_HIGH: float = 8.0
SCORE_FLOOR: float = 6.0


@dataclass
class SourceIntake:
    source_id: str
    name: str
    category: str
    license_id: str
    pii_class: str
    provenance: str
    reproducible: bool
    reviewer: str | None = None
    review_date: str | None = None

    def __str__(self) -> str:
        return f"SourceIntake({self.source_id}: {self.name}, license={self.license_id}, pii={self.pii_class})"


@dataclass
class PilotReport:
    source_id: str
    sample_size: int
    population_size: int
    schema_coverage_pct: float
    dedup_rate: float
    therapeutic_relevance_score: int
    overall_pilot_score: float
    notes: str = ""

    def __str__(self) -> str:
        return f"PilotReport({self.source_id}: {self.sample_size}/{self.population_size} samples, score={self.overall_pilot_score})"


@dataclass
class CurationExitReport:
    source_id: str
    net_retention_pct: float
    schema_validation_pct: float
    manifest_signed: bool
    records_passed: int
    records_rejected: int

    def __str__(self) -> str:
        return (
            f"CurationExitReport({self.source_id}: retention={self.net_retention_pct}%, signed={self.manifest_signed})"
        )


@dataclass
class AcquisitionScore:
    therapeutic_relevance: int
    data_structure_quality: int
    training_integration: int
    ethical_accessibility: int
    overall_score: float

    @property
    def priority_tier(self) -> PriorityTier:
        if self.overall_score >= SCORE_TIER_HIGH:
            return PriorityTier.HIGH
        if self.overall_score >= SCORE_FLOOR:
            return PriorityTier.MEDIUM
        return PriorityTier.LOW

    @property
    def passes_score_floor(self) -> bool:
        return self.overall_score >= SCORE_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "therapeutic_relevance": self.therapeutic_relevance,
            "data_structure_quality": self.data_structure_quality,
            "training_integration": self.training_integration,
            "ethical_accessibility": self.ethical_accessibility,
            "overall_score": self.overall_score,
            "priority_tier": self.priority_tier.value,
            "passes_score_floor": self.passes_score_floor,
        }


@dataclass
class GateResult:
    gate: str
    decision: GateDecision
    details: str

    def to_dict(self) -> dict[str, str]:
        return {"gate": self.gate, "decision": self.decision.value, "details": self.details}


@dataclass
class IntakeDecision:
    source: SourceIntake
    qualifying: list[str]
    blocking: list[str]
    passed: bool
    exception_granted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source.source_id,
            "passed": self.passed,
            "exception_granted": self.exception_granted,
            "qualifying": self.qualifying,
            "blocking": self.blocking,
        }


@dataclass
class PilotDecision:
    report: PilotReport
    gates: list[GateResult]
    passed: bool
    exception_granted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.report.source_id,
            "passed": self.passed,
            "exception_granted": self.exception_granted,
            "gates": [g.to_dict() for g in self.gates],
        }


@dataclass
class CurationExitDecision:
    report: CurationExitReport
    gates: list[GateResult]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.report.source_id,
            "passed": self.passed,
            "gates": [g.to_dict() for g in self.gates],
        }


APPROVED_LICENSES: set[str] = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "unlicense",
    "odc-by",
    "odbl-1.0",
}

EXCEPTION_LICENSES: set[str] = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
}

SCORE_WEIGHTS: dict[str, float] = {
    "therapeutic_relevance": 0.35,
    "data_structure_quality": 0.25,
    "training_integration": 0.20,
    "ethical_accessibility": 0.20,
}

GATE_1_SCORE_FLOOR: float = 6.0
GATE_1_RELEVANCE_FLOOR: int = 6
GATE_1_SCHEMA_FLOOR: float = 95.0
GATE_1_DEDUP_CEILING: float = 50.0

GATE_2_RETENTION_FLOOR: float = 30.0
GATE_2_SCHEMA_FLOOR: float = 99.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def calculate_overall_score(
    therapeutic_relevance: int,
    data_structure_quality: int,
    training_integration: int,
    ethical_accessibility: int,
) -> AcquisitionScore:
    overall = (
        therapeutic_relevance * SCORE_WEIGHTS["therapeutic_relevance"]
        + data_structure_quality * SCORE_WEIGHTS["data_structure_quality"]
        + training_integration * SCORE_WEIGHTS["training_integration"]
        + ethical_accessibility * SCORE_WEIGHTS["ethical_accessibility"]
    )
    return AcquisitionScore(
        therapeutic_relevance=_clamp(therapeutic_relevance, 1, 10),
        data_structure_quality=_clamp(data_structure_quality, 1, 10),
        training_integration=_clamp(training_integration, 1, 10),
        ethical_accessibility=_clamp(ethical_accessibility, 1, 10),
        overall_score=round(overall, 2),
    )


def score_from_evaluation(
    therapeutic_relevance: int, data_structure_quality: int, training_integration: int, ethical_accessibility: int
) -> AcquisitionScore:
    return calculate_overall_score(
        therapeutic_relevance=therapeutic_relevance,
        data_structure_quality=data_structure_quality,
        training_integration=training_integration,
        ethical_accessibility=ethical_accessibility,
    )


class AcquisitionRubric:
    def evaluate_intake(self, source: SourceIntake) -> IntakeDecision:
        qualifying: list[str] = []
        blocking: list[str] = []

        if source.license_id in APPROVED_LICENSES:
            qualifying.append(f"license {source.license_id} is approved")
        elif source.license_id in EXCEPTION_LICENSES:
            blocking.append(f"license {source.license_id} requires an explicit exception grant before intake")
        else:
            blocking.append(f"license {source.license_id} is not approved and not exception-eligible")

        if source.pii_class == "none":
            qualifying.append("no PII/PHI risk")
        elif source.pii_class in ("low", "medium"):
            qualifying.append(f"PII class {source.pii_class} — mitigations required before Gate 1")
        else:
            blocking.append(f"PII class {source.pii_class} exceeds acceptable intake risk")

        if source.provenance in ("trusted", "verified"):
            qualifying.append(f"provenance is {source.provenance}")
        else:
            blocking.append(f"provenance {source.provenance} — insufficient source traceability")

        if source.reproducible:
            qualifying.append("acquisition is reproducible")
        else:
            blocking.append("acquisition is not reproducible")

        if source.reviewer and source.review_date:
            qualifying.append(f"reviewed by {source.reviewer} on {source.review_date}")
        else:
            blocking.append("source lacks reviewer sign-off")

        passed = len(blocking) == 0

        return IntakeDecision(
            source=source,
            qualifying=qualifying,
            blocking=blocking,
            passed=passed,
        )

    def grant_intake_exception(self, source: SourceIntake, reason: str) -> IntakeDecision:
        base = self.evaluate_intake(source)
        if source.license_id not in EXCEPTION_LICENSES:
            base.exception_granted = False
            return base

        blocked_after_exception = [
            b for b in base.blocking if "license" not in b.lower() and "exception" not in b.lower()
        ]
        if blocked_after_exception:
            base.blocking[:] = blocked_after_exception
            base.passed = False
            base.exception_granted = False
            return base

        base.blocking.clear()
        base.qualifying.append(f"exception granted: {reason}")
        base.passed = True
        base.exception_granted = True
        return base

    def evaluate_pilot(self, report: PilotReport) -> PilotDecision:
        gates: list[GateResult] = []

        if report.overall_pilot_score >= GATE_1_SCORE_FLOOR:
            gates.append(
                GateResult(
                    "pilot_score_floor",
                    GateDecision.PASS,
                    f"overall score {report.overall_pilot_score} >= {GATE_1_SCORE_FLOOR}",
                )
            )
        else:
            gates.append(
                GateResult(
                    "pilot_score_floor",
                    GateDecision.BLOCK,
                    f"overall score {report.overall_pilot_score} < {GATE_1_SCORE_FLOOR}; floor is non-exceptable",
                )
            )

        if report.therapeutic_relevance_score >= GATE_1_RELEVANCE_FLOOR:
            gates.append(
                GateResult(
                    "therapeutic_relevance",
                    GateDecision.PASS,
                    f"relevance {report.therapeutic_relevance_score} >= {GATE_1_RELEVANCE_FLOOR}",
                )
            )
        else:
            gates.append(
                GateResult(
                    "therapeutic_relevance",
                    GateDecision.BLOCK,
                    f"Therapeutic Relevance {report.therapeutic_relevance_score} < {GATE_1_RELEVANCE_FLOOR}",
                )
            )

        if report.schema_coverage_pct >= GATE_1_SCHEMA_FLOOR:
            gates.append(
                GateResult(
                    "schema_coverage",
                    GateDecision.PASS,
                    f"coverage {report.schema_coverage_pct}% >= {GATE_1_SCHEMA_FLOOR}%",
                )
            )
        else:
            gates.append(
                GateResult(
                    "schema_coverage",
                    GateDecision.BLOCK,
                    f"schema coverage {report.schema_coverage_pct}% < {GATE_1_SCHEMA_FLOOR}%",
                )
            )

        if report.dedup_rate < GATE_1_DEDUP_CEILING:
            gates.append(
                GateResult(
                    "dedup_rate", GateDecision.PASS, f"dedup rate {report.dedup_rate}% < {GATE_1_DEDUP_CEILING}%"
                )
            )
        else:
            gates.append(
                GateResult(
                    "dedup_rate", GateDecision.BLOCK, f"dedup rate {report.dedup_rate}% >= {GATE_1_DEDUP_CEILING}%"
                )
            )

        passed = all(g.decision == GateDecision.PASS for g in gates)

        return PilotDecision(report=report, gates=gates, passed=passed)

    def evaluate_curation_exit(self, report: CurationExitReport) -> CurationExitDecision:
        gates: list[GateResult] = []

        if report.net_retention_pct >= GATE_2_RETENTION_FLOOR:
            gates.append(
                GateResult(
                    "net_retention",
                    GateDecision.PASS,
                    f"net retention {report.net_retention_pct}% >= {GATE_2_RETENTION_FLOOR}%",
                )
            )
        else:
            gates.append(
                GateResult(
                    "net_retention",
                    GateDecision.BLOCK,
                    f"net retention {report.net_retention_pct}% < {GATE_2_RETENTION_FLOOR}%",
                )
            )

        if report.schema_validation_pct >= GATE_2_SCHEMA_FLOOR:
            gates.append(
                GateResult(
                    "schema_validation",
                    GateDecision.PASS,
                    f"schema validation {report.schema_validation_pct}% >= {GATE_2_SCHEMA_FLOOR}%",
                )
            )
        else:
            gates.append(
                GateResult(
                    "schema_validation",
                    GateDecision.BLOCK,
                    f"schema validation {report.schema_validation_pct}% < {GATE_2_SCHEMA_FLOOR}%",
                )
            )

        if report.manifest_signed:
            gates.append(GateResult("manifest_signed", GateDecision.PASS, "batch manifest signed"))
        else:
            gates.append(GateResult("manifest_signed", GateDecision.BLOCK, "batch manifest not signed"))

        passed = all(g.decision == GateDecision.PASS for g in gates)

        return CurationExitDecision(report=report, gates=gates, passed=passed)

    def promote(
        self, intake: SourceIntake, pilot: PilotReport | None = None, curation_exit: CurationExitReport | None = None
    ) -> list[GateResult]:
        gates: list[GateResult] = []

        intake_result = self.evaluate_intake(intake)
        if intake_result.passed:
            gates.append(GateResult("intake", GateDecision.PASS, "intake passed all qualifying checks"))
        else:
            gates.append(
                GateResult("intake", GateDecision.BLOCK, f"intake blocked: {'; '.join(intake_result.blocking)}")
            )
            return gates

        if pilot is not None:
            if pilot.source_id != intake.source_id:
                gates.append(
                    GateResult(
                        "pilot",
                        GateDecision.BLOCK,
                        f"source_id mismatch: pilot={pilot.source_id} != intake={intake.source_id}",
                    )
                )
                return gates
            pilot_result = self.evaluate_pilot(pilot)
            gates.append(
                GateResult(
                    "pilot",
                    GateDecision.PASS if pilot_result.passed else GateDecision.BLOCK,
                    f"pilot {'passed' if pilot_result.passed else 'failed'} "
                    f"({pilot.sample_size}/{pilot.population_size} samples)",
                )
            )
            if not pilot_result.passed:
                return gates

        if curation_exit is not None:
            if curation_exit.source_id != intake.source_id:
                gates.append(
                    GateResult(
                        "curation_exit",
                        GateDecision.BLOCK,
                        f"source_id mismatch: curation_exit={curation_exit.source_id} != intake={intake.source_id}",
                    )
                )
                return gates
            exit_result = self.evaluate_curation_exit(curation_exit)
            gates.append(
                GateResult(
                    "curation_exit",
                    GateDecision.PASS if exit_result.passed else GateDecision.BLOCK,
                    f"curation exit {'passed' if exit_result.passed else 'failed'}",
                )
            )

        return gates


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Acquisition Rubric CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    score_p = sub.add_parser("score", help="Calculate overall score for 4 dimensions")
    for _dim in ("therapeutic_relevance", "data_structure_quality", "training_integration", "ethical_accessibility"):
        score_p.add_argument(f"--{_dim.replace('_', '-')}", type=int, required=True, help=f"{_dim.replace('_', ' ')} (1-10)")

    intake_p = sub.add_parser("intake", help="Evaluate Gate 0 intake for a source")
    intake_p.add_argument("--source-id", required=True)
    intake_p.add_argument("--name", required=True)
    intake_p.add_argument("--category", default="academic")
    intake_p.add_argument("--license-id", required=True)
    intake_p.add_argument("--pii-class", default="none")
    intake_p.add_argument("--provenance", default="trusted")
    intake_p.add_argument("--reproducible", action="store_true")
    intake_p.add_argument("--reviewer", default=None)
    intake_p.add_argument("--review-date", default=None)

    args = parser.parse_args()

    rubric = AcquisitionRubric()

    if args.command == "score":
        score = calculate_overall_score(
            therapeutic_relevance=args.therapeutic_relevance,
            data_structure_quality=args.data_structure_quality,
            training_integration=args.training_integration,
            ethical_accessibility=args.ethical_accessibility,
        )
        sys.stdout.write(
            f"overall_score={score.overall_score}, "
            f"priority_tier={score.priority_tier.value}, "
            f"passes_score_floor={score.passes_score_floor}\n"
        )
        sys.exit(0)

    if args.command == "intake":
        source = SourceIntake(
            source_id=args.source_id,
            name=args.name,
            category=args.category,
            license_id=args.license_id,
            pii_class=args.pii_class,
            provenance=args.provenance,
            reproducible=args.reproducible,
            reviewer=args.reviewer,
            review_date=args.review_date,
        )
        decision = rubric.evaluate_intake(source)
        for _q in decision.qualifying:
            pass
        for _b in decision.blocking:
            pass
        sys.exit(0 if decision.passed else 1)


if __name__ == "__main__":
    _run_cli()


__all__ = [
    "APPROVED_LICENSES",
    "EXCEPTION_LICENSES",
    "AcquisitionRubric",
    "AcquisitionScore",
    "CurationExitDecision",
    "CurationExitReport",
    "GateDecision",
    "GateResult",
    "IntakeDecision",
    "PilotDecision",
    "PilotReport",
    "PriorityTier",
    "SourceIntake",
    "calculate_overall_score",
    "score_from_evaluation",
]
