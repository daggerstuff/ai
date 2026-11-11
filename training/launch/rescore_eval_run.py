"""Re-score stored paired-eval results against the current case fixtures.

The harness records one verdict per deterministic check but not the check
itself, so a rubric change cannot be applied to results already on disk. This
tool re-runs the response-based checks from the fixture files against the stored
responses and reports the recorded verdict next to the re-scored one, so a
rubric revision can be evaluated on evidence already collected instead of
re-running every arm.

Checks that read tool or memory state are carried over unchanged: the harness
stores only a hash of the final state, not the state itself, so they cannot be
recomputed here. For the crisis suites every check is response-based, so the
carry-over does not apply to them.

A run recorded under an older rubric has a different number of checks per case
than the fixture now has, so there is no position-by-position correspondence to
merge on. Those rows are reported as unreconciled rather than guessed at.

Resource-name, coddling-phrase, and brochure-phrase frequency are reported
alongside the verdicts. They describe what the arms say; they are not scoring
inputs, and they are computed from the stored reply alone, so they still work
on runs recorded under an older rubric.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa.validation.therapy_bench.generate_crisis_capability_v1 import CODDLING, PAMPHLET
from qa.validation.therapy_bench.paired_eval import score_state_checks
from training.launch.summarize_eval_run import load_manifest, run_labels, run_suite_and_pair

RESPONSE_CHECK_TYPES = frozenset({"response_contains", "response_contains_any", "response_omits", "response_enum"})

# Reported, never scored: how often a reply names a crisis resource.
RESOURCE_TERMS = (
    "988",
    "crisis line",
    "hotline",
    "911",
    "emergency",
    "poison control",
    "hospital",
    "ambulance",
    "safety plan",
    "are you safe",
)


@dataclass
class CaseScore:
    """One case result for one arm, before and after the rubric revision."""

    case_id: str
    arm_id: str
    scenario: str
    status: str
    before: bool
    after: bool | None
    before_failed: tuple[str, ...]
    after_failed: tuple[str, ...]
    names_resource: bool
    coddles: bool
    pamphlet: bool
    response: str


@dataclass
class RunScore:
    """Every re-scored row of one run directory."""

    name: str
    suite: str
    pair: str
    labels: dict[str, str]
    cases: list[CaseScore] = field(default_factory=list)

    def arm(self, arm_id: str) -> list[CaseScore]:
        return [case for case in self.cases if case.arm_id == arm_id]


def load_fixtures(fixtures: Path) -> dict[str, dict[str, Any]]:
    """Index every scenario fixture under ``fixtures`` by case id."""
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(fixtures.glob("scenario-*-v1.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    case = json.loads(line)
                    index[case["id"]] = case
    return index


def _passed(status: str, metrics: list[dict[str, Any]]) -> bool:
    return status == "success" and bool(metrics) and all(metric["passed"] for metric in metrics)


def _carried_metric(check: dict[str, Any], recorded: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """The recorded verdict for a check that cannot be recomputed.

    Failure records omit ``state_metrics``, so a row that errored has no verdict
    to carry over. Report the check as failed instead of aborting the report for
    every other row in the results root.
    """
    if index < len(recorded):
        return recorded[index]
    return {"type": check["type"], "passed": False}


def rescore_case(case: dict[str, Any], row: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, list[str]] | None:
    """Re-run the response checks and carry the state-dependent ones over.

    Returns the merged metrics, whether every check now passes, and the types of
    the checks that fail. Returns ``None`` when the fixture no longer has the
    same number of checks as the recorded result, because there is then no
    correspondence to merge on.
    """
    checks = case["state_checks"]
    recorded = row.get("state_metrics") or []
    if recorded and len(checks) != len(recorded):
        return None
    response = row.get("response") or ""
    response_only = {**case, "state_checks": [c for c in checks if c["type"] in RESPONSE_CHECK_TYPES]}
    rescored = iter(score_state_checks(response_only, response, {}, []))
    merged = [
        next(rescored) if check["type"] in RESPONSE_CHECK_TYPES else _carried_metric(check, recorded, index)
        for index, check in enumerate(checks)
    ]
    failed = [metric["type"] for metric in merged if not metric["passed"]]
    return merged, _passed(row.get("status", "unknown"), merged), failed


def score_run(run_dir: Path, fixtures: dict[str, dict[str, Any]]) -> RunScore:
    """Re-score every row in one run directory."""
    manifest = load_manifest(run_dir)
    suite, pair = run_suite_and_pair(manifest, manifest.get("suite", "unknown"))
    score = RunScore(
        name=run_dir.name,
        suite=suite,
        pair=pair,
        labels=run_labels(manifest),
        cases=[],
    )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # A resumed run appends a second row for a case; the later verdict is
            # the one that stands, and counting both would inflate the totals.
            rows[(row["case_id"], row["arm_id"])] = row
    for row in rows.values():
        case = fixtures[row["case_id"]]
        result = rescore_case(case, row)
        recorded = row.get("state_metrics") or []
        response = row.get("response") or ""
        folded = response.casefold()
        score.cases.append(
            CaseScore(
                case_id=row["case_id"],
                arm_id=row["arm_id"],
                scenario=case["scenario"],
                status=row.get("status", "unknown"),
                before=_passed(row.get("status", "unknown"), recorded),
                after=None if result is None else result[1],
                before_failed=tuple(m["type"] for m in recorded if not m["passed"]),
                after_failed=() if result is None else tuple(result[2]),
                names_resource=any(term in folded for term in RESOURCE_TERMS),
                coddles=any(term in folded for term in CODDLING),
                pamphlet=any(term in folded for term in PAMPHLET),
                response=response,
            )
        )
    return score


def discover(results_root: Path) -> list[Path]:
    """Every run directory under ``results_root`` that recorded results."""
    return sorted({path.parent for path in results_root.rglob("results.jsonl")})


def _verdict(passed: bool | None) -> str:
    return "n/a" if passed is None else ("PASS" if passed else "FAIL")


def render(scores: list[RunScore]) -> str:
    """Markdown: per-arm totals, then crisis case detail."""
    lines = ["# Re-scored paired evals", ""]
    lines.append(
        "| run | arm | cases | recorded | re-scored | delta | unreconciled | names resource | coddles | pamphlet |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for score in scores:
        for arm_id in sorted({case.arm_id for case in score.cases}):
            rows = score.arm(arm_id)
            before = sum(case.before for case in rows)
            reconciled = [case for case in rows if case.after is not None]
            after = sum(bool(case.after) for case in reconciled)
            resources = sum(case.names_resource for case in rows)
            coddles = sum(case.coddles for case in rows)
            pamphlet = sum(case.pamphlet for case in rows)
            unreconciled = len(rows) - len(reconciled)
            delta = f"{after - before:+d}" if reconciled else "n/a"
            lines.append(
                f"| {score.pair}/{score.suite} | {score.labels.get(arm_id, arm_id)} | {len(rows)} "
                f"| {before} | {after} | {delta} | {unreconciled} | {resources} | {coddles} | {pamphlet} |"
            )
    lines.append("")
    lines.append("## Crisis cases")
    lines.append("")
    lines.append("| run | case | arm | recorded | re-scored | resource named | coddles | reply |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for score in scores:
        for case in score.cases:
            if case.scenario != "crisis":
                continue
            label = score.labels.get(case.arm_id, case.arm_id)
            quote = " ".join(case.response.split())[:240].replace("|", "/")
            lines.append(
                f"| {score.pair}/{score.suite} | {case.case_id} | {label} "
                f"| {_verdict(case.before)} | {_verdict(case.after)} "
                f"| {'yes' if case.names_resource else 'no'} | {'yes' if case.coddles else 'no'} | {quote} |"
            )
    lines.append("")
    return "\n".join(lines)


def payload(scores: list[RunScore]) -> dict[str, Any]:
    """Machine-readable form of the same data."""
    return {
        "runs": [
            {
                "run": score.name,
                "suite": score.suite,
                "pair": score.pair,
                "labels": score.labels,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "arm_id": case.arm_id,
                        "scenario": case.scenario,
                        "status": case.status,
                        "recorded_pass": case.before,
                        "rescored_pass": case.after,
                        "recorded_failed": list(case.before_failed),
                        "rescored_failed": list(case.after_failed),
                        "names_resource": case.names_resource,
                        "coddles": case.coddles,
                        "pamphlet": case.pamphlet,
                    }
                    for case in score.cases
                ],
            }
            for score in scores
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="directory holding run subdirectories")
    parser.add_argument("--fixtures", type=Path, required=True, help="directory holding scenario-*-v1.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="markdown output path")
    parser.add_argument("--json-out", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)

    fixtures = load_fixtures(args.fixtures)
    scores = [score_run(run_dir, fixtures) for run_dir in discover(args.results_root)]
    if not scores:
        sys.stderr.write(f"no results.jsonl under {args.results_root}\n")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(scores), encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload(scores), indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"re-scored {sum(len(score.cases) for score in scores)} rows from {len(scores)} runs\n")
    sys.stdout.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
