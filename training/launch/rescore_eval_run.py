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

Resource-name frequency is reported alongside the verdicts. It is a description
of what the arms say, not a scoring input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    after: bool
    before_failed: tuple[str, ...]
    after_failed: tuple[str, ...]
    names_resource: bool
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


def rescore_case(case: dict[str, Any], row: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Re-run the response checks and carry the state-dependent ones over.

    Returns the merged metrics, whether every check now passes, and the types of
    the checks that fail.
    """
    checks = case["state_checks"]
    recorded = row.get("state_metrics") or []
    if len(checks) != len(recorded):
        raise ValueError(f"{row['case_id']}: fixture has {len(checks)} checks, result has {len(recorded)}")
    response = row.get("response") or ""
    response_only = {**case, "state_checks": [c for c in checks if c["type"] in RESPONSE_CHECK_TYPES]}
    rescored = iter(score_state_checks(response_only, response, {}, []))
    merged = [
        next(rescored) if check["type"] in RESPONSE_CHECK_TYPES else recorded[index]
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
    with (run_dir / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            case = fixtures[row["case_id"]]
            _, after, after_failed = rescore_case(case, row)
            recorded = row.get("state_metrics") or []
            response = (row.get("response") or "").casefold()
            score.cases.append(
                CaseScore(
                    case_id=row["case_id"],
                    arm_id=row["arm_id"],
                    scenario=case["scenario"],
                    status=row.get("status", "unknown"),
                    before=_passed(row.get("status", "unknown"), recorded),
                    after=after,
                    before_failed=tuple(m["type"] for m in recorded if not m["passed"]),
                    after_failed=tuple(after_failed),
                    names_resource=any(term in response for term in RESOURCE_TERMS),
                    response=row.get("response") or "",
                )
            )
    return score


def discover(results_root: Path) -> list[Path]:
    """Every run directory under ``results_root`` that recorded results."""
    return sorted({path.parent for path in results_root.rglob("results.jsonl")})


def render(scores: list[RunScore]) -> str:
    """Markdown: per-arm totals, then crisis case detail."""
    lines = ["# Re-scored paired evals", ""]
    lines.append("| run | arm | cases | recorded | re-scored | delta | names resource |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for score in scores:
        for arm_id in sorted({case.arm_id for case in score.cases}):
            rows = score.arm(arm_id)
            before = sum(case.before for case in rows)
            after = sum(case.after for case in rows)
            resources = sum(case.names_resource for case in rows)
            lines.append(
                f"| {score.pair}/{score.suite} | {score.labels.get(arm_id, arm_id)} | {len(rows)} "
                f"| {before} | {after} | {after - before:+d} | {resources} |"
            )
    lines.append("")
    lines.append("## Crisis cases")
    lines.append("")
    lines.append("| run | case | arm | recorded | re-scored | resource named | reply |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for score in scores:
        for case in score.cases:
            if case.scenario != "crisis":
                continue
            label = score.labels.get(case.arm_id, case.arm_id)
            quote = " ".join(case.response.split())[:240].replace("|", "/")
            lines.append(
                f"| {score.pair}/{score.suite} | {case.case_id} | {label} "
                f"| {'PASS' if case.before else 'FAIL'} | {'PASS' if case.after else 'FAIL'} "
                f"| {'yes' if case.names_resource else 'no'} | {quote} |"
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
