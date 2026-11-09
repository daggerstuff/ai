"""Summarize (and diff) paired-eval result directories.

One ``--run-dir`` renders a per-case pass table for the two arms in that run.
Two ``--run-dir`` arguments render a case-by-case comparison, labelling each arm
by the model its manifest recorded -- so ``v13`` and ``v14`` are read from the
evidence rather than assumed from the arm slot. ``--results-root`` renders a
matrix over every finished run under a results directory.

Arm labels come from the run id (``wandb-{suite}-{release}-vs-{compare}-{digest}``)
which records which model sat in which slot, so no per-release constants need
editing when a new release is evaluated.

Pass/fail comes from the harness itself: a case passes when the row succeeded
and every recorded ``state_metrics`` check passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "Qwen/Qwen3.6-35B-A3B": "base",
}
PAIR_RUN_ID = re.compile(r"^wandb-(?P<suite>.+)-(?P<release>v\d+)-vs-(?P<compare>base|v\d+)-[0-9a-f]{8}$")
DIFF_ARG_COUNT = 2
COMPLETE_STATUS = "complete"
QUOTE_LIMIT = 360


@dataclass(frozen=True)
class ArmRow:
    """One case result for one arm."""

    passed: bool
    status: str
    failed_checks: tuple[str, ...]
    finish_reason: str | None
    response: str | None = None


@dataclass(frozen=True)
class RunSummary:
    """Everything the matrix needs about one run directory."""

    name: str
    suite: str
    pair: str
    status: str
    labels: dict[str, str]
    arm_ids: tuple[str, ...]
    passes: dict[str, int]
    case_count: int


def arm_label(model: str) -> str:
    """Map a model URI to a short human label."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return model.rsplit("/", 1)[-1]


def run_labels(manifest: dict[str, Any]) -> dict[str, str]:
    """Name each arm from the run id, falling back to the recorded models.

    A paired run records ``v{release}-vs-{compare}`` in its run id, with the
    compared model in the base slot and the release under test in the sft slot.
    Deriving the labels here keeps the summarizer release-agnostic.
    """
    match = PAIR_RUN_ID.match(manifest.get("run_id", ""))
    models = arm_models(manifest)
    if match:
        labels = {"base": match["compare"], "sft": match["release"]}
        if set(labels) == set(models):
            return {arm: labels[arm] for arm in models}
    return {arm: arm_label(model) for arm, model in models.items()}


def run_suite_and_pair(manifest: dict[str, Any], fallback: str) -> tuple[str, str]:
    match = PAIR_RUN_ID.match(manifest.get("run_id", ""))
    if match:
        return match["suite"], f"{match['release']}-vs-{match['compare']}"
    return fallback, "unknown"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def load_rows(run_dir: Path) -> dict[tuple[str, str], ArmRow]:
    """Return {(case_id, arm_id): ArmRow} for one run directory."""
    rows: dict[tuple[str, str], ArmRow] = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            checks = row.get("state_metrics") or []
            failed = tuple(check.get("type", "check") for check in checks if not check.get("passed", False))
            rows[(row["case_id"], row["arm_id"])] = ArmRow(
                passed=row.get("status") == "success" and bool(checks) and not failed,
                status=row.get("status", "unknown"),
                failed_checks=failed,
                finish_reason=row.get("finish_reason"),
                response=row.get("response"),
            )
    return rows


def arm_models(manifest: dict[str, Any]) -> dict[str, str]:
    return {arm["id"]: arm["model"] for arm in manifest["arms"]}


def _verdict(row: ArmRow | None) -> str:
    if row is None:
        return "missing"
    if row.passed:
        return "PASS"
    detail = f"{row.status}" if row.status != "success" else "no checks"
    if row.failed_checks:
        detail = ",".join(row.failed_checks)
    return f"FAIL({detail})"


def _pass_count(rows: dict[tuple[str, str], ArmRow], cases: list[str], arm: str) -> int:
    return sum(1 for case_id in cases if (row := rows.get((case_id, arm))) is not None and row.passed)


def render_single(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    labels = run_labels(manifest)
    rows = load_rows(run_dir)
    arm_ids = manifest["arm_ids"]
    case_ids = list(manifest["case_ids"])
    lines = [
        f"# {run_dir.name}  run_id={manifest['run_id']}  status={manifest['status']}",
        "",
        "| case | " + " | ".join(labels[arm] for arm in arm_ids) + " |",
        "|---|" + "---|" * len(arm_ids),
    ]
    for case_id in case_ids:
        cells = [_verdict(rows.get((case_id, arm))) for arm in arm_ids]
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")
    lines.append("")
    for arm in arm_ids:
        lines.append(f"{labels[arm]}: {_pass_count(rows, case_ids, arm)}/{len(case_ids)} pass")
    lines.extend(["", _delta_section(run_dir.name, manifest, rows)])
    return "\n".join(lines)


def _delta_table(
    title: str,
    cases: list[str],
    left: tuple[str, dict[tuple[str, str], ArmRow], str],
    right: tuple[str, dict[tuple[str, str], ArmRow], str],
) -> str:
    """Render a case-by-case delta between two (arm_id, rows, label) arms."""
    left_arm, left_rows, left_label = left
    right_arm, right_rows, right_label = right
    lines = [
        f"# {title}",
        "",
        f"| case | {left_label} | {right_label} | delta |",
        "|---|---|---|---|",
    ]
    changed = 0
    regressions: list[tuple[str, ArmRow]] = []
    for case_id in cases:
        left_row = left_rows.get((case_id, left_arm))
        right_row = right_rows.get((case_id, right_arm))
        left_ok = bool(left_row and left_row.passed)
        right_ok = bool(right_row and right_row.passed)
        delta = "same" if left_ok == right_ok else ("improved" if right_ok else "regressed")
        changed += left_ok != right_ok
        if left_ok and not right_ok and right_row is not None:
            regressions.append((case_id, right_row))
        lines.append(f"| {case_id} | {_verdict(left_row)} | {_verdict(right_row)} | {delta} |")
    lines.extend(["", f"{changed} of {len(cases)} cases changed"])
    lines.extend(_regression_block(regressions, right_label))
    return "\n".join(lines)


def _quote(text: str, limit: int = QUOTE_LIMIT) -> str:
    """Flatten a model reply to one readable line for the scoreboard."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _regression_block(regressions: list[tuple[str, ArmRow]], label: str) -> list[str]:
    """Show what the regressed arm actually said, so the failure is inspectable."""
    if not regressions:
        return []
    lines = ["", "## regressed cases", ""]
    for case_id, row in regressions:
        detail = ",".join(row.failed_checks) or row.status
        lines.append(f"### {case_id}")
        lines.append(f"- failed: {detail}")
        if row.response:
            lines.append(f"- {label} said: {_quote(row.response)}")
        lines.append("")
    return lines[:-1] if lines[-1] == "" else lines


def _delta_section(
    name: str,
    manifest: dict[str, Any],
    rows: dict[tuple[str, str], ArmRow],
) -> str:
    """Compare the two arms of a paired run against each other.

    A paired run already holds both models, so the interesting comparison for a
    ``v14-vs-v13`` suite is inside one directory: the compared model occupies the
    base slot and the release under test the sft slot.
    """
    first_arm, second_arm = manifest["arm_ids"]
    labels = run_labels(manifest)
    return _delta_table(
        f"{name}: {labels[first_arm]} vs {labels[second_arm]}",
        list(manifest["case_ids"]),
        (first_arm, rows, labels[first_arm]),
        (second_arm, rows, labels[second_arm]),
    )


def render_diff(left_dir: Path, right_dir: Path) -> str:
    left_manifest, right_manifest = load_manifest(left_dir), load_manifest(right_dir)
    left_labels, right_labels = run_labels(left_manifest), run_labels(right_manifest)
    left_rows, right_rows = load_rows(left_dir), load_rows(right_dir)
    left_arm, right_arm = left_manifest["arm_ids"][-1], right_manifest["arm_ids"][-1]
    left_label = left_labels[left_arm]
    right_label = right_labels[right_arm]
    if left_label == right_label:
        raise ValueError(
            f"both run dirs supervise {left_label}; comparing a model against itself is not a diff. "
            "For two arms inside one paired run, summarize that run dir on its own."
        )
    cases = [case for case in left_manifest["case_ids"] if case in right_manifest["case_ids"]]
    return _delta_table(
        f"{left_label} ({left_dir.name}) vs {right_label} ({right_dir.name})",
        cases,
        (left_arm, left_rows, left_label),
        (right_arm, right_rows, right_label),
    )


def summarize_run(run_dir: Path) -> RunSummary:
    manifest = load_manifest(run_dir)
    rows = load_rows(run_dir)
    arm_ids = tuple(manifest["arm_ids"])
    case_ids = list(manifest["case_ids"])
    suite, pair = run_suite_and_pair(manifest, run_dir.name)
    return RunSummary(
        name=run_dir.name,
        suite=suite,
        pair=pair,
        status=manifest["status"],
        labels=run_labels(manifest),
        arm_ids=arm_ids,
        passes={arm: _pass_count(rows, case_ids, arm) for arm in arm_ids},
        case_count=len(case_ids),
    )


def collect_runs(root: Path) -> list[tuple[Path, RunSummary]]:
    """Summarize every run directory under ``root`` that has a manifest."""
    run_dirs = sorted(path for path in root.iterdir() if (path / "manifest.json").is_file())
    return [(run_dir, summarize_run(run_dir)) for run_dir in run_dirs]


def regressed_cases(run_dir: Path) -> list[tuple[str, ArmRow]]:
    """Cases the release arm fails while the arm it is compared to passes."""
    manifest = load_manifest(run_dir)
    rows = load_rows(run_dir)
    reference_arm, release_arm = manifest["arm_ids"]
    found: list[tuple[str, ArmRow]] = []
    for case_id in manifest["case_ids"]:
        reference = rows.get((case_id, reference_arm))
        release = rows.get((case_id, release_arm))
        if reference is not None and release is not None and reference.passed and not release.passed:
            found.append((case_id, release))
    return found


def render_matrix(root: Path) -> str:
    """Render every paired run under ``root`` as a suite-by-suite matrix."""
    collected = collect_runs(root)
    if not collected:
        raise ValueError(f"no run directories with a manifest under {root}")
    runs = [summary for _, summary in collected]
    lines = [f"# {root}", "", "| pair | suite | arm | pass | arm | pass |", "|---|---|---|---|---|---|"]
    for run in sorted(runs, key=lambda item: (item.pair, item.suite)):
        first, second = run.arm_ids
        note = "" if run.status == COMPLETE_STATUS else f" ({run.status})"
        lines.append(
            f"| {run.pair} | {run.suite}{note} "
            f"| {run.labels[first]} | {run.passes[first]}/{run.case_count} "
            f"| {run.labels[second]} | {run.passes[second]}/{run.case_count} |"
        )
    lines.extend(["", "## totals by pair and arm", "", "| pair | arm | suites | pass |", "|---|---|---|---|"])
    totals: dict[tuple[str, str], list[int]] = {}
    for run in runs:
        for arm in run.arm_ids:
            entry = totals.setdefault((run.pair, run.labels[arm]), [0, 0, 0])
            entry[0] += run.passes[arm]
            entry[1] += run.case_count
            entry[2] += 1
    for (pair, label), (passed, total, suites) in sorted(totals.items()):
        lines.append(f"| {pair} | {label} | {suites} | {passed}/{total} |")
    lines.extend(_matrix_regressions(collected))
    return "\n".join(lines)


def _matrix_regressions(collected: list[tuple[Path, RunSummary]]) -> list[str]:
    """Surface every case the release loses to the model it is measured against."""
    lines: list[str] = []
    for run_dir, summary in sorted(collected, key=lambda item: (item[1].pair, item[1].suite)):
        lost = regressed_cases(run_dir)
        if not lost:
            continue
        release = summary.labels[summary.arm_ids[1]]
        lines.extend(["", f"### {summary.pair} / {summary.suite} — {release} loses {len(lost)}"])
        for case_id, row in lost:
            detail = ",".join(row.failed_checks) or row.status
            lines.append(f"- {case_id}: failed {detail}")
            if row.response:
                lines.append(f"  - {release} said: {_quote(row.response)}")
    if not lines:
        return []
    return ["", "## cases the release loses against the compared arm", *lines]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--results-root", type=Path, help="render a matrix over every run under this directory")
    args = parser.parse_args()
    if args.results_root is not None:
        if args.run_dir:
            parser.error("--results-root and --run-dir are mutually exclusive")
        report = render_matrix(args.results_root)
    elif len(args.run_dir) == 1:
        report = render_single(args.run_dir[0])
    elif len(args.run_dir) == DIFF_ARG_COUNT:
        report = render_diff(args.run_dir[0], args.run_dir[1])
    else:
        parser.error("pass one run dir to summarize, two to diff, or --results-root for a matrix")
    sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
