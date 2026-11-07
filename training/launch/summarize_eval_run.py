"""Summarize (and diff) paired-eval result directories.

One ``--run-dir`` renders a per-case pass table for the two arms in that run.
Two ``--run-dir`` arguments render a case-by-case comparison, labelling each arm
by the model its manifest recorded -- so ``v13`` and ``v14`` are read from the
evidence rather than assumed from the arm slot.

Pass/fail comes from the harness itself: a case passes when the row succeeded
and every recorded ``state_metrics`` check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "Qwen/Qwen3.6-35B-A3B": "base",
}
LABEL_PREFIX = "wandb-artifact:///wutang/pixelated-empathy-kan28/"
V13_SUFFIX = "arc-sft-baseline-ecf35d5e5384"
V14_SUFFIX = "arc-sft-baseline-8ccb925b683e"
DIFF_ARG_COUNT = 2


@dataclass(frozen=True)
class ArmRow:
    """One case result for one arm."""

    passed: bool
    status: str
    failed_checks: tuple[str, ...]
    finish_reason: str | None


def arm_label(model: str) -> str:
    """Map a model URI to a short human label."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    for suffix, label in ((V13_SUFFIX, "v13"), (V14_SUFFIX, "v14")):
        if model.startswith(LABEL_PREFIX) and model.removeprefix(LABEL_PREFIX).startswith(suffix):
            return label
    return model.rsplit("/", 1)[-1]


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


def render_single(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    models = arm_models(manifest)
    rows = load_rows(run_dir)
    arm_ids = manifest["arm_ids"]
    labels = [arm_label(models[arm]) for arm in arm_ids]
    lines = [
        f"# {run_dir.name}  run_id={manifest['run_id']}  status={manifest['status']}",
        "",
        "| case | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(arm_ids),
    ]
    for case_id in manifest["case_ids"]:
        cells = [_verdict(rows.get((case_id, arm))) for arm in arm_ids]
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")
    lines.append("")
    for arm in arm_ids:
        passed = sum(1 for case_id in manifest["case_ids"] if _verdict(rows.get((case_id, arm))) == "PASS")
        lines.append(f"{arm_label(models[arm])}: {passed}/{len(manifest['case_ids'])} pass")
    lines.extend(["", _delta_section(run_dir.name, manifest, models, rows)])
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
    for case_id in cases:
        left_row = left_rows.get((case_id, left_arm))
        right_row = right_rows.get((case_id, right_arm))
        left_ok = bool(left_row and left_row.passed)
        right_ok = bool(right_row and right_row.passed)
        delta = "same" if left_ok == right_ok else ("improved" if right_ok else "regressed")
        changed += left_ok != right_ok
        lines.append(f"| {case_id} | {_verdict(left_row)} | {_verdict(right_row)} | {delta} |")
    lines.extend(["", f"{changed} of {len(cases)} cases changed"])
    return "\n".join(lines)


def _delta_section(
    name: str,
    manifest: dict[str, Any],
    models: dict[str, str],
    rows: dict[tuple[str, str], ArmRow],
) -> str:
    """Compare the two arms of a paired run against each other.

    A paired run already holds both models, so the interesting comparison for a
    ``v14-vs-v13`` suite is inside one directory: the v13 adapter occupies the
    base slot and v14 the sft slot.
    """
    first_arm, second_arm = manifest["arm_ids"]
    return _delta_table(
        f"{name}: {arm_label(models[first_arm])} vs {arm_label(models[second_arm])}",
        list(manifest["case_ids"]),
        (first_arm, rows, arm_label(models[first_arm])),
        (second_arm, rows, arm_label(models[second_arm])),
    )


def render_diff(left_dir: Path, right_dir: Path) -> str:
    left_manifest, right_manifest = load_manifest(left_dir), load_manifest(right_dir)
    left_models, right_models = arm_models(left_manifest), arm_models(right_manifest)
    left_rows, right_rows = load_rows(left_dir), load_rows(right_dir)
    left_arm, right_arm = left_manifest["arm_ids"][-1], right_manifest["arm_ids"][-1]
    left_label = arm_label(left_models[left_arm])
    right_label = arm_label(right_models[right_arm])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.run_dir) == 1:
        report = render_single(args.run_dir[0])
    elif len(args.run_dir) == DIFF_ARG_COUNT:
        report = render_diff(args.run_dir[0], args.run_dir[1])
    else:
        parser.error("pass one run dir to summarize, or two to diff")
    sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
