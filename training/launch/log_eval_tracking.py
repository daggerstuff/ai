"""Publish paired-eval results to W&B so eval history is queryable, not greppable.

The harness writes ``results.jsonl`` + ``manifest.json`` per run and the
scoreboard is hand-rendered markdown, so a release's eval history only exists in
the docs tree: you cannot ask "how did every arm score on crisis-sealed across
v13..v19" without reading files. This module turns a results root into one
machine-readable metrics document and one W&B run holding per-suite/per-arm
scalars, a per-case table, and an artifact with the raw results.

Arm labels come from the run id (``wandb-{suite}-{release}-vs-{compare}-{digest}``),
the same derivation the summarizer uses, so nothing here is release-specific and
a label always names the model that produced the row.

Usage:

    python -m training.launch.log_eval_tracking \
        --results-root docs/training/execution-2026-11-10/evals/results \
        --release-label v19

``--dry-run`` writes the metrics document without contacting W&B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.launch.summarize_eval_run import (
    arm_models,
    load_manifest,
    load_rows,
    run_labels,
    run_suite_and_pair,
)

REPO = Path(__file__).resolve().parents[3]
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
ARTIFACT_TYPE = "eval-results"
COMPLETE_STATUS = "complete"
METRICS_SCHEMA = "paired-eval-metrics-v1"
FAILURE_QUOTE_LIMIT = 240


@dataclass(frozen=True)
class ArmMetrics:
    """One arm's reading for one suite in one run."""

    label: str
    model: str
    passes: int
    cases: int
    failures: tuple[str, ...]

    @property
    def pass_rate(self) -> float:
        return self.passes / self.cases if self.cases else 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "passes": self.passes,
            "cases": self.cases,
            "pass_rate": round(self.pass_rate, 6),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class RunMetrics:
    """One paired run, reduced to the numbers a scoreboard needs."""

    run: str
    suite: str
    pair: str
    status: str
    arms: tuple[ArmMetrics, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "suite": self.suite,
            "pair": self.pair,
            "status": self.status,
            "arms": [arm.payload() for arm in self.arms],
        }


def _failures(rows: dict[tuple[str, str], Any], case_ids: list[str], arm_id: str) -> tuple[tuple[str, ...], int]:
    failures: list[str] = []
    passes = 0
    for case_id in case_ids:
        row = rows.get((case_id, arm_id))
        if row is not None and row.passed:
            passes += 1
        else:
            failures.append(case_id)
    return tuple(failures), passes


def summarize(run_dir: Path) -> RunMetrics:
    """Reduce one run directory to per-arm pass counts and failing cases."""
    manifest = load_manifest(run_dir)
    rows = load_rows(run_dir)
    labels = run_labels(manifest)
    models = arm_models(manifest)
    case_ids = list(manifest["case_ids"])
    suite, pair = run_suite_and_pair(manifest, run_dir.name)
    arms: list[ArmMetrics] = []
    for arm_id in manifest["arm_ids"]:
        failures, passes = _failures(rows, case_ids, arm_id)
        arms.append(
            ArmMetrics(
                label=labels[arm_id],
                model=models[arm_id],
                passes=passes,
                cases=len(case_ids),
                failures=failures,
            )
        )
    return RunMetrics(
        run=run_dir.name,
        suite=suite,
        pair=pair,
        status=manifest.get("status", "unknown"),
        arms=tuple(arms),
    )


def discover(root: Path) -> list[RunMetrics]:
    """Summarize every run directory under ``root`` that has a manifest."""
    run_dirs = sorted(path for path in root.iterdir() if (path / "manifest.json").is_file())
    return [summarize(run_dir) for run_dir in run_dirs]


def totals(runs: list[RunMetrics]) -> dict[str, dict[str, Any]]:
    """Aggregate pass counts per arm label across every suite."""
    acc: dict[str, dict[str, Any]] = {}
    for run in runs:
        for arm in run.arms:
            entry = acc.setdefault(arm.label, {"model": arm.model, "passes": 0, "cases": 0, "suites": {}})
            entry["passes"] += arm.passes
            entry["cases"] += arm.cases
            entry["suites"][run.suite] = {"passes": arm.passes, "cases": arm.cases}
    for entry in acc.values():
        entry["pass_rate"] = round(entry["passes"] / entry["cases"], 6) if entry["cases"] else 0.0
    return acc


def build_metrics(root: Path, release_label: str) -> dict[str, Any]:
    """Build the machine-readable metrics document for one results root."""
    runs = discover(root)
    if not runs:
        raise ValueError(f"no run directories with a manifest under {root}")
    return {
        "schema": METRICS_SCHEMA,
        "release": release_label,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_root": str(root),
        "runs": [run.payload() for run in runs],
        "totals": totals(runs),
    }


def write_metrics(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def scalar_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten the metrics document into W&B scalars.

    Keys are ``eval/{suite}/{label}/...`` for the per-suite readings and
    ``eval/total/{label}/...`` for the aggregate, so a W&B panel can chart one
    arm across releases without knowing the run layout.
    """
    scalars: dict[str, float] = {}
    for run in payload["runs"]:
        for arm in run["arms"]:
            prefix = f"eval/{run['suite']}/{arm['label']}"
            scalars[f"{prefix}/passes"] = float(arm["passes"])
            scalars[f"{prefix}/cases"] = float(arm["cases"])
            scalars[f"{prefix}/pass_rate"] = float(arm["pass_rate"])
    for label, entry in payload["totals"].items():
        scalars[f"eval/total/{label}/passes"] = float(entry["passes"])
        scalars[f"eval/total/{label}/cases"] = float(entry["cases"])
        scalars[f"eval/total/{label}/pass_rate"] = float(entry["pass_rate"])
    return scalars


def case_table_rows(root: Path) -> list[list[Any]]:
    """One row per (run, case, arm): the evidence behind every scalar."""
    rows: list[list[Any]] = []
    for run in discover(root):
        manifest = load_manifest(root / run.run)
        labels = run_labels(manifest)
        case_rows = load_rows(root / run.run)
        for case_id in manifest["case_ids"]:
            for arm_id in manifest["arm_ids"]:
                row = case_rows.get((case_id, arm_id))
                rows.append(
                    [
                        run.pair,
                        run.suite,
                        labels[arm_id],
                        case_id,
                        bool(row is not None and row.passed),
                        "" if row is None else row.status,
                        "" if row is None else ",".join(row.failed_checks),
                        "" if row is None else (row.response or "")[:FAILURE_QUOTE_LIMIT],
                    ]
                )
    return rows


def root_label(root: Path) -> str:
    """Name a results root by what distinguishes it from the release default.

    ``execution-.../evals/results`` is the main batch and needs no suffix,
    ``evals-vs-v16/results`` a supplementary one, and a scratch root like
    ``crisis-rerun-v3/evals/results`` names itself after its parent directory.
    """
    parent = root.parent.name
    if parent == "evals":
        outer = root.parent.parent.name
        return "" if outer.startswith("execution-") else outer
    if parent.startswith("evals-"):
        return parent[len("evals-") :]
    return parent


def _api_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("WANDB_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("WANDB_API_KEY not found in environment or .env")


def log_to_wandb(
    payload: dict[str, Any],
    root: Path,
    *,
    metrics_path: Path,
    extra_files: tuple[Path, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Log one W&B run plus an eval-results artifact for a results root."""
    scalars = scalar_metrics(payload)
    report = {
        "release": payload["release"],
        "results_root": str(root),
        "metrics": str(metrics_path),
        "scalar_count": len(scalars),
        "dry_run": dry_run,
    }
    if dry_run:
        report["case_rows"] = len(case_table_rows(root))
        return report

    import wandb

    os.environ["WANDB_API_KEY"] = _api_key()
    table_rows = case_table_rows(root)
    suffix = root_label(root)
    name = f"eval-{payload['release']}" + (f"-{suffix}" if suffix else "")
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="paired-eval",
        name=name,
        mode="online",
        config={
            "release": payload["release"],
            "results_root": str(root),
            "schema": payload["schema"],
            "run_count": len(payload["runs"]),
            "case_rows": len(table_rows),
        },
    )
    try:
        run.log(scalars)
        run.log(
            {
                "eval/cases": wandb.Table(
                    columns=[
                        "pair",
                        "suite",
                        "arm",
                        "case",
                        "passed",
                        "status",
                        "failed_checks",
                        "response",
                    ],
                    data=table_rows,
                )
            }
        )
        artifact = wandb.Artifact(name, type=ARTIFACT_TYPE)
        artifact.add_file(str(metrics_path), name=metrics_path.name)
        for path in extra_files:
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        artifact.add_dir(str(root), name="results")
        logged = run.log_artifact(artifact)
        logged.wait()
        report["run_id"] = run.id
        report["run_name"] = name
        # wandb stamps the version onto the name once the artifact is logged.
        report["artifact"] = artifact.name
        report["case_rows"] = len(table_rows)
    finally:
        run.finish()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, required=True, help="directory holding one subdir per run")
    parser.add_argument("--release-label", required=True, help="release under test, e.g. v19")
    parser.add_argument(
        "--exec-dir",
        type=Path,
        default=None,
        help="where to write <release>-metrics.json (default: the results root's parent)",
    )
    parser.add_argument("--dry-run", action="store_true", help="write metrics only, do not contact W&B")
    args = parser.parse_args(argv)

    root = args.results_root if args.results_root.is_absolute() else REPO / args.results_root
    if not root.is_dir():
        parser.error(f"--results-root does not exist: {root}")
    exec_dir = args.exec_dir or root.parent
    if not exec_dir.is_absolute():
        exec_dir = REPO / exec_dir

    payload = build_metrics(root, args.release_label)
    metrics_path = write_metrics(exec_dir / f"{args.release_label}-metrics.json", payload)
    scoreboard = exec_dir / f"{args.release_label}-scoreboard.md"
    report = log_to_wandb(
        payload,
        root,
        metrics_path=metrics_path,
        extra_files=(scoreboard,),
        dry_run=args.dry_run,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
