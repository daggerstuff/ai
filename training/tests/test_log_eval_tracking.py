"""Tests for publishing paired-eval results to W&B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.launch import log_eval_tracking as tracking

DIGEST = "0123abcd"


def write_run(
    root: Path,
    name: str,
    *,
    release: str = "v19",
    compare: str = "base",
    suite: str = "crisis-dev",
    rows: list[dict[str, Any]],
    arm_ids: tuple[str, ...] = ("base", "sft"),
    status: str = "complete",
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "run_id": f"wandb-{suite}-{release}-vs-{compare}-{DIGEST}",
        "status": status,
        "case_ids": sorted({row["case_id"] for row in rows}),
        "arm_ids": list(arm_ids),
        "arms": [
            {"id": "base", "model": "Qwen/Qwen3.6-35B-A3B"},
            {"id": "sft", "model": f"wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-{release}:v1"},
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run_dir


def row(
    case_id: str,
    arm_id: str,
    *,
    passed: bool = True,
    response: str = "reply",
    status: str = "success",
) -> dict[str, Any]:
    metrics = [{"type": "response_contains", "passed": passed}]
    return {
        "case_id": case_id,
        "arm_id": arm_id,
        "status": status,
        "response": response,
        "state_metrics": metrics,
    }


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    root = tmp_path / "evals" / "results"
    root.mkdir(parents=True)
    write_run(
        root,
        "crisis-dev-v19-vs-base",
        rows=[
            row("case-a", "base", passed=True),
            row("case-a", "sft", passed=True),
            row("case-b", "base", passed=False, response="I can't help with that."),
            row("case-b", "sft", passed=True),
        ],
    )
    write_run(
        root,
        "literal-dev-v19-vs-v18",
        compare="v18",
        suite="literal-dev",
        rows=[
            row("case-c", "base", passed=False),
            row("case-c", "sft", passed=True),
        ],
    )
    return root


def test_metrics_document_names_schema_release_and_runs(results_root: Path) -> None:
    payload = tracking.build_metrics(results_root, "v19")

    assert payload["schema"] == "paired-eval-metrics-v1"
    assert payload["release"] == "v19"
    assert [run["run"] for run in payload["runs"]] == ["crisis-dev-v19-vs-base", "literal-dev-v19-vs-v18"]
    assert payload["runs"][0]["pair"] == "v19-vs-base"
    assert payload["runs"][1]["pair"] == "v19-vs-v18"


def test_arms_carry_pass_counts_and_failing_cases(results_root: Path) -> None:
    run = tracking.build_metrics(results_root, "v19")["runs"][0]
    arms = {arm["label"]: arm for arm in run["arms"]}

    assert arms["base"]["passes"] == 1
    assert arms["base"]["cases"] == 2
    assert arms["base"]["failures"] == ["case-b"]
    assert arms["v19"]["passes"] == 2
    assert arms["v19"]["failures"] == []
    assert arms["v19"]["model"].endswith("arc-sft-baseline-v19:v1")


def test_labels_come_from_the_run_id_not_the_slot(results_root: Path) -> None:
    run = tracking.build_metrics(results_root, "v19")["runs"][1]
    assert [arm["label"] for arm in run["arms"]] == ["v18", "v19"]


def test_totals_aggregate_every_suite_per_label(results_root: Path) -> None:
    totals = tracking.build_metrics(results_root, "v19")["totals"]

    assert totals["base"]["passes"] == 1
    assert totals["base"]["cases"] == 2
    assert totals["v18"]["passes"] == 0
    assert totals["v18"]["cases"] == 1
    assert totals["v19"]["passes"] == 3
    assert totals["v19"]["cases"] == 3
    assert totals["v19"]["suites"]["crisis-dev"] == {"passes": 2, "cases": 2}
    assert totals["v19"]["suites"]["literal-dev"] == {"passes": 1, "cases": 1}
    assert totals["v19"]["pass_rate"] == 1.0


def test_scalars_are_namespaced_by_suite_and_total(results_root: Path) -> None:
    scalars = tracking.scalar_metrics(tracking.build_metrics(results_root, "v19"))

    assert scalars["eval/crisis-dev/v19/passes"] == 2.0
    assert scalars["eval/crisis-dev/v19/cases"] == 2.0
    assert scalars["eval/crisis-dev/v19/pass_rate"] == 1.0
    assert scalars["eval/literal-dev/v18/pass_rate"] == 0.0
    assert scalars["eval/total/v19/pass_rate"] == 1.0
    assert all(isinstance(value, float) for value in scalars.values())


def test_case_table_holds_one_row_per_run_case_and_arm(results_root: Path) -> None:
    rows = tracking.case_table_rows(results_root)

    assert len(rows) == 4 + 2  # crisis 2 cases x 2 arms, literal 1 case x 2 arms
    failed = [line for line in rows if line[4] is False]
    assert {line[3] for line in failed} == {"case-b", "case-c"}
    assert failed[0][6] == "response_contains"


def test_case_table_quotes_responses_without_flooding_the_table(tmp_path: Path) -> None:
    root = tmp_path / "evals" / "results"
    root.mkdir(parents=True)
    write_run(
        root,
        "crisis-dev-v19-vs-base",
        rows=[row("case-a", "sft", response="x" * 5_000)],
    )
    quoted = [line for line in tracking.case_table_rows(root) if line[2] == "v19"]
    assert len(quoted[0][7]) == tracking.FAILURE_QUOTE_LIMIT


def test_missing_arm_row_counts_as_a_failure(results_root: Path) -> None:
    (results_root / "crisis-dev-v19-vs-base" / "results.jsonl").write_text(
        json.dumps(row("case-a", "sft")) + "\n",
        encoding="utf-8",
    )
    run = tracking.summarize(results_root / "crisis-dev-v19-vs-base")
    arms = {arm.label: arm for arm in run.arms}

    assert arms["base"].passes == 0
    assert arms["base"].failures == ("case-a", "case-b")


def test_manifest_status_is_preserved_for_the_scoreboard(results_root: Path) -> None:
    write_run(
        results_root,
        "contract-dev-v19-vs-base",
        suite="contract-dev",
        rows=[row("case-d", "sft")],
        status="incomplete",
    )
    runs = {run.run: run for run in tracking.discover(results_root)}
    assert runs["contract-dev-v19-vs-base"].status == "incomplete"


def test_discover_skips_directories_without_a_manifest(results_root: Path) -> None:
    (results_root / "attempt1-discard").mkdir()
    assert len(tracking.discover(results_root)) == 2


def test_root_label_only_names_what_distinguishes_a_batch(tmp_path: Path) -> None:
    main = tmp_path / "execution-2026-11-10" / "evals" / "results"
    extra = tmp_path / "execution-2026-11-09" / "evals-vs-v16" / "results"
    scratch = tmp_path / "crisis-rerun-v3" / "evals" / "results"
    other = tmp_path / "somewhere" / "batch-a" / "results"

    assert tracking.root_label(main) == ""
    assert tracking.root_label(extra) == "vs-v16"
    assert tracking.root_label(scratch) == "crisis-rerun-v3"
    assert tracking.root_label(other) == "batch-a"


def test_build_metrics_refuses_an_empty_root(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    with pytest.raises(ValueError, match="no run directories"):
        tracking.build_metrics(empty, "v19")


def test_write_metrics_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    target = tracking.write_metrics(tmp_path / "nested" / "v19-metrics.json", {"b": 1, "a": 2})
    text = target.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')


def test_dry_run_writes_metrics_without_contacting_wandb(results_root: Path, tmp_path: Path) -> None:
    payload = tracking.build_metrics(results_root, "v19")
    metrics = tracking.write_metrics(tmp_path / "v19-metrics.json", payload)

    report = tracking.log_to_wandb(payload, results_root, metrics_path=metrics, dry_run=True)

    assert report["dry_run"] is True
    assert report["scalar_count"] == len(tracking.scalar_metrics(payload))
    assert report["case_rows"] == len(tracking.case_table_rows(results_root))
    assert "run_id" not in report
    assert metrics.is_file()


def test_main_writes_metrics_and_reports_dry_run(results_root: Path, tmp_path: Path, capsys) -> None:
    rc = tracking.main(
        [
            "--results-root",
            str(results_root),
            "--release-label",
            "v19",
            "--exec-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert (tmp_path / "v19-metrics.json").is_file()


def test_api_key_prefers_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "env-key")
    monkeypatch.setattr(tracking, "REPO", tmp_path)
    assert tracking._api_key() == "env-key"


def test_api_key_falls_back_to_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "REPO", tmp_path)
    (tmp_path / ".env").write_text("WANDB_API_KEY=dotenv-key\n", encoding="utf-8")
    assert tracking._api_key() == "dotenv-key"


def test_api_key_absent_is_a_clean_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "REPO", tmp_path)
    (tmp_path / ".env").write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        tracking._api_key()
