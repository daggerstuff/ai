"""Tests for the paired-eval result summarizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.launch import summarize_eval_run as summary

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
V13_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-ecf35d5e5384:v1"
V14_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-8ccb925b683e:v1"
V15_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-73504a99ed05:v1"
DIGEST = "0123abcd"


def write_run(
    run_dir: Path,
    case_ids: list[str],
    models: dict[str, str],
    passes: dict[tuple[str, str], bool],
    manifest_extra: dict[str, Any] | None = None,
) -> Path:
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": f"run-{run_dir.name}",
        "status": "complete",
        "case_ids": case_ids,
        "arm_ids": ["base", "sft"],
        "arms": [{"id": arm_id, "model": models[arm_id]} for arm_id in ("base", "sft")],
        **(manifest_extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for case_id in case_ids:
        for arm_id in ("base", "sft"):
            passed = passes.get((case_id, arm_id), True)
            rows.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "status": "success",
                    "finish_reason": "stop",
                    "state_metrics": [
                        {"type": "response_enum", "passed": passed},
                    ],
                }
            )
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run_dir


def pair_run_id(suite: str, release: str, compare: str) -> str:
    return f"wandb-{suite}-{release}-vs-{compare}-{DIGEST}"


def test_arm_label_names_the_base_model() -> None:
    assert summary.arm_label(BASE_MODEL) == "base"


def test_arm_label_falls_back_to_the_artifact_name() -> None:
    assert summary.arm_label(V13_URI) == "arc-sft-baseline-ecf35d5e5384:v1"


def test_run_labels_read_the_release_and_compare_from_the_run_id() -> None:
    manifest = {
        "run_id": pair_run_id("crisis-dev", "v15", "v14"),
        "arms": [{"id": "base", "model": V14_URI}, {"id": "sft", "model": V15_URI}],
    }

    assert summary.run_labels(manifest) == {"base": "v14", "sft": "v15"}


def test_run_labels_keep_the_base_slot_named_for_the_base_model() -> None:
    manifest = {
        "run_id": pair_run_id("literal-sealed", "v15", "base"),
        "arms": [{"id": "base", "model": BASE_MODEL}, {"id": "sft", "model": V15_URI}],
    }

    assert summary.run_labels(manifest) == {"base": "base", "sft": "v15"}


def test_run_labels_fall_back_when_the_run_id_is_not_a_pair() -> None:
    manifest = {
        "run_id": "wandb-v13-synthetic-fixed-contracts-20261105-20261105",
        "arms": [{"id": "base", "model": BASE_MODEL}, {"id": "sft", "model": V13_URI}],
    }

    assert summary.run_labels(manifest)["base"] == "base"
    assert summary.run_labels(manifest)["sft"] == "arc-sft-baseline-ecf35d5e5384:v1"


def test_load_rows_marks_failed_state_checks(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path / "run",
        ["case-a"],
        {"base": BASE_MODEL, "sft": V14_URI},
        {("case-a", "sft"): False},
    )

    rows = summary.load_rows(run_dir)

    assert rows[("case-a", "base")].passed is True
    assert rows[("case-a", "sft")].passed is False
    assert rows[("case-a", "sft")].failed_checks == ("response_enum",)


def test_render_single_counts_passes_per_arm(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path / "capability-dev-v14-vs-base",
        ["case-a", "case-b"],
        {"base": BASE_MODEL, "sft": V14_URI},
        {("case-a", "base"): False, ("case-b", "sft"): False},
        manifest_extra={"run_id": pair_run_id("capability-dev", "v14", "base")},
    )

    output = summary.render_single(run_dir)

    assert "| case-a | FAIL(response_enum) | PASS |" in output
    assert "| case-b | PASS | FAIL(response_enum) |" in output
    assert "base: 1/2 pass" in output
    assert "v14: 1/2 pass" in output


def test_render_diff_flags_a_regression(tmp_path: Path) -> None:
    v13_run = write_run(
        tmp_path / "v13-run",
        ["case-a"],
        {"base": BASE_MODEL, "sft": V13_URI},
        {},
        manifest_extra={"run_id": pair_run_id("capability-dev", "v13", "base")},
    )
    v14_run = write_run(
        tmp_path / "v14-run",
        ["case-a"],
        {"base": BASE_MODEL, "sft": V14_URI},
        {("case-a", "sft"): False},
        manifest_extra={"run_id": pair_run_id("capability-dev", "v14", "base")},
    )

    output = summary.render_diff(v13_run, v14_run)

    assert "| case-a | PASS | FAIL(response_enum) | regressed |" in output
    assert "1 of 1 cases changed" in output


def test_render_single_compares_the_two_arms_of_a_paired_run(tmp_path: Path) -> None:
    # A v15-vs-v14 suite keeps the v14 adapter in the base slot, so the arm
    # comparison lives inside one run directory rather than across two.
    run_dir = write_run(
        tmp_path / "crisis-dev-v15-vs-v14",
        ["case-a", "case-b"],
        {"base": V14_URI, "sft": V15_URI},
        {("case-a", "sft"): False},
        manifest_extra={"run_id": pair_run_id("crisis-dev", "v15", "v14")},
    )

    output = summary.render_single(run_dir)

    assert "# crisis-dev-v15-vs-v14: v14 vs v15" in output
    assert "| case-a | PASS | FAIL(response_enum) | regressed |" in output
    assert "| case-b | PASS | PASS | same |" in output
    assert "1 of 2 cases changed" in output


def test_render_diff_refuses_to_compare_a_model_against_itself(tmp_path: Path) -> None:
    arms = {"base": BASE_MODEL, "sft": V14_URI}
    run_id = pair_run_id("contract-dev", "v14", "base")
    first = write_run(tmp_path / "first", ["case-a"], arms, {}, {"run_id": run_id})
    second = write_run(tmp_path / "second", ["case-a"], arms, {}, {"run_id": run_id})

    with pytest.raises(ValueError, match="against itself is not a diff"):
        summary.render_diff(first, second)


def test_render_matrix_lists_every_run_by_pair_and_suite(tmp_path: Path) -> None:
    write_run(
        tmp_path / "capability-dev-v15-vs-base",
        ["case-a", "case-b"],
        {"base": BASE_MODEL, "sft": V15_URI},
        {("case-a", "sft"): False},
        manifest_extra={"run_id": pair_run_id("capability-dev", "v15", "base")},
    )
    write_run(
        tmp_path / "capability-dev-v15-vs-v14",
        ["case-a", "case-b"],
        {"base": V14_URI, "sft": V15_URI},
        {},
        manifest_extra={"run_id": pair_run_id("capability-dev", "v15", "v14")},
    )

    output = summary.render_matrix(tmp_path)

    assert "| v15-vs-base | capability-dev | base | 2/2 | v15 | 1/2 |" in output
    assert "| v15-vs-v14 | capability-dev | v14 | 2/2 | v15 | 2/2 |" in output
    assert "| v15-vs-base | v15 | 1 | 1/2 |" in output
    assert "| v15-vs-v14 | v15 | 1 | 2/2 |" in output


def test_render_matrix_marks_runs_that_have_not_finished(tmp_path: Path) -> None:
    write_run(
        tmp_path / "crisis-dev-v15-vs-base",
        ["case-a"],
        {"base": BASE_MODEL, "sft": V15_URI},
        {},
        manifest_extra={"run_id": pair_run_id("crisis-dev", "v15", "base"), "status": "running"},
    )

    output = summary.render_matrix(tmp_path)

    assert "crisis-dev (running)" in output


def test_render_matrix_rejects_an_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no run directories"):
        summary.render_matrix(tmp_path)
