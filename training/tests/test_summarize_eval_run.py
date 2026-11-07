"""Tests for the paired-eval result summarizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.launch import summarize_eval_run as summary

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
V13_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-ecf35d5e5384:v1"
V14_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-8ccb925b683e:v1"


def write_run(
    root: Path,
    name: str,
    case_ids: list[str],
    models: dict[str, str],
    passes: dict[tuple[str, str], bool],
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": f"run-{name}",
        "status": "complete",
        "case_ids": case_ids,
        "arm_ids": ["base", "sft"],
        "arms": [{"id": arm_id, "model": models[arm_id]} for arm_id in ("base", "sft")],
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


def test_arm_label_recognises_the_known_artifacts() -> None:
    assert summary.arm_label(BASE_MODEL) == "base"
    assert summary.arm_label(V13_URI) == "v13"
    assert summary.arm_label(V14_URI) == "v14"


def test_arm_label_falls_back_to_the_artifact_name() -> None:
    assert summary.arm_label("wandb-artifact:///wutang/pixelated-empathy-kan28/other-run:v0") == "other-run:v0"


def test_load_rows_marks_failed_state_checks(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path,
        "run",
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
        tmp_path,
        "run",
        ["case-a", "case-b"],
        {"base": BASE_MODEL, "sft": V14_URI},
        {("case-a", "base"): False, ("case-b", "sft"): False},
    )

    output = summary.render_single(run_dir)

    assert "| case-a | FAIL(response_enum) | PASS |" in output
    assert "| case-b | PASS | FAIL(response_enum) |" in output
    assert "base: 1/2 pass" in output
    assert "v14: 1/2 pass" in output


def test_render_diff_flags_a_regression(tmp_path: Path) -> None:
    v13_run = write_run(tmp_path, "v13-run", ["case-a"], {"base": BASE_MODEL, "sft": V13_URI}, {})
    v14_run = write_run(
        tmp_path,
        "v14-run",
        ["case-a"],
        {"base": BASE_MODEL, "sft": V14_URI},
        {("case-a", "sft"): False},
    )

    output = summary.render_diff(v13_run, v14_run)

    assert "| case-a | PASS | FAIL(response_enum) | regressed |" in output
    assert "1 of 1 cases changed" in output


def test_render_single_compares_the_two_arms_of_a_paired_run(tmp_path: Path) -> None:
    # A v14-vs-v13 suite keeps the v13 adapter in the base slot, so the arm
    # comparison lives inside one run directory rather than across two.
    run_dir = write_run(
        tmp_path,
        "crisis-dev-v14-vs-v13",
        ["case-a", "case-b"],
        {"base": V13_URI, "sft": V14_URI},
        {("case-a", "sft"): False},
    )

    output = summary.render_single(run_dir)

    assert "# crisis-dev-v14-vs-v13: v13 vs v14" in output
    assert "| case-a | PASS | FAIL(response_enum) | regressed |" in output
    assert "| case-b | PASS | PASS | same |" in output
    assert "1 of 2 cases changed" in output


def test_render_diff_refuses_to_compare_a_model_against_itself(tmp_path: Path) -> None:
    arms = {"base": BASE_MODEL, "sft": V14_URI}
    first = write_run(tmp_path, "first", ["case-a"], arms, {})
    second = write_run(tmp_path, "second", ["case-a"], arms, {})

    with pytest.raises(ValueError, match="against itself is not a diff"):
        summary.render_diff(first, second)
