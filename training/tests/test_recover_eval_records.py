"""Tests for the log-to-results recovery tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.launch import recover_eval_records as recover


def _log_line(kind: str, body: dict[str, object], *, prefix: bool = True) -> str:
    marker = "[job] " if prefix else ""
    return marker + kind + " " + json.dumps(body)


def test_records_are_attributed_to_the_run_named_by_the_next_summary() -> None:
    lines = [
        _log_line("RECORD", {"case_id": "a", "arm_id": "base", "passed": True}),
        _log_line("RECORD", {"case_id": "b", "arm_id": "sft", "passed": False}),
        _log_line("SUMMARY", {"run": "eval-bf16-dev", "arm": "sft", "total": 2, "passed": 1}),
        _log_line("RECORD", {"case_id": "c", "arm_id": "sft", "passed": True}),
        _log_line("SUMMARY", {"run": "eval-nf4-dev", "arm": "sft", "total": 1, "passed": 1}),
    ]

    runs = recover.recover(lines)

    assert sorted(runs) == ["eval-bf16-dev", "eval-nf4-dev"]
    assert [row["case_id"] for row in runs["eval-bf16-dev"]] == ["a", "b"]
    assert [row["case_id"] for row in runs["eval-nf4-dev"]] == ["c"]


def test_the_job_prefix_is_optional_and_noise_lines_are_ignored() -> None:
    lines = [
        "[job] Qwen3-8B loaded",
        _log_line("RECORD", {"case_id": "a", "arm_id": "base", "passed": True}, prefix=False),
        _log_line("SUMMARY", {"run": "eval-bf16-dev"}, prefix=False),
    ]

    runs = recover.recover(lines)

    assert list(runs) == ["eval-bf16-dev"]
    assert runs["eval-bf16-dev"][0]["case_id"] == "a"


def test_records_after_the_last_summary_are_kept_as_unattributed() -> None:
    lines = [
        _log_line("RECORD", {"case_id": "a", "arm_id": "base", "passed": True}),
        _log_line("SUMMARY", {"run": "eval-bf16-dev"}),
        _log_line("RECORD", {"case_id": "b", "arm_id": "sft", "passed": False}),
    ]

    runs = recover.recover(lines)

    assert list(runs) == ["eval-bf16-dev", recover.UNATTRIBUTED]
    assert runs[recover.UNATTRIBUTED][0]["case_id"] == "b"


def test_write_runs_writes_one_results_file_per_run_and_counts_passes(tmp_path: Path) -> None:
    runs = {
        "eval-bf16-dev": [
            {"case_id": "a", "arm_id": "base", "passed": True},
            {"case_id": "b", "arm_id": "base", "passed": False},
            {"case_id": "a", "arm_id": "sft", "passed": True},
        ]
    }

    counts = recover.write_runs(runs, tmp_path)

    written = (tmp_path / "eval-bf16-dev" / "results.jsonl").read_text().splitlines()
    assert len(written) == 3
    assert counts == {"eval-bf16-dev": {"base": 1, "sft": 1}}


def test_write_runs_refuses_to_follow_a_results_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("keep\n")
    run = tmp_path / "eval-bf16-dev"
    run.mkdir()
    (run / "results.jsonl").symlink_to(outside)

    with pytest.raises(OSError, match="Errno 40"):
        recover.write_runs({"eval-bf16-dev": [{"case_id": "a", "arm_id": "base", "passed": True}]}, tmp_path)

    assert outside.read_text() == "keep\n"
