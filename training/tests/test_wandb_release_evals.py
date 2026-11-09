"""Tests for the unattended release-eval pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from training.launch import wandb_release_evals as pipeline

RELEASE_LABEL = "v17"
RELEASE_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-b6886c9f8dde:v1"
PREVIOUS_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-562d43347133:v1"
OLDER_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-73504a99ed05:v1"
SUITE_COUNT = 8
ONE_ATTEMPT_PLUS_RETRY = 2


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_parse_compare_reads_a_label_and_artifact_uri() -> None:
    assert pipeline.parse_compare(f"v16={PREVIOUS_URI}") == ("v16", PREVIOUS_URI)


@pytest.mark.parametrize("value", ["v16", f"={PREVIOUS_URI}", "v16=/tmp/model", ""])
def test_parse_compare_rejects_anything_that_is_not_a_label_and_artifact(value: str) -> None:
    with pytest.raises(Exception, match="label=wandb-artifact"):
        pipeline.parse_compare(value)


def test_build_commands_dedupes_the_vs_base_reading(tmp_path: Path) -> None:
    """The prep plan emits a vs-base pair for every compare; only one is run."""
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()

    commands = pipeline.build_commands(
        RELEASE_LABEL,
        RELEASE_URI,
        [("v16", PREVIOUS_URI), ("v15", OLDER_URI)],
        exec_dir,
    )

    labels = [label for label, _ in commands]
    assert len(labels) == len(set(labels))
    assert len(commands) == SUITE_COUNT * 3
    assert sum("-vs-base" in label for label in labels) == SUITE_COUNT
    assert sum("-vs-v16" in label for label in labels) == SUITE_COUNT
    assert sum("-vs-v15" in label for label in labels) == SUITE_COUNT


def test_build_commands_points_every_run_at_its_own_results_directory(tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()

    commands = pipeline.build_commands(RELEASE_LABEL, RELEASE_URI, [("v16", PREVIOUS_URI)], exec_dir)

    for label, argv in commands:
        results_dir = Path(argv[argv.index("--results-dir") + 1])
        assert results_dir == exec_dir / "evals" / "results" / label
        assert argv[argv.index("--config") + 1].endswith(f"{label}.config.json")


def test_a_transient_provider_error_gets_one_clean_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "partial.json").write_text("{}", encoding="utf-8")
    argv = ["runner", "--results-dir", str(results_dir)]
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            return _Completed(1, stderr="ProviderHTTPError: http_502 bad gateway")
        return _Completed(0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("crisis-dev-v17-vs-base", argv, "key")

    assert (label, rc) == ("crisis-dev-v17-vs-base", 0)
    assert len(calls) == ONE_ATTEMPT_PLUS_RETRY
    assert "--continue-on-transient-http-errors" in calls[1]
    # the retry starts from a clean directory: a partial run's manifest records
    # the flag set it was launched with, so resuming it is rejected.
    assert not results_dir.exists()


def test_a_real_failure_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["runner", "--results-dir", str(tmp_path / "results")]
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return _Completed(2, stderr="EvaluationInputError: case identity changed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("literal-sealed-v17-vs-base", argv, "key")

    assert (label, rc) == ("literal-sealed-v17-vs-base", 2)
    assert len(calls) == 1
