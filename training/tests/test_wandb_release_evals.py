"""Tests for the unattended release-eval pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from training.launch import prepare_release_eval as prep, wandb_release_evals as pipeline

RELEASE_LABEL = "v17"
RELEASE_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-b6886c9f8dde:v1"
PREVIOUS_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-562d43347133:v1"
OLDER_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-73504a99ed05:v1"
SUITE_COUNT = 8
ONE_ATTEMPT_PLUS_RETRY = 2
CLEAN_RESTART_CALL = 2


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_cases_dir(root: Path) -> Path:
    cases_dir = root / "cases"
    cases_dir.mkdir()
    for _, case_file, _ in prep.SUITES:
        (cases_dir / case_file).write_text('{"id": "case-1"}\n', encoding="utf-8")
    return cases_dir


def test_parse_compare_reads_a_label_and_artifact_uri() -> None:
    assert pipeline.parse_compare(f"v16={PREVIOUS_URI}") == ("v16", PREVIOUS_URI)


def test_parse_compare_rejects_anything_that_is_not_a_label_and_artifact() -> None:
    for value in ["v16", f"={PREVIOUS_URI}", "v16=/tmp/model", ""]:
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
        cases_dir=make_cases_dir(tmp_path),
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

    commands = pipeline.build_commands(
        RELEASE_LABEL,
        RELEASE_URI,
        [("v16", PREVIOUS_URI)],
        exec_dir,
        cases_dir=make_cases_dir(tmp_path),
    )

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

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
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

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(command))
        return _Completed(2, stderr="EvaluationInputError: case identity changed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("literal-sealed-v17-vs-base", argv, "key")

    assert (label, rc) == ("literal-sealed-v17-vs-base", 2)
    assert len(calls) == 1


def _write_result(results_dir: Path, **record: Any) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"arm_id": "release", "case_id": "self_harm", **record})
    with (results_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def test_transient_failures_ignores_successful_records(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="success", error_type=None)
    _write_result(results_dir, status="failed", error_type="ValueError")

    assert pipeline.transient_failures(results_dir) == []


def test_transient_failures_reports_a_read_timeout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="failed", error_type="TimeoutError")

    assert pipeline.transient_failures(results_dir) == ["release/self_harm=TimeoutError"]


def test_transient_failures_reports_a_provider_5xx(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="failed", error_type="ProviderHTTPError", provider_http_status=503)

    assert pipeline.transient_failures(results_dir) == ["release/self_harm=ProviderHTTPError"]


def test_transient_failures_tolerates_a_missing_results_file(tmp_path: Path) -> None:
    assert pipeline.transient_failures(tmp_path / "nothing-here") == []


def test_transient_failures_ignores_a_timeout_that_was_later_answered(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="error", error_type="TimeoutError")
    _write_result(results_dir, status="success", error_type=None, retry_count=1)

    assert pipeline.transient_failures(results_dir) == []


def test_a_read_timeout_resumes_in_place_without_wiping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="failed", error_type="TimeoutError")
    argv = ["runner", "--results-dir", str(results_dir)]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(command))
        if len(calls) == 1:
            return _Completed(1, stdout="case failed")
        return _Completed(0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("crisis-dev-v21-vs-base", argv, "key")

    assert (label, rc) == ("crisis-dev-v21-vs-base", 0)
    assert len(calls) == ONE_ATTEMPT_PLUS_RETRY
    # a timeout is not fatal, so the harness resumes and only the failed cases
    # are re-asked: the directory and its completed records must survive.
    assert "--continue-on-transient-http-errors" not in calls[1]
    assert results_dir.exists()


def test_a_read_timeout_retry_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="failed", error_type="TimeoutError")
    argv = ["runner", "--results-dir", str(results_dir)]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(command))
        return _Completed(1, stdout="still timing out")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("crisis-dev-v21-vs-base", argv, "key")

    assert (label, rc) == ("crisis-dev-v21-vs-base", 1)
    assert len(calls) == 1 + pipeline.MAX_TRANSIENT_RETRIES


def test_resume_blocked_reads_the_fatal_flag(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="error", error_type="ProviderHTTPError", provider_http_status=503, fatal=True)

    assert pipeline.resume_blocked(results_dir) is True


def test_resume_blocked_is_false_for_a_transport_timeout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="failed", error_type="TimeoutError", fatal=False)

    assert pipeline.resume_blocked(results_dir) is False


def test_a_fatal_provider_5xx_restarts_clean_instead_of_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The harness refuses to resume a run whose record is fatal, so re-asking the
    # same directory cannot help; the launcher has to start from a clean one.
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="error", error_type="ProviderHTTPError", provider_http_status=503, fatal=True)
    argv = ["runner", "--results-dir", str(results_dir)]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(command))
        return _Completed(1, stdout="resume blocked after fatal provider metadata failure: ProviderHTTPError")

    monkeypatch.setattr(subprocess, "run", fake_run)

    label, rc = pipeline.run_one("crisis-dev-v21-vs-base", argv, "key")

    assert (label, rc) == ("crisis-dev-v21-vs-base", 1)
    assert "--continue-on-transient-http-errors" in calls[1]
    assert not results_dir.exists()


def test_a_resume_after_a_clean_restart_keeps_the_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The resume manifest is compared against the recorded options, so the flag the
    # clean restart added has to stay on every later invocation.
    results_dir = tmp_path / "results"
    _write_result(results_dir, status="error", error_type="ProviderHTTPError", provider_http_status=503, fatal=True)
    argv = ["runner", "--results-dir", str(results_dir)]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> _Completed:
        calls.append(list(command))
        if len(calls) == CLEAN_RESTART_CALL:
            _write_result(results_dir, status="failed", error_type="TimeoutError", fatal=False)
        return _Completed(1, stdout="ProviderHTTPError")

    monkeypatch.setattr(subprocess, "run", fake_run)

    pipeline.run_one("crisis-dev-v21-vs-base", argv, "key")

    assert "--continue-on-transient-http-errors" in calls[1]
    assert "--continue-on-transient-http-errors" in calls[2]


def test_resolve_compares_keeps_the_default_predecessor() -> None:
    """An explicit --compare must not drop the reading against the default prior adapter."""
    default = ("v16", PREVIOUS_URI)
    compares = pipeline.resolve_compares([("v15", OLDER_URI)], default, RELEASE_LABEL)

    assert [label for label, _ in compares] == ["v15", "v16"]


def test_resolve_compares_does_not_repeat_a_requested_default() -> None:
    default = ("v16", PREVIOUS_URI)
    compares = pipeline.resolve_compares([("v16", PREVIOUS_URI)], default, RELEASE_LABEL)

    assert compares == [("v16", PREVIOUS_URI)]


def test_resolve_compares_never_compares_a_release_with_itself() -> None:
    default = (RELEASE_LABEL, RELEASE_URI)
    compares = pipeline.resolve_compares([(RELEASE_LABEL, RELEASE_URI), ("v16", PREVIOUS_URI)], default, RELEASE_LABEL)

    assert compares == [("v16", PREVIOUS_URI)]


def test_resolve_compares_can_opt_out_of_the_default() -> None:
    default = ("v16", PREVIOUS_URI)
    compares = pipeline.resolve_compares([("v15", OLDER_URI)], default, RELEASE_LABEL, include_default=False)

    assert compares == [("v15", OLDER_URI)]


def test_resolve_compares_refuses_a_set_with_nothing_to_run() -> None:
    with pytest.raises(SystemExit, match="no comparisons to run"):
        pipeline.resolve_compares([(RELEASE_LABEL, RELEASE_URI)], (RELEASE_LABEL, RELEASE_URI), RELEASE_LABEL)


def test_build_commands_runs_the_resolved_set(tmp_path: Path) -> None:
    """The pipeline's default run is the requested compare plus the default prior adapter."""
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    compares = pipeline.resolve_compares(
        [("v15", OLDER_URI)],
        ("v16", PREVIOUS_URI),
        RELEASE_LABEL,
    )

    commands = pipeline.build_commands(
        RELEASE_LABEL,
        RELEASE_URI,
        compares,
        exec_dir,
        cases_dir=make_cases_dir(tmp_path),
    )

    labels = [label for label, _ in commands]
    assert len(commands) == SUITE_COUNT * 3
    assert sum("-vs-v16" in label for label in labels) == SUITE_COUNT
