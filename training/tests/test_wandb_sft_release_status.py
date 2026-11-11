import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE_NAME = "wandb_sft_release"
MODULE_PATH = Path(__file__).parents[1] / "launch" / f"{MODULE_NAME}.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load test module at {MODULE_PATH}")
release = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = release
SPEC.loader.exec_module(release)

g04 = release.g04

RELEASE_ID = "wandb-full-corpus-current-master-20261110-v99"
TASK_ID = "a" * 64
JOB_ID = "827fb6fa-6816-493b-b1cf-48b2809fd627"
MODEL_ID = "f0aa8b33-4309-4be3-ac4e-4530c0ade9f8"
REQUESTED_POLLS = 3
FOLLOW_POLL_CALLS = 4


class FakeAPI:
    """Returns the scripted statuses in order, repeating the last one."""

    def __init__(self, statuses: list[str]):
        self.statuses = statuses
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, _body: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, path))
        index = min(len(self.calls), len(self.statuses)) - 1
        return {"id": JOB_ID, "status": self.statuses[index]}


def make_state_dir(tmp_path: Path, *, release_id: str = RELEASE_ID) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = {
        "task_id": TASK_ID,
        "release_id": release_id,
        "arm": "baseline",
        "phase": "job_created",
        "last_status": "PENDING",
        "training_job_id": JOB_ID,
        "model_id": MODEL_ID,
        "payload_sha256": "b" * 64,
    }
    (state_dir / f"wandb-sft-{TASK_ID}.json").write_text(json.dumps(state), encoding="utf-8")
    return state_dir


def make_campaign_files(tmp_path: Path, *, release_id: str = RELEASE_ID) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"release_id": release_id}), encoding="utf-8")
    config_path = tmp_path / "v99-launch-config.json"
    config_path.write_text(json.dumps({"manifest": str(manifest_path), "train_rows": 2}), encoding="utf-8")
    return config_path, manifest_path


def make_request(state_dir: Path, api: FakeAPI, *, max_polls: int | None, tmp_path: Path) -> Any:
    return g04.ExecutionRequest(
        manifest_path=tmp_path / "manifest.json",
        arm="baseline",
        state_dir=state_dir,
        work_dir=tmp_path / "work",
        api=api,
        api_key="fake-key-for-test",
        poll_interval=0,
        max_polls=max_polls,
    )


def read_state(state_dir: Path) -> dict[str, Any]:
    return json.loads((state_dir / f"wandb-sft-{TASK_ID}.json").read_text(encoding="utf-8"))


def test_poll_budget_follows_by_default_only_where_asked() -> None:
    follow = release._parse_args(["--release", "v21", "--execute"])
    assert release._poll_budget(follow, follow_by_default=True) is None
    assert release._poll_budget(follow, follow_by_default=False) == 1
    bounded = release._parse_args(["--release", "v21", "--execute", "--polls", "3"])
    assert release._poll_budget(bounded, follow_by_default=True) == REQUESTED_POLLS
    no_polls = release._parse_args(["--release", "v21", "--status", "--polls", "0"])
    assert release._poll_budget(no_polls, follow_by_default=False) == 0


def test_status_and_execute_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        release._parse_args(["--release", "v21", "--execute", "--status"])


def test_a_single_poll_after_creation_records_that_the_job_is_running(tmp_path: Path) -> None:
    """The defect this guards: one poll used to leave the state file saying PENDING."""
    state_dir = make_state_dir(tmp_path)
    api = FakeAPI(["RUNNING"])
    state = read_state(state_dir)
    release.g04._poll_job(make_request(state_dir, api, max_polls=1, tmp_path=tmp_path), state, JOB_ID, TASK_ID)
    assert len(api.calls) == 1
    recorded = read_state(state_dir)
    assert recorded["phase"] == "job_running"
    assert recorded["last_status"] == "RUNNING"


def test_following_polls_until_the_job_is_terminal(tmp_path: Path) -> None:
    state_dir = make_state_dir(tmp_path)
    api = FakeAPI(["PENDING", "RUNNING", "RUNNING", "COMPLETED"])
    state = read_state(state_dir)
    release.g04._poll_job(make_request(state_dir, api, max_polls=None, tmp_path=tmp_path), state, JOB_ID, TASK_ID)
    assert len(api.calls) == FOLLOW_POLL_CALLS
    recorded = read_state(state_dir)
    assert recorded["phase"] == "completed"
    assert recorded["last_status"] == "COMPLETED"


def test_terminal_detection_is_case_insensitive() -> None:
    """W&B reports upper-case statuses; a lower-case set never matched them."""
    for status in ("COMPLETED", "Succeeded", "FAILED", "canceled", "CANCELLED", "error"):
        assert g04.is_terminal_status(status), status
    for status in ("PENDING", "RUNNING", "queued", "", None, 0):
        assert not g04.is_terminal_status(status), status


def test_status_reconciles_a_recorded_run_without_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = make_campaign_files(tmp_path)
    state_dir = make_state_dir(tmp_path)
    api = FakeAPI(["RUNNING"])
    monkeypatch.setattr(release, "_ensure_api_key", lambda: "fake-key-for-test")
    monkeypatch.setattr(g04, "TrainingAPI", lambda _key: api)
    args = release._parse_args(["--config", str(config_path), "--status", "--state-dir", str(state_dir)])
    assert release._status(args) == 0
    assert [method for method, _ in api.calls] == ["GET"]
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "status"
    assert report["last_status"] == "RUNNING"
    assert report["terminal"] is False
    assert report["training_job_id"] == JOB_ID
    assert read_state(state_dir)["phase"] == "job_running"


def test_status_reports_a_completed_job_as_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = make_campaign_files(tmp_path)
    state_dir = make_state_dir(tmp_path)
    api = FakeAPI(["COMPLETED"])
    monkeypatch.setattr(release, "_ensure_api_key", lambda: "fake-key-for-test")
    monkeypatch.setattr(g04, "TrainingAPI", lambda _key: api)
    args = release._parse_args(["--config", str(config_path), "--status", "--state-dir", str(state_dir)])
    assert release._status(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["terminal"] is True
    assert report["phase"] == "completed"


def test_reconcile_is_mutually_exclusive_with_the_other_modes() -> None:
    with pytest.raises(SystemExit):
        release._parse_args(["--release", "v21", "--reconcile", "--execute"])
    with pytest.raises(SystemExit):
        release._parse_args(["--release", "v21", "--reconcile", "--status"])


def test_reconcile_refuses_a_run_that_never_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, manifest_path = make_campaign_files(tmp_path)
    state_dir = tmp_path / "empty-state"
    state_dir.mkdir()
    monkeypatch.setattr(release, "_ensure_api_key", lambda: "fake-key-for-test")
    monkeypatch.setattr(g04, "make_training_artifact", lambda *_args, **_kwargs: {"artifact_bytes": 1})
    monkeypatch.setattr(g04, "_verify_remote_access", lambda _request: None)
    monkeypatch.setattr(release, "_artifact_uploader", lambda *_args, **_kwargs: None)
    campaign = release.Campaign(
        config_path=config_path,
        manifest_path=manifest_path,
        manifest={"release_id": RELEASE_ID},
        payload=g04.Payload(path=manifest_path, sha256="c" * 64, rows=1, split="train"),
        config={},
        task_id=TASK_ID,
    )
    args = release._parse_args(["--config", str(config_path), "--reconcile", "--state-dir", str(state_dir)])

    with pytest.raises(g04.QualificationError, match="nothing to reconcile"):
        release._execute(args, campaign, reconcile=True)


def test_status_refuses_a_release_with_no_recorded_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _ = make_campaign_files(tmp_path)
    state_dir = make_state_dir(tmp_path, release_id="some-other-release")
    monkeypatch.setattr(release, "_ensure_api_key", lambda: "fake-key-for-test")
    args = release._parse_args(["--config", str(config_path), "--status", "--state-dir", str(state_dir)])
    with pytest.raises(g04.QualificationError, match="no launcher state"):
        release._status(args)


def test_status_refuses_an_ambiguous_state_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _ = make_campaign_files(tmp_path)
    state_dir = make_state_dir(tmp_path)
    duplicate = json.loads((state_dir / f"wandb-sft-{TASK_ID}.json").read_text(encoding="utf-8"))
    (state_dir / "wandb-sft-other.json").write_text(json.dumps(duplicate), encoding="utf-8")
    monkeypatch.setattr(release, "_ensure_api_key", lambda: "fake-key-for-test")
    args = release._parse_args(["--config", str(config_path), "--status", "--state-dir", str(state_dir)])
    with pytest.raises(g04.QualificationError, match="more than one launcher state"):
        release._status(args)
