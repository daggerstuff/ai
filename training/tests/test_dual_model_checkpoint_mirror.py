from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import wandb
from training import backup_dual_model_sft_checkpoints as mirror


def test_saved_checkpoint_is_rehashed_before_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "gemma4/step-00000000-v0"
    destination.mkdir(parents=True)
    (destination / "adapter_model.safetensors").write_bytes(b"changed")
    manifest = {
        "local_files": [
            {
                "name": "adapter_model.safetensors",
                "size_bytes": 7,
                "sha256": hashlib.sha256(b"initial").hexdigest(),
            }
        ]
    }
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="integrity"):
        mirror.backup_checkpoint(
            None,
            tmp_path,
            "gemma4",
            {
                "step": 0,
                "version": 0,
                "artifact_name": "model",
            },
        )


def test_checkpoint_poll_reads_all_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class API:
        def request(self, _method: str, path: str) -> dict[str, Any]:
            calls.append(path)
            if "training-jobs" in path:
                return {"status": "RUNNING"}
            step = 1 if "after=" in path else 0
            return {"data": [{"step": step}], "has_more": step == 0, "last_id": "cursor-0"}

    def backup(_api: object, _root: Path, _label: str, checkpoint: dict[str, Any], _campaign: str) -> dict[str, Any]:
        return {"checkpoint_step": checkpoint["step"], "checkpoint_version": 0, "artifact_size_bytes": 8}

    monkeypatch.setattr(mirror, "backup_checkpoint", backup)
    result = mirror.poll_model(API(), None, tmp_path, "gemma4", {"model_id": "m", "training_job_id": "j"})
    assert [item["step"] for item in result["checkpoints"]] == [0, 1]
    assert calls[-1].endswith("?after=cursor-0")


def test_transient_error_does_not_stop_other_model_or_next_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"model_id": "m", "training_job_id": "j"}))
    calls = []

    def poll(_training: object, _public: object, _root: Path, label: str, _state: dict[str, Any]) -> dict[str, Any]:
        calls.append(label)
        if len(calls) == 1:
            raise TimeoutError("transport error")
        return {"status": "RUNNING", "checkpoints": []}

    monkeypatch.setattr(mirror, "TrainingAPI", lambda _: None)
    monkeypatch.setattr(wandb, "Api", lambda **_: None)
    monkeypatch.setattr(mirror, "poll_model", poll)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror",
            "--backup-root",
            str(tmp_path / "backup"),
            "--max-polls",
            "2",
            "--state-file",
            f"gemma4={state}",
            "--state-file",
            f"nemotron35={state}",
        ],
    )
    assert mirror.main() == 0
    assert calls == ["gemma4", "nemotron35", "gemma4", "nemotron35"]


def test_completed_job_waits_for_delayed_trained_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"model_id": "m", "training_job_id": "j"}))
    observed_steps: list[int] = []

    def poll(*_args: object) -> dict[str, Any]:
        step = int(bool(observed_steps))
        observed_steps.append(step)
        return {"status": "COMPLETED", "checkpoints": [{"step": step}]}

    monkeypatch.setattr(mirror, "TrainingAPI", lambda _: None)
    monkeypatch.setattr(wandb, "Api", lambda **_: None)
    monkeypatch.setattr(mirror, "poll_model", poll)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror",
            "--backup-root",
            str(tmp_path / "backup"),
            "--max-polls",
            "2",
            "--state-file",
            f"gemma4={state}",
        ],
    )
    assert mirror.main() == 0
    assert observed_steps == [0, 1]
