from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import wandb
from training import (
    backup_wandb_full_corpus_lora as lora_backup,
    backup_wandb_full_corpus_train_state as train_state_backup,
)

EXPECTED_INCOMPLETE_POLL_EXIT = 2


class _StatusUnavailable:
    def request(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("status endpoint unavailable")


class _NoArtifacts:
    def __init__(self, **_kwargs: object) -> None:
        pass


def _seed_local_checkpoint(root: Path, label: str, version: str) -> dict[str, object]:
    checkpoint_dir = root / label / version
    checkpoint_dir.mkdir(parents=True)
    manifest = {
        "artifact_uri": f"wandb-artifact:///local/{label}:{version}",
        "version": version,
        "completed_batches": 123,
        "artifact_size_bytes": 7,
        "local_files": [{"name": "adapter.safetensors", "size_bytes": 7, "sha256": "local"}],
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_lora_backup_preserves_verified_index_when_status_and_artifacts_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for label in lora_backup.JOBS:
        _seed_local_checkpoint(backup_root, label, "v123")
        (backup_root / f"{label}-latest.json").write_text(
            json.dumps({"checkpoint_version": "v123"}), encoding="utf-8"
        )
    monkeypatch.setattr(lora_backup, "TrainingAPI", lambda _key: _StatusUnavailable())
    monkeypatch.setattr(wandb, "Api", _NoArtifacts)
    monkeypatch.setattr(lora_backup, "latest_artifact", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["backup", "--backup-root", str(backup_root), "--max-polls", "1"])
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    assert lora_backup.main() == EXPECTED_INCOMPLETE_POLL_EXIT

    index = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert set(index["checkpoints"]) == set(lora_backup.JOBS)
    for label in lora_backup.JOBS:
        latest = json.loads((backup_root / f"{label}-latest.json").read_text(encoding="utf-8"))
        observation = json.loads((backup_root / f"{label}-observation.json").read_text(encoding="utf-8"))
        assert latest["checkpoint_version"] == "v123"
        assert observation["status"] == "UNKNOWN"
        assert observation["checkpoint_version"] is None


def test_train_state_backup_preserves_verified_manifest_when_no_remote_versions_are_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    prior_checkpoints = {}
    for label in train_state_backup.JOBS:
        backup_label = f"{label}-train-state"
        manifest = _seed_local_checkpoint(backup_root, backup_label, "v123")
        prior_checkpoints[label] = {
            "artifact_uri": manifest["artifact_uri"],
            "version": manifest["version"],
            "completed_batches": manifest["completed_batches"],
            "size_bytes": manifest["artifact_size_bytes"],
            "local_path": str(backup_root / backup_label / "v123"),
        }
    (backup_root / "train-state-manifest.json").write_text(
        json.dumps({"checkpoints": prior_checkpoints}), encoding="utf-8"
    )
    monkeypatch.setattr(train_state_backup, "TrainingAPI", lambda _key: _StatusUnavailable())
    monkeypatch.setattr(wandb, "Api", _NoArtifacts)
    monkeypatch.setattr(train_state_backup, "latest_train_state_artifact", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["backup", "--backup-root", str(backup_root), "--max-polls", "1"])
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    assert train_state_backup.main() == EXPECTED_INCOMPLETE_POLL_EXIT

    saved = json.loads((backup_root / "train-state-manifest.json").read_text(encoding="utf-8"))
    assert saved["checkpoints"] == prior_checkpoints
    for label in train_state_backup.JOBS:
        observation = json.loads(
            (backup_root / f"{label}-train-state-observation.json").read_text(encoding="utf-8")
        )
        assert observation["status"] == "UNKNOWN"
        assert observation["checkpoint_version"] is None
