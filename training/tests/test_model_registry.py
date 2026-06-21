"""Tests for the model registry CLI (scripts/devops/model-registry.py)."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

# Load model_registry.py by path — scripts/ isn't a package directory.
_model_registry_path = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "devops" / "model-registry.py"
_spec = importlib.util.spec_from_file_location("model_registry", _model_registry_path)
_model_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_registry)

_load_manifest = _model_registry._load_manifest
_save_manifest = _model_registry._save_manifest
cmd_list = _model_registry.cmd_list
cmd_show = _model_registry.cmd_show
cmd_tag = _model_registry.cmd_tag
cmd_rollback = _model_registry.cmd_rollback


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch, tmp_path: Path) -> Path:
    """Redirect REGISTRY_PATH to a temp dir for test isolation."""
    test_registry = tmp_path / "registry"
    test_registry.mkdir()
    manifest_path = test_registry / "models.json"
    monkeypatch.setattr(_model_registry, "REGISTRY_PATH", manifest_path)
    monkeypatch.setattr(_model_registry, "DEFAULT_CHECKPOINT_DIR", tmp_path / "checkpoints")
    return manifest_path


# ---------------------------------------------------------------------------
# _load_manifest / _save_manifest
# ---------------------------------------------------------------------------


class TestManifestIO:
    def test_load_returns_default_when_missing(self):
        manifest = _load_manifest()
        assert manifest == {"schema_version": "1.0", "active_run_id": None, "checkpoints": []}

    def test_save_and_load_roundtrip(self):
        expected = {
            "schema_version": "1.0",
            "active_run_id": "run-001",
            "checkpoints": [{"run_id": "run-001", "base_model": "test"}],
        }
        _save_manifest(expected)
        loaded = _load_manifest()
        assert loaded == expected

    def test_save_creates_parent_directories(self):
        _save_manifest({"checkpoints": []})
        assert _model_registry.REGISTRY_PATH.exists()


# ---------------------------------------------------------------------------
# cmd_tag
# ---------------------------------------------------------------------------


class TestTag:
    def test_tag_adds_checkpoint(self):
        args = Namespace(
            run_id="grpo-001",
            base_model="Mistral-Nemo-Instruct-2407",
            dataset_version="v2",
            clinical_validity_score=0.85,
            metrics=None,
            force=False,
            set_active=False,
        )
        cmd_tag(args)
        manifest = _load_manifest()
        assert len(manifest["checkpoints"]) == 1
        assert manifest["checkpoints"][0]["run_id"] == "grpo-001"

    def test_tag_rejects_duplicate_without_force(self):
        cmd_tag(
            Namespace(
                run_id="dup",
                base_model="x",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=False,
                set_active=False,
            )
        )
        with pytest.raises(SystemExit):
            cmd_tag(
                Namespace(
                    run_id="dup",
                    base_model="x",
                    dataset_version="v1",
                    clinical_validity_score=0.5,
                    metrics=None,
                    force=False,
                    set_active=False,
                )
            )

    def test_tag_force_overwrites_duplicate(self):
        cmd_tag(
            Namespace(
                run_id="dup",
                base_model="old",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=False,
                set_active=False,
            )
        )
        cmd_tag(
            Namespace(
                run_id="dup",
                base_model="new",
                dataset_version="v2",
                clinical_validity_score=0.9,
                metrics=None,
                force=True,
                set_active=False,
            )
        )
        manifest = _load_manifest()
        assert len(manifest["checkpoints"]) == 1
        assert manifest["checkpoints"][0]["base_model"] == "new"

    def test_tag_sets_active(self):
        cmd_tag(
            Namespace(
                run_id="active-run",
                base_model="x",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=False,
                set_active=True,
            )
        )
        manifest = _load_manifest()
        assert manifest["active_run_id"] == "active-run"

    def test_tag_with_optional_metrics(self):
        metrics_json = '{"eval_loss": 0.45, "eval_accuracy": 0.92}'
        cmd_tag(
            Namespace(
                run_id="metrics-run",
                base_model="x",
                dataset_version="v1",
                clinical_validity_score=0.7,
                metrics=metrics_json,
                force=False,
                set_active=False,
            )
        )
        manifest = _load_manifest()
        assert manifest["checkpoints"][0]["metrics"] == {"eval_loss": 0.45, "eval_accuracy": 0.92}


# ---------------------------------------------------------------------------
# cmd_show
# ---------------------------------------------------------------------------


class TestShow:
    def test_show_returns_checkpoint(self, capsys):
        cmd_tag(
            Namespace(
                run_id="show-me",
                base_model="test-model",
                dataset_version="v1",
                clinical_validity_score=0.8,
                metrics=None,
                force=False,
                set_active=False,
            )
        )
        cmd_show(Namespace(run_id="show-me"))
        captured = capsys.readouterr()
        assert "show-me" in captured.out
        assert "test-model" in captured.out

    def test_show_missing_exits(self):
        with pytest.raises(SystemExit):
            cmd_show(Namespace(run_id="nonexistent"))


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_shows_checkpoints(self, capsys):
        cmd_tag(
            Namespace(
                run_id="list-run",
                base_model="m",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=False,
                set_active=True,
            )
        )
        cmd_list(Namespace())
        captured = capsys.readouterr()
        assert "list-run" in captured.out

    def test_list_empty_message(self, capsys):
        _save_manifest({"schema_version": "1.0", "active_run_id": None, "checkpoints": []})
        cmd_list(Namespace())
        captured = capsys.readouterr()
        assert "No checkpoints registered" in captured.out


# ---------------------------------------------------------------------------
# cmd_rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_updates_active_run_id(self, tmp_path: Path):
        cmd_tag(
            Namespace(
                run_id="roll-run",
                base_model="m",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=False,
                set_active=False,
            )
        )
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "roll-run").mkdir(parents=True)

        cmd_rollback(Namespace(run_id="roll-run", checkpoint_dir=checkpoint_dir))
        manifest = _load_manifest()
        assert manifest["active_run_id"] == "roll-run"

    def test_rollback_missing_run_id_exits(self):
        with pytest.raises(SystemExit):
            cmd_rollback(Namespace(run_id="ghost", checkpoint_dir=Path("/tmp")))

    def test_rollback_creates_active_symlink(self, tmp_path: Path):
        cmd_tag(
            Namespace(
                run_id="link-run",
                base_model="m",
                dataset_version="v1",
                clinical_validity_score=0.5,
                metrics=None,
                force=True,
                set_active=True,
            )
        )
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "link-run").mkdir(parents=True)

        cmd_rollback(Namespace(run_id="link-run", checkpoint_dir=checkpoint_dir))
        dest = tmp_path / "checkpoints" / "active"
        assert dest.exists()
