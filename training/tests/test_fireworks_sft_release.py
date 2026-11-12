"""Tests for the Fireworks Training API launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import fireworks_sft_release as launcher
from training.tinker_sft_release import Plan


def make_plan(dataset_dir: Path, *, release_id: str | None) -> Plan:
    return Plan(
        dataset_dir=dataset_dir,
        base_model=launcher.FIREWORKS_BASE_MODEL,
        datums=1,
        trained_tokens=1,
        batch_size=1,
        steps=1,
        schedule=[5e-5],
        rank=32,
        seed=1,
        release_id=release_id,
        recipe_degenerate=True,
    )


def test_checkpoint_name_fits_the_dns_label_budget_with_the_sdk_suffix(tmp_path: Path) -> None:
    release_id = "wandb-full-corpus-current-master-20261110-v22"
    plan = make_plan(tmp_path, release_id=release_id)

    name = launcher.checkpoint_name(plan)

    # The SDK appends "-<8 hex>" of its own, and the label limit is 63.
    assert len(name) + 9 <= 63
    # The release version is what tells two runs apart, so it must survive.
    assert name.endswith("v22")
    assert name.startswith(f"{launcher.CHECKPOINT_PREFIX}-")


def test_checkpoint_name_sanitizes_a_dataset_directory_name(tmp_path: Path) -> None:
    plan = make_plan(tmp_path / "Wandb_Full.Corpus v22", release_id=None)

    name = launcher.checkpoint_name(plan)

    assert name == "fw-wandb-full-corpus-v22"


def test_checkpoint_name_keeps_a_short_release_id_intact(tmp_path: Path) -> None:
    plan = make_plan(tmp_path, release_id="v22")

    assert launcher.checkpoint_name(plan) == "fw-v22"


def test_describe_model_reads_tunability_from_the_model_and_arch_from_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "state": "READY",
        "supervisedLoraTunable": True,
        "supportsLora": True,
        "supportsServerlessTraining": True,
        "trainingContextLength": 131072,
        "contextLength": 131072,
        "baseModelDetails": {"parameterCount": "29776626688", "moe": False},
    }

    def fake_get(url: str, api_key: str | None, timeout: float = 30.0) -> tuple[int, str]:
        return 200, json.dumps(payload)

    monkeypatch.setattr(launcher, "http_get", fake_get)

    report = launcher.describe_model("fw-test")

    assert report["status"] == 200
    assert report["supervisedLoraTunable"] is True
    assert report["supportsServerlessTraining"] is True
    assert report["trainingContextLength"] == 131072
    assert report["parameter_count"] == "29776626688"
    assert report["moe"] is False


def test_describe_model_reports_a_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, api_key: str | None, timeout: float = 30.0) -> tuple[int, str]:
        return 412, "Account linencloset is suspended"

    monkeypatch.setattr(launcher, "http_get", fake_get)

    report = launcher.describe_model("fw-test")

    assert report["status"] == 412
    assert "suspended" in report["reason"]
    assert "supervisedLoraTunable" not in report


def test_build_service_turns_on_trainer_cleanup(tmp_path: Path) -> None:
    """A run that leaves its trainer job up keeps billing GPUs."""
    captured: dict[str, object] = {}

    class StubService:
        @classmethod
        def from_firetitan_config(cls, **kwargs: object) -> str:
            captured.update(kwargs)
            return "service"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "fireworks.training.sdk.FiretitanServiceClient",
        StubService,
        raising=False,
    )
    try:
        service = launcher.build_service(
            make_plan(tmp_path, release_id="v22"),
            api_key="fw-test",
            region="US_OHIO_3",
            max_context_length=32768,
        )
    finally:
        monkeypatch.undo()

    assert service == "service"
    assert captured["cleanup_trainer_on_close"] is True
    assert captured["create_deployment"] is False
    assert captured["base_model"] == launcher.FIREWORKS_BASE_MODEL
    assert captured["tokenizer_model"] == launcher.TOKENIZER_MODEL
    assert captured["lora_rank"] == 32
    assert captured["lora_alpha"] == 32
