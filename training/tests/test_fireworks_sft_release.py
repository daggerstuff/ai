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


def test_build_service_uses_the_free_serverless_gateway() -> None:
    """The managed path provisions a dedicated trainer, which bills the account."""
    captured: dict[str, object] = {"enabled_retry": False}

    class StubService:
        _managed_config = None
        _managed_base_url = launcher.SERVERLESS_BASE_URL

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def _enable_serverless_supervised_409_retry(self) -> None:
            captured["enabled_retry"] = True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "fireworks.training.sdk.FiretitanServiceClient",
        StubService,
        raising=False,
    )
    try:
        service = launcher.build_service(api_key="fw-test")
    finally:
        monkeypatch.undo()

    assert isinstance(service, StubService)
    assert captured["api_key"] == "fw-test"
    assert captured["base_url"] == launcher.SERVERLESS_BASE_URL
    assert "managed_config" not in captured
    assert captured["enabled_retry"] is True


def test_assert_serverless_rejects_a_managed_service() -> None:
    """Fail closed before a run can reach a billed accelerator."""

    class Managed:
        _managed_config = object()
        _managed_base_url = launcher.SERVERLESS_BASE_URL

    with pytest.raises(RuntimeError, match="managed path"):
        launcher.assert_serverless(Managed())

    class Dedicated:
        _managed_config = None
        _managed_base_url = "https://api.fireworks.ai"

    with pytest.raises(RuntimeError, match="serverless gateway"):
        launcher.assert_serverless(Dedicated())


def test_promote_target_must_be_a_bare_model_name() -> None:
    """The SDK rejects a resource path only after training has finished."""
    assert launcher.check_promote_target("pixelated-glimmer-v22") == []

    problems = launcher.check_promote_target("accounts/linencloset/models/pixelated-glimmer-v22")

    assert problems
    assert "invalid characters" in problems[0]


def test_promote_uses_the_session_checkpoint_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A serverless run has no trainer job, so promotion is session-scoped."""
    promoted: dict[str, object] = {}
    session = "accounts/daggerstuff/trainingSessions/ts-abc123"

    class StubService:
        training_session_name = session

    class StubClient:
        def __init__(self, **kwargs: object) -> None:
            promoted.update(kwargs)

        def list_training_session_checkpoints(self, name: str) -> list[dict[str, object]]:
            promoted["listed"] = name
            return [
                {"name": f"{name}/checkpoints/step-1", "promotable": True, "createTime": "2026-11-12T07:00:00Z"},
                {"name": f"{name}/checkpoints/step-2", "promotable": True, "createTime": "2026-11-12T08:00:00Z"},
                {"name": f"{name}/checkpoints/step-3", "promotable": False, "createTime": "2026-11-12T09:00:00Z"},
            ]

        def promote_session_checkpoint(self, **kwargs: object) -> dict[str, object]:
            promoted["call"] = kwargs
            return {"name": "accounts/daggerstuff/models/pixelated-glimmer-v22"}

    monkeypatch.setattr("fireworks.training.sdk.fireworks_client.FireworksClient", StubClient)

    result = launcher.promote(StubService(), "pixelated-glimmer-v22", api_key="fw-test")

    assert promoted["listed"] == session
    assert promoted["api_key"] == "fw-test"
    call = promoted["call"]
    assert isinstance(call, dict)
    # Newest promotable wins; the newer non-promotable row is skipped.
    assert call["name"] == f"{session}/checkpoints/step-2"
    assert call["output_model_id"] == "pixelated-glimmer-v22"
    assert result["name"] == "accounts/daggerstuff/models/pixelated-glimmer-v22"


def test_promote_refuses_a_service_without_a_session() -> None:
    class StubService:
        training_session_name = None

    with pytest.raises(RuntimeError, match="no serverless training session"):
        launcher.promote(StubService(), "pixelated-glimmer-v22", api_key="fw-test")
