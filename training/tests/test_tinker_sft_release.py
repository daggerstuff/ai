"""Tests for the Tinker SFT trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import tinker_sft_release as trainer
from training.launch.make_launch_config import PEAK_LR, learning_rate_schedule


class FakeDatum:
    """Stand-in for ``tinker.Datum``."""

    def __init__(self, model_input: Any, loss_fn_inputs: dict[str, Any]) -> None:
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class FakeModelInput:
    """Stand-in for ``tinker.ModelInput``."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    @classmethod
    def from_ints(cls, ids: list[int]) -> FakeModelInput:
        return cls(list(ids))


class FakeAdamParams:
    """Stand-in for ``tinker.AdamParams``."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate


FAKE_TINKER = SimpleNamespace(
    Datum=FakeDatum,
    ModelInput=FakeModelInput,
    AdamParams=FakeAdamParams,
)


class FakeFuture:
    """Minimal stand-in for Tinker's APIFuture."""

    def __init__(self, value: Any = None, metrics: dict[str, Any] | None = None) -> None:
        if value is None:
            value = SimpleNamespace(path="weights://fake")
        self._value = SimpleNamespace(**vars(value)) if metrics else value
        self._metrics = metrics or {}

    def result(self) -> Any:
        if self._metrics:
            return SimpleNamespace(**vars(self._value), metrics=self._metrics)
        return self._value


class FakeClient:
    """Records what the trainer asked Tinker to do."""

    def __init__(self) -> None:
        self.batches: list[list[FakeDatum]] = []
        self.learning_rates: list[float] = []
        self.saved_states: list[str] = []
        self.sampled: list[str] = []

    def forward_backward(self, data: list[FakeDatum], loss_fn: str) -> FakeFuture:
        assert loss_fn == "cross_entropy"
        self.batches.append(list(data))
        return FakeFuture(metrics={"loss:mean": 1.0 / (len(self.batches) + 1)})

    def optim_step(self, params: FakeAdamParams) -> FakeFuture:
        self.learning_rates.append(params.learning_rate)
        return FakeFuture()

    def save_state(self, name: str) -> FakeFuture:
        self.saved_states.append(name)
        return FakeFuture(SimpleNamespace(path=f"state://{name}"))

    def save_weights_for_sampler(self, name: str) -> FakeFuture:
        self.sampled.append(name)
        return FakeFuture(SimpleNamespace(path=f"sampler://{name}"))


class FakeService:
    """Records the LoRA client request."""

    def __init__(self) -> None:
        self.client = FakeClient()
        self.requests: list[dict[str, Any]] = []

    def create_lora_training_client(self, **kwargs: Any) -> FakeClient:
        self.requests.append(kwargs)
        return self.client


class FakeRun:
    """Records W&B scalars."""

    def __init__(self) -> None:
        self.scalars: list[dict[str, float]] = []
        self.config: dict[str, Any] = {}
        self.finished = False

    def log(self, scalars: dict[str, float]) -> None:
        self.scalars.append(scalars)

    def finish(self) -> None:
        self.finished = True


def datum(input_ids: list[int], weights: list[int]) -> dict[str, Any]:
    """Build one exported datum line."""
    return {
        "input_ids": input_ids,
        "target_ids": input_ids[1:] + [0],
        "weights": weights,
    }


def write_export(tmp_path: Path, rows: list[dict[str, Any]], *, datums: int | None = None) -> Path:
    """Write an export dir with a record that matches the datum file."""
    path = tmp_path / trainer.DATUM_NAME
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "schema": "tinker-tokenized-datums-v1",
        "base_model": "Qwen/Qwen3-8B",
        "sha256": sha,
        "datums": datums if datums is not None else len(rows),
        "source_release_id": "stub-release",
        "counts": {"trained_tokens": sum(sum(row["weights"]) for row in rows)},
    }
    (tmp_path / trainer.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def many_rows(count: int) -> list[dict[str, Any]]:
    """Return ``count`` datums with one trained token each."""
    return [datum([10, 11, 12], [0, 1, 0]) for _ in range(count)]


def test_plan_reuses_the_release_lr_recipe(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(48))

    plan = trainer.plan_run(dataset, batch_size=1)

    assert plan.steps == 48
    assert plan.schedule == learning_rate_schedule(48)
    assert plan.trained_tokens == 48
    assert plan.recipe_degenerate is False


def test_plan_holds_peak_lr_when_a_run_is_too_short_for_the_recipe(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(8))

    plan = trainer.plan_run(dataset, batch_size=16)

    assert plan.steps == 1
    assert plan.schedule == [PEAK_LR]
    assert plan.recipe_degenerate is True


def test_plan_rejects_a_datum_file_that_no_longer_matches_its_record(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(4))
    (dataset / trainer.DATUM_NAME).write_text(json.dumps(datum([1, 2], [0, 1])) + "\n", encoding="utf-8")

    with pytest.raises(trainer.PlanError, match="sha256"):
        trainer.plan_run(dataset)


def test_plan_rejects_a_miscounted_export(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(4), datums=99)

    with pytest.raises(trainer.PlanError, match="datums"):
        trainer.plan_run(dataset)


def test_plan_rejects_a_missing_export(tmp_path: Path) -> None:
    with pytest.raises(trainer.PlanError, match="missing"):
        trainer.plan_run(tmp_path)


def test_limit_skips_the_full_export_verification(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(64))

    plan = trainer.plan_run(dataset, batch_size=16, limit=16)

    assert plan.datums == 16
    assert plan.steps == 1


def test_datum_carries_the_exported_mask(tmp_path: Path) -> None:
    row = datum([10, 11, 12, 13], [0, 1, 1, 0])

    built = trainer.datum_from_row(FAKE_TINKER, row)

    assert built.model_input.ids == [10, 11, 12, 13]
    assert built.loss_fn_inputs["target_tokens"] == [11, 12, 13, 0]
    assert built.loss_fn_inputs["weights"] == [0, 1, 1, 0]


def test_train_replays_the_schedule_and_saves_weights(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(48))
    plan = trainer.plan_run(dataset, batch_size=16)
    service = FakeService()
    run = FakeRun()

    record = trainer.train(
        plan,
        service=service,
        tinker=FAKE_TINKER,
        run=run,
        log=lambda *_: None,
    )

    assert service.requests == [{"base_model": "Qwen/Qwen3-8B", "rank": 32, "seed": 20261110}]
    assert service.client.learning_rates == plan.schedule
    assert [len(batch) for batch in service.client.batches] == [16, 16, 16]
    assert service.client.sampled == ["tinker-stub-release"]
    assert record["sampling_path"] == "sampler://tinker-stub-release"
    assert record["state_path"] == "state://tinker-stub-release-final"
    assert len(record["losses"]) == plan.steps
    assert [scalar["train/lr"] for scalar in run.scalars] == plan.schedule
    assert run.scalars[-1]["train/loss"] == pytest.approx(0.25)


def test_train_saves_intermediate_state_when_asked(tmp_path: Path) -> None:
    dataset = write_export(tmp_path, many_rows(48))
    plan = trainer.plan_run(dataset, batch_size=16)
    service = FakeService()

    trainer.train(
        plan,
        service=service,
        tinker=FAKE_TINKER,
        log=lambda *_: None,
        save_every=1,
    )

    assert service.client.saved_states == [
        "tinker-stub-release-step1",
        "tinker-stub-release-step2",
        "tinker-stub-release-final",
    ]


def test_check_api_reports_a_billing_block(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, api_key: str | None, timeout: float = 30.0) -> tuple[int, str]:
        if url.endswith("/healthz"):
            return 200, "ok"
        return 402, "Access is blocked due to billing status."

    monkeypatch.setattr(trainer, "_get", fake_get)

    report = trainer.check_api(api_key="tml-test")

    assert report["health_status"] == 200
    assert report["blocked"] is True
    assert "billing" in report["reason"]


def test_check_api_reports_whether_the_base_model_is_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = {
        "supported_models": [
            {"model_name": "Qwen/Qwen3-8B"},
            {"model_name": "meta-llama/Llama-3.1-8B"},
        ]
    }

    def fake_get(url: str, api_key: str | None, timeout: float = 30.0) -> tuple[int, str]:
        if url.endswith("/healthz"):
            return 200, "ok"
        return 200, json.dumps(capabilities)

    monkeypatch.setattr(trainer, "_get", fake_get)

    report = trainer.check_api(base_model="Qwen/Qwen3-8B")

    assert report["blocked"] is False
    assert report["model_supported"] is True
    assert report["model_count"] == 2
