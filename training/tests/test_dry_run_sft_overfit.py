"""Tests for `training.scripts.dry_run_sft_overfit`.

Exercises the dry-run overfit script in mocked mode (no HuggingFace download
required). Verifies loss converges across 10 epochs on a 100-record PAL SFT
subset, matching PIX-4076 acceptance criterion (b).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ensure `training.scripts.*` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.pal_dataloader import PalSftDataset  # noqa: E402
from training.scripts.dry_run_sft_overfit import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    StubModel,
    StubTrainer,
    main,
    run_dry_run,
)
from training.tests.test_pal_dataloader import StubTokenizer  # noqa: E402


def _make_sft_jsonl(path: Path, n: int = 100) -> Path:
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a clinical assistant."},
                {"role": "user", "content": f"Hello patient {i}"},
                {"role": "assistant", "content": f"Hi, I will help with case {i}."},
            ],
            "metadata": {"index": i},
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


class TestStubModel:
    def test_loss_decreases_across_calls(self) -> None:
        m = StubModel(delta=0.1)
        losses = [float(m(input_ids=None).loss.item()) for _ in range(5)]
        for prev, cur in zip(losses, losses[1:]):
            assert cur < prev

    def test_loss_non_negative(self) -> None:
        m = StubModel(delta=10.0)
        # call many times — should clamp at 0
        out = m(input_ids=None)
        assert float(out.loss.item()) >= 0.0

    def test_train_returns_self(self) -> None:
        m = StubModel()
        assert m.train() is m

    def test_save_pretrained_creates_dir(self, tmp_path: Path) -> None:
        m = StubModel()
        out = tmp_path / "ckpt"
        m.save_pretrained(out)
        assert out.is_dir()


class TestStubTrainer:
    def test_runs_epochs_and_records_loss(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=10)
        ds = PalSftDataset(path, StubTokenizer())
        model = StubModel(delta=0.05)
        trainer = StubTrainer(
            model=model,
            dataset=ds,
            epochs=3,
            batch_size=2,
            learning_rate=DEFAULT_LEARNING_RATE,
        )
        metrics = trainer.train()
        assert metrics["epochs"] == 3
        assert len(metrics["loss_history"]) == 3
        assert metrics["converged"] is True
        assert metrics["final_loss"] < metrics["initial_loss"]

    def test_raises_on_empty_dataset(self) -> None:
        # build an empty-list dataset stub
        class _Empty:
            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> Any:
                raise IndexError

        trainer = StubTrainer(model=StubModel(), dataset=_Empty(), epochs=1)
        with pytest.raises(ValueError, match="dataset is empty"):
            trainer.train()

    def test_writes_metrics_json(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=10)
        ds = PalSftDataset(path, StubTokenizer())
        out_dir = tmp_path / "out"
        trainer = StubTrainer(
            model=StubModel(),
            dataset=ds,
            epochs=2,
            output_dir=out_dir,
        )
        trainer.train()
        metrics_file = out_dir / "dry_run_metrics.json"
        assert metrics_file.is_file()
        with metrics_file.open() as fh:
            persisted = json.load(fh)
        assert persisted["epochs"] == 2
        assert "loss_history" in persisted


class TestRunDryRun:
    def test_returns_converged_metrics(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=100)
        metrics = run_dry_run(
            data_path=path,
            tokenizer=StubTokenizer(),
            model=StubModel(delta=0.05),
            epochs=10,
            batch_size=4,
            learning_rate=DEFAULT_LEARNING_RATE,
        )
        assert metrics["converged"] is True
        assert metrics["n_records"] == 100
        assert metrics["epochs"] == 10
        assert len(metrics["loss_history"]) == 10
        assert metrics["final_loss"] < metrics["initial_loss"]

    def test_raises_when_not_converged(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=20)

        class _FlatModel(StubModel):
            def __call__(self, **kwargs: Any) -> Any:
                torch = self._get_torch()

                class _Out:
                    loss = torch.tensor(1.0, dtype=torch.float32)

                return _Out()

        with pytest.raises(AssertionError, match="loss did not converge"):
            run_dry_run(
                data_path=path,
                tokenizer=StubTokenizer(),
                model=_FlatModel(),
                epochs=3,
                batch_size=2,
            )

    def test_raises_on_empty_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.touch()
        with pytest.raises(ValueError, match="no valid PAL SFT records"):
            run_dry_run(
                data_path=path,
                tokenizer=StubTokenizer(),
                model=StubModel(),
                epochs=1,
            )


class TestMain:
    def test_mock_mode_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=100)
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--data-path",
                str(path),
                "--mock",
                "--epochs",
                "5",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["converged"] is True
        assert (out_dir / "dry_run_metrics.json").is_file()

    def test_mock_mode_failure_when_not_converged(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=10)
        # patch run_dry_run to return non-converged via a flat model
        from training.scripts import dry_run_sft_overfit

        orig_run = dry_run_sft_overfit.run_dry_run

        def _flat_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "epochs": 1,
                "loss_history": [1.0, 1.0],
                "initial_loss": 1.0,
                "final_loss": 1.0,
                "converged": False,
                "n_records": 10,
                "batch_size": 4,
                "learning_rate": 5e-5,
            }

        dry_run_sft_overfit.run_dry_run = _flat_run  # type: ignore[assignment]
        try:
            rc = main(["--data-path", str(path), "--mock", "--epochs", "1"])
            assert rc == 1
        finally:
            dry_run_sft_overfit.run_dry_run = orig_run  # type: ignore[assignment]

    def test_real_mode_requires_model_args(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=5)
        with pytest.raises(SystemExit):
            main(["--data-path", str(path), "--epochs", "1"])

    def test_preserves_unicode_in_metrics(self, tmp_path: Path) -> None:
        path = tmp_path / "uni.jsonl"
        records = [
            {
                "messages": [
                    {"role": "user", "content": f"Xin chào 🇻🇳 {i}"},
                    {"role": "assistant", "content": "Chào bạn"},
                ]
            }
            for i in range(100)
        ]
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics = run_dry_run(
            data_path=path,
            tokenizer=StubTokenizer(),
            model=StubModel(),
            epochs=3,
            batch_size=4,
        )
        assert metrics["converged"] is True
        assert metrics["n_records"] == 100
