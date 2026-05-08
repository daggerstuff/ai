"""Tests for the DPO trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.dpo_trainer import (
    MIN_SAMPLES,
    CheckpointVerificationCallback,
    load_preference_dataset,
    save_metrics,
)


class TestLoadPreferenceDataset:

    def _make_jsonl(self, path: Path, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_loads_valid_pairs(self, tmp_path: Path):
        data_path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": f"Q{i}", "chosen": f"Good answer {i}", "rejected": f"Bad answer {i}"}
            for i in range(25)
        ]
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == 25

    def test_includes_all_records(self, tmp_path: Path):
        """Safety filter disabled — all records pass through including crisis content."""
        data_path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": "Tell me about therapy", "chosen": "I want to kill myself", "rejected": "Bad answer"},
        ]
        for i in range(MIN_SAMPLES):
            records.append({"prompt": f"Q{i}", "chosen": f"Safe answer {i}", "rejected": f"Bad answer {i}"})
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == len(records)

    def test_includes_all_rejected(self, tmp_path: Path):
        """Safety filter disabled — even unsafe rejected content passes through."""
        data_path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": "Tell me about therapy", "chosen": "Safe answer", "rejected": "I want to die tonight"},
        ]
        for i in range(MIN_SAMPLES):
            records.append({"prompt": f"Q{i}", "chosen": f"Safe answer {i}", "rejected": f"Bad answer {i}"})
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == len(records)

    def test_raises_on_insufficient_samples(self, tmp_path: Path):
        data_path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
        ]
        self._make_jsonl(data_path, records)

        with pytest.raises(ValueError, match="Only 2 samples"):
            load_preference_dataset(
                data_path, 1024, logging.getLogger("test"),
            )

    def test_missing_data_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_preference_dataset(
                tmp_path / "nonexistent.jsonl", 1024, logging.getLogger("test"),
            )

    def test_missing_fields_skipped(self, tmp_path: Path):
        data_path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": "Q", "chosen": "A"},
        ]
        for i in range(MIN_SAMPLES):
            records.append({"prompt": f"Q{i}", "chosen": f"A{i}", "rejected": f"B{i}"})
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == MIN_SAMPLES


class TestCheckpointVerificationCallback:

    def test_verifies_existing_files(self, tmp_path: Path):
        callback = CheckpointVerificationCallback()
        output_dir = tmp_path / "checkpoint"
        output_dir.mkdir()
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "adapter_model.safetensors").write_text("data", encoding="utf-8")

        result = callback.verify(output_dir)
        assert all(result.values())

    def test_verifies_missing_files(self, tmp_path: Path):
        callback = CheckpointVerificationCallback()
        output_dir = tmp_path / "empty_checkpoint"
        output_dir.mkdir()

        result = callback.verify(output_dir)
        assert not any(result.values())


class TestSaveMetrics:

    def test_saves_metrics_json(self, tmp_path: Path):
        metrics = {"train_loss": 0.5, "train_runtime": 100.0}
        save_metrics(tmp_path, metrics, beta=0.1)

        metrics_path = tmp_path / "dpo_metrics.json"
        assert metrics_path.exists()
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert report["beta"] == 0.1
        assert report["metrics"]["train_loss"] == 0.5
        assert "generated_at" in report


class TestBuildParser:

    def test_beta_default(self):
        from training.dpo_trainer import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "/tmp/test.jsonl",
            "--base_model_checkpoint", "model",
            "--output_dir", "/tmp/out",
        ])
        assert args.beta == 0.1

    def test_lora_args_registered(self):
        from training.dpo_trainer import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "/tmp/test.jsonl",
            "--base_model_checkpoint", "model",
            "--output_dir", "/tmp/out",
        ])
        assert args.lora_r == 8
        assert args.lora_alpha == 16


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(
        safe_prompt=st.text(min_size=1, max_size=50),
        safe_chosen=st.text(min_size=1, max_size=50),
        safe_rejected=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_hypothesis_pairs_preserved(safe_prompt: str, safe_chosen: str, safe_rejected: str):
        tmp = Path("/tmp") / "hypo_dpo_test"
        tmp.mkdir(exist_ok=True)
        data_path = tmp / "safe_pairs.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for i in range(MIN_SAMPLES):
                f.write(json.dumps({"prompt": safe_prompt, "chosen": safe_chosen, "rejected": safe_rejected}) + "\n")
        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == MIN_SAMPLES

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_pairs_preserved():
        raise AssertionError("Skipped when hypothesis is unavailable")
