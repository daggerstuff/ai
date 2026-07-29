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

    def test_coerces_conversational_message_list_schema(self, tmp_path: Path):
        """PAL ``generate_dpo_pairs.py`` emits chosen/rejected as message
        lists (TRL conversational DPO format). The loader must coerce these
        to the standard string schema ``run_dpo`` expects.
        """
        data_path = tmp_path / "pal_dpo.jsonl"
        records = [
            {
                "prompt": f"Given this persona: P{i}\n\nGenerate the next response.",
                "chosen": [{"role": "assistant", "content": f"in-character {i}"}],
                "rejected": [{"role": "assistant", "content": f"jargon {i}"}],
            }
            for i in range(MIN_SAMPLES)
        ]
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == MIN_SAMPLES
        for i, sample in enumerate(result):
            assert sample["prompt"] == records[i]["prompt"]
            assert sample["chosen"] == f"in-character {i}"
            assert sample["rejected"] == f"jargon {i}"

    def test_mixed_schema_file_loads_both(self, tmp_path: Path):
        """A dataset mixing standard-string and conversational records must
        load both without error, coercing only the message-list ones.
        """
        data_path = tmp_path / "mixed.jsonl"
        records = [{"prompt": "Q0", "chosen": "A0", "rejected": "B0"}]
        for i in range(1, MIN_SAMPLES):
            records.append(
                {
                    "prompt": f"Q{i}",
                    "chosen": [{"role": "assistant", "content": f"A{i}"}],
                    "rejected": [{"role": "assistant", "content": f"B{i}"}],
                }
            )
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == MIN_SAMPLES
        assert result[0]["chosen"] == "A0"
        assert result[1]["chosen"] == "A1"

    def test_conversational_record_missing_assistant_turn_skipped(self, tmp_path: Path):
        """A conversational record with no assistant turn yields no response
        text and must be skipped, not crash.
        """
        data_path = tmp_path / "bad.jsonl"
        records = [
            {
                "prompt": "Q0",
                "chosen": [{"role": "user", "content": "no assistant turn"}],
                "rejected": [{"role": "user", "content": "no assistant turn"}],
            },
        ]
        for i in range(1, MIN_SAMPLES + 1):
            records.append({"prompt": f"Q{i}", "chosen": f"A{i}", "rejected": f"B{i}"})
        self._make_jsonl(data_path, records)

        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        # The one bad conversational record is skipped; the rest pass.
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
            for _i in range(MIN_SAMPLES):
                f.write(json.dumps({"prompt": safe_prompt, "chosen": safe_chosen, "rejected": safe_rejected}) + "\n")
        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test"),
        )
        assert len(result) == MIN_SAMPLES

if st is not None:

    @given(
        prompt=st.text(min_size=1, max_size=50),
        chosen=st.text(min_size=1, max_size=50),
        rejected=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_hypothesis_dpo_safety_filter_prop1(prompt: str, chosen: str, rejected: str):
        """Prop 1 — No-filter policy: mixed safe/unsafe pairs are preserved.

        The DPO trainer is configured with safety filtering disabled
        (all content allowed). This property verifies that the loader
        does NOT drop pairs based on content: every generated pair
        that has prompt/chosen/rejected fields is returned unchanged.
        """
        tmp = Path("/tmp") / "hypo_dpo_prop1"
        tmp.mkdir(exist_ok=True)
        data_path = tmp / "mixed_pairs.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for _i in range(MIN_SAMPLES):
                f.write(
                    json.dumps(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                    )
                    + "\n"
                )
        result = load_preference_dataset(
            data_path, 1024, logging.getLogger("test_prop1")
        )
        # Under no-filter policy, every well-formed pair is kept
        assert len(result) == MIN_SAMPLES
        for sample in result:
            assert sample["prompt"] == prompt
            assert sample["chosen"] == chosen
            assert sample["rejected"] == rejected
