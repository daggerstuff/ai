"""Tests for the Kaggle therapy dataset adapter."""

from __future__ import annotations

import csv as csv_mod
import json
from pathlib import Path

import pytest

from ai.sourcing.dataset_adapters.kaggle_therapy_adapter import (
    KaggleTherapyAdapter,
    _parse_python_list,
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Create a minimal Kaggle raw directory with sample files."""
    # combined_dataset.json
    combined = tmp_path / "combined_dataset.json"
    combined.write_text(
        json.dumps({"Context": "I feel sad.", "Response": "I hear you."}) + "\n"
        + json.dumps({"Context": "", "Response": "skip me"}) + "\n",
        encoding="utf-8",
    )

    # train.csv (synthetic therapy conversations with Python repr format)
    train = tmp_path / "train.csv"
    with open(train, "w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["conversations"])
        writer.writerow(["[{'from': 'human', 'value': 'I am stressed.'}\n {'from': 'gpt', 'value': 'Tell me more.'}]"])
        writer.writerow(["[{'from': 'human', 'value': 'Hello'}]"])

    # archive CSVs
    archive = tmp_path / "archive"
    archive.mkdir()
    counsel = archive / "counsel_chat2.csv"
    counsel.write_text(
        "questionText,answerText\n"
        '"How do I cope?","Try breathing exercises."\n'
        '"","Skip this one"\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def adapter(raw_dir: Path, tmp_path: Path) -> KaggleTherapyAdapter:
    return KaggleTherapyAdapter(
        dataset_name="kaggle_therapy",
        output_dir=tmp_path / "output",
        raw_dir=raw_dir,
    )


class TestParsePythonList:
    def test_valid_list_with_newlines(self) -> None:
        s = "[{'from': 'human', 'value': 'hello'}\n {'from': 'gpt', 'value': 'hi'}]"
        result = _parse_python_list(s)
        assert result is not None
        assert len(result) == 2
        assert result[0]["from"] == "human"
        assert result[1]["from"] == "gpt"

    def test_valid_list_with_commas(self) -> None:
        s = "[{'from': 'human', 'value': 'hello'}, {'from': 'gpt', 'value': 'hi'}]"
        result = _parse_python_list(s)
        assert result is not None
        assert len(result) == 2

    def test_invalid_string(self) -> None:
        assert _parse_python_list("not a list") is None

    def test_empty_string(self) -> None:
        assert _parse_python_list("") is None


class TestKaggleTherapyAdapter:
    def test_extract_returns_all_sources(self, adapter: KaggleTherapyAdapter) -> None:
        records = adapter.extract()
        # 1 from combined_dataset.json (1 valid, 1 skipped)
        # 1 from train.csv (1 valid 2-turn, 1 skipped — too short)
        # 1 from counsel_chat2.csv (1 valid, 1 skipped)
        assert len(records) == 3

    def test_extract_synthetic_train_messages(self, adapter: KaggleTherapyAdapter) -> None:
        records = adapter.extract()
        train_records = [r for r in records if r["_source_file"] == "train.csv"]
        assert len(train_records) == 1
        # system + user + assistant = 3 messages
        assert len(train_records[0]["messages"]) == 3

    def test_convert_to_chatml_produces_valid_schema(self, adapter: KaggleTherapyAdapter) -> None:
        raw = adapter.extract()
        chatml = adapter.convert_to_chatml(raw)
        assert len(chatml) == 3
        for record in chatml:
            assert "messages" in record
            assert record["source"] == "kaggle_therapy"
            assert record["task_type"] == "therapy_response_generation"
            assert record["clinical_reviewed"] is False
            assert "provenance" in record
            assert len(record["messages"]) >= 3
            assert record["messages"][0]["role"] == "system"
            assert record["messages"][1]["role"] == "user"
            assert record["messages"][2]["role"] == "assistant"

    def test_run_produces_jsonl_file(self, adapter: KaggleTherapyAdapter) -> None:
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record["source"] == "kaggle_therapy"

    def test_download_is_noop(self, adapter: KaggleTherapyAdapter) -> None:
        # Download should not raise even though there's nothing to download
        adapter.download()

    def test_missing_files_returns_empty(self, tmp_path: Path) -> None:
        adapter = KaggleTherapyAdapter(
            dataset_name="kaggle_therapy",
            output_dir=tmp_path / "out",
            raw_dir=tmp_path / "nonexistent",
        )
        assert adapter.extract() == []

    def test_multi_turn_conversation(self, tmp_path: Path) -> None:
        """Test that multi-turn conversations are properly extracted."""
        raw = tmp_path
        train = raw / "train.csv"
        with open(train, "w", encoding="utf-8", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["conversations"])
            writer.writerow(
                ["[{'from': 'human', 'value': 'I am sad.'}\n"
                " {'from': 'gpt', 'value': 'Tell me more.'}\n"
                " {'from': 'human', 'value': 'I lost my job.'}\n"
                " {'from': 'gpt', 'value': 'That sounds really hard.'}]"]
            )
        adapter = KaggleTherapyAdapter(
            dataset_name="kaggle_therapy",
            output_dir=tmp_path / "out",
            raw_dir=raw,
        )
        records = adapter.extract()
        train_records = [r for r in records if r["_source_file"] == "train.csv"]
        assert len(train_records) == 1
        assert len(train_records[0]["messages"]) == 5  # system + 2 turns
