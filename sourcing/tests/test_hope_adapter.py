"""Tests for the HOPE dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from ai.sourcing.dataset_adapters.hope_adapter import HOPEAdapter


@pytest.fixture
def sample_csv_rows():
    """Sample HOPE CSV file rows (as read from file)."""
    return [
        {"ID": "1_0", "Type": "T", "Utterance": "Hello, how are you today?", "Dialog_Act": "gt"},
        {"ID": "1_1", "Type": "P", "Utterance": "I've been feeling anxious", "Dialog_Act": "id"},
        {"ID": "1_2", "Type": "T", "Utterance": "Tell me more about that anxiety", "Dialog_Act": "irq"},
        {"ID": "1_3", "Type": "P", "Utterance": "It started last week at work", "Dialog_Act": "id"},
    ]


@pytest.fixture
def sample_csv_content(sample_csv_rows):
    """Sample HOPE utterances in extract() output format (lowercase keys)."""
    return [
        {"id": r["ID"], "type": r["Type"], "utterance": r["Utterance"], "dialog_act": r["Dialog_Act"]}
        for r in sample_csv_rows
    ]


@pytest.fixture
def adapter(tmp_path):
    return HOPEAdapter("hope", tmp_path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Type", "Utterance", "Dialog_Act"])
        writer.writeheader()
        writer.writerows(rows)


class TestHOPEAdapter:
    def test_download_skips_if_exists(self, adapter):
        repo_dir = adapter._raw_dir / "SPARTA_WSDM2022"
        repo_dir.mkdir(parents=True)
        with patch("subprocess.run") as mock_run:
            adapter.download()
            mock_run.assert_not_called()

    def test_extract_returns_sessions(self, adapter, sample_csv_rows):
        repo_dir = adapter._raw_dir / "SPARTA_WSDM2022" / "HOPE_data" / "HOPE_therapy_session_transcripts"
        repo_dir.mkdir(parents=True)
        _write_csv(repo_dir / "42.csv", sample_csv_rows)

        sessions = adapter.extract()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "42"
        assert len(sessions[0]["utterances"]) == 4

    def test_convert_to_chatml_basic(self, adapter, sample_csv_content):
        raw = [{"session_id": "42", "utterances": sample_csv_content}]
        records = adapter.convert_to_chatml(raw)

        assert len(records) == 1
        record = records[0]
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "assistant"
        assert record["messages"][1]["content"] == "Hello, how are you today?"
        assert record["messages"][2]["role"] == "user"
        assert record["source"] == "hope"
        assert record["task_type"] == "therapy_response_generation"
        assert "gt" in record["dac_labels"]
        assert "id" in record["dac_labels"]

    def test_short_session_skipped(self, adapter):
        raw = [{"session_id": "1", "utterances": [{"type": "T", "utterance": "Hello", "dialog_act": "gt"}]}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_no_user_turns_skipped(self, adapter):
        raw = [
            {
                "session_id": "1",
                "utterances": [
                    {"type": "T", "utterance": "Hello", "dialog_act": "gt"},
                    {"type": "T", "utterance": "How are you?", "dialog_act": "gt"},
                ],
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_csv_content):
        raw = [{"session_id": "42", "utterances": sample_csv_content}]
        records = adapter.convert_to_chatml(raw)
        assert "provenance" in records[0]
        assert records[0]["provenance"]["source_url"].startswith("https://github.com")

    def test_full_run(self, adapter, sample_csv_rows, tmp_path):
        repo_dir = adapter._raw_dir / "SPARTA_WSDM2022" / "HOPE_data" / "HOPE_therapy_session_transcripts"
        repo_dir.mkdir(parents=True)
        _write_csv(repo_dir / "42.csv", sample_csv_rows)

        with patch("subprocess.run"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "hope"
