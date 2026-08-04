"""Tests for the MEMO dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.memo_adapter import MEMOAdapter

_CSV_FIELDS = ["ID", "Type", "Utterance", "Dialog_Act", "Component"]


@pytest.fixture
def sample_csv_rows_raw():
    """CSV format with capitalized keys (as read from file)."""
    return [
        {
            "ID": "0_0",
            "Type": "T",
            "Utterance": "What brings you in today?",
            "Dialog_Act": "irq",
            "Component": "symptom_and_history",
        },
        {
            "ID": "0_1",
            "Type": "P",
            "Utterance": "I've been feeling down",
            "Dialog_Act": "id",
            "Component": "symptom_and_history",
        },
        {
            "ID": "0_2",
            "Type": "T",
            "Utterance": "Can you tell me more?",
            "Dialog_Act": "irq",
            "Component": "patient_discovery",
        },
        {
            "ID": "0_3",
            "Type": "P",
            "Utterance": "It started after I lost my job",
            "Dialog_Act": "id",
            "Component": "patient_discovery",
        },
    ]


@pytest.fixture
def sample_csv_rows():
    """Extract output format with lowercase keys (as produced by extract())."""
    return [
        {
            "id": "0_0",
            "type": "T",
            "utterance": "What brings you in today?",
            "dialog_act": "irq",
            "component": "symptom_and_history",
        },
        {
            "id": "0_1",
            "type": "P",
            "utterance": "I've been feeling down",
            "dialog_act": "id",
            "component": "symptom_and_history",
        },
        {
            "id": "0_2",
            "type": "T",
            "utterance": "Can you tell me more?",
            "dialog_act": "irq",
            "component": "patient_discovery",
        },
        {
            "id": "0_3",
            "type": "P",
            "utterance": "It started after I lost my job",
            "dialog_act": "id",
            "component": "patient_discovery",
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    return MEMOAdapter("memo", tmp_path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestMEMOAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "forms.gle" in readme.read_text(encoding="utf-8")

    def test_download_idempotent(self, adapter):
        adapter.download()
        first = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        adapter.download()
        second = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        assert first == second

    def test_extract_returns_sessions(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "session_1.csv", sample_csv_rows_raw)
        sessions = adapter.extract()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session_1"
        assert len(sessions[0]["utterances"]) == 4

    def test_convert_to_chatml_basic(self, adapter, sample_csv_rows):
        raw = [{"session_id": "s1", "utterances": sample_csv_rows}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        record = records[0]
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "assistant"
        assert record["messages"][2]["role"] == "user"
        assert record["source"] == "memo"
        assert "symptom_and_history" in record["psychotherapy_elements"]
        assert "patient_discovery" in record["psychotherapy_elements"]

    def test_short_session_skipped(self, adapter):
        raw = [{"session_id": "s1", "utterances": [{"type": "T", "utterance": "Hello", "component": ""}]}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_access_method_request(self, adapter, sample_csv_rows):
        raw = [{"session_id": "s1", "utterances": sample_csv_rows}]
        records = adapter.convert_to_chatml(raw)
        assert records[0]["provenance"]["access_method"] == "request"

    def test_full_run(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "session_1.csv", sample_csv_rows_raw)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "memo"
