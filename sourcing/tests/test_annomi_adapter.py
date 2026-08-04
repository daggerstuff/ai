"""Tests for the AnnoMI dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.annomi_adapter import AnnoMIAdapter

_CSV_FIELDS = [
    "utterance_id",
    "transcript_id",
    "interlocutor",
    "utterance",
    "mi_quality",
    "main_therapist_behaviour",
    "client_talk_type",
    "topic",
]


@pytest.fixture
def adapter(tmp_path):
    return AnnoMIAdapter("annomi", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    """CSV format (as read from file)."""
    return [
        {
            "utterance_id": "1_0",
            "transcript_id": "t1",
            "interlocutor": "therapist",
            "utterance": "What brings you here today?",
            "mi_quality": "high",
            "main_therapist_behaviour": "question",
            "client_talk_type": "",
            "topic": "alcohol",
        },
        {
            "utterance_id": "1_1",
            "transcript_id": "t1",
            "interlocutor": "client",
            "utterance": "I want to cut back on drinking.",
            "mi_quality": "high",
            "main_therapist_behaviour": "",
            "client_talk_type": "change",
            "topic": "alcohol",
        },
        {
            "utterance_id": "1_2",
            "transcript_id": "t1",
            "interlocutor": "therapist",
            "utterance": "Tell me more about that.",
            "mi_quality": "high",
            "main_therapist_behaviour": "reflection",
            "client_talk_type": "",
            "topic": "alcohol",
        },
        {
            "utterance_id": "1_3",
            "transcript_id": "t1",
            "interlocutor": "client",
            "utterance": "It's been affecting my family.",
            "mi_quality": "high",
            "main_therapist_behaviour": "",
            "client_talk_type": "change",
            "topic": "alcohol",
        },
    ]


@pytest.fixture
def sample_utterances():
    """Extract output format."""
    return [
        {
            "utterance_id": "1_0",
            "transcript_id": "t1",
            "interlocutor": "therapist",
            "utterance": "What brings you here today?",
            "mi_quality": "high",
            "main_therapist_behaviour": "question",
            "client_talk_type": "",
            "topic": "alcohol",
            "is_full": False,
        },
        {
            "utterance_id": "1_1",
            "transcript_id": "t1",
            "interlocutor": "client",
            "utterance": "I want to cut back on drinking.",
            "mi_quality": "high",
            "main_therapist_behaviour": "",
            "client_talk_type": "change",
            "topic": "alcohol",
            "is_full": False,
        },
        {
            "utterance_id": "1_2",
            "transcript_id": "t1",
            "interlocutor": "therapist",
            "utterance": "Tell me more about that.",
            "mi_quality": "high",
            "main_therapist_behaviour": "reflection",
            "client_talk_type": "",
            "topic": "alcohol",
            "is_full": False,
        },
        {
            "utterance_id": "1_3",
            "transcript_id": "t1",
            "interlocutor": "client",
            "utterance": "It's been affecting my family.",
            "mi_quality": "high",
            "main_therapist_behaviour": "",
            "client_talk_type": "change",
            "topic": "alcohol",
            "is_full": False,
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestAnnoMIAdapter:
    def test_download_skips_if_exists(self, adapter, monkeypatch):
        # Pre-create file to test skip
        (adapter._raw_dir / "AnnoMI-simple.csv").write_text("exists", encoding="utf-8")
        (adapter._raw_dir / "AnnoMI-full.csv").write_text("exists", encoding="utf-8")
        called = []
        monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: called.append(a))
        adapter.download()
        assert len(called) == 0

    def test_extract_returns_utterances(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "AnnoMI-simple.csv", sample_csv_rows_raw)
        utterances = adapter.extract()
        assert len(utterances) == 4
        assert utterances[0]["transcript_id"] == "t1"
        assert utterances[0]["is_full"] is False

    def test_convert_basic(self, adapter, sample_utterances):
        records = adapter.convert_to_chatml(sample_utterances)
        assert len(records) == 1
        rec = records[0]
        assert rec["source"] == "annomi"
        assert rec["task_type"] == "therapy_response_generation"
        assert rec["diagnostic_tag"] == "alcohol_use"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "assistant"
        assert rec["messages"][2]["role"] == "user"
        assert "question" in rec["therapist_behaviours"]
        assert "change" in rec["client_talk_types"]
        assert rec["mi_quality"] == "high"

    def test_convert_multiple_transcripts(self, adapter):
        raw = [
            {
                "utterance_id": "1_0",
                "transcript_id": "t1",
                "interlocutor": "therapist",
                "utterance": "Hello",
                "mi_quality": "high",
                "main_therapist_behaviour": "",
                "client_talk_type": "",
                "topic": "",
                "is_full": False,
            },
            {
                "utterance_id": "1_1",
                "transcript_id": "t1",
                "interlocutor": "client",
                "utterance": "Hi",
                "mi_quality": "high",
                "main_therapist_behaviour": "",
                "client_talk_type": "",
                "topic": "",
                "is_full": False,
            },
            {
                "utterance_id": "2_0",
                "transcript_id": "t2",
                "interlocutor": "therapist",
                "utterance": "Welcome",
                "mi_quality": "low",
                "main_therapist_behaviour": "",
                "client_talk_type": "",
                "topic": "",
                "is_full": False,
            },
            {
                "utterance_id": "2_1",
                "transcript_id": "t2",
                "interlocutor": "client",
                "utterance": "Thanks",
                "mi_quality": "low",
                "main_therapist_behaviour": "",
                "client_talk_type": "",
                "topic": "",
                "is_full": False,
            },
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 2
        assert records[0]["transcript_id"] == "t1"
        assert records[1]["transcript_id"] == "t2"

    def test_empty_utterance_skipped(self, adapter):
        raw = [
            {
                "utterance_id": "1_0",
                "transcript_id": "t1",
                "interlocutor": "therapist",
                "utterance": "",
                "mi_quality": "",
                "main_therapist_behaviour": "",
                "client_talk_type": "",
                "topic": "",
                "is_full": False,
            },
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_utterances):
        records = adapter.convert_to_chatml(sample_utterances)
        assert records[0]["provenance"]["access_method"] == "github"
        assert records[0]["provenance"]["source_url"] == "https://github.com/uccollab/AnnoMI"

    def test_full_run(self, adapter, sample_csv_rows_raw, monkeypatch):
        _write_csv(adapter._raw_dir / "AnnoMI-simple.csv", sample_csv_rows_raw)
        _write_csv(adapter._raw_dir / "AnnoMI-full.csv", sample_csv_rows_raw)
        monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: None)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "annomi"
