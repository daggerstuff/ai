"""Tests for the DMTCorpus adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.dmtcorpus_adapter import DMTCorpusAdapter


@pytest.fixture
def adapter(tmp_path):
    return DMTCorpusAdapter("dmtcorpus", tmp_path)


@pytest.fixture
def sample_session_data():
    """Sample JSON data as written to file."""
    return [
        {
            "condition_id": "c-001",
            "condition": "major depressive disorder",
            "session_number": 1,
            "total_sessions": 6,
            "homework_items": [{"description": "Mood diary"}],
            "cross_session_state": {"prior_homework": "completed"},
            "dialog": [
                {"speaker": "Therapist", "utterance": "What brings you in today?"},
                {"speaker": "Patient", "utterance": "I've been feeling hopeless."},
                {"speaker": "Therapist", "utterance": "Let's explore that feeling."},
                {"speaker": "Patient", "utterance": "It started three months ago."},
            ],
        },
        {
            "condition_id": "c-001",
            "condition": "major depressive disorder",
            "session_number": 2,
            "total_sessions": 6,
            "homework_items": [],
            "cross_session_state": {},
            "dialog": [
                {"speaker": "Therapist", "utterance": "How was the mood diary?"},
                {"speaker": "Patient", "utterance": "I filled it out every day."},
            ],
        },
    ]


@pytest.fixture
def sample_records(sample_session_data):
    """Extract output format (with _source_file added)."""
    return [{**s, "_source_file": "sessions.json"} for s in sample_session_data]


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestDMTCorpusAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_download_idempotent(self, adapter):
        adapter.download()
        adapter.download()
        # README should still exist
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_session_data):
        _write_json(adapter._raw_dir / "sessions.json", sample_session_data)
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["condition_id"] == "c-001"
        assert records[0]["_source_file"] == "sessions.json"

    def test_convert_basic(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert len(records) == 2
        rec = records[0]
        assert rec["source"] == "dmtcorpus"
        assert rec["task_type"] == "therapy_response_generation"
        assert rec["diagnostic_tag"] == "major depressive disorder"
        assert rec["is_synthetic"] is True
        assert rec["session_number"] == 1
        assert rec["total_sessions"] == 6
        assert rec["messages"][0]["role"] == "system"
        assert "Session 1 of 6" in rec["messages"][0]["content"]
        # Therapist -> assistant, Patient -> user
        assert rec["messages"][1]["role"] == "assistant"
        assert rec["messages"][2]["role"] == "user"

    def test_convert_empty_dialog_skipped(self, adapter):
        raw = [{"condition_id": "c-2", "dialog": []}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_convert_no_user_skipped(self, adapter):
        raw = [
            {
                "condition_id": "c-3",
                "dialog": [
                    {"speaker": "Therapist", "utterance": "Hello"},
                    {"speaker": "counselor", "utterance": "Welcome"},
                ],
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["provenance"]["access_method"] == "request"
        assert "arxiv.org" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_session_data):
        _write_json(adapter._raw_dir / "sessions.json", sample_session_data)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "dmtcorpus"
