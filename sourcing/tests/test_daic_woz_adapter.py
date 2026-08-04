"""Tests for the DAIC-WOZ adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.daic_woz_adapter import DAICWozAdapter

_TRANSCRIPT_FIELDS = ["participant_id", "speaker", "transcript", "timestamp"]
_LABEL_FIELDS = ["participant_id", "phq8_score", "pcl_c_score"]


@pytest.fixture
def adapter(tmp_path):
    return DAICWozAdapter("daic_woz", tmp_path)


@pytest.fixture
def sample_transcript_rows():
    return [
        {"participant_id": "300", "speaker": "Ellie", "transcript": "How are you feeling today?", "timestamp": "0.0"},
        {
            "participant_id": "300",
            "speaker": "Participant",
            "transcript": "I've been feeling down.",
            "timestamp": "2.0",
        },
        {"participant_id": "300", "speaker": "Ellie", "transcript": "Can you tell me more?", "timestamp": "4.0"},
        {
            "participant_id": "300",
            "speaker": "Participant",
            "transcript": "Nothing seems to matter.",
            "timestamp": "6.0",
        },
    ]


@pytest.fixture
def sample_label_rows():
    return [
        {"participant_id": "300", "phq8_score": "15", "pcl_c_score": "35"},
    ]


@pytest.fixture
def sample_session(sample_transcript_rows, sample_label_rows):
    """Extract output format."""
    return {
        "session_id": "300",
        "utterances": [
            {**{k.lower(): v for k, v in r.items()}, "_source_file": "300_TRANSCRIPT"} for r in sample_transcript_rows
        ],
        "labels": {k.lower(): v for k, v in sample_label_rows[0].items()},
        "_source_file": "300_TRANSCRIPT",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestDAICWozAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_download_idempotent(self, adapter):
        adapter.download()
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_sessions(self, adapter, sample_transcript_rows, sample_label_rows):
        _write_csv(adapter._raw_dir / "300_TRANSCRIPT.csv", sample_transcript_rows, _TRANSCRIPT_FIELDS)
        _write_csv(adapter._raw_dir / "labels.csv", sample_label_rows, _LABEL_FIELDS)
        sessions = adapter.extract()
        assert len(sessions) >= 1
        assert sessions[0]["labels"]["phq8_score"] == "15"

    def test_convert_basic(self, adapter, sample_session):
        records = adapter.convert_to_chatml([sample_session])
        assert len(records) == 1
        rec = records[0]
        assert rec["source"] == "daic_woz"
        assert rec["task_type"] == "severity_estimation"
        assert rec["diagnostic_tag"] == "depression"
        assert rec["phq8_score"] == "15"
        assert rec["severity"] == "moderately_severe"
        assert rec["messages"][0]["role"] == "system"
        assert "PHQ-8" in rec["messages"][0]["content"]
        assert rec["messages"][1]["role"] == "assistant"
        assert rec["messages"][2]["role"] == "user"

    def test_convert_skips_no_user(self, adapter):
        session = {
            "session_id": "301",
            "utterances": [
                {"speaker": "ellie", "transcript": "Hello", "_source_file": "t"},
                {"speaker": "interviewer", "transcript": "How are you?", "_source_file": "t"},
            ],
            "labels": {},
        }
        records = adapter.convert_to_chatml([session])
        assert len(records) == 0

    def test_phq8_severity_mapping(self, adapter):
        session = {
            "session_id": "302",
            "utterances": [
                {"speaker": "ellie", "transcript": "Hi", "_source_file": "t"},
                {"speaker": "participant", "transcript": "Hello", "_source_file": "t"},
            ],
            "labels": {"phq8_score": "22", "pcl_c_score": ""},
        }
        records = adapter.convert_to_chatml([session])
        assert records[0]["severity"] == "severe"

    def test_provenance_present(self, adapter, sample_session):
        records = adapter.convert_to_chatml([sample_session])
        assert records[0]["provenance"]["access_method"] == "request"
        assert "dcapswoz" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_transcript_rows, sample_label_rows):
        _write_csv(adapter._raw_dir / "300_TRANSCRIPT.csv", sample_transcript_rows, _TRANSCRIPT_FIELDS)
        _write_csv(adapter._raw_dir / "labels.csv", sample_label_rows, _LABEL_FIELDS)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["source"] == "daic_woz"
