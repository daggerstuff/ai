"""Tests for the MI-TAGS dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.mitags_adapter import MITAGSAdapter

_UTTERANCE_FIELDS = ["id", "Video Title", "Turn", "Speaker", "Text", "Code", "Annotator", "Normalized Turn"]
_GLOBAL_FIELDS = [
    "id",
    "Video Title",
    "Annotator",
    "Empathy",
    "SofteningSustainTalk",
    "CultivatingChangeTalk",
    "Partnership",
    "Only Text",
    "Tagged Text",
    "Only Tags",
]


@pytest.fixture
def sample_utterances_raw():
    """CSV format with capitalized keys (as read from file)."""
    return [
        {
            "id": "1",
            "Video Title": "MI Role-Play - Opioid Use",
            "Turn": "1",
            "Speaker": "T",
            "Text": "Tell me about your opioid use",
            "Code": "Question Open",
            "Annotator": "A1",
            "Normalized Turn": "0.01",
        },
        {
            "id": "2",
            "Video Title": "MI Role-Play - Opioid Use",
            "Turn": "2",
            "Speaker": "C",
            "Text": "I take about 3 pills a day",
            "Code": "",
            "Annotator": "A1",
            "Normalized Turn": "0.02",
        },
        {
            "id": "3",
            "Video Title": "MI Role-Play - Opioid Use",
            "Turn": "3",
            "Speaker": "T",
            "Text": "That's quite a lot. How does it make you feel?",
            "Code": "Complex Reflection",
            "Annotator": "A1",
            "Normalized Turn": "0.03",
        },
        {
            "id": "4",
            "Video Title": "MI Role-Play - Opioid Use",
            "Turn": "4",
            "Speaker": "C",
            "Text": "It helps with the pain but I worry about dependency",
            "Code": "",
            "Annotator": "A1",
            "Normalized Turn": "0.04",
        },
    ]


@pytest.fixture
def sample_utterances():
    """Extract output format with lowercase keys (as produced by extract())."""
    return [
        {
            "turn": "1",
            "speaker": "T",
            "text": "Tell me about your opioid use",
            "code": "Question Open",
            "normalized_turn": "0.01",
        },
        {"turn": "2", "speaker": "C", "text": "I take about 3 pills a day", "code": "", "normalized_turn": "0.02"},
        {
            "turn": "3",
            "speaker": "T",
            "text": "That's quite a lot. How does it make you feel?",
            "code": "Complex Reflection",
            "normalized_turn": "0.03",
        },
        {
            "turn": "4",
            "speaker": "C",
            "text": "It helps with the pain but I worry about dependency",
            "code": "",
            "normalized_turn": "0.04",
        },
    ]


@pytest.fixture
def sample_globals():
    return [
        {
            "id": "1",
            "Video Title": "MI Role-Play - Opioid Use",
            "Annotator": "A1",
            "Empathy": "4",
            "SofteningSustainTalk": "3",
            "CultivatingChangeTalk": "4",
            "Partnership": "3",
            "Only Text": "",
            "Tagged Text": "",
            "Only Tags": "",
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    return MITAGSAdapter("mitags", tmp_path)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestMITAGSAdapter:
    def test_download_skips_if_exists(self, adapter):
        repo_dir = adapter._raw_dir / "MI-TAGS"
        repo_dir.mkdir(parents=True)
        with patch("subprocess.run") as mock_run:
            adapter.download()
            mock_run.assert_not_called()

    def test_extract_returns_sessions(self, adapter, sample_utterances_raw, sample_globals):
        repo_dir = adapter._raw_dir / "MI-TAGS"
        repo_dir.mkdir(parents=True)
        _write_csv(repo_dir / "sample_utterances.csv", _UTTERANCE_FIELDS, sample_utterances_raw)
        _write_csv(repo_dir / "sample_global_mitis.csv", _GLOBAL_FIELDS, sample_globals)

        sessions = adapter.extract()
        assert len(sessions) == 1
        assert len(sessions[0]["utterances"]) == 4
        assert sessions[0]["global_scores"]["empathy"] == 4

    def test_convert_to_chatml_basic(self, adapter, sample_utterances):
        raw = [
            {
                "video_title": "MI Role-Play - Opioid Use",
                "utterances": sample_utterances,
                "global_scores": {"empathy": 4},
            }
        ]
        records = adapter.convert_to_chatml(raw)

        assert len(records) == 1
        record = records[0]
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "assistant"
        assert record["messages"][2]["role"] == "user"
        assert record["source"] == "mitags"
        assert record["task_type"] == "therapy_response_generation"
        assert "Question Open" in record["miti_codes"]
        assert record["global_scores"]["empathy"] == 4

    def test_short_session_single_utterance(self, adapter):
        raw = [
            {
                "video_title": "test",
                "utterances": [{"speaker": "T", "text": "Hello", "code": "Affirm"}],
                "global_scores": {},
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        assert len(records[0]["messages"]) == 3

    def test_empty_session_skipped(self, adapter):
        raw = [
            {
                "video_title": "test",
                "utterances": [],
                "global_scores": {},
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_same_speaker_session(self, adapter):
        raw = [
            {
                "video_title": "test",
                "utterances": [
                    {"speaker": "T", "text": "Hello", "code": ""},
                    {"speaker": "T", "text": "More text", "code": ""},
                ],
                "global_scores": {},
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        assert len(records[0]["messages"]) == 4

    def test_provenance_present(self, adapter, sample_utterances):
        raw = [{"video_title": "test", "utterances": sample_utterances, "global_scores": {}}]
        records = adapter.convert_to_chatml(raw)
        assert "provenance" in records[0]
        assert records[0]["provenance"]["source_url"].startswith("https://github.com")

    def test_full_run(self, adapter, sample_utterances_raw, sample_globals):
        repo_dir = adapter._raw_dir / "MI-TAGS"
        repo_dir.mkdir(parents=True)
        _write_csv(repo_dir / "sample_utterances.csv", _UTTERANCE_FIELDS, sample_utterances_raw)
        _write_csv(repo_dir / "sample_global_mitis.csv", _GLOBAL_FIELDS, sample_globals)

        with patch("subprocess.run"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "mitags"
