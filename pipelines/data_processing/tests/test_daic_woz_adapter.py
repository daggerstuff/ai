"""Tests for the DAIC-WOZ adapter (HuggingFace mirror)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.daic_woz_adapter import DAICWozAdapter


@pytest.fixture
def adapter(tmp_path):
    return DAICWozAdapter("daic_woz", tmp_path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_utterance(pid: int, speaker: str, text: str, **extra) -> dict[str, Any]:
    row = {
        "participant_id": pid,
        "speaker": speaker,
        "text": text,
        "start_time": 0.0,
        "stop_time": 1.0,
        "PHQ8_Binary": 1.0,
        "PHQ8_Score": 15.0,
        "PTSD_severity": 30.0,
        "PTSD_label": 1.0,
        "Gender": 0.0,
        "age": 41.0,
    }
    row.update(extra)
    return row


@pytest.fixture
def sample_utterances():
    """Utterance-level rows from HF parquet (audio column already removed)."""
    return [
        _make_utterance(303, "Ellie", "hi i'm ellie thanks for coming in today"),
        _make_utterance(303, "Participant", "hi nice to meet you"),
        _make_utterance(303, "Ellie", "how are you feeling today"),
        _make_utterance(303, "Participant", "i've been feeling down"),
        _make_utterance(304, "Ellie", "hello welcome"),
        _make_utterance(304, "Participant", "thank you"),
    ]


@pytest.fixture
def sample_session(sample_utterances):
    """Extract output format: session grouped by participant_id."""
    pid_303 = [u for u in sample_utterances if u["participant_id"] == 303]
    return {
        "session_id": "303",
        "utterances": pid_303,
        "labels": {
            "PHQ8_Binary": 1.0,
            "PHQ8_Score": 15.0,
            "PTSD_severity": 30.0,
            "PTSD_label": 1.0,
            "Gender": 0.0,
            "age": 41.0,
        },
    }


class TestDAICWozAdapter:
    def test_download_creates_readme_on_failure(self, adapter, monkeypatch):
        """When HF download fails, README is created."""
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no HF")))
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_download_skips_if_data_exists(self, adapter):
        """If JSONL files already exist, download is a no-op."""
        _write_jsonl(adapter._raw_dir / "train.jsonl", [_make_utterance(303, "Ellie", "hi")])
        adapter.download()
        # Should NOT create README since data already exists
        assert not (adapter._raw_dir / "README.txt").exists()

    def test_extract_groups_by_participant(self, adapter, sample_utterances):
        _write_jsonl(adapter._raw_dir / "train.jsonl", sample_utterances)
        sessions = adapter.extract()
        assert len(sessions) == 2  # participants 303 and 304
        s303 = [s for s in sessions if s["session_id"] == "303"][0]
        assert len(s303["utterances"]) == 4
        assert s303["labels"]["PHQ8_Score"] == 15.0

    def test_convert_basic(self, adapter, sample_session):
        records = adapter.convert_to_chatml([sample_session])
        assert len(records) == 1
        rec = records[0]
        assert rec["source"] == "daic_woz"
        assert rec["task_type"] == "severity_estimation"
        assert rec["diagnostic_tag"] == "depression"  # PHQ8_Binary=1
        assert rec["phq8_score"] == "15.0"
        assert rec["severity"] == "moderately_severe"  # score 15
        assert rec["messages"][0]["role"] == "system"
        assert "PHQ-8" in rec["messages"][0]["content"]
        assert rec["messages"][1]["role"] == "assistant"  # Ellie first
        assert rec["messages"][2]["role"] == "user"  # Participant

    def test_convert_skips_empty_session(self, adapter):
        session = {"session_id": "999", "utterances": [], "labels": {}}
        records = adapter.convert_to_chatml([session])
        assert len(records) == 0

    def test_convert_single_speaker_user_only(self, adapter):
        """Single speaker (user only) gets synthetic assistant message."""
        session = {
            "session_id": "305",
            "utterances": [
                {"speaker": "Participant", "text": "i feel sad"},
                {"speaker": "Participant", "text": "nothing matters"},
            ],
            "labels": {"PHQ8_Score": 20, "PHQ8_Binary": 1},
        }
        records = adapter.convert_to_chatml([session])
        assert len(records) == 1
        roles = [m["role"] for m in records[0]["messages"]]
        assert "user" in roles
        assert "assistant" in roles  # synthetic

    def test_convert_single_speaker_assistant_only(self, adapter):
        """Single speaker (assistant only) gets synthetic user message."""
        session = {
            "session_id": "306",
            "utterances": [
                {"speaker": "Ellie", "text": "how are you"},
                {"speaker": "Ellie", "text": "tell me more"},
            ],
            "labels": {"PHQ8_Score": 5, "PHQ8_Binary": 0},
        }
        records = adapter.convert_to_chatml([session])
        assert len(records) == 1
        roles = [m["role"] for m in records[0]["messages"]]
        assert "assistant" in roles
        assert "user" in roles  # synthetic

    def test_phq8_severity_mapping(self, adapter):
        session = {
            "session_id": "302",
            "utterances": [
                {"speaker": "ellie", "text": "Hi"},
                {"speaker": "participant", "text": "Hello"},
            ],
            "labels": {"PHQ8_Score": 22, "PHQ8_Binary": 1},
        }
        records = adapter.convert_to_chatml([session])
        assert records[0]["severity"] == "severe"

    def test_provenance_huggingface(self, adapter, sample_session):
        records = adapter.convert_to_chatml([sample_session])
        assert records[0]["provenance"]["access_method"] == "huggingface"
        assert "saeedzou/DAIC-WOZ" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_utterances, monkeypatch):
        _write_jsonl(adapter._raw_dir / "train.jsonl", sample_utterances)
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip")))
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # 2 participants
        record = json.loads(lines[0])
        assert record["source"] == "daic_woz"
