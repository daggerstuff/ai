"""Tests for the GitHubRepoAdapter.

Tests cover:
  - CSV conversation pattern (user/assistant columns)
  - CSV classification pattern (text/label columns)
  - JSON conversation pattern (dialog/turns)
  - JSON ShareGPT pattern (conversations key)
  - JSON pre-formatted messages pattern
  - JSON text+label classification pattern
  - JSONL line-by-line parsing
  - Provenance metadata
  - Full run() with mocked download
  - Helper functions (_classify_csv, _find_column, _should_skip_path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.pipelines.data_processing.dataset_adapters.github_repo_adapter import (
    GitHubRepoAdapter,
    _classify_csv,
    _find_column,
    _is_data_file,
    _should_skip_path,
)

# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_csv_conversation(tmp_path: Path) -> Path:
    """CSV with user/assistant columns (conversation pattern)."""
    path = tmp_path / "conv.csv"
    path.write_text(
        "user,assistant\n"
        '"I feel hopeless","I hear you. Can you tell me more about what\'s going on?"\n'
        '"Nobody cares about me","That sounds really painful. You\'re not alone in this."\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_csv_classification(tmp_path: Path) -> Path:
    """CSV with text/label columns (classification pattern)."""
    path = tmp_path / "labels.csv"
    path.write_text(
        "text,label\n"
        '"I want to end it all",suicidal_ideation\n'
        '"I feel okay today",depression_negative\n'
        '"I am anxious about everything",anxiety\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_json_conversation() -> list[dict[str, Any]]:
    """JSON conversation data with dialog turns."""
    return [
        {
            "situation": "Student overwhelmed with exams",
            "emotion_type": "anxiety",
            "dialog": [
                {"speaker": "seeker", "utterance": "I can't handle the pressure"},
                {"speaker": "supporter", "utterance": "It sounds like you're under a lot of stress. What's been happening?"},
                {"speaker": "seeker", "utterance": "I have three exams next week"},
                {"speaker": "supporter", "utterance": "That's a lot. Let's talk about how to break this down."},
            ],
        }
    ]


@pytest.fixture
def sample_sharegpt() -> list[dict[str, Any]]:
    """ShareGPT format conversation."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "I feel so depressed"},
                {"from": "gpt", "value": "I'm sorry you're feeling this way. How long has this been going on?"},
                {"from": "human", "value": "A few months now"},
                {"from": "gpt", "value": "It takes courage to share that. Have you talked to anyone about this?"},
            ]
        }
    ]


@pytest.fixture
def sample_preformatted_chatml() -> list[dict[str, Any]]:
    """Already-ChatML-formatted messages."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a crisis counselor."},
                {"role": "user", "content": "I'm having a panic attack"},
                {"role": "assistant", "content": "I'm here with you. Let's focus on your breathing."},
            ],
            "task_type": "therapy_response_generation",
        }
    ]


@pytest.fixture
def sample_json_classification() -> list[dict[str, Any]]:
    """Simple text+label JSON."""
    return [
        {"text": "I can't sleep and I feel worthless", "label": "depression"},
        {"text": "Everything is going to be fine", "label": "no_concern"},
    ]


@pytest.fixture
def adapter(tmp_path: Path) -> GitHubRepoAdapter:
    """Create a GitHubRepoAdapter with mocked repo."""
    return GitHubRepoAdapter(
        "test_repo",
        tmp_path,
        repo_full_name="test-owner/test-repo",
        branch="main",
    )


# ─── Helper function tests ─────────────────────────────────────────────


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_find_column_exact_match(self):
        assert _find_column(["text", "label", "id"], ["text", "body"]) == "text"

    def test_find_column_case_insensitive(self):
        assert _find_column(["Text", "Label"], ["text"]) == "Text"

    def test_find_column_partial_match(self):
        assert _find_column(["user_message", "timestamp"], ["message"]) == "user_message"

    def test_find_column_no_match(self):
        assert _find_column(["foo", "bar"], ["text", "content"]) is None

    def test_should_skip_path(self):
        assert _should_skip_path("node_modules/lib/data.csv")
        assert _should_skip_path(".github/workflows/ci.yml")
        assert not _should_skip_path("data/train.csv")
        assert not _should_skip_path("src/utils/helpers.py")

    def test_is_data_file(self):
        assert _is_data_file("data.csv")
        assert _is_data_file("output.json")
        assert _is_data_file("records.jsonl")
        assert _is_data_file("data.tsv")
        assert not _is_data_file("script.py")
        assert not _is_data_file("README.md")

    def test_classify_csv_conversation(self, sample_csv_conversation: Path):
        rows, classification = _classify_csv(sample_csv_conversation)
        assert classification == "conversation"
        assert len(rows) == 2

    def test_classify_csv_text_label(self, sample_csv_classification: Path):
        rows, classification = _classify_csv(sample_csv_classification)
        assert classification == "text_label"
        assert len(rows) == 3


# ─── CSV conversion tests ──────────────────────────────────────────────


class TestCsvConversion:
    """Tests for CSV -> ChatML conversion."""

    def test_csv_conversation_pattern(self, adapter, sample_csv_conversation: Path):
        """CSV with user/assistant columns produces therapy_response_generation records."""
        rows, _ = _classify_csv(sample_csv_conversation)
        records = adapter.convert_to_chatml([
            {"type": "csv", "rows": rows, "file_info": {"path": "data/conv.csv"}},
        ])

        assert len(records) == 2
        for r in records:
            assert r["messages"][0]["role"] == "user"
            assert r["messages"][1]["role"] == "assistant"
            assert r["task_type"] == "therapy_response_generation"
            assert "provenance" in r

    def test_csv_classification_pattern(self, adapter, sample_csv_classification: Path):
        """CSV with text/label columns produces symptom_classification records."""
        rows, _ = _classify_csv(sample_csv_classification)
        records = adapter.convert_to_chatml([
            {"type": "csv", "rows": rows, "file_info": {"path": "data/labels.csv"}},
        ])

        assert len(records) == 3
        assert records[0]["task_type"] == "symptom_classification"
        assert records[0]["diagnostic_tag"] == "suicidal_ideation"
        assert records[1]["diagnostic_tag"] == "depression_negative"
        assert records[2]["diagnostic_tag"] == "anxiety"

    def test_csv_provenance(self, adapter, sample_csv_conversation: Path):
        """Records include provenance with GitHub source URL."""
        rows, _ = _classify_csv(sample_csv_conversation)
        records = adapter.convert_to_chatml([
            {"type": "csv", "rows": rows, "file_info": {"path": "data/conv.csv"}},
        ])
        assert records[0]["provenance"]["access_method"] == "github"
        assert "test-owner/test-repo" in records[0]["provenance"]["source_url"]
        assert records[0]["provenance"]["original_format"] == "csv"


# ─── JSON conversion tests ─────────────────────────────────────────────


class TestJsonConversion:
    """Tests for JSON -> ChatML conversion."""

    def test_json_conversation_pattern(self, adapter, sample_json_conversation):
        """JSON with dialog turns produces therapy_response_generation records."""
        records = adapter.convert_to_chatml([
            {"type": "json", "data": sample_json_conversation, "file_info": {"path": "data/conv.json"}},
        ])
        assert len(records) == 1
        r = records[0]
        assert r["messages"][0]["role"] == "system"
        assert "anxiety" in r["diagnostic_tag"]
        assert r["messages"][1]["role"] == "user"
        assert r["messages"][2]["role"] == "assistant"

    def test_sharegpt_pattern(self, adapter, sample_sharegpt):
        """ShareGPT format conversations are converted correctly."""
        records = adapter.convert_to_chatml([
            {"type": "json", "data": sample_sharegpt, "file_info": {"path": "data/sharegpt.json"}},
        ])
        assert len(records) == 1
        r = records[0]
        assert r["messages"][0]["role"] == "user"
        assert r["messages"][1]["role"] == "assistant"
        assert r["messages"][2]["role"] == "user"
        assert r["messages"][3]["role"] == "assistant"

    def test_preformatted_chatml(self, adapter, sample_preformatted_chatml):
        """Already-ChatML records pass through with provenance added."""
        records = adapter.convert_to_chatml([
            {"type": "json", "data": sample_preformatted_chatml, "file_info": {"path": "data/chatml.json"}},
        ])
        assert len(records) == 1
        r = records[0]
        assert r["messages"][0]["role"] == "system"
        assert r["messages"][1]["role"] == "user"
        assert r["messages"][2]["role"] == "assistant"
        assert r["task_type"] == "therapy_response_generation"

    def test_json_classification(self, adapter, sample_json_classification):
        """Text+label JSON items become symptom_classification records."""
        records = adapter.convert_to_chatml([
            {"type": "json", "data": sample_json_classification, "file_info": {"path": "data/labels.json"}},
        ])
        assert len(records) == 2
        assert records[0]["task_type"] == "symptom_classification"
        assert records[0]["diagnostic_tag"] == "depression"
        assert records[1]["diagnostic_tag"] == "no_concern"


# ─── Integration / run() tests ─────────────────────────────────────────


class TestRunIntegration:
    """Tests for full run() pipeline with mocked downloads."""

    def test_full_run_with_csv(self, adapter, sample_csv_conversation):
        """Full run() with pre-placed CSV produces JSONL output."""
        # Pre-populate raw dir
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        target = adapter._raw_dir / "data_conv.csv"
        target.write_text(sample_csv_conversation.read_text(), encoding="utf-8")
        adapter._file_list = [{"path": "data/conv.csv", "local_file": "data_conv.csv", "size": 100, "sha": "abc"}]

        with patch.object(adapter, "download"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "test-owner/test-repo"
        assert "provenance" in record

    def test_full_run_with_json(self, adapter, sample_json_conversation):
        """Full run() with pre-placed JSON produces JSONL output."""
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        target = adapter._raw_dir / "data_conv.json"
        target.write_text(json.dumps(sample_json_conversation), encoding="utf-8")
        adapter._file_list = [{"path": "data/conv.json", "local_file": "data_conv.json", "size": 200, "sha": "def"}]

        with patch.object(adapter, "download"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["messages"][0]["role"] == "system"

    def test_full_run_with_jsonl(self, adapter, sample_json_classification):
        """Full run() with pre-placed JSONL produces output."""
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        target = adapter._raw_dir / "data_records.jsonl"
        with open(target, "w", encoding="utf-8") as f:
            for item in sample_json_classification:
                f.write(json.dumps(item) + "\n")
        adapter._file_list = [{"path": "data/records.jsonl", "local_file": "data_records.jsonl", "size": 300, "sha": "ghi"}]

        with patch.object(adapter, "download"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_empty_repo_skips_gracefully(self, adapter):
        """Repo with no data files produces no output."""
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        adapter._file_list = []

        with patch.object(adapter, "download"):
            output_path = adapter.run()

        # Output file exists but is empty
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8").strip()
            assert content == ""
