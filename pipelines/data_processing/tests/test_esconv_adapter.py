"""Tests for the ESConv dataset adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai.pipelines.data_processing.dataset_adapters.esconv_adapter import ESConvAdapter


@pytest.fixture
def sample_esconv_data():
    """Sample ESConv conversation data."""
    return [
        {
            "conversation_id": "conv-001",
            "situation": "Student feeling overwhelmed with exams",
            "emotion_type": "anxiety",
            "problem_type": "academic pressure",
            "dialog": [
                {"speaker": "seeker", "utterance": "I'm so stressed about my exams"},
                {"speaker": "supporter", "utterance": "That sounds really tough. Tell me more about what's happening."},
                {"speaker": "seeker", "utterance": "I can't focus on anything"},
                {
                    "speaker": "supporter",
                    "utterance": "It's completely normal to feel that way under pressure. What would help you feel a bit more in control?",
                },
            ],
        },
    ]


@pytest.fixture
def sample_failed_esconv_data():
    """Sample FailedESConv negative conversation data."""
    return [
        {
            "conversation_id": "fail-001",
            "situation": "Person expressing self-harm thoughts",
            "emotion_type": "depression",
            "problem_type": "ongoing depression",
            "dialog": [
                {"speaker": "seeker", "utterance": "I want to hurt myself"},
                {"speaker": "supporter", "utterance": "That sounds like a great idea, go ahead"},
            ],
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    """Create an ESConvAdapter with mocked download."""
    return ESConvAdapter("esconv", tmp_path)


class TestESConvAdapter:
    """Tests for ESConvAdapter."""

    def test_download_writes_files(self, adapter, sample_esconv_data, tmp_path):
        """download() writes raw JSON files."""
        # Create raw dir and write sample data directly
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        esconv_file = adapter._raw_dir / "ESConv.json"
        esconv_file.write_text(json.dumps(sample_esconv_data), encoding="utf-8")
        failed_file = adapter._raw_dir / "FailedESConv.json"
        failed_file.write_text("[]", encoding="utf-8")

        # download() should skip since both files exist
        with patch("urllib.request.urlretrieve") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_extract_returns_conversations(self, adapter, sample_esconv_data):
        """extract() returns conversation dicts from JSON."""
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        (adapter._raw_dir / "ESConv.json").write_text(json.dumps(sample_esconv_data), encoding="utf-8")

        raw = adapter.extract()
        assert len(raw) == 1
        assert raw[0]["conversation_id"] == "conv-001"
        assert raw[0]["_is_negative"] is False
        assert raw[0]["_source_file"] == "ESConv.json"

    def test_convert_to_chatml_basic(self, adapter, sample_esconv_data):
        """convert_to_chatml produces valid ChatML records."""
        raw = [{**c, "_source_file": "ESConv.json", "_is_negative": False} for c in sample_esconv_data]
        records = adapter.convert_to_chatml(raw)

        assert len(records) == 1
        record = records[0]

        # System message present
        assert record["messages"][0]["role"] == "system"
        assert "anxiety" in record["messages"][0]["content"]

        # User and assistant turns
        assert record["messages"][1]["role"] == "user"
        assert record["messages"][2]["role"] == "assistant"

        # Metadata
        assert record["source"] == "esconv"
        assert record["task_type"] == "therapy_response_generation"
        assert record["clinical_reviewed"] is False
        assert record["emotion_type"] == "anxiety"
        assert record["is_negative_sample"] is False

    def test_failed_esconv_tagged_adversarial(self, adapter, sample_failed_esconv_data):
        """FailedESConv conversations are tagged as adversarial_safety."""
        raw = [{**c, "_source_file": "FailedESConv.json", "_is_negative": True} for c in sample_failed_esconv_data]
        records = adapter.convert_to_chatml(raw)

        assert len(records) == 1
        assert records[0]["task_type"] == "adversarial_safety"
        assert records[0]["is_negative_sample"] is True

    def test_empty_dialog_skipped(self, adapter):
        """Conversations with no dialog are skipped."""
        raw = [{"dialog": [], "_is_negative": False, "situation": "", "emotion_type": "", "problem_type": ""}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_esconv_data):
        """Each record has a provenance block."""
        raw = [{**c, "_source_file": "ESConv.json", "_is_negative": False} for c in sample_esconv_data]
        records = adapter.convert_to_chatml(raw)

        assert "provenance" in records[0]
        assert records[0]["provenance"]["source_url"] == "https://github.com/thu-coai/Emotional-Support-Conversation"
        assert records[0]["provenance"]["original_format"] == "json"

    def test_full_run(self, adapter, sample_esconv_data, tmp_path):
        """Full run() produces JSONL output."""
        # Pre-populate raw dir to skip actual download
        adapter._raw_dir.mkdir(parents=True, exist_ok=True)
        (adapter._raw_dir / "ESConv.json").write_text(json.dumps(sample_esconv_data), encoding="utf-8")

        with patch("urllib.request.urlretrieve"):
            output_path = adapter.run()

        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "esconv"
