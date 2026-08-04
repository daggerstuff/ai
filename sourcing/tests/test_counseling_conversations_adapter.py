"""Tests for the Counseling Conversations dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.counseling_conversations_adapter import (
    CounselingConversationsAdapter,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def sample_rows():
    return [
        {"Context": "I feel anxious all the time.", "Response": "Let's explore that anxiety together."},
        {"Context": "I can't sleep at night.", "Response": "Sleep difficulties are common. Let's discuss."},
    ]


@pytest.fixture
def adapter(tmp_path):
    return CounselingConversationsAdapter("counseling_conversations", tmp_path)


class TestCounselingConversationsAdapter:
    def test_download_skips_when_file_exists(self, adapter, sample_rows):
        _write_jsonl(adapter._raw_dir / "combined_dataset.json", sample_rows)
        # Should not attempt download
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_download_writes_readme_on_failure(self, adapter):
        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("network error")):
            adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "Download failed" in readme.read_text()

    def test_extract_reads_jsonl(self, adapter, sample_rows):
        _write_jsonl(adapter._raw_dir / "combined_dataset.json", sample_rows)
        result = adapter.extract()
        assert len(result) == 2
        assert result[0]["Context"] == "I feel anxious all the time."

    def test_extract_empty_when_no_file(self, adapter):
        assert adapter.extract() == []

    def test_convert_basic(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert len(records) == 2
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "I feel anxious all the time."
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Let's explore that anxiety together."

    def test_convert_skips_empty_fields(self, adapter):
        rows = [
            {"Context": "", "Response": "response"},
            {"Context": "context", "Response": ""},
            {"Context": "ok", "Response": "ok"},
        ]
        records = adapter.convert_to_chatml(rows)
        assert len(records) == 1

    def test_convert_system_prompt(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert "counselor" in records[0]["messages"][0]["content"].lower()

    def test_provenance(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        prov = records[0]["provenance"]
        assert prov["access_method"] == "huggingface"
        assert prov["original_format"] == "jsonl"
        assert "huggingface.co" in prov["source_url"]

    def test_full_run(self, adapter, sample_rows, monkeypatch):
        _write_jsonl(adapter._raw_dir / "combined_dataset.json", sample_rows)
        monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: None)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "counseling_conversations"
        assert record["task_type"] == "therapy_response_generation"

    def test_factory_registration(self):
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("counseling_conversations", "/tmp/test_cc")
        assert isinstance(a, CounselingConversationsAdapter)
