"""Tests for the PsyDial privacy-preserving counseling dataset adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.psydial_adapter import PsyDialAdapter


@pytest.fixture
def sample_entries():
    """Sample entries matching HF PsyDial-D101 JSON format."""
    return [
        {
            "idx": 0,
            "messages": [{"role": "user", "content": "你好。"}],
            "golden": {"role": "assistant", "content": "你好，你今天想要讨论哪方面的话题呢？"},
        },
        {
            "idx": 1,
            "messages": [
                {"role": "user", "content": "我最近压力很大。"},
            ],
            "golden": {"role": "assistant", "content": "能理解你的感受，能具体说说是什么方面的压力吗？"},
        },
        {
            "idx": 2,
            "messages": [
                {"role": "user", "content": "我和父母吵架了。"},
                {"role": "assistant", "content": "听起来让你很难过。"},
                {"role": "user", "content": "是的，我不知道该怎么办。"},
            ],
            "golden": {"role": "assistant", "content": "让我们一起来看看可能的解决方案。"},
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    return PsyDialAdapter("psydial", tmp_path)


def _populate_raw(adapter, entries, filename="PsyDial-D101.json"):
    """Populate raw dir. D101 gets entries; D4+D1 get empty arrays."""
    adapter._raw_dir.mkdir(parents=True, exist_ok=True)
    (adapter._raw_dir / "PsyDial-D101.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    for fname in ("PsyDial-D4.json", "PsyDial-D1.json"):
        (adapter._raw_dir / fname).write_text("[]", encoding="utf-8")


class TestPsyDialAdapter:
    def test_download_skips_when_files_present(self, adapter, sample_entries):
        _populate_raw(adapter, sample_entries)
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_extract_reads_json_entries(self, adapter, sample_entries):
        _populate_raw(adapter, sample_entries)
        raw = adapter.extract()
        assert len(raw) == 3
        assert raw[0]["idx"] == 0
        assert raw[0]["_source_file"] == "PsyDial-D101.json"

    def test_convert_basic_entry(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        assert len(records) == 3

        rec0 = records[0]
        assert rec0["source"] == "psydial"
        assert rec0["task_type"] == "therapy_response_generation"
        assert rec0["privacy_preserving"] is True
        assert rec0["messages"][0]["role"] == "system"
        assert "RMRR" in rec0["messages"][0]["content"]
        assert rec0["messages"][1]["role"] == "user"
        assert rec0["messages"][2]["role"] == "assistant"
        assert rec0["messages"][2]["content"] == "你好，你今天想要讨论哪方面的话题呢？"
        assert rec0["language"] == "zh"

    def test_system_prompt_includes_rmrr_note(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        for rec in records:
            assert "RMRR" in rec["rmrr_methodology"]
            assert "Reconstruct" in rec["rmrr_methodology"]
            assert "RMRR" in rec["messages"][0]["content"]

    def test_empty_messages_skipped(self, adapter):
        raw = [{"idx": 0, "messages": [], "golden": {}}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_empty_golden_skipped(self, adapter):
        raw = [{"idx": 0, "messages": [{"role": "user", "content": "test"}], "golden": {}}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        prov = records[0]["provenance"]
        assert prov["source_url"] == "https://huggingface.co/qiuhuachuan/PsyDial-D101"
        assert prov["access_method"] == "huggingface"
        assert prov["original_format"] == "json"

    def test_full_run(self, adapter, sample_entries):
        _populate_raw(adapter, sample_entries)
        with patch("huggingface_hub.hf_hub_download"):
            output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["source"] == "psydial"
        assert first["task_type"] == "therapy_response_generation"

    def test_factory_registration(self, tmp_path):
        from ai.sourcing.dataset_adapters.adapter_factory import (
            get_adapter,
            list_available_adapters,
        )

        assert "psydial" in list_available_adapters()
        adapter = get_adapter("PSYDIAL", tmp_path)
        assert isinstance(adapter, PsyDialAdapter)
