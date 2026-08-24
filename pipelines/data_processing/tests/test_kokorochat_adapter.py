"""Tests for the KokoroChat adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.kokorochat_adapter import KokoroChatAdapter


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_dialogue(turns: list[tuple[str, str]]) -> dict:
    return {"dialogue": [{"role": role, "time": "2020-01-01T00:00:00", "utterance": text} for role, text in turns]}


@pytest.fixture
def adapter(tmp_path):
    return KokoroChatAdapter("kokorochat", tmp_path)


@pytest.fixture
def sample_dialogues():
    return [
        {
            "_source_file": "001",
            "dialogue": [
                {"role": "counselor", "time": "2020-01-01T00:00:00", "utterance": "こんにちは"},
                {"role": "client", "time": "2020-01-01T00:00:10", "utterance": "よろしくお願いします"},
                {"role": "counselor", "time": "2020-01-01T00:00:20", "utterance": "ご相談ありがとうございます"},
            ],
        },
        {
            "_source_file": "002",
            "dialogue": [
                {"role": "client", "time": "2020-01-01T00:00:00", "utterance": "困っています"},
                {"role": "counselor", "time": "2020-01-01T00:00:10", "utterance": "話を聞かせてください"},
            ],
        },
    ]


def _populate_raw(adapter, dialogues=None):
    if dialogues is None:
        dialogues = [
            _make_dialogue([("counselor", "こんにちは"), ("client", "助けてください"), ("counselor", "話してください")])
        ]
    for i, d in enumerate(dialogues):
        _write_json(adapter._raw_dir / f"{i:04d}.json", d)


class TestKokoroChatAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_reads_json(self, adapter):
        _write_json(adapter._raw_dir / "001.json", _make_dialogue([("counselor", "A"), ("client", "B")]))
        records = adapter.extract()
        assert len(records) == 1
        assert records[0]["_source_file"] == "001"
        assert len(records[0]["dialogue"]) == 2

    def test_convert_basic(self, adapter, sample_dialogues):
        records = adapter.convert_to_chatml(sample_dialogues)
        assert len(records) == 2
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "assistant"  # counselor first
        assert msgs[2]["role"] == "user"  # client second
        assert msgs[3]["role"] == "assistant"

    def test_client_first(self, adapter):
        row = {
            "_source_file": "001",
            "dialogue": [
                {"role": "client", "time": "T", "utterance": "助けて"},
                {"role": "counselor", "time": "T", "utterance": "はい"},
            ],
        }
        records = adapter.convert_to_chatml([row])
        assert len(records) == 1
        msgs = records[0]["messages"]
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_empty_utterance_skipped(self, adapter):
        row = {
            "_source_file": "001",
            "dialogue": [
                {"role": "counselor", "time": "T", "utterance": ""},
                {"role": "client", "time": "T", "utterance": "OK"},
                {"role": "counselor", "time": "T", "utterance": "はい"},
            ],
        }
        records = adapter.convert_to_chatml([row])
        assert len(records) == 1

    def test_missing_roles_skipped(self, adapter):
        row = {
            "_source_file": "001",
            "dialogue": [
                {"role": "counselor", "time": "T", "utterance": "A"},
                {"role": "counselor", "time": "T", "utterance": "B"},
            ],
        }
        records = adapter.convert_to_chatml([row])
        assert len(records) == 0  # no user message

    def test_empty_dialogue_skipped(self, adapter):
        row = {"_source_file": "001", "dialogue": []}
        records = adapter.convert_to_chatml([row])
        assert len(records) == 0

    def test_provenance(self, adapter, sample_dialogues):
        records = adapter.convert_to_chatml(sample_dialogues)
        assert records[0]["provenance"]["access_method"] == "s3"
        assert records[0]["provenance"]["original_format"] == "json"

    def test_full_run(self, adapter, monkeypatch):
        _populate_raw(adapter)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        output = adapter.run()
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "kokorochat"

    def test_factory_registration(self):
        from ai.pipelines.data_processing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("kokorochat", "/tmp/test_kokoro")
        assert isinstance(a, KokoroChatAdapter)
