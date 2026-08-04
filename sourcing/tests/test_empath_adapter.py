"""Tests for the Empath adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ai.sourcing.dataset_adapters.empath_adapter import EmpathAdapter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_row(
    idx: int = 1,
    chat_history: list[str] | None = None,
    sys_response: str = "I understand how you feel.",
    situation: str = "I went shopping today.",
    emotion: str = "joyful",
    behavior: str = "I'm in a positive mood, please congratulate me.",
) -> dict:
    if chat_history is None:
        chat_history = ["I had a great day!"]
    return {
        "id": idx,
        "chat_history": str(chat_history),
        "sys_response": sys_response,
        "situation": situation,
        "emotion": emotion,
        "question or not": "[None]",
        "behavior": behavior,
    }


@pytest.fixture
def adapter(tmp_path):
    return EmpathAdapter("empath", tmp_path)


@pytest.fixture
def sample_rows():
    return [
        _make_row(1, ["I had a great day!", "That's wonderful!"], "Keep enjoying!"),
        _make_row(2, ["I feel sad."], "I'm sorry to hear that."),
    ]


def _populate_raw(adapter, rows=None):
    if rows is None:
        rows = [_make_row()]
    _write_jsonl(adapter._raw_dir / "train.jsonl", rows)
    _write_jsonl(adapter._raw_dir / "validation.jsonl", [_make_row(99)])
    _write_jsonl(adapter._raw_dir / "test.jsonl", [_make_row(100)])


class TestEmpathAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_reads_jsonl(self, adapter, sample_rows):
        _write_jsonl(adapter._raw_dir / "train.jsonl", sample_rows)
        _write_jsonl(adapter._raw_dir / "validation.jsonl", [])
        _write_jsonl(adapter._raw_dir / "test.jsonl", [])
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[0]["_split"] == "train"

    def test_convert_basic(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert len(records) == 2
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "Keep enjoying!"

    def test_multi_turn_alternating(self, adapter):
        row = _make_row(1, ["Hello", "Hi there!", "How are you?"], "I'm great!")
        records = adapter.convert_to_chatml([row])
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[4]["role"] == "assistant"

    def test_empty_sys_response_skipped(self, adapter):
        row = _make_row(1, ["Hello"], "")
        records = adapter.convert_to_chatml([row])
        assert len(records) == 0

    def test_emotion_in_system(self, adapter):
        row = _make_row(1, ["I feel angry"], "I hear you.", emotion="angry")
        records = adapter.convert_to_chatml([row])
        assert "angry" in records[0]["messages"][0]["content"]

    def test_behavior_none_handled(self, adapter):
        row = _make_row(1, ["Hi"], "Hello!", behavior="[None]")
        records = adapter.convert_to_chatml([row])
        assert "Behavioral framing" not in records[0]["messages"][0]["content"]

    def test_provenance(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert records[0]["provenance"]["access_method"] == "huggingface"
        assert (
            records[0]["provenance"]["source_url"] == "https://huggingface.co/datasets/Adapting/empathetic_dialogues_v2"
        )

    def test_full_run(self, adapter, sample_rows, monkeypatch):
        _populate_raw(adapter, sample_rows)
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: None)
        output = adapter.run()
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4  # 2 train + 1 validation + 1 test
        record = json.loads(lines[0])
        assert record["source"] == "empath"
        assert record["task_type"] == "therapy_response_generation"

    def test_factory_registration(self):
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("empath", "/tmp/test_empath")
        assert isinstance(a, EmpathAdapter)
