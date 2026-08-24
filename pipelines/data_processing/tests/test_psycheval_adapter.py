"""Tests for the PsychEval adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.sourcing.dataset_adapters.psycheval_adapter import PsychEvalAdapter, _strip_meta_tags


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_session(num: int, dialogue: list[tuple[str, str]]) -> dict:
    return {
        "session_number": num,
        "session_goals": {"overall_stage": "test", "session_focus": {}},
        "suggest_skills": [],
        "session_dialogue": [{"role": role, "text": text} for role, text in dialogue],
        "session_summary": {},
        "client_info_last": {},
    }


def _make_file(
    modality: str = "bt",
    client_id: int = 1,
    sessions: int = 1,
    turns_per_session: int = 4,
) -> dict:
    dialogue = [
        ("Counselor", "<assessment>test</assessment>你好、ようこそ。"),
        ("Client", "こんにちは。"),
        ("Counselor", "<skill>123</skill>お話しください。"),
        ("Client", "困っています。"),
    ]
    return {
        "theoretical": modality,
        "client_id": client_id,
        "client_info": {
            "static_traits": {"name": "テスト", "age": "30", "gender": "男"},
            "main_problem": "テスト問題",
            "topic": "テスト",
        },
        "global_plan": [],
        "sessions": [_make_session(i + 1, dialogue[:turns_per_session]) for i in range(sessions)],
    }


@pytest.fixture
def adapter(tmp_path):
    return PsychEvalAdapter("psycheval", tmp_path)


def _populate_raw(adapter, modality="bt", files=None):
    if files is None:
        files = [_make_file(modality)]
    data_dir = adapter._raw_dir / "PsychEval" / "data" / modality
    for i, f in enumerate(files):
        _write_json(data_dir / f"{i + 1}.json", f)


class TestPsychEvalAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_reads_json(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        assert len(records) == 1
        assert records[0]["_modality"] == "bt"
        assert records[0]["client_id"] == 1

    def test_convert_basic(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 1
        msgs = chatml[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] in ("user", "assistant")

    def test_multiple_sessions(self, adapter):
        _populate_raw(adapter, files=[_make_file(sessions=3)])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 3

    def test_meta_tags_stripped(self, adapter):
        _populate_raw(adapter, files=[_make_file(turns_per_session=2)])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        counselor_msg = [m for m in chatml[0]["messages"] if m["role"] == "assistant"][0]
        assert "<assessment>" not in counselor_msg["content"]
        assert "你好" in counselor_msg["content"]

    def test_strip_meta_tags_function(self):
        text = "<assessment>test</assessment><skill>123</skill>実メッセージ"
        result = _strip_meta_tags(text)
        assert "実メッセージ" in result
        assert "<" not in result

    def test_empty_dialogue_skipped(self, adapter):
        file_data = _make_file()
        file_data["sessions"][0]["session_dialogue"] = []
        _populate_raw(adapter, files=[file_data])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 0

    def test_multiple_modalities(self, adapter):
        for mod in ["bt", "cbt", "het"]:
            _populate_raw(adapter, modality=mod)
        records = adapter.extract()
        modalities = {r["_modality"] for r in records}
        assert modalities == {"bt", "cbt", "het"}

    def test_provenance(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert chatml[0]["provenance"]["access_method"] == "s3"

    def test_full_run(self, adapter, monkeypatch):
        _populate_raw(adapter)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        output = adapter.run()
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

    def test_factory_registration(self):
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("psycheval", "/tmp/test_psycheval")
        assert isinstance(a, PsychEvalAdapter)
