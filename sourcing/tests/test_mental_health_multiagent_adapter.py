"""Tests for the mental_health_multiagent adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.sourcing.dataset_adapters.mental_health_multiagent_adapter import MentalHealthMultiagentAdapter


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_chat(roles: list[tuple[str, str]]) -> dict:
    return {
        "timestamp": "2025-01-01T00:00:00",
        "questionnaire": "test",
        "conversation": [{"role": r, "content": c} for r, c in roles],
    }


@pytest.fixture
def adapter(tmp_path):
    return MentalHealthMultiagentAdapter("mental_health_multiagent", tmp_path)


def _populate_raw(adapter, chats=None):
    if chats is None:
        chats = [_make_chat([("assistant", "Hello"), ("user", "Hi"), ("assistant", "How are you?")])]
    for i, chat in enumerate(chats):
        _write_json(adapter._raw_dir / f"chat_{i}.json", chat)


class TestMentalHealthMultiagentAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        assert len(records) == 1
        assert len(records[0]["conversation"]) == 3

    def test_convert_basic(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 1
        msgs = chatml[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_empty_conversation_skipped(self, adapter):
        _populate_raw(adapter, [_make_chat([])])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 0

    def test_missing_roles_skipped(self, adapter):
        _populate_raw(adapter, [_make_chat([("assistant", "Only assistant")])])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 0

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

        a = get_adapter("mental_health_multiagent", "/tmp/test_mhma")
        assert isinstance(a, MentalHealthMultiagentAdapter)
