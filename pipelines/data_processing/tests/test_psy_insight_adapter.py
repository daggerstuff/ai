"""Tests for the psy_insight adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.psy_insight_adapter import PsyInsightAdapter


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_entry(
    dialog_id: str = "001",
    psychotherapy: str = "CBT",
    topic: str = "Anxiety",
    background: str = "Client feels anxious.",
    dialog: list[dict] | None = None,
) -> dict:
    if dialog is None:
        dialog = [
            {"speaker": "Seeker", "participant": "Client", "content": "I feel anxious.", "id": 1, "observation": ""},
            {
                "speaker": "Supporter",
                "participant": "Therapist",
                "content": "Tell me more.",
                "id": 2,
                "observation": "",
            },
            {"speaker": "Seeker", "participant": "Client", "content": "It's hard.", "id": 3, "observation": ""},
        ]
    return {
        "dialog_id": dialog_id,
        "theme": f"{psychotherapy}: test",
        "psychotherapy": psychotherapy,
        "topic": topic,
        "stage": "",
        "guide": "",
        "is_same_qa": 0,
        "is_same_session": 0,
        "background": background,
        "reasoning": "",
        "dialog": dialog,
        "summary": "",
    }


@pytest.fixture
def adapter(tmp_path):
    return PsyInsightAdapter("psy_insight", tmp_path)


def _populate_raw(adapter, entries=None, lang="en"):
    if entries is None:
        entries = [_make_entry()]
    data_dir = adapter._raw_dir / "Psy-Insight" / "data"
    _write_json(data_dir / f"{lang}_data_version7.json", entries)


class TestPsyInsightAdapter:
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
        assert records[0]["_language"] == "en"

    def test_convert_basic(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 1
        msgs = chatml[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert "CBT" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_empty_dialog_skipped(self, adapter):
        _populate_raw(adapter, [_make_entry(dialog=[])])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 0

    def test_missing_roles_skipped(self, adapter):
        _populate_raw(
            adapter,
            [
                _make_entry(
                    dialog=[
                        {
                            "speaker": "Supporter",
                            "participant": "Therapist",
                            "content": "Only therapist",
                            "id": 1,
                            "observation": "",
                        },
                    ]
                )
            ],
        )
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
        from ai.pipelines.data_processing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("psy_insight", "/tmp/test_psi")
        assert isinstance(a, PsyInsightAdapter)
