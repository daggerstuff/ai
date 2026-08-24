"""Tests for the dsm_vector_space adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.dsm_vector_space_adapter import DsmVectorSpaceAdapter


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_diagnosis(
    name: str = "Panic Disorder",
    code: str = "F41.0",
    chapter: str = "Anxiety Disorders",
    threshold: str = "Repeated unexpected panic attacks",
    duration: str = "At least 1 month",
    symptoms: str = "Palpitations, sweating, trembling",
) -> dict:
    return {
        "diagnosis_id": "DX_PANIC",
        "diagnosis_name": name,
        "diagnostic_code": code,
        "chapter_category": chapter,
        "threshold_count": threshold,
        "duration_rule": duration,
        "symptoms": symptoms,
    }


@pytest.fixture
def adapter(tmp_path):
    return DsmVectorSpaceAdapter("dsm_vector_space", tmp_path)


def _populate_raw(adapter, diagnoses=None, filename="anxiety_disorders.json"):
    if diagnoses is None:
        diagnoses = [_make_diagnosis()]
    data_dir = adapter._raw_dir / "dsm-in-vector-space" / "data"
    _write_json(data_dir / filename, diagnoses)


class TestDsmVectorSpaceAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_list_format(self, adapter):
        _populate_raw(adapter, [_make_diagnosis(), _make_diagnosis("GAD", "F41.1")])
        records = adapter.extract()
        assert len(records) == 2
        assert "diagnosis_name" in records[0]

    def test_convert_basic(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 1
        msgs = chatml[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert "Panic Disorder" in msgs[0]["content"]
        assert "F41.0" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_empty_entry_skipped(self, adapter):
        _populate_raw(adapter, [{"diagnosis_name": ""}])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 0

    def test_multiple_diagnoses(self, adapter):
        _populate_raw(
            adapter,
            [
                _make_diagnosis("Panic Disorder"),
                _make_diagnosis("GAD", "F41.1", "Anxiety Disorders"),
                _make_diagnosis("Social Anxiety", "F40.1", "Anxiety Disorders"),
            ],
        )
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) == 3

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

        a = get_adapter("dsm_vector_space", "/tmp/test_dsm")
        assert isinstance(a, DsmVectorSpaceAdapter)
