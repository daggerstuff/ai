"""Tests for the MentalChat16K adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.mentalchat16k_adapter import MentalChat16KAdapter


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_row(
    instruction: str = "You are a helpful mental health counselling assistant.",
    input: str = "I've been feeling anxious lately.",
    output: str = "I understand. Let's explore that together.",
) -> dict:
    return {"instruction": instruction, "input": input, "output": output}


@pytest.fixture
def adapter(tmp_path):
    return MentalChat16KAdapter("mentalchat16k", tmp_path)


@pytest.fixture
def sample_rows():
    return [
        _make_row(),
        _make_row(
            input="I can't sleep at night.",
            output="Let's talk about your sleep hygiene.",
        ),
    ]


def _populate_raw(adapter, rows=None):
    if rows is None:
        rows = [_make_row()]
    _write_csv(adapter._raw_dir / "Interview_Data_6K.csv", rows)
    _write_csv(adapter._raw_dir / "Synthetic_Data_10K.csv", [])


class TestMentalChat16KAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_reads_csv(self, adapter, sample_rows):
        _write_csv(adapter._raw_dir / "Interview_Data_6K.csv", sample_rows)
        _write_csv(adapter._raw_dir / "Synthetic_Data_10K.csv", [])
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["input"] == "I've been feeling anxious lately."
        assert records[0]["_source_file"] == "Interview_Data_6K.csv"

    def test_convert_basic(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert len(records) == 2
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[0]["content"] == "You are a helpful mental health counselling assistant."

    def test_empty_input_skipped(self, adapter):
        row = _make_row(input="")
        records = adapter.convert_to_chatml([row])
        assert len(records) == 0

    def test_empty_output_skipped(self, adapter):
        row = _make_row(output="")
        records = adapter.convert_to_chatml([row])
        assert len(records) == 0

    def test_default_system_when_no_instruction(self, adapter):
        row = _make_row(instruction="")
        records = adapter.convert_to_chatml([row])
        assert "compassionate" in records[0]["messages"][0]["content"]

    def test_provenance(self, adapter, sample_rows):
        records = adapter.convert_to_chatml(sample_rows)
        assert records[0]["provenance"]["access_method"] == "s3"
        assert records[0]["provenance"]["original_format"] == "csv"

    def test_full_run(self, adapter, sample_rows, monkeypatch):
        _populate_raw(adapter, sample_rows)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        output = adapter.run()
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "mentalchat16k"

    def test_factory_registration(self):
        from ai.pipelines.data_processing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("mentalchat16k", "/tmp/test_mc16k")
        assert isinstance(a, MentalChat16KAdapter)
