"""Tests for the Reddit Mental NLP (Kaggle) adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.reddit_mental_nlp_adapter import RedditMentalNLPAdapter

_CSV_FIELDS = ["text", "label"]


@pytest.fixture
def adapter(tmp_path):
    return RedditMentalNLPAdapter("reddit_mental_nlp", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    return [
        {"text": "I feel hopeless every single day", "label": "depression"},
        {"text": "I can't focus on anything for more than 5 minutes", "label": "ADHD"},
        {"text": "My mood swings are unbearable", "label": "bpd"},
    ]


@pytest.fixture
def sample_records(sample_csv_rows_raw):
    return [{**{k.lower(): v for k, v in r.items()}, "_source_file": "mental_disorders"} for r in sample_csv_rows_raw]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestRedditMentalNLPAdapter:
    def test_download_skips_if_csv_exists(self, adapter, monkeypatch):
        (adapter._raw_dir / "data.csv").write_text("exists", encoding="utf-8")
        called: list[Any] = []
        monkeypatch.setattr("subprocess.run", lambda *a, **k: called.append(a))
        adapter.download()
        assert len(called) == 0

    def test_download_creates_readme_on_failure(self, adapter, monkeypatch):
        def fake_run(*_a, **_k):
            raise FileNotFoundError("kaggle not found")

        monkeypatch.setattr("subprocess.run", fake_run)
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "mental_disorders.csv", sample_csv_rows_raw)
        records = adapter.extract()
        assert len(records) == 3
        assert records[0]["text"] == "I feel hopeless every single day"
        assert records[0]["_source_file"] == "mental_disorders"

    def test_convert_basic(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert len(records) == 3
        rec = records[0]
        assert rec["source"] == "reddit_mental_nlp"
        assert rec["task_type"] == "symptom_classification"
        assert rec["diagnostic_tag"] == "major_depressive_disorder"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["linguistic_style"] == "informal"

    def test_convert_maps_labels(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["diagnostic_tag"] == "major_depressive_disorder"
        assert records[1]["diagnostic_tag"] == "adhd"
        assert records[2]["diagnostic_tag"] == "borderline_personality_disorder"

    def test_convert_skips_empty_text(self, adapter):
        raw = [{"text": "", "label": "depression", "_source_file": "f"}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["provenance"]["access_method"] == "kaggle"
        assert "kaggle.com" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_csv_rows_raw, monkeypatch):
        _write_csv(adapter._raw_dir / "mental_disorders.csv", sample_csv_rows_raw)
        # Skip download since file exists
        monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["source"] == "reddit_mental_nlp"
