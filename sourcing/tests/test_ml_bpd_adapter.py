"""Tests for the machine_learning_BPD dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.ml_bpd_adapter import MLBPDAdapter

_CSV_FIELDS = [
    "patient_id",
    "age",
    "gender",
    "bis11_score",
    "symptom_severity",
    "treatment_outcome",
]


@pytest.fixture
def adapter(tmp_path):
    return MLBPDAdapter("ml_bpd", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    """CSV format (as read from file)."""
    return [
        {
            "patient_id": "1",
            "age": "28",
            "gender": "female",
            "bis11_score": "72",
            "symptom_severity": "severe",
            "treatment_outcome": " responder",
        },
        {
            "patient_id": "2",
            "age": "35",
            "gender": "male",
            "bis11_score": "65",
            "symptom_severity": "moderate",
            "treatment_outcome": "non-responder",
        },
    ]


@pytest.fixture
def sample_records():
    """Extract output format (lowercase keys + _source_file)."""
    return [
        {
            "patient_id": "1",
            "age": "28",
            "gender": "female",
            "bis11_score": "72",
            "symptom_severity": "severe",
            "treatment_outcome": "responder",
            "_source_file": "data",
        },
        {
            "patient_id": "2",
            "age": "35",
            "gender": "male",
            "bis11_score": "65",
            "symptom_severity": "moderate",
            "treatment_outcome": "non-responder",
            "_source_file": "data",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestMLBPDAdapter:
    def test_download_skips_if_exists(self, adapter, monkeypatch):
        for fname in ("data.csv", "BPD_data.csv", "treatment_outcome.csv", "bis11_scores.csv", "symptom_severity.csv"):
            (adapter._raw_dir / fname).write_text("exists", encoding="utf-8")
        called: list[Any] = []
        monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: called.append(a))
        adapter.download()
        assert len(called) == 0

    def test_download_creates_readme_on_failure(self, adapter, monkeypatch):
        def fake_retrieve(*_args):
            raise ConnectionError("no network")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_retrieve)
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "data.csv", sample_csv_rows_raw)
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["patient_id"] == "1"
        assert records[0]["_source_file"] == "data"

    def test_convert_basic(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert len(records) == 2
        rec = records[0]
        assert rec["source"] == "ml_bpd"
        assert rec["task_type"] == "severity_estimation"
        assert rec["diagnostic_tag"] == "borderline_personality_disorder"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["clinical_reviewed"] is True
        assert "gender_female" in rec["demographic_tags"]
        assert "age_26_45" in rec["demographic_tags"]

    def test_convert_builds_description_from_fields(self, adapter):
        raw = [
            {
                "patient_id": "3",
                "age": "40",
                "gender": "f",
                "bis11_score": "70",
                "symptom_severity": "moderate",
                "treatment_outcome": "",
                "_source_file": "bpd_data",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        # Description built from available fields
        user_content = records[0]["messages"][1]["content"]
        assert "bis11_score" in user_content or "70" in user_content

    def test_convert_skips_empty(self, adapter):
        raw = [
            {
                "patient_id": "4",
                "age": "",
                "gender": "",
                "bis11_score": "",
                "symptom_severity": "",
                "treatment_outcome": "",
                "_source_file": "empty",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["provenance"]["access_method"] == "github"
        assert "github.com/saidejp/machine_learning_BPD" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_csv_rows_raw, monkeypatch):
        _write_csv(adapter._raw_dir / "data.csv", sample_csv_rows_raw)
        # Skip download since file already exists
        monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: None)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "ml_bpd"
