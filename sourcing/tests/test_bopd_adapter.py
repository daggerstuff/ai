"""Tests for the BoPD dataset adapter."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.bopd_adapter import BoPDAdapter

_CSV_FIELDS = ["patient_id", "age", "gender", "icd10_code", "criteria_met", "screening_score"]


@pytest.fixture
def adapter(tmp_path):
    return BoPDAdapter("bopd", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    return [
        {
            "patient_id": "1",
            "age": "30",
            "gender": "female",
            "icd10_code": "F60.3",
            "criteria_met": "true",
            "screening_score": "8",
        },
        {
            "patient_id": "2",
            "age": "45",
            "gender": "male",
            "icd10_code": "Z13.3",
            "criteria_met": "false",
            "screening_score": "2",
        },
    ]


@pytest.fixture
def sample_records():
    return [
        {
            "patient_id": "1",
            "age": "30",
            "gender": "female",
            "icd10_code": "F60.3",
            "criteria_met": "true",
            "screening_score": "8",
            "_source_file": "bopd_data",
        },
        {
            "patient_id": "2",
            "age": "45",
            "gender": "male",
            "icd10_code": "Z13.3",
            "criteria_met": "false",
            "screening_score": "2",
            "_source_file": "bopd_data",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestBoPDAdapter:
    def test_download_skips_if_exists(self, adapter, monkeypatch):
        (adapter._raw_dir / "bopd_repo").mkdir()
        called: list[Any] = []
        monkeypatch.setattr("subprocess.run", lambda *a, **k: called.append(a))
        adapter.download()
        assert len(called) == 0

    def test_download_creates_readme_on_failure(self, adapter, monkeypatch):
        def fake_run(*_a, **_k):
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr("subprocess.run", fake_run)
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "bopd_data.csv", sample_csv_rows_raw)
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["patient_id"] == "1"
        assert records[0]["_source_file"] == "bopd_data"

    def test_convert_basic(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert len(records) == 2
        rec = records[0]
        assert rec["source"] == "bopd"
        assert rec["task_type"] == "symptom_classification"
        assert rec["diagnostic_tag"] == "borderline_personality_disorder"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert "borderline_personality_disorder" in rec["messages"][2]["content"]
        assert rec["clinical_reviewed"] is True
        assert "gender_female" in rec["demographic_tags"]
        assert "age_26_45" in rec["demographic_tags"]

    def test_convert_infers_label_from_criteria(self, adapter):
        raw = [
            {
                "patient_id": "3",
                "age": "22",
                "gender": "m",
                "icd10_code": "F60.3",
                "criteria_met": "false",
                "screening_score": "1",
                "_source_file": "data",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        assert records[0]["messages"][2]["content"] == "no_bpd"

    def test_convert_skips_empty(self, adapter):
        raw = [
            {
                "patient_id": "4",
                "age": "",
                "gender": "",
                "icd10_code": "",
                "criteria_met": "",
                "screening_score": "",
                "_source_file": "empty",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["provenance"]["access_method"] == "github"
        assert "BoPD" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_csv_rows_raw, monkeypatch):
        _write_csv(adapter._raw_dir / "bopd_data.csv", sample_csv_rows_raw)
        # Make download skip by creating fake repo dir
        (adapter._raw_dir / "bopd_repo").mkdir()
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "bopd"
