"""Tests for the machine_learning_BPD dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.ml_bpd_adapter import MLBPDAdapter

_CSV_FIELDS = [
    "GPO.",
    "EXP.",
    "Sexo",
    "Edad",
    "Hijos",
    "Educación",
    "Años_estudio",
    "BDI_TOTAL_PRE",
    "BAI_TOTAL_PRE",
    "BIS_TOTAL_PRE",
    "DERS_TOTAL_PRE",
    "BEST_TOTAL_PRE",
]


@pytest.fixture
def adapter(tmp_path):
    return MLBPDAdapter("ml_bpd", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    return [
        {
            "GPO.": "DBT",
            "EXP.": "001",
            "Sexo": "Mujer",
            "Edad": "28",
            "Hijos": "0",
            "Educación": "3",
            "Años_estudio": "12",
            "BDI_TOTAL_PRE": "32",
            "BAI_TOTAL_PRE": "25",
            "BIS_TOTAL_PRE": "72",
            "DERS_TOTAL_PRE": "45",
            "BEST_TOTAL_PRE": "38",
        },
        {
            "GPO.": "DBT",
            "EXP.": "002",
            "Sexo": "Hombre",
            "Edad": "35",
            "Hijos": "1",
            "Educación": "2",
            "Años_estudio": "10",
            "BDI_TOTAL_PRE": "18",
            "BAI_TOTAL_PRE": "12",
            "BIS_TOTAL_PRE": "65",
            "DERS_TOTAL_PRE": "30",
            "BEST_TOTAL_PRE": "22",
        },
    ]


@pytest.fixture
def sample_records():
    return [
        {
            "Sexo": "Mujer",
            "Edad": "28",
            "BDI_TOTAL_PRE": "32",
            "BAI_TOTAL_PRE": "25",
            "BIS_TOTAL_PRE": "72",
            "DERS_TOTAL_PRE": "45",
            "BEST_TOTAL_PRE": "38",
            "_source_file": "training_noSession.csv",
        },
        {
            "Sexo": "Hombre",
            "Edad": "35",
            "BDI_TOTAL_PRE": "18",
            "BAI_TOTAL_PRE": "12",
            "BIS_TOTAL_PRE": "65",
            "DERS_TOTAL_PRE": "30",
            "BEST_TOTAL_PRE": "22",
            "_source_file": "training_noSession.csv",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


_DEFAULT_ROWS: list[dict[str, str]] = [
    {
        "GPO.": "DBT",
        "EXP.": "001",
        "Sexo": "Mujer",
        "Edad": "28",
        "Hijos": "0",
        "Educación": "3",
        "Años_estudio": "12",
        "BDI_TOTAL_PRE": "32",
        "BAI_TOTAL_PRE": "25",
        "BIS_TOTAL_PRE": "72",
        "DERS_TOTAL_PRE": "45",
        "BEST_TOTAL_PRE": "38",
    },
    {
        "GPO.": "DBT",
        "EXP.": "002",
        "Sexo": "Hombre",
        "Edad": "35",
        "Hijos": "1",
        "Educación": "2",
        "Años_estudio": "10",
        "BDI_TOTAL_PRE": "18",
        "BAI_TOTAL_PRE": "12",
        "BIS_TOTAL_PRE": "65",
        "DERS_TOTAL_PRE": "30",
        "BEST_TOTAL_PRE": "22",
    },
]


def _make_repo(adapter: MLBPDAdapter, rows: list[dict[str, str]] | None = None) -> Path:
    repo_dir = adapter._raw_dir / "machine_learning_BPD"
    repo_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(repo_dir / "training_noSession.csv", rows if rows is not None else _DEFAULT_ROWS)
    return repo_dir


class TestMLBPDAdapter:
    def test_download_skips_if_exists(self, adapter):
        repo_dir = adapter._raw_dir / "machine_learning_BPD"
        repo_dir.mkdir(parents=True)
        (repo_dir / "training_noSession.csv").write_text("exists", encoding="utf-8")
        adapter.download()
        assert not (adapter._raw_dir / "README.txt").exists()

    def test_download_creates_readme_on_failure(self, adapter, monkeypatch):
        def fake_run(*_a, **_kw):
            raise Exception("no network")

        monkeypatch.setattr("subprocess.run", fake_run)
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_csv_rows_raw):
        _make_repo(adapter, sample_csv_rows_raw)
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["Edad"] == "28"
        assert records[0]["_source_file"] == "training_noSession.csv"

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

    def test_convert_builds_description_from_clinical_cols(self, adapter):
        raw = [
            {
                "Sexo": "Mujer",
                "Edad": "40",
                "BDI_TOTAL_PRE": "25",
                "BAI_TOTAL_PRE": "18",
                "BIS_TOTAL_PRE": "70",
                "DERS_TOTAL_PRE": "35",
                "BEST_TOTAL_PRE": "28",
                "_source_file": "testing_noSession.csv",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        user_content = records[0]["messages"][1]["content"]
        assert "BDI_TOTAL_PRE" in user_content
        assert "25" in user_content

    def test_convert_severity_from_bdi(self, adapter):
        raw = [
            {
                "Sexo": "Mujer",
                "Edad": "22",
                "BDI_TOTAL_PRE": "35",
                "BAI_TOTAL_PRE": "20",
                "BIS_TOTAL_PRE": "68",
                "_source_file": "training_noSession.csv",
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 1
        assert records[0]["messages"][2]["content"] == "severe"

    def test_convert_skips_empty(self, adapter):
        raw = [
            {
                "Sexo": "",
                "Edad": "",
                "BDI_TOTAL_PRE": "",
                "BAI_TOTAL_PRE": "",
                "BIS_TOTAL_PRE": "",
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
        _make_repo(adapter, sample_csv_rows_raw)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "ml_bpd"
