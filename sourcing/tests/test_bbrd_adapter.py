"""Tests for the BBRD adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.bbrd_adapter import BBRDAdapter

_CSV_FIELDS = ["user_id", "post_text", "label", "timestamp"]


@pytest.fixture
def adapter(tmp_path):
    return BBRDAdapter("bbrd", tmp_path)


@pytest.fixture
def sample_csv_rows_raw():
    return [
        {"user_id": "user1", "post_text": "I can't take it anymore", "label": "suicidality", "timestamp": "2020-01-15"},
        {"user_id": "user2", "post_text": "I relapsed and cut again", "label": "self_harm", "timestamp": "2021-03-20"},
        {
            "user_id": "user3",
            "post_text": "Therapy is helping slowly",
            "label": "therapy_behavior",
            "timestamp": "2022-06-10",
        },
    ]


@pytest.fixture
def sample_records(sample_csv_rows_raw):
    """Extract output format (lowercase keys + _source_file)."""
    return [{**{k.lower(): v for k, v in r.items()}, "_source_file": "bbrd_posts"} for r in sample_csv_rows_raw]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestBBRDAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_download_idempotent(self, adapter):
        adapter.download()
        adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_returns_records(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "bbrd_posts.csv", sample_csv_rows_raw)
        records = adapter.extract()
        assert len(records) == 3
        assert records[0]["user_id"] == "user1"
        assert records[0]["_source_file"] == "bbrd_posts"

    def test_convert_basic(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert len(records) == 3
        rec = records[0]
        assert rec["source"] == "bbrd"
        assert rec["task_type"] == "symptom_classification"
        assert rec["diagnostic_tag"] == "suicidal_ideation"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["linguistic_style"] == "informal"
        assert rec["clinical_reviewed"] is True

    def test_convert_maps_labels(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["diagnostic_tag"] == "suicidal_ideation"
        assert records[1]["diagnostic_tag"] == "self_harm"
        assert records[2]["diagnostic_tag"] == "therapy_behavior"

    def test_convert_skips_empty_text(self, adapter):
        raw = [{"user_id": "u", "post_text": "", "label": "suicidality", "_source_file": "f"}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_records):
        records = adapter.convert_to_chatml(sample_records)
        assert records[0]["provenance"]["access_method"] == "request"
        assert "lancaster" in records[0]["provenance"]["source_url"]

    def test_full_run(self, adapter, sample_csv_rows_raw):
        _write_csv(adapter._raw_dir / "bbrd_posts.csv", sample_csv_rows_raw)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["source"] == "bbrd"
