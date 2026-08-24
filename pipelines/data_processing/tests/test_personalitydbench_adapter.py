"""Tests for the PersonalityDBench dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.personalitydbench_adapter import PersonalityDBenchAdapter


@pytest.fixture
def adapter(tmp_path):
    return PersonalityDBenchAdapter("personalitydbench", tmp_path)


@pytest.fixture
def sample_entries_raw():
    """Raw JSON format."""
    return [
        {
            "type": "prisma",
            "text": "My emotions spiral out of control constantly.",
            "dsm_criteria": "Borderline PD: affective instability, impulsivity",
            "pd_label": "borderline",
        },
        {
            "type": "steering",
            "persona": "A 30-year-old with grandiose self-image",
            "prompt": "Describe your typical day.",
            "response": "I am superior to everyone around me.",
            "pd_label": "narcissistic",
        },
        {
            "type": "prisma",
            "text": "",
            "dsm_criteria": "",
            "pd_label": "antisocial",
        },
    ]


@pytest.fixture
def sample_entries():
    """Extract output format."""
    return [
        {
            "type": "prisma",
            "text": "My emotions spiral out of control constantly.",
            "dsm_criteria": "Borderline PD: affective instability, impulsivity",
            "pd_label": "borderline",
            "_source_file": "prisma_data",
        },
        {
            "type": "steering",
            "persona": "A 30-year-old with grandiose self-image",
            "prompt": "Describe your typical day.",
            "response": "I am superior to everyone around me.",
            "pd_label": "narcissistic",
            "_source_file": "steering_data",
        },
    ]


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestPersonalityDBenchAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "aclanthology.org" in readme.read_text(encoding="utf-8")

    def test_download_idempotent(self, adapter):
        adapter.download()
        first = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        adapter.download()
        second = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        assert first == second

    def test_extract_returns_entries(self, adapter, sample_entries_raw):
        _write_json(adapter._raw_dir / "prisma_data.json", sample_entries_raw)
        entries = adapter.extract()
        assert len(entries) == 3
        assert entries[0]["type"] == "prisma"

    def test_convert_prisma(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        assert len(records) == 2
        prisma = records[0]
        assert prisma["task_type"] == "symptom_classification"
        assert prisma["diagnostic_tag"] == "borderline_personality_disorder"
        assert prisma["component"] == "prisma"
        assert prisma["clinical_reviewed"] is True

    def test_convert_steering(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        steering = [r for r in records if r["component"] == "steering"][0]
        assert steering["task_type"] == "therapy_response_generation"
        assert steering["diagnostic_tag"] == "narcissistic_personality_disorder"
        assert steering["messages"][-1]["role"] == "assistant"

    def test_empty_text_skipped(self, adapter):
        raw = [{"type": "prisma", "text": "", "pd_label": "borderline"}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_entries):
        records = adapter.convert_to_chatml(sample_entries)
        assert records[0]["provenance"]["source_url"] == "https://aclanthology.org/2026.acl-long.1395/"
        assert records[0]["provenance"]["access_method"] == "request"

    def test_full_run(self, adapter, sample_entries_raw):
        _write_json(adapter._raw_dir / "data.json", sample_entries_raw)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "personalitydbench"
