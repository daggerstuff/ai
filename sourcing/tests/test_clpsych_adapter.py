"""Tests for the CLPsych dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.sourcing.dataset_adapters.clpsych_adapter import CLPsychAdapter


@pytest.fixture
def adapter(tmp_path):
    return CLPsychAdapter("clpsych", tmp_path)


@pytest.fixture
def sample_posts_raw():
    """Raw JSON format (as read from file)."""
    return [
        {
            "user_id": "u1",
            "post_id": "p1",
            "text": "I can't take it anymore, I just want it to end.",
            "label": "suicide_risk_high",
            "task_year": "2024",
            "response": "High suicide risk detected based on lethal ideation language.",
        },
        {
            "user_id": "u2",
            "post_id": "p2",
            "text": "Feeling restless and can't focus on anything today.",
            "label": "adhd_symptoms_moderate",
            "task_year": "2026",
        },
        {
            "user_id": "u3",
            "post_id": "p3",
            "text": "",
            "label": "abcd_affect",
            "task_year": "2025",
        },
    ]


@pytest.fixture
def sample_posts():
    """Extract output format (same as raw for JSON adapters)."""
    return [
        {
            "user_id": "u1",
            "post_id": "p1",
            "text": "I can't take it anymore, I just want it to end.",
            "label": "suicide_risk_high",
            "task_year": "2024",
            "response": "High suicide risk detected based on lethal ideation language.",
            "_source_file": "task2024",
        },
        {
            "user_id": "u2",
            "post_id": "p2",
            "text": "Feeling restless and can't focus on anything today.",
            "label": "adhd_symptoms_moderate",
            "task_year": "2026",
            "_source_file": "task2026",
        },
    ]


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestCLPsychAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "clpsych.org" in readme.read_text(encoding="utf-8")

    def test_download_idempotent(self, adapter):
        adapter.download()
        first = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        adapter.download()
        second = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        assert first == second

    def test_extract_returns_posts(self, adapter, sample_posts_raw):
        _write_json(adapter._raw_dir / "task2024.json", sample_posts_raw)
        posts = adapter.extract()
        assert len(posts) == 3
        assert posts[0]["user_id"] == "u1"
        assert posts[0]["_source_file"] == "task2024"

    def test_convert_suicide_risk(self, adapter, sample_posts):
        records = adapter.convert_to_chatml(sample_posts)
        assert len(records) == 2  # empty text skipped
        rec = records[0]
        assert rec["task_type"] == "risk_assessment"
        assert rec["diagnostic_tag"] == "suicide_risk"
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"  # has response

    def test_convert_adhd(self, adapter, sample_posts):
        records = adapter.convert_to_chatml(sample_posts)
        adhd_rec = [r for r in records if r.get("clpsych_label", "").startswith("adhd")][0]
        assert adhd_rec["task_type"] == "symptom_classification"
        assert adhd_rec["diagnostic_tag"] == "adhd_symptoms"
        # Label used as assistant answer
        assert len(adhd_rec["messages"]) == 3

    def test_empty_text_skipped(self, adapter):
        raw = [{"user_id": "x", "post_id": "y", "text": "", "label": "abcd_affect"}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_posts):
        records = adapter.convert_to_chatml(sample_posts)
        assert records[0]["provenance"]["access_method"] == "request"
        assert records[0]["provenance"]["source_url"] == "https://clpsych.org/shared-task/"

    def test_full_run(self, adapter, sample_posts_raw):
        _write_json(adapter._raw_dir / "task2024.json", sample_posts_raw)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "clpsych"
