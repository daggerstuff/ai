"""Tests for the eRisk dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.pipelines.data_processing.dataset_adapters.erisk_adapter import ERISKAdapter


@pytest.fixture
def adapter(tmp_path):
    return ERISKAdapter("erisk", tmp_path)


@pytest.fixture
def sample_users_raw():
    """Raw JSON format."""
    return [
        {
            "user_id": "u1",
            "task": "depression",
            "label": "moderate",
            "posts": [
                {"date": "2024-01-01", "text": "I feel so empty inside."},
                {"date": "2024-01-05", "text": "Nothing matters anymore."},
            ],
        },
        {
            "user_id": "u2",
            "task": "self_harm",
            "label": "high_risk",
            "posts": [
                {"date": "2024-02-01", "text": "I can't stop the urges."},
            ],
        },
        {
            "user_id": "u3",
            "task": "anorexia",
            "posts": [],
        },
    ]


@pytest.fixture
def sample_users():
    """Extract output format."""
    return [
        {
            "user_id": "u1",
            "task": "depression",
            "label": "moderate",
            "posts": [
                {"date": "2024-01-01", "text": "I feel so empty inside."},
                {"date": "2024-01-05", "text": "Nothing matters anymore."},
            ],
            "_source_file": "depression_task",
        },
        {
            "user_id": "u2",
            "task": "self_harm",
            "label": "high_risk",
            "posts": [
                {"date": "2024-02-01", "text": "I can't stop the urges."},
            ],
            "_source_file": "selfharm_task",
        },
    ]


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestERISKAdapter:
    def test_download_creates_readme(self, adapter):
        adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "erisk.irlab.org" in readme.read_text(encoding="utf-8")

    def test_download_idempotent(self, adapter):
        adapter.download()
        first = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        adapter.download()
        second = (adapter._raw_dir / "README.txt").read_text(encoding="utf-8")
        assert first == second

    def test_extract_returns_users(self, adapter, sample_users_raw):
        _write_json(adapter._raw_dir / "depression_task.json", sample_users_raw)
        users = adapter.extract()
        assert len(users) == 3
        assert users[0]["user_id"] == "u1"

    def test_convert_depression(self, adapter, sample_users):
        records = adapter.convert_to_chatml(sample_users)
        assert len(records) == 2
        dep = records[0]
        assert dep["task_type"] == "severity_estimation"
        assert dep["diagnostic_tag"] == "depression"
        assert dep["messages"][0]["role"] == "system"
        assert dep["num_posts"] == 2
        assert dep["messages"][-1]["role"] == "assistant"

    def test_convert_self_harm(self, adapter, sample_users):
        records = adapter.convert_to_chatml(sample_users)
        sh = [r for r in records if r["erisk_task"] == "self_harm"][0]
        assert sh["task_type"] == "risk_assessment"
        assert sh["diagnostic_tag"] == "self_harm"

    def test_empty_posts_skipped(self, adapter):
        raw = [{"user_id": "x", "task": "anorexia", "posts": []}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_users):
        records = adapter.convert_to_chatml(sample_users)
        assert records[0]["provenance"]["access_method"] == "request"
        assert records[0]["provenance"]["source_url"] == "https://erisk.irlab.org/"
        assert records[0]["erde_metric"] == "latency-aware"

    def test_full_run(self, adapter, sample_users_raw):
        _write_json(adapter._raw_dir / "task.json", sample_users_raw)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "erisk"
