"""Tests for the Reddit Mental Health Posts dataset adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.pipelines.data_processing.dataset_adapters.reddit_mental_health_posts_adapter import (
    RedditMentalHealthPostsAdapter,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _sample_csv_rows(subreddit: str = "depression") -> list[dict[str, Any]]:
    return [
        {
            "author": "user1",
            "body": "I feel so down lately.",
            "created_utc": "2024-01-01T00:00:00Z",
            "id": "abc123",
            "num_comments": "5",
            "score": "10",
            "subreddit": subreddit,
            "title": "Feeling lost",
            "upvote_ratio": "0.9",
            "url": "https://reddit.com/r/depression/comments/abc123",
        },
        {
            "author": "user2",
            "body": "",
            "created_utc": "2024-01-02T00:00:00Z",
            "id": "def456",
            "num_comments": "2",
            "score": "3",
            "subreddit": subreddit,
            "title": "Just a title today",
            "upvote_ratio": "0.8",
            "url": "https://reddit.com/r/depression/comments/def456",
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    return RedditMentalHealthPostsAdapter("reddit_mental_health_posts", tmp_path)


class TestRedditMentalHealthPostsAdapter:
    def test_download_skips_when_files_exist(self, adapter):
        for fname in ["adhd.csv", "aspergers.csv", "depression.csv", "ocd.csv", "ptsd.csv"]:
            (adapter._raw_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            (adapter._raw_dir / fname).write_text("col\nval")
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_download_writes_readme_on_failure(self, adapter):
        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("network error")):
            adapter.download()
        readme = adapter._raw_dir / "README.txt"
        assert readme.exists()
        assert "Download failed" in readme.read_text()

    def test_extract_reads_csvs(self, adapter):
        _write_csv(adapter._raw_dir / "depression.csv", _sample_csv_rows("depression"))
        _write_csv(adapter._raw_dir / "adhd.csv", _sample_csv_rows("adhd"))
        result = adapter.extract()
        assert len(result) == 4

    def test_extract_empty_when_no_files(self, adapter):
        assert adapter.extract() == []

    def test_convert_basic(self, adapter):
        rows = _sample_csv_rows("depression")
        records = adapter.convert_to_chatml(rows)
        assert len(records) == 2
        msgs = records[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "I feel so down lately."
        assert msgs[2]["role"] == "assistant"

    def test_convert_body_fallback_to_title(self, adapter):
        rows = _sample_csv_rows("depression")
        records = adapter.convert_to_chatml(rows)
        # Second row has empty body, should use title
        assert records[1]["messages"][1]["content"] == "Just a title today"

    def test_convert_diagnosis_mapping(self, adapter):
        rows = _sample_csv_rows("ocd")
        records = adapter.convert_to_chatml(rows)
        assert records[0]["diagnostic_tag"] == "obsessive_compulsive_disorder"
        assert records[0]["label"] == "ocd"

    def test_convert_skips_empty_both(self, adapter):
        rows = [{"body": "", "title": "", "subreddit": "depression", "author": "x"}]
        records = adapter.convert_to_chatml(rows)
        assert len(records) == 0

    def test_provenance(self, adapter):
        rows = _sample_csv_rows("depression")
        records = adapter.convert_to_chatml(rows)
        prov = records[0]["provenance"]
        assert prov["access_method"] == "huggingface"
        assert prov["original_format"] == "csv"

    def test_full_run(self, adapter):
        _write_csv(adapter._raw_dir / "depression.csv", _sample_csv_rows("depression"))
        # Pre-create other CSVs to prevent real download
        for fname in ["adhd.csv", "aspergers.csv", "ocd.csv", "ptsd.csv"]:
            (adapter._raw_dir / fname).write_text("author,body\n")
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["source"] == "reddit_mental_health_posts"
        assert record["task_type"] == "symptom_classification"

    def test_factory_registration(self):
        from ai.pipelines.data_processing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("reddit_mental_health_posts", "/tmp/test_rmh")
        assert isinstance(a, RedditMentalHealthPostsAdapter)
