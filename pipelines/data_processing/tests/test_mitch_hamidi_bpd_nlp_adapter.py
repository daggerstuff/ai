"""Tests for the mitch_hamidi_bpd_nlp adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.mitch_hamidi_bpd_nlp_adapter import MitchHamidiBpdNlpAdapter


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_COMMENT_FIELDS = ["Unnamed: 0", "Comment", "Author", "Post"]
_SEARCH_FIELDS = [
    "Unnamed: 0",
    "Title",
    "Post Text",
    "Post Creation Time",
    "Content Type",
    "Post Score",
    "Number of Comments",
    "Post Author",
]


def _make_comment(comment: str = "I feel you.", post: str = "I'm struggling") -> dict:
    return {"Unnamed: 0": 0, "Comment": comment, "Author": "user1", "Post": post}


def _make_search(title: str = "Need help", body: str = "I don't know what to do") -> dict:
    return {
        "Unnamed: 0": 0,
        "Title": title,
        "Post Text": body,
        "Post Creation Time": "2024-01-01",
        "Content Type": "text",
        "Post Score": 5,
        "Number of Comments": 3,
        "Post Author": "user2",
    }


@pytest.fixture
def adapter(tmp_path):
    return MitchHamidiBpdNlpAdapter("mitch_hamidi_bpd_nlp", tmp_path)


def _populate_raw(adapter, comments=None, search=None):
    target = adapter._raw_dir / "Scraping_Reddit_for_Data"
    if comments is None:
        comments = [_make_comment()]
    if search is None:
        search = [_make_search()]
    _write_csv(target / "diagnosed_comments.csv", comments, _COMMENT_FIELDS)
    _write_csv(target / "diagnosed_search.csv", search, _SEARCH_FIELDS)


class TestMitchHamidiAdapter:
    def test_download_skips_when_files_present(self, adapter, monkeypatch):
        _populate_raw(adapter)
        called = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: called.append(True))
        adapter.download()
        assert not called

    def test_extract_reads_csv(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        assert len(records) == 2
        assert records[0]["_is_comment"] is True
        assert records[1]["_is_comment"] is False

    def test_convert_comment(self, adapter):
        _populate_raw(adapter, comments=[_make_comment("Great advice", "I need therapy")])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert len(chatml) >= 1
        comment_rec = [r for r in chatml if r["diagnostic_tag"] == "diagnosed" and r["linguistic_style"] == "informal"][
            0
        ]
        msgs = comment_rec["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Great advice"

    def test_convert_search(self, adapter):
        _populate_raw(adapter, search=[_make_search("Help me", "I'm lost")])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        search_recs = [r for r in chatml if not r.get("_is_comment", True)]
        assert len(chatml) >= 1

    def test_empty_comment_skipped(self, adapter):
        _populate_raw(adapter, comments=[_make_comment("", "post")])
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        comment_recs = [r for r in chatml if "Comment" not in str(r.get("_is_comment"))]
        # Only search record should remain
        assert len(chatml) == 1

    def test_provenance(self, adapter):
        _populate_raw(adapter)
        records = adapter.extract()
        chatml = adapter.convert_to_chatml(records)
        assert chatml[0]["provenance"]["access_method"] == "s3"
        assert chatml[0]["provenance"]["original_format"] == "csv"

    def test_full_run(self, adapter, monkeypatch):
        _populate_raw(adapter)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        output = adapter.run()
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1

    def test_factory_registration(self):
        from ai.pipelines.data_processing.dataset_adapters.adapter_factory import get_adapter

        a = get_adapter("mitch_hamidi_bpd_nlp", "/tmp/test_mh")
        assert isinstance(a, MitchHamidiBpdNlpAdapter)
