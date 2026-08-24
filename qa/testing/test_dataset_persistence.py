"""Tests for dataset persistence provenance storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.inference.deployment import persistence


def _record() -> dict[str, Any]:
    return {
        "instruction": "Use this transcript excerpt as source material.",
        "output": "A real transcript excerpt about trauma recovery.",
        "provenance": {
            "source_path": "training/youtube_transcripts/Test/video.txt",
            "source_type": "youtube",
            "license": "NOASSERTION",
        },
    }


def test_store_training_records_requires_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    store = persistence.DatasetPersistence()

    with pytest.raises(ValueError, match="provenance"):
        store.store_training_records("youtube", [{"instruction": "missing"}], local_dir=tmp_path)


def test_store_training_records_writes_local_audit_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    store = persistence.DatasetPersistence()

    result = store.store_training_records("youtube", [_record()], version="test", local_dir=tmp_path)

    assert result["mongo_enabled"] is False
    assert result["records_stored"] == 1
    local_path = Path(result["local_path"])
    assert local_path.exists()
    written = json.loads(local_path.read_text(encoding="utf-8").strip())
    assert written["dataset"] == "youtube"
    assert written["provenance"]["source_type"] == "youtube"
    assert written["record"]["output"].startswith("A real transcript excerpt")


def test_store_training_records_upserts_to_mongodb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAdmin:
        def command(self, _command: str) -> None:
            return None

    class FakeCollection:
        def __init__(self) -> None:
            self.upserts: list[dict[str, Any]] = []

        def update_one(self, query: dict[str, Any], update: dict[str, Any], *, upsert: bool) -> None:
            self.upserts.append({"query": query, "update": update, "upsert": upsert})

    class FakeDb:
        def __init__(self) -> None:
            self.collections: dict[str, FakeCollection] = {}

        def __getitem__(self, name: str) -> FakeCollection:
            return self.collections.setdefault(name, FakeCollection())

    class FakeClient:
        last_db: FakeDb | None = None

        def __init__(self, _uri: str, **_kwargs: Any) -> None:
            self.admin = FakeAdmin()
            self.db = FakeDb()
            FakeClient.last_db = self.db

        def __getitem__(self, _name: str) -> FakeDb:
            return self.db

    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setattr(persistence, "MongoClient", FakeClient)

    store = persistence.DatasetPersistence()
    result = store.store_training_records("youtube", [_record()], version="test", local_dir=tmp_path)

    assert result["mongo_enabled"] is True
    assert FakeClient.last_db is not None
    upserts = FakeClient.last_db["training_records"].upserts
    assert len(upserts) == 1
    assert upserts[0]["upsert"] is True
    assert upserts[0]["update"]["$set"]["provenance"]["license"] == "NOASSERTION"
