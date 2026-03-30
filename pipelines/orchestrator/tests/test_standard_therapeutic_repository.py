from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.pipelines.orchestrator.orchestration.standard_therapeutic_repository import (
    StandardTherapeuticRepository,
)


def test_standard_therapeutic_repository_loads_conversations_array_from_object(
    tmp_path: Path,
):
    payload = {
        "metadata": {"source": "example"},
        "conversations": [
            {"text": "one"},
            {"conversation": [{"role": "user", "content": "two"}]},
        ],
    }
    file_path = tmp_path / "training_dataset.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    records = StandardTherapeuticRepository().load_json_file(file_path)

    assert records == payload["conversations"]


def test_standard_therapeutic_repository_raises_when_conversations_array_is_missing(
    tmp_path: Path,
):
    file_path = tmp_path / "training_dataset.json"
    file_path.write_text(json.dumps({"metadata": {"source": "broken"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="conversations"):
        StandardTherapeuticRepository().load_json_file(file_path)


def test_standard_therapeutic_repository_loads_single_record_dict_root(
    tmp_path: Path,
):
    payload = {
        "text": "Therapeutic example",
        "metadata": {"source": "single-record"},
    }
    file_path = tmp_path / "training_dataset.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    records = StandardTherapeuticRepository().load_json_file(file_path)

    assert records == [payload]


def test_standard_therapeutic_repository_rejects_oversized_json_fallback(
    tmp_path: Path,
    monkeypatch,
):
    file_path = tmp_path / "training_dataset.json"
    file_path.write_text(json.dumps({"conversations": []}), encoding="utf-8")
    monkeypatch.setattr(
        StandardTherapeuticRepository,
        "_MAX_EAGER_JSON_BYTES",
        1,
    )

    with pytest.raises(ValueError, match="too large"):
        StandardTherapeuticRepository().load_json_file(file_path)
