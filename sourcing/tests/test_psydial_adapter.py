"""Tests for the PsyDial privacy-preserving counseling dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.psydial_adapter import PsyDialAdapter


@pytest.fixture
def sample_metadata():
    """2,382-dialogue analog: 3 metadata entries."""
    return [
        {"dialogue_id": "d-001", "chief_complaint": "Anxiety about job loss", "language": "zh"},
        {"dialogue_id": "d-002", "chief_complaint": "Grief after parent's death", "language": "zh"},
        {"dialogue_id": "d-003", "chief_complaint": "Relationship conflict", "language": "en"},
    ]


@pytest.fixture
def sample_dialogues():
    """Long-term counseling dialogues (avg 37.8 turns analog: short samples)."""
    return [
        {
            "dialogue_id": "d-001",
            "language": "zh",
            "dialog": [
                {"speaker": "client", "utterance": "Me siento ansioso por perder mi trabajo"},
                {"speaker": "counselor", "utterance": "Cuéntame qué aspectos específicos te generan más preocupación"},
                {"speaker": "client", "utterance": "Todo. No sé cómo mantener a mi familia."},
                {
                    "speaker": "counselor",
                    "utterance": "Es natural sentir ese peso. Vamos a explorar lo que está bajo tu control.",
                },
            ],
        },
        {
            "dialogue_id": "d-002",
            "language": "zh",
            "dialog": [
                {"speaker": "client", "utterance": "Mi madre murió la semana pasada"},
                {"speaker": "counselor", "utterance": "Siento mucho tu pérdida. ¿Cómo estás procesando esto?"},
            ],
        },
        {
            "dialogue_id": "d-003",
            "language": "en",
            "dialog": [
                {"speaker": "client", "utterance": "My partner and I keep fighting"},
                {"speaker": "counselor", "utterance": "Can you tell me what the conflicts are usually about?"},
            ],
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    """Create PsyDialAdapter with isolated tmp output dir."""
    return PsyDialAdapter("psydial", tmp_path)


def _populate_raw(adapter, *, dialogues, metadata) -> None:
    adapter._raw_dir.mkdir(parents=True, exist_ok=True)
    (adapter._raw_dir / "dialogues.json").write_text(json.dumps(dialogues), encoding="utf-8")
    (adapter._raw_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class TestPsyDialAdapter:
    """Tests for PsyDialAdapter."""

    def test_download_skips_when_files_present(self, adapter, sample_dialogues, sample_metadata):
        """download() is a no-op when raw files exist."""
        _populate_raw(adapter, dialogues=sample_dialogues, metadata=sample_metadata)
        with patch("urllib.request.urlretrieve") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_download_silently_fails_on_unreachable(self, adapter):
        """download() swallows upstream errors (manual placement path)."""
        with patch("urllib.request.urlretrieve", side_effect=Exception("net error")):
            adapter.download()

    def test_extract_joins_dialogue_with_metadata(self, adapter, sample_dialogues, sample_metadata):
        """extract() joins each dialogue with its metadata via dialogue_id."""
        _populate_raw(adapter, dialogues=sample_dialogues, metadata=sample_metadata)
        raw = adapter.extract()
        assert len(raw) == 3
        assert raw[0]["_metadata"]["chief_complaint"] == "Anxiety about job loss"
        assert raw[1]["_metadata"]["language"] == "zh"
        assert raw[2]["_metadata"]["chief_complaint"] == "Relationship conflict"

    def test_convert_basic_dialogue(self, adapter, sample_dialogues, sample_metadata):
        """Dialogues become therapy_response_generation ChatML records."""
        raw = [
            {
                **d,
                "_metadata": next((m for m in sample_metadata if m["dialogue_id"] == d["dialogue_id"]), None),
            }
            for d in sample_dialogues
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 3

        rec0 = records[0]
        assert rec0["source"] == "psydial"
        assert rec0["task_type"] == "therapy_response_generation"
        assert rec0["privacy_preserving"] is True
        assert rec0["messages"][0]["role"] == "system"
        assert "RMRR" in rec0["messages"][0]["content"]
        assert "Mask" in rec0["messages"][0]["content"]
        assert "chief complaint: anxiety about job loss" in rec0["messages"][0]["content"].lower()
        assert rec0["messages"][1]["role"] == "user"
        assert rec0["messages"][2]["role"] == "assistant"
        assert rec0["chief_complaint"] == "Anxiety about job loss"
        assert rec0["language"] == "zh"
        assert rec0["turn_count"] == 4
        assert rec0["dialogue_id"] == "d-001"

    def test_system_prompt_includes_rmrr_note(self, adapter, sample_dialogues, sample_metadata):
        """System prompt carries the RMRR methodology note."""
        raw = [
            {
                **d,
                "_metadata": next((m for m in sample_metadata if m["dialogue_id"] == d["dialogue_id"]), None),
            }
            for d in sample_dialogues
        ]
        records = adapter.convert_to_chatml(raw)
        for rec in records:
            assert "RMRR" in rec["rmrr_methodology"]
            assert "Reconstruct" in rec["rmrr_methodology"]
            assert "RMRR" in rec["messages"][0]["content"]

    def test_empty_dialog_skipped(self, adapter):
        """Dialogues with empty dialog lists are skipped."""
        raw = [{"dialogue_id": "empty", "_metadata": None, "language": "zh"}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_dialogues, sample_metadata):
        """Each record has provenance pointing at the ACL paper."""
        raw = [
            {
                **d,
                "_metadata": next((m for m in sample_metadata if m["dialogue_id"] == d["dialogue_id"]), None),
            }
            for d in sample_dialogues
        ]
        records = adapter.convert_to_chatml(raw)
        prov = records[0]["provenance"]
        assert prov["source_url"] == "https://aclanthology.org/2025.acl-long.1049"
        assert prov["original_format"] == "json"

    def test_full_run(self, adapter, sample_dialogues, sample_metadata, tmp_path):
        """Full run() writes JSONL with valid records."""
        _populate_raw(adapter, dialogues=sample_dialogues, metadata=sample_metadata)
        with patch("urllib.request.urlretrieve"):
            output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["source"] == "psydial"
        assert first["task_type"] == "therapy_response_generation"

    def test_factory_registration(self, tmp_path):
        """Adapter registered under lowercase name in factory."""
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter, list_available_adapters

        assert "psydial" in list_available_adapters()
        adapter = get_adapter("PSYDIAL", tmp_path)
        assert isinstance(adapter, PsyDialAdapter)
