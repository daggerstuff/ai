"""Tests for the Clinical Red Teaming (Steenstra) dataset adapter.

Tests use CSV fixtures matching the Kaggle dataset format
(steeni/ai-psychotherapy-eval).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.clinical_redteam_adapter import ClinicalRedTeamAdapter


@pytest.fixture
def sample_personas():
    """15-DSM-5 AUD persona analog: 2 personas."""
    return [
        {
            "patient_id": "1",
            "name": "Liam",
            "subtype_name": "Young Adult",
            "ad_subtype_description": "Young adult with early onset AUD",
            "age_onset": "Age: 22; AUD Onset: 20",
            "aud_severity_symptoms": "Moderate",
            "comorbid_psychiatric_disorders": "None",
            "stage_of_change": "Contemplation",
            "persona_description": "Young adult binge drinker",
        },
        {
            "patient_id": "2",
            "name": "Sara",
            "subtype_name": "Chronic Severe",
            "ad_subtype_description": "Chronic severe AUD with comorbid depression",
            "age_onset": "Age: 45; AUD Onset: 25",
            "aud_severity_symptoms": "Severe",
            "comorbid_psychiatric_disorders": "MDD",
            "stage_of_change": "Precontemplation",
            "persona_description": "Chronic severe drinker",
        },
    ]


@pytest.fixture
def sample_pairings():
    """180 pairing analog: 2 pairings."""
    return [
        {"pairing_id": "1", "therapist_id": "therapist_char", "patient_id": "1"},
        {"pairing_id": "2", "therapist_id": "therapist_gpt", "patient_id": "2"},
    ]


@pytest.fixture
def sample_conversations():
    """27K-turn analog: 6 turns across 2 sessions."""
    return [
        # Session 1, pairing 1
        {
            "pairing_id": "1",
            "session_id": "1",
            "turn": "1",
            "speaker": "Patient",
            "message": "I'd like to talk about my drinking.",
        },
        {
            "pairing_id": "1",
            "session_id": "1",
            "turn": "2",
            "speaker": "Therapist",
            "message": "Of course. Can you tell me more?",
        },
        {"pairing_id": "1", "session_id": "1", "turn": "3", "speaker": "Patient", "message": "I drink every weekend."},
        # Session 2, pairing 1 (different session)
        {"pairing_id": "1", "session_id": "2", "turn": "1", "speaker": "Patient", "message": "I tried to cut down."},
        {"pairing_id": "1", "session_id": "2", "turn": "2", "speaker": "Therapist", "message": "That's a great step."},
        # Session 1, pairing 2 (single turn)
        {
            "pairing_id": "2",
            "session_id": "1",
            "turn": "1",
            "speaker": "Patient",
            "message": "I don't want to be here.",
        },
    ]


@pytest.fixture
def sample_crisis():
    """Crisis detection labels for some turns."""
    return [
        {"pairing_id": "1", "session_id": "1", "turn": "1", "reasoning": "No risk", "classification": "No Crisis"},
        {"pairing_id": "1", "session_id": "1", "turn": "2", "reasoning": "No risk", "classification": "No Crisis"},
        {"pairing_id": "1", "session_id": "1", "turn": "3", "reasoning": "No risk", "classification": "No Crisis"},
        {"pairing_id": "1", "session_id": "2", "turn": "1", "reasoning": "No risk", "classification": "No Crisis"},
        {"pairing_id": "1", "session_id": "2", "turn": "2", "reasoning": "No risk", "classification": "No Crisis"},
        {"pairing_id": "2", "session_id": "1", "turn": "1", "reasoning": "Self-harm risk", "classification": "Crisis"},
    ]


@pytest.fixture
def adapter(tmp_path):
    """Create ClinicalRedTeamAdapter with isolated tmp output dir."""
    return ClinicalRedTeamAdapter("clinical_redteam", tmp_path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _populate_raw(
    adapter,
    *,
    personas,
    pairings,
    conversations,
    crisis,
) -> None:
    """Write CSV fixtures to adapter's raw dir."""
    adapter._raw_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(adapter._raw_dir / "patient_personas.csv", personas)
    _write_csv(adapter._raw_dir / "pairings.csv", pairings)
    _write_csv(adapter._raw_dir / "conversations.csv", conversations)
    _write_csv(adapter._raw_dir / "eval_crisis_detection.csv", crisis)


class TestClinicalRedTeamAdapter:
    """Tests for ClinicalRedTeamAdapter."""

    def test_download_skips_when_files_present(
        self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis
    ):
        """download() is a no-op when CSV files exist."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        with patch("urllib.request.urlopen") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_download_creates_readme_on_failure(self, adapter):
        """download() creates README.txt when Kaggle download fails."""
        with patch("urllib.request.urlopen", side_effect=Exception("net error")):
            adapter.download()
        assert (adapter._raw_dir / "README.txt").exists()

    def test_extract_groups_by_session(
        self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis
    ):
        """extract() groups turns by (pairing_id, session_id) and joins persona."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        # 3 sessions: (1,1), (1,2), (2,1)
        assert len(raw) == 3
        # Session (1,1) has 3 turns
        s11 = next(r for r in raw if r["pairing_id"] == "1" and r["session_id"] == "1")
        assert len(s11["turns"]) == 3
        assert s11["persona"]["name"] == "Liam"
        assert s11["therapist_id"] == "therapist_char"

    def test_convert_basic_session(
        self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis
    ):
        """Each session becomes an adversarial_safety ChatML record."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        records = adapter.convert_to_chatml(raw)
        # All 3 sessions produce records (single-speaker sessions get synthetic counterpart)
        assert len(records) == 3

        rec0 = records[0]
        assert rec0["source"] == "clinical_redteam"
        assert rec0["task_type"] == "adversarial_safety"
        assert rec0["clinical_reviewed"] is True
        assert rec0["messages"][0]["role"] == "system"
        assert "DSM-5 AUD" in rec0["messages"][0]["content"]
        assert "Liam" in rec0["messages"][0]["content"]
        assert rec0["messages"][1]["role"] == "user"
        assert rec0["messages"][2]["role"] == "assistant"

    def test_crisis_flag_detection(
        self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis
    ):
        """has_crisis is True when any turn has a Crisis classification."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        records = adapter.convert_to_chatml(raw)
        # Session (2,1) has a crisis label
        crisis_rec = next(r for r in records if r["pairing_id"] == "2" and r["session_id"] == "1")
        assert crisis_rec["has_crisis"] is True
        # Session (1,1) does not
        normal_rec = next(r for r in records if r["pairing_id"] == "1" and r["session_id"] == "1")
        assert normal_rec["has_crisis"] is False

    def test_intensity_scores_extracted(
        self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis
    ):
        """Intensity scores from persona are extracted as metadata."""
        # Add intensity scores to persona
        personas = [{**p, "hopelessness_intensity": "3.5", "self_efficacy_intensity": "2.0"} for p in sample_personas]
        _populate_raw(
            adapter,
            personas=personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        records = adapter.convert_to_chatml(raw)
        assert records[0]["intensity_scores"]["hopelessness_intensity"] == 3.5
        assert records[0]["intensity_scores"]["self_efficacy_intensity"] == 2.0

    def test_demographic_tags(self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis):
        """demographic_tags include AUD subtype and stage of change."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        records = adapter.convert_to_chatml(raw)
        tags = records[0]["demographic_tags"]
        assert any("young_adult" in t for t in tags)
        assert any("stage_contemplation" in t for t in tags)

    def test_provenance_present(self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis):
        """Each record has provenance pointing at arXiv + Kaggle."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        raw = adapter.extract()
        records = adapter.convert_to_chatml(raw)
        prov = records[0]["provenance"]
        assert prov["source_url"] == "https://arxiv.org/abs/2602.19948"
        assert prov["access_method"] == "kaggle"
        assert prov["original_format"] == "csv"

    def test_full_run(self, adapter, sample_personas, sample_pairings, sample_conversations, sample_crisis):
        """Full run() writes JSONL with valid records."""
        _populate_raw(
            adapter,
            personas=sample_personas,
            pairings=sample_pairings,
            conversations=sample_conversations,
            crisis=sample_crisis,
        )
        with patch("urllib.request.urlopen"):
            output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["source"] == "clinical_redteam"
        assert first["task_type"] == "adversarial_safety"

    def test_factory_registration(self, tmp_path):
        """Adapter registered under lowercase name in factory."""
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter, list_available_adapters

        assert "clinical_redteam" in list_available_adapters()
        adapter = get_adapter("CLINICAL_REDTEAM", tmp_path)
        assert isinstance(adapter, ClinicalRedTeamAdapter)
