"""Tests for the EMPATH dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.empath_adapter import EmpathAdapter


@pytest.fixture
def sample_personas():
    """34-persona analog: 2 personas (Mexican Spanish + US English)."""
    return [
        {
            "persona_id": "p-001",
            "name": "Maria",
            "language": "Mexican Spanish",
            "background": "32-year-old immigrant mother",
            "age_range": "30_39",
            "gender": "female",
            "cultural_context": "Mexican-American, bilingual household",
            "diagnostic_tag": "adjustment_disorder",
        },
        {
            "persona_id": "p-002",
            "name": "James",
            "language": "US English",
            "background": "45-year-old veteran",
            "age_range": "40_49",
            "gender": "male",
            "cultural_context": "US Midwest, VA care",
            "diagnostic_tag": "ptsd",
        },
    ]


@pytest.fixture
def sample_metrics():
    """19-metric analog: 3 metrics spanning 2 dimensions."""
    return [
        {"metric_id": "crisis_01", "name": "suicide_risk_screen", "dimension": "crisis_handling"},
        {"metric_id": "ther_01", "name": "reflective_listening", "dimension": "therapeutic_quality"},
        {"metric_id": "cult_01", "name": "cultural_humility", "dimension": "cultural_adaptation"},
    ]


@pytest.fixture
def sample_seeds():
    """140-seed analog: 3 seeds (2 persona-bearing, 1 metric-only)."""
    return [
        {
            "seed_id": "s-001",
            "persona_id": "p-001",
            "chief_complaint": "Siento que no puedo mas con mis hijos",
            "language": "Mexican Spanish",
            "cultural_context": "Familismo, marianismo expectations",
            "dialog": [
                {"speaker": "client", "utterance": "Estoy abrumada"},
                {"speaker": "counselor", "utterance": "Cuéntame más sobre lo que estás sintiendo"},
            ],
            "scores": {"crisis_01": 0.1, "ther_01": 0.9, "cult_01": 0.8},
        },
        {
            "seed_id": "s-002",
            "persona_id": "p-002",
            "chief_complaint": "I keep having flashbacks",
            "language": "US English",
            "cultural_context": "Military culture",
            "dialog": [
                {"speaker": "client", "utterance": "I can't sleep, the dreams won't stop"},
                {
                    "speaker": "counselor",
                    "utterance": "That sounds incredibly hard. Let's work on grounding techniques.",
                },
            ],
        },
        {
            "seed_id": "s-003",
            "chief_complaint": "Patient expresses passive suicidal ideation",
            "language": "US English",
            "scores": {"crisis_01": 0.95, "ther_01": 0.4},
        },
    ]


@pytest.fixture
def adapter(tmp_path):
    """Create EmpathAdapter with isolated tmp output dir."""
    return EmpathAdapter("empath", tmp_path)


def _populate_raw(adapter: EmpathAdapter, *, personas, seeds, metrics) -> None:
    adapter._raw_dir.mkdir(parents=True, exist_ok=True)
    (adapter._raw_dir / "personas.json").write_text(json.dumps(personas), encoding="utf-8")
    (adapter._raw_dir / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
    (adapter._raw_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class TestEmpathAdapter:
    """Tests for EmpathAdapter."""

    def test_download_skips_when_files_present(self, adapter, sample_personas, sample_seeds, sample_metrics):
        """download() is a no-op when raw files already exist."""
        _populate_raw(adapter, personas=sample_personas, seeds=sample_seeds, metrics=sample_metrics)
        with patch("urllib.request.urlretrieve") as mock_dl:
            adapter.download()
            mock_dl.assert_not_called()

    def test_download_silently_fails_on_unreachable(self, adapter):
        """download() swallows unreachable upstream errors (manual placement path)."""
        with patch("urllib.request.urlretrieve", side_effect=Exception("net error")):
            adapter.download()  # should not raise

    def test_extract_joins_persona_with_seeds(self, adapter, sample_personas, sample_seeds, sample_metrics):
        """extract() returns seeds joined with their persona via persona_id."""
        _populate_raw(adapter, personas=sample_personas, seeds=sample_seeds, metrics=sample_metrics)
        raw = adapter.extract()
        assert len(raw) == 3
        assert raw[0]["_persona"]["persona_id"] == "p-001"
        assert raw[1]["_persona"]["name"] == "James"
        assert raw[2]["_persona"] is None  # metric-only seed
        assert isinstance(raw[0]["_metrics"], list)

    def test_convert_basic_persona_record(self, adapter, sample_personas, sample_seeds, sample_metrics):
        """Persona-bearing seeds become therapy_response_generation records."""
        raw = adapter.extract() if False else None  # noqa - placeholder skipped
        # Build raw directly to isolate conversion logic.
        raw = [
            {
                **seed,
                "_persona": next((p for p in sample_personas if p["persona_id"] == seed.get("persona_id")), None),
                "_metrics": sample_metrics,
            }
            for seed in sample_seeds
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 3

        rec0 = records[0]
        assert rec0["source"] == "empath"
        assert rec0["task_type"] == "therapy_response_generation"
        assert rec0["messages"][0]["role"] == "system"
        assert "Maria" in rec0["messages"][0]["content"]
        assert "Familismo" in rec0["messages"][0]["content"]
        assert rec0["messages"][1]["role"] == "user"
        assert rec0["messages"][2]["role"] == "assistant"
        assert rec0["persona_id"] == "p-001"
        assert rec0["language"] == "Mexican Spanish"
        assert rec0["metric_scores"] == {"crisis_01": 0.1, "ther_01": 0.9, "cult_01": 0.8}
        assert "age_30_39" in rec0["demographic_tags"]
        assert "gender_female" in rec0["demographic_tags"]

    def test_metric_only_seed_tagged_empathy_scoring(self, adapter, sample_personas, sample_seeds, sample_metrics):
        """Seeds without persona are tagged empathy_scoring."""
        raw = [
            {
                **seed,
                "_persona": next((p for p in sample_personas if p["persona_id"] == seed.get("persona_id")), None),
                "_metrics": sample_metrics,
            }
            for seed in sample_seeds
        ]
        records = adapter.convert_to_chatml(raw)
        metric_only = next(
            r
            for r in records
            if r["seed_id"] == "s-003"
            or r.get("persona_id") is None
            and r["messages"][0]["content"].startswith("Evaluation-only")
        )
        assert metric_only["task_type"] == "empathy_scoring"
        assert metric_only["persona_id"] is None
        assert "Evaluation-only record" in metric_only["messages"][0]["content"]

    def test_empty_seed_skipped(self, adapter):
        """Empty seeds (no complaint, no dialog) are skipped."""
        raw = [{"seed_id": "empty", "_persona": None, "_metrics": []}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_personas, sample_seeds, sample_metrics):
        """Each record has a provenance block pointing at arXiv."""
        raw = [
            {
                **seed,
                "_persona": next((p for p in sample_personas if p["persona_id"] == seed.get("persona_id")), None),
                "_metrics": sample_metrics,
            }
            for seed in sample_seeds
        ]
        records = adapter.convert_to_chatml(raw)
        prov = records[0]["provenance"]
        assert prov["source_url"] == "https://arxiv.org/abs/2606.30256"
        assert prov["original_format"] == "json"
        assert prov["access_method"] == "request"

    def test_full_run(self, adapter, sample_personas, sample_seeds, sample_metrics, tmp_path):
        """Full run() produces JSONL output with valid records."""
        _populate_raw(adapter, personas=sample_personas, seeds=sample_seeds, metrics=sample_metrics)
        with patch("urllib.request.urlretrieve"):
            output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["source"] == "empath"
        assert first["task_type"] in {"therapy_response_generation", "empathy_scoring"}

    def test_factory_registration(self, tmp_path):
        """Adapter is registered under lowercase name in factory."""
        from ai.sourcing.dataset_adapters.adapter_factory import get_adapter, list_available_adapters

        assert "empath" in list_available_adapters()
        adapter = get_adapter("EMPATH", tmp_path)
        assert isinstance(adapter, EmpathAdapter)
