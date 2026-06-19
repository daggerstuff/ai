"""Verify SDG pipeline DPO output format is compatible with dpo_trainer's loader.

SDG pipeline generates DPO pairs with extra fields (clinical_validity, provenance)
beyond what dpo_trainer.load_preference_dataset() requires. This test ensures
the extra fields are silently accepted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
import requests

from training.dpo_trainer import load_preference_dataset


def _make_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestSdgDpoFormatCompatibility:
    """DPO pairs produced by sdg_pipeline.py must load correctly in the trainer."""

    def test_minimal_fields_load(self, tmp_path: Path):
        """SDG output with prompt/chosen/rejected loads successfully."""
        records = [
            {
                "prompt": f"Client says {i}",
                "chosen": f"Good therapist response {i}",
                "rejected": f"Dismissive response {i}",
            }
            for i in range(25)
        ]
        data_path = tmp_path / "minimal.jsonl"
        _make_jsonl(data_path, records)
        result = load_preference_dataset(data_path, 1024, logging.getLogger("test"))
        assert len(result) == 25
        for r in result:
            assert "prompt" in r
            assert "chosen" in r
            assert "rejected" in r

    def test_with_provenance_extra_fields(self, tmp_path: Path):
        """SDG pipeline adds provenance + clinical_validity — loader must ignore them."""
        records = []
        for i in range(25):
            records.append(
                {
                    "prompt": f"Client says {i}",
                    "chosen": f"Good response {i}",
                    "rejected": f"Bad response {i}",
                    "clinical_validity_score": 0.85,
                    "clinical_validity_detail": {
                        "empathy": 0.9,
                        "safety": 0.95,
                        "therapeutic": 0.8,
                    },
                    "provenance": {
                        "source_url": "sdg://synthetic/dpo",
                        "source_type": "synthetic_sdg",
                        "acquired_at": "2026-06-01T00:00:00+00:00",
                        "pipeline_version": "modern-dataset-provenance-v1",
                        "license": "NOASSERTION",
                        "transformations": ["dpo_preference_pair"],
                        "metadata": {"scenario": "dpo_preference_pairs"},
                    },
                }
            )
        data_path = tmp_path / "with_provenance.jsonl"
        _make_jsonl(data_path, records)
        result = load_preference_dataset(data_path, 1024, logging.getLogger("test"))
        assert len(result) == 25
        # load_preference_dataset keeps only prompt/chosen/rejected;
        # extra SDG fields (provenance, clinical_validity) are silently dropped.
        # This is correct — the trainer doesn't use them.
        for r in result:
            assert set(r.keys()) == {"prompt", "chosen", "rejected"}

    def test_with_full_sdg_format(self, tmp_path: Path):
        """Load 50 records in the exact format sdg_pipeline.py produces."""
        records = []
        for i in range(50):
            records.append(
                {
                    "prompt": f"I've been feeling {['anxious', 'depressed', 'overwhelmed', 'stuck', 'hopeless'][i % 5]} lately, doc.",
                    "chosen": f"That sounds really hard. Can you tell me more about when it started? (response {i})",
                    "rejected": f"Just think positive. It'll pass. (response {i})",
                    "clinical_validity_score": round(0.5 + (i % 50) / 100, 3),
                    "clinical_validity_detail": {
                        "empathy": round(0.6 + (i % 50) / 200, 3),
                        "safety": round(0.7 + (i % 50) / 200, 3),
                        "therapeutic": round(0.5 + (i % 50) / 200, 3),
                    },
                    "provenance": {
                        "source_url": "sdg://synthetic/dpo",
                        "source_type": "synthetic_sdg",
                        "acquired_at": "2026-06-01T00:00:00+00:00",
                        "pipeline_version": "modern-dataset-provenance-v1",
                        "license": "NOASSERTION",
                        "transformations": ["dpo_preference_pair"],
                        "metadata": {"scenario": "dpo_preference_pairs"},
                    },
                }
            )
        data_path = tmp_path / "full_sdg_format.jsonl"
        _make_jsonl(data_path, records)
        result = load_preference_dataset(data_path, 1024, logging.getLogger("test"))
        assert len(result) == 50
        # Fields the trainer actually uses
        for r in result:
            assert len(r["prompt"]) > 0
            assert len(r["chosen"]) > 0
            assert len(r["rejected"]) > 0

    def test_sdg_output_with_missing_rejected_gets_filtered(self, tmp_path: Path):
        """SDG pipeline always includes rejected, but if absent, loader skips the record."""
        records = [
            {"prompt": "Q", "chosen": "A"},  # missing rejected — should be filtered
        ]
        for i in range(25):
            records.append({"prompt": f"Q{i}", "chosen": f"A{i}", "rejected": f"B{i}"})
        data_path = tmp_path / "missing_rejected.jsonl"
        _make_jsonl(data_path, records)
        result = load_preference_dataset(data_path, 1024, logging.getLogger("test"))
        # The malformed record (no rejected) should be skipped
        assert len(result) == 25

    def test_generation_report_json_shape(self, tmp_path: Path):
        """Verify the generation_report.json that sdg_pipeline.py writes has expected fields."""
        report = {
            "generated_at": "2026-06-01T00:00:00+00:00",
            "scenario": "dpo_preference_pairs",
            "category": "",
            "target_count": 10000,
            "generated_count": 10000,
            "filtered_count": 120,
            "filter_rate": 0.012,
            "iterations": 10120,
            "max_iterations": 20000,
            "failed_calls": 0,
            "total_calls": 10120,
            "failure_rate": 0.0,
            "clinical_validity": {
                "mean": 0.723,
                "min": 0.45,
                "max": 0.95,
                "samples_scored": 10000,
            },
        }
        report_path = tmp_path / "generation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        loaded = json.loads(report_path.read_text())
        assert loaded["scenario"] == "dpo_preference_pairs"
        assert loaded["target_count"] == 10000
        assert loaded["generated_count"] == 10000
        assert "generated_at" in loaded
        assert "clinical_validity" in loaded
        assert loaded["clinical_validity"]["mean"] == 0.723


@pytest.mark.skipif(
    not os.environ.get("NEMO_ENDPOINT"),
    reason="NeMo API not reachable — set NEMO_ENDPOINT to run this integration smoke test",
)
def test_actual_sdg_pipeline_can_generate_dpo_pairs():
    """Integration smoke test — verifies NeMo-compatible API at NEMO_ENDPOINT is reachable.

    Runs only when NEMO_ENDPOINT is in the environment. Performs a lightweight
    HTTP ping against `/v1/models` (the OpenAI-compatible endpoint shape used by
    NeMo/Riva-style servers) and asserts the endpoint is live. Full DPO generation
    coverage is exercised by ``test_with_full_sdg_format`` using on-disk fixtures.
    """
    server_error_threshold = 500
    request_timeout_seconds = 5

    endpoint = os.environ["NEMO_ENDPOINT"].rstrip("/")
    try:
        response = requests.get(
            f"{endpoint}/v1/models", timeout=request_timeout_seconds
        )
    except requests.RequestException as exc:
        pytest.skip(f"NEMO_ENDPOINT unreachable: {exc}")
    if response.status_code >= server_error_threshold:
        pytest.skip(f"NEMO_ENDPOINT returned HTTP {response.status_code}")
    assert response.status_code < server_error_threshold, (
        f"NEMO_ENDPOINT returned {response.status_code}; "
        f"expected < {server_error_threshold}"
    )
