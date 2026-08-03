"""Tests for ``training.pal_persona_consistency_eval``.

PIX-4222 — Phase 5.2: persona-consistency evaluation test suite.

Covers: heuristic NLI prediction rules, ``score_example`` / ``score_pairs`` /
``score_records``, ``_extract_persona_response``, ``_aggregate``, JSONL reader,
and CLI. All tests use the deterministic heuristic backend — no model download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from training.pal_persona_consistency_eval import (
    ConsistencyExample,
    ConsistencyReport,
    _aggregate,
    _extract_persona_response,
    _HeuristicNli,
    _iter_jsonl,
    build_nli_backend,
    main,
    score_example,
    score_pairs,
    score_records,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _heuristic() -> _HeuristicNli:
    return _HeuristicNli()


# ---------------------------------------------------------------------------
# _HeuristicNli
# ---------------------------------------------------------------------------


class TestHeuristicNli:
    def test_predict_entailment_health_seeking_preference(self) -> None:
        """Persona mentioning a health-seeking preference echoed in the
        response ⇒ entailment."""
        backend = _heuristic()
        result = backend.predict(
            "The patient prefers traditional medicine for their care.",
            "I recommend traditional medicine for your condition.",
        )
        assert result == "entailment"

    def test_predict_entailment_location_echo(self) -> None:
        """Persona mentioning a location echoed in the response ⇒ entailment."""
        backend = _heuristic()
        result = backend.predict(
            "The patient lives in Hanoi.",
            "Based on your location in Hanoi, here are some options.",
        )
        assert result == "entailment"

    def test_predict_contradiction_low_literacy_plus_jargon(self) -> None:
        """Low-literacy persona + clinical jargon response ⇒ contradiction."""
        backend = _heuristic()
        result = backend.predict(
            "The patient has low health literacy.",
            "We recommend a differential diagnosis at a tertiary academic medical center.",
        )
        assert result == "contradiction"

    def test_predict_neutral_empty_premise(self) -> None:
        backend = _heuristic()
        assert backend.predict("", "some response") == "neutral"

    def test_predict_neutral_empty_hypothesis(self) -> None:
        backend = _heuristic()
        assert backend.predict("some persona", "") == "neutral"

    def test_predict_neutral_no_overlap(self) -> None:
        backend = _heuristic()
        assert (
            backend.predict(
                "The patient is 45 years old.",
                "Exercise is good for cardiovascular health.",
            )
            == "neutral"
        )

    def test_name_attribute_is_heuristic(self) -> None:
        assert _HeuristicNli.name == "heuristic"


# ---------------------------------------------------------------------------
# build_nli_backend
# ---------------------------------------------------------------------------


class TestBuildNliBackend:
    def test_force_heuristic_returns_heuristic(self) -> None:
        backend = build_nli_backend(force_heuristic=True)
        assert isinstance(backend, _HeuristicNli)
        assert backend.name == "heuristic"

    def test_crossencoder_fallback_on_exception(self) -> None:
        """When the CrossEncoder cannot load, fallback to heuristic."""
        with patch(
            "training.pal_persona_consistency_eval._CrossEncoderNli",
            side_effect=RuntimeError("model not found"),
        ):
            backend = build_nli_backend(force_heuristic=False)
        assert isinstance(backend, _HeuristicNli)

    def test_force_heuristic_skips_crossencoder(self) -> None:
        """force_heuristic=True must never attempt to load the CrossEncoder."""
        with patch(
            "training.pal_persona_consistency_eval._CrossEncoderNli",
            side_effect=AssertionError("should not be called"),
        ) as mock_ce:
            backend = build_nli_backend(force_heuristic=True)
        mock_ce.assert_not_called()
        assert isinstance(backend, _HeuristicNli)


# ---------------------------------------------------------------------------
# score_example
# ---------------------------------------------------------------------------


class TestScoreExample:
    def test_returns_consistency_example_with_correct_fields(self) -> None:
        backend = _heuristic()
        ex = score_example(
            "The patient prefers modern medicine.",
            "I suggest modern medicine for your treatment.",
            backend,
        )
        assert isinstance(ex, ConsistencyExample)
        assert ex.persona == "The patient prefers modern medicine."
        assert ex.response == "I suggest modern medicine for your treatment."
        assert ex.label == "entailment"
        assert ex.score == 1
        assert ex.backend == "heuristic"

    def test_raises_on_empty_persona(self) -> None:
        backend = _heuristic()
        with pytest.raises(ValueError, match="persona must be a non-empty string"):
            score_example("", "a response", backend)

    def test_raises_on_empty_response(self) -> None:
        backend = _heuristic()
        with pytest.raises(ValueError, match="response must be a non-empty string"):
            score_example("a persona", "   ", backend)

    def test_raises_on_unknown_label_from_backend(self) -> None:
        backend = _heuristic()
        with (
            patch.object(backend, "predict", return_value="unknown_label"),
            pytest.raises(ValueError, match="backend returned unknown label"),
        ):
            score_example("persona", "response", backend)


# ---------------------------------------------------------------------------
# score_pairs
# ---------------------------------------------------------------------------


class TestScorePairs:
    def test_scores_multiple_pairs_returns_report(self) -> None:
        backend = _heuristic()
        pairs = [
            ("The patient prefers modern medicine.", "I recommend modern medicine."),
            ("The patient has low health literacy.", "Consider a differential diagnosis."),
            ("The patient is calm.", "That is good to hear."),
        ]
        report = score_pairs(pairs, backend)
        assert isinstance(report, ConsistencyReport)
        assert report.n == 3
        assert len(report.examples) == 3

    def test_empty_iterable_returns_empty_report(self) -> None:
        backend = _heuristic()
        report = score_pairs([], backend)
        assert report.n == 0
        assert report.c_score == 0.0
        assert report.backend == "unknown"

    def test_report_has_correct_c_score(self) -> None:
        backend = _heuristic()
        pairs = [
            ("The patient prefers modern medicine.", "I recommend modern medicine."),  # entailment → +1
            ("The patient is calm.", "That is good to hear."),  # neutral → 0
            ("The patient has low health literacy.", "Consider a differential diagnosis."),  # contradiction → -1
        ]
        report = score_pairs(pairs, backend)
        assert report.c_score == pytest.approx(0.0)
        assert report.entail_rate == pytest.approx(1 / 3)
        assert report.neutral_rate == pytest.approx(1 / 3)
        assert report.contradict_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# score_records
# ---------------------------------------------------------------------------


class TestScoreRecords:
    def test_flat_records_with_persona_response(self) -> None:
        backend = _heuristic()
        records = [
            {"persona": "The patient prefers modern medicine.", "response": "I recommend modern medicine."},
            {"persona": "The patient is calm.", "response": "That is good to hear."},
        ]
        report = score_records(records, backend)
        assert report.n == 2
        assert report.backend == "heuristic"

    def test_sft_records_with_messages_and_metadata(self) -> None:
        backend = _heuristic()
        records = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "I recommend modern medicine."},
                ],
                "metadata": {"persona_string": "The patient prefers modern medicine."},
            },
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "That is good to hear."},
                ],
                "metadata": {"persona_string": "The patient is calm."},
            },
        ]
        report = score_records(records, backend)
        assert report.n == 2
        assert report.examples[0].persona == "The patient prefers modern medicine."
        assert report.examples[0].response == "I recommend modern medicine."

    def test_mixed_flat_and_sft_records(self) -> None:
        backend = _heuristic()
        records = [
            {"persona": "The patient prefers modern medicine.", "response": "I recommend modern medicine."},
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "That is good to hear."},
                ],
                "metadata": {"persona_string": "The patient is calm."},
            },
        ]
        report = score_records(records, backend)
        assert report.n == 2

    def test_raises_on_missing_persona_in_record(self) -> None:
        backend = _heuristic()
        records = [{"response": "some response"}]
        with pytest.raises(ValueError, match="record missing persona"):
            score_records(records, backend)


# ---------------------------------------------------------------------------
# _extract_persona_response
# ---------------------------------------------------------------------------


class TestExtractPersonaResponse:
    def test_flat_record(self) -> None:
        persona, response = _extract_persona_response({"persona": "A patient persona", "response": "A response"})
        assert persona == "A patient persona"
        assert response == "A response"

    def test_sft_record_extracts_last_assistant_turn(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "first user message"},
                {"role": "assistant", "content": "first assistant response"},
                {"role": "user", "content": "second user message"},
                {"role": "assistant", "content": "second assistant response"},
            ],
            "metadata": {"persona_string": "SFT persona"},
        }
        persona, response = _extract_persona_response(record)
        assert persona == "SFT persona"
        assert response == "second assistant response"

    def test_sft_record_no_assistant_turn_raises(self) -> None:
        record = {
            "messages": [{"role": "user", "content": "no assistant here"}],
            "metadata": {"persona_string": "SFT persona"},
        }
        with pytest.raises(ValueError, match="record missing response"):
            _extract_persona_response(record)

    def test_empty_persona_raises(self) -> None:
        record = {"persona": "  ", "response": "a response"}
        with pytest.raises(ValueError, match="record missing persona"):
            _extract_persona_response(record)


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_empty_list_returns_zero_report(self) -> None:
        report = _aggregate([])
        assert report.n == 0
        assert report.c_score == 0.0
        assert report.entail_rate == 0.0
        assert report.neutral_rate == 0.0
        assert report.contradict_rate == 0.0
        assert report.backend == "unknown"

    def test_all_entailment_c_score_one(self) -> None:
        examples = [
            ConsistencyExample(persona="p", response="r", label="entailment", score=1, backend="heuristic"),
            ConsistencyExample(persona="p", response="r", label="entailment", score=1, backend="heuristic"),
        ]
        report = _aggregate(examples)
        assert report.c_score == 1.0
        assert report.entail_rate == 1.0
        assert report.neutral_rate == 0.0
        assert report.contradict_rate == 0.0
        assert report.backend == "heuristic"

    def test_mixed_labels_correct_rates_and_backend(self) -> None:
        examples = [
            ConsistencyExample(persona="p", response="r", label="entailment", score=1, backend="heuristic"),
            ConsistencyExample(persona="p", response="r", label="neutral", score=0, backend="heuristic"),
            ConsistencyExample(persona="p", response="r", label="contradiction", score=-1, backend="heuristic"),
        ]
        report = _aggregate(examples)
        assert report.n == 3
        assert report.c_score == pytest.approx(0.0)
        assert report.entail_rate == pytest.approx(1 / 3)
        assert report.neutral_rate == pytest.approx(1 / 3)
        assert report.contradict_rate == pytest.approx(1 / 3)
        assert report.backend == "heuristic"
        assert len(report.examples) == 3


# ---------------------------------------------------------------------------
# _iter_jsonl
# ---------------------------------------------------------------------------


class TestIterJsonl:
    def test_reads_valid_jsonl_skips_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"a": 1}) + "\n\n\n" + json.dumps({"b": 2}) + "\n",
            encoding="utf-8",
        )
        records = list(_iter_jsonl(path))
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps({"good": True}) + "\n{bad json}\n" + json.dumps({"also_good": True}) + "\n",
            encoding="utf-8",
        )
        records = list(_iter_jsonl(path))
        assert len(records) == 2
        assert records[0] == {"good": True}
        assert records[1] == {"also_good": True}


# ---------------------------------------------------------------------------
# main / CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_missing_input_returns_exit_1(self, tmp_path: Path) -> None:
        rc = main([str(tmp_path / "nonexistent.jsonl"), "--force-heuristic"])
        assert rc == 1

    def test_force_heuristic_writes_output_and_exits_0(self, tmp_path: Path) -> None:
        data_path = _write_jsonl(
            tmp_path / "input.jsonl",
            [
                {"persona": "The patient prefers modern medicine.", "response": "I recommend modern medicine."},
                {"persona": "The patient is calm.", "response": "That is good to hear."},
            ],
        )
        out_path = tmp_path / "report.json"
        rc = main(
            [
                str(data_path),
                "--force-heuristic",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.exists()
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["n"] == 2
        assert report["backend"] == "heuristic"
        assert "c_score" in report
        assert "entail_rate" in report
        assert "neutral_rate" in report
        assert "contradict_rate" in report
