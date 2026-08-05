"""Tests for ``training.pal_persona_consistency_eval`` (PAL §C.score, Phase 5.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.pal_persona_consistency_eval import (
    ConsistencyExample,
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


@pytest.fixture
def heuristic() -> _HeuristicNli:
    return _HeuristicNli()


@pytest.fixture
def backend(heuristic: _HeuristicNli):
    return heuristic


class TestHeuristicPredict:
    def test_low_literacy_plus_jargon_is_contradiction(self, heuristic: _HeuristicNli) -> None:
        persona = "Patient with low health literacy in Hanoi."
        response = "Refer to a tertiary academic medical center for expedited neuroimaging."
        assert heuristic.predict(persona, response) == "contradiction"

    def test_low_literacy_alias_triggers_contradiction(self, heuristic: _HeuristicNli) -> None:
        persona = "low literacy adult"
        response = "multi-disciplinary differential diagnosis per clinical guidelines"
        assert heuristic.predict(persona, response) == "contradiction"

    def test_preference_traditional_medicine_entailment(self, heuristic: _HeuristicNli) -> None:
        persona = "Prefers traditional medicine for chronic pain."
        response = "We can explore traditional medicine options together."
        assert heuristic.predict(persona, response) == "entailment"

    def test_preference_modern_medicine_entailment(self, heuristic: _HeuristicNli) -> None:
        persona = "Trusts modern medicine for acute issues."
        response = "Modern medicine offers fast diagnostics here."
        assert heuristic.predict(persona, response) == "entailment"

    def test_preference_integrated_medicine_entailment(self, heuristic: _HeuristicNli) -> None:
        persona = "Uses integrated medicine approaches."
        response = "Integrated medicine can balance both worlds."
        assert heuristic.predict(persona, response) == "entailment"

    def test_location_hanoi_entailment(self, heuristic: _HeuristicNli) -> None:
        persona = "Lives in hanoi with mild anxiety."
        response = "Clinics in hanoi can support follow-up."
        assert heuristic.predict(persona, response) == "entailment"

    def test_location_hcmc_entailment(self, heuristic: _HeuristicNli) -> None:
        persona = "Resident of hcmc seeking counseling."
        response = "Therapists in hcmc are available."
        assert heuristic.predict(persona, response) == "entailment"

    def test_unrelated_defaults_neutral(self, heuristic: _HeuristicNli) -> None:
        persona = "Enjoys gardening on weekends."
        response = "Try breathing exercises twice daily."
        assert heuristic.predict(persona, response) == "neutral"

    def test_empty_premise_is_neutral(self, heuristic: _HeuristicNli) -> None:
        assert heuristic.predict("", "some response") == "neutral"

    def test_empty_hypothesis_is_neutral(self, heuristic: _HeuristicNli) -> None:
        assert heuristic.predict("some persona", "") == "neutral"


class TestScoreExample:
    def test_scores_entailment_pair(self, backend) -> None:
        ex = score_example("integrated medicine fan", "integrated medicine helps", backend)
        assert ex.label == "entailment"
        assert ex.score == 1
        assert ex.backend == "heuristic"

    def test_rejects_empty_persona(self, backend) -> None:
        with pytest.raises(ValueError, match="persona"):
            score_example("", "response", backend)

    def test_rejects_empty_response(self, backend) -> None:
        with pytest.raises(ValueError, match="response"):
            score_example("persona", "   ", backend)

    def test_preserves_persona_and_response_text(self, backend) -> None:
        persona = "hanoi resident"
        response = "visit hanoi clinic"
        ex = score_example(persona, response, backend)
        assert ex.persona == persona
        assert ex.response == response


class TestScorePairs:
    def test_scores_multiple_pairs(self, backend) -> None:
        pairs = [
            ("integrated medicine", "integrated medicine"),
            ("hanoi", "live in hanoi"),
        ]
        report = score_pairs(pairs, backend)
        assert report.n == 2
        assert all(e.label == "entailment" for e in report.examples)

    def test_empty_pairs_returns_zero_n(self, backend) -> None:
        report = score_pairs([], backend)
        assert report.n == 0
        assert report.c_score == 0.0

    def test_mixed_labels_aggregate_rates(self, backend) -> None:
        pairs = [
            ("integrated medicine", "integrated medicine"),
            ("low health literacy", "tertiary academic medical center"),
            ("gardening", "breathing exercises"),
        ]
        report = score_pairs(pairs, backend)
        assert report.n == 3
        assert report.entail_rate == 1 / 3
        assert report.contradict_rate == 1 / 3
        assert report.neutral_rate == 1 / 3


class TestScoreRecords:
    def test_flat_persona_response_schema(self, backend) -> None:
        records = [{"persona": "hanoi", "response": "clinic in hanoi"}]
        report = score_records(records, backend)
        assert report.n == 1
        assert report.examples[0].label == "entailment"

    def test_sft_messages_and_metadata_schema(self, backend) -> None:
        records = [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "integrated medicine may help"},
                ],
                "metadata": {"persona_string": "prefers integrated medicine"},
            }
        ]
        report = score_records(records, backend)
        assert report.n == 1
        assert report.examples[0].label == "entailment"

    def test_missing_persona_raises(self, backend) -> None:
        with pytest.raises(ValueError, match="persona"):
            score_records([{"response": "only response"}], backend)

    def test_missing_response_raises(self, backend) -> None:
        with pytest.raises(ValueError, match="response"):
            score_records([{"persona": "only persona"}], backend)


class TestExtractPersonaResponse:
    def test_flat_record(self) -> None:
        persona, response = _extract_persona_response({"persona": "p", "response": "r"})
        assert persona == "p"
        assert response == "r"

    def test_sft_metadata_and_assistant_turn(self) -> None:
        persona, response = _extract_persona_response(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "assistant says hanoi"},
                ],
                "metadata": {"persona_string": "lives in hanoi"},
            }
        )
        assert persona == "lives in hanoi"
        assert response == "assistant says hanoi"

    def test_raises_when_persona_missing(self) -> None:
        with pytest.raises(ValueError, match="persona"):
            _extract_persona_response({"messages": [{"role": "assistant", "content": "x"}]})

    def test_raises_when_response_missing(self) -> None:
        with pytest.raises(ValueError, match="response"):
            _extract_persona_response({"persona": "p", "messages": [{"role": "user", "content": "u"}]})


class TestAggregate:
    def test_single_entailment(self) -> None:
        examples = [
            ConsistencyExample("p", "r", "entailment", 1, "heuristic"),
        ]
        report = _aggregate(examples)
        assert report.n == 1
        assert report.c_score == 1.0
        assert report.entail_rate == 1.0

    def test_mixed_label_rates(self) -> None:
        examples = [
            ConsistencyExample("p1", "r1", "entailment", 1, "heuristic"),
            ConsistencyExample("p2", "r2", "neutral", 0, "heuristic"),
            ConsistencyExample("p3", "r3", "contradiction", -1, "heuristic"),
        ]
        report = _aggregate(examples)
        assert report.c_score == 0.0
        assert report.entail_rate == 1 / 3
        assert report.neutral_rate == 1 / 3
        assert report.contradict_rate == 1 / 3

    def test_empty_examples(self) -> None:
        report = _aggregate([])
        assert report.n == 0
        assert report.backend == "unknown"

    def test_c_score_mean(self) -> None:
        examples = [
            ConsistencyExample("p1", "r1", "entailment", 1, "heuristic"),
            ConsistencyExample("p2", "r2", "contradiction", -1, "heuristic"),
        ]
        report = _aggregate(examples)
        assert report.c_score == 0.0


class TestIterJsonl:
    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"persona": "hanoi", "response": "hanoi clinic"}\n\n{"persona": "p", "response": "r"}\n',
            encoding="utf-8",
        )
        records = list(_iter_jsonl(path))
        assert len(records) == 2

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"persona": "hanoi", "response": "hanoi"}\nnot-json\n{"persona": "p", "response": "r"}\n',
            encoding="utf-8",
        )
        records = list(_iter_jsonl(path))
        assert len(records) == 2


class TestBuildNliBackend:
    def test_force_heuristic_returns_heuristic(self) -> None:
        backend = build_nli_backend(force_heuristic=True)
        assert backend.name == "heuristic"


class TestCli:
    def _write_jsonl(self, path: Path) -> Path:
        path.write_text(
            '{"persona": "integrated medicine", "response": "integrated medicine"}\n',
            encoding="utf-8",
        )
        return path

    def test_writes_report_file(self, tmp_path: Path) -> None:
        inp = self._write_jsonl(tmp_path / "in.jsonl")
        out = tmp_path / "report.json"
        code = main([str(inp), "--force-heuristic", "--output", str(out)])
        assert code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["n"] == 1
        assert "c_score" in data

    def test_missing_input_exit_code_1(self, tmp_path: Path) -> None:
        code = main([str(tmp_path / "missing.jsonl"), "--force-heuristic"])
        assert code == 1

    def test_limit_flag(self, tmp_path: Path) -> None:
        inp = tmp_path / "in.jsonl"
        inp.write_text(
            '{"persona": "hanoi", "response": "hanoi"}\n{"persona": "hcmc", "response": "hcmc"}\n',
            encoding="utf-8",
        )
        out = tmp_path / "report.json"
        code = main([str(inp), "--force-heuristic", "--limit", "1", "--output", str(out)])
        assert code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["n"] == 1

    def test_model_flag_accepted(self, tmp_path: Path) -> None:
        inp = self._write_jsonl(tmp_path / "in.jsonl")
        code = main(
            [
                str(inp),
                "--force-heuristic",
                "--model",
                "cross-encoder/nli-deberta-v3-base",
            ]
        )
        assert code == 0

    def test_force_heuristic_exit_code_0(self, tmp_path: Path) -> None:
        inp = self._write_jsonl(tmp_path / "in.jsonl")
        code = main([str(inp), "--force-heuristic"])
        assert code == 0

    def test_heuristic_without_force_exit_code_2(self, tmp_path: Path) -> None:
        inp = self._write_jsonl(tmp_path / "in.jsonl")
        code = main([str(inp), "--force-heuristic"])
        # Simulate heuristic fallback without explicit flag by reusing forced backend
        # then verifying contract: when backend is heuristic and flag absent → exit 2.
        # build_nli_backend(force_heuristic=True) always returns heuristic; main checks flag.
        assert code == 0
        # Exercise the real exit-2 path: force heuristic backend but omit --force-heuristic
        # by patching build to return heuristic while CLI flag is off.
        from unittest.mock import patch

        with patch(
            "training.pal_persona_consistency_eval.build_nli_backend",
            return_value=build_nli_backend(force_heuristic=True),
        ):
            code2 = main([str(inp)])
        assert code2 == 2
