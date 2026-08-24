"""
Comprehensive tests for the DiagnosisArena evaluation suite (PIX-3909).

Covers: types, benchmark IO, HeuristicJudge (tier + 4-dim + MCQ), LLMJudge,
error taxonomy, reporter (summarize/to_json/to_markdown/write_report),
leaderboard (SystemEvaluation, Leaderboard ranking, top_errors, write_leaderboard),
BenchmarkArtifactStore, runner (evaluate_case, run end-to-end, MCQ format),
openai_judge helpers (_score_for, _parse_judge_output, _normalize_dimension_scores,
_majority_tier, _majority_dimensions, inter_rater_agreement), and the
OpenAIBenchmarkPipeline orchestration.

Run: AI_DISABLE_SAFETY_ML_MODELS=1 uv run pytest ai/evals/test_diagnosis_arena.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.qa.validation.diagnosis_arena import (
    DIAGNOSTIC_DIMENSIONS,
    DIMENSION_WEIGHTS,
    ERROR_TAXONOMY,
    BenchmarkArtifactStore,
    BenchmarkSummary,
    CaseScore,
    ClinicalCase,
    Difficulty,
    DimensionScore,
    ErrorTaxonomy,
    GeneratedDiagnosis,
    HeuristicJudge,
    Judgment,
    JudgmentResult,
    Leaderboard,
    LLMJudge,
    ModelResponse,
    OpenAIBenchmarkPipeline,
    ResponseFormat,
    SystemEvaluation,
    TierScore,
    case_from_dict,
    classify_errors,
    evaluate_case,
    inter_rater_agreement,
    run,
    run_multi_system_benchmark,
    solve_case_for_system,
    summarize,
    to_json,
    to_markdown,
    top_errors,
    write_leaderboard,
    write_report,
)
from ai.qa.validation.diagnosis_arena.benchmark import DiagnosisArenaBenchmark
from ai.qa.validation.diagnosis_arena.openai_judge import (
    _majority_dimensions,
    _majority_tier,
    _normalize_dimension_scores,
    _parse_judge_output,
    _score_for,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str = "c1",
    difficulty: Difficulty = Difficulty.SIMPLE,
    final_diagnosis: str = "Streptococcal pharyngitis",
    differential_diagnoses: tuple[str, ...] = (
        "Streptococcal pharyngitis",
        "Viral pharyngitis",
        "Infectious mononucleosis",
    ),
    supporting_evidence: tuple[str, ...] = (
        "rapid strep positive",
        "tonsillar exudate",
        "tender anterior cervical LAD",
    ),
    key_differentiators: tuple[str, ...] = (
        "rapid strep positive",
        "exudate",
        "cervical adenopathy",
    ),
    mcq_options: tuple[str, ...] = (
        "Streptococcal pharyngitis",
        "Viral pharyngitis",
        "Infectious mononucleosis",
        "Peritonsillar abscess",
        "Diphtheria",
    ),
    presentation: str = "35-year-old with sore throat and fever.",
) -> ClinicalCase:
    return ClinicalCase(
        case_id=case_id,
        difficulty=difficulty,
        presentation=presentation,
        mcq_options=mcq_options,
        final_diagnosis=final_diagnosis,
        differential_diagnoses=differential_diagnoses,
        supporting_evidence=supporting_evidence,
        key_differentiators=key_differentiators,
    )


def _make_response(
    case_id: str = "c1",
    response_id: str = "r1",
    format: ResponseFormat = ResponseFormat.OPEN_ENDED,
    final_diagnosis: str = "streptococcal pharyngitis",
    differential_list: tuple[str, ...] = (
        "Streptococcal pharyngitis",
        "Viral pharyngitis",
    ),
    hypothesis_list: tuple[str, ...] = ("Streptococcal pharyngitis",),
    evidence_cited: tuple[str, ...] = ("rapid strep positive", "tonsillar exudate"),
    reasoning: str = "History consistent with bacterial pharyngitis; rapid strep positive confirms.",
    mcq_selected: str = "",
) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        case_id=case_id,
        format=format,
        final_diagnosis=final_diagnosis,
        differential_list=differential_list,
        hypothesis_list=hypothesis_list,
        evidence_cited=evidence_cited,
        reasoning=reasoning,
        mcq_selected=mcq_selected,
    )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestTypes:
    def test_tier_score_numeric_values(self) -> None:
        assert TierScore.IDENTICAL.numeric == 1.0
        assert TierScore.RELEVANT.numeric == 0.5
        assert TierScore.IRRELEVANT.numeric == 0.0

    def test_dimension_weights_sum_close_to_one(self) -> None:
        # Per the paper weights sum to 1.0
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9

    def test_diagnostic_dimensions_match_weights(self) -> None:
        assert set(DIAGNOSTIC_DIMENSIONS) == set(DIMENSION_WEIGHTS.keys())

    def test_judgment_tier_numeric_property(self) -> None:
        j = Judgment(
            response_id="r1",
            case_id="c1",
            tier=TierScore.IDENTICAL,
            dimensions=(),
        )
        assert j.tier_numeric == 1.0

    def test_judgment_aggregate_dimension_score_blank(self) -> None:
        j = Judgment(response_id="r", case_id="c", tier=TierScore.IDENTICAL)
        assert j.aggregate_dimension_score == 0.0

    def test_judgment_aggregate_dimension_score_weighted(self) -> None:
        dims = tuple(DimensionScore(name=n, score=1.0) for n in DIAGNOSTIC_DIMENSIONS)
        j = Judgment(response_id="r", case_id="c", tier=TierScore.IDENTICAL, dimensions=dims)
        assert abs(j.aggregate_dimension_score - 1.0) < 1e-9

    def test_judgment_result_tier_numeric(self) -> None:
        j = JudgmentResult(response_id="r", case_id="c", tier=TierScore.RELEVANT)
        assert j.tier_numeric == 0.5
        assert j.aggregate_dimension_score == 0.0

    def test_generated_diagnosis_mirrors_model_response(self) -> None:
        g = GeneratedDiagnosis(
            response_id="r",
            case_id="c",
            format=ResponseFormat.OPEN_ENDED,
            final_diagnosis="dx",
        )
        assert g.final_diagnosis == "dx"

    def test_clinical_eq(self) -> None:
        c1 = _make_case()
        c2 = _make_case()
        assert c1 == c2


# ---------------------------------------------------------------------------
# Benchmark IO
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_case_from_dict_minimum(self) -> None:
        case = case_from_dict({"case_id": "x", "difficulty": "simple", "presentation": "p"})
        assert case.case_id == "x"
        assert case.difficulty is Difficulty.SIMPLE
        assert case.presentation == "p"

    def test_case_from_dict_missing_required(self) -> None:
        with pytest.raises(ValueError, match="missing fields"):
            case_from_dict({"case_id": "x"})

    def test_case_from_dict_invalid_difficulty(self) -> None:
        with pytest.raises(ValueError, match="impossible"):
            case_from_dict({"case_id": "x", "difficulty": "impossible", "presentation": "p"})

    def test_benchmark_round_trip_jsonl(self, tmp_path: Path) -> None:
        bm = DiagnosisArenaBenchmark([_make_case("c1"), _make_case("c2", difficulty=Difficulty.MODERATE)])
        jsonl = tmp_path / "cases.jsonl"
        bm.to_jsonl(jsonl)
        loaded = DiagnosisArenaBenchmark.from_jsonl(jsonl)
        assert len(loaded) == 2
        assert loaded.get("c2").difficulty is Difficulty.MODERATE

    def test_benchmark_get_unknown(self) -> None:
        bm = DiagnosisArenaBenchmark([_make_case("c1")])
        with pytest.raises(KeyError):
            bm.get("nope")

    def test_benchmark_by_difficulty(self) -> None:
        bm = DiagnosisArenaBenchmark(
            [
                _make_case("s", difficulty=Difficulty.SIMPLE),
                _make_case("m", difficulty=Difficulty.MODERATE),
                _make_case("x", difficulty=Difficulty.COMPLEX),
            ]
        )
        assert len(bm.by_difficulty(Difficulty.SIMPLE)) == 1
        assert len(bm.by_difficulty(Difficulty.COMPLEX)) == 1

    def test_benchmark_extend_add(self) -> None:
        bm = DiagnosisArenaBenchmark()
        bm.add(_make_case("a"))
        bm.extend([_make_case("b"), _make_case("c")])
        assert len(bm) == 3

    def test_benchmark_iter(self) -> None:
        bm = DiagnosisArenaBenchmark([_make_case("a"), _make_case("b")])
        ids = [c.case_id for c in bm]
        assert ids == ["a", "b"]

    def test_benchmark_from_json_array_file(self, tmp_path: Path) -> None:
        payload = [
            {
                "case_id": "j1",
                "difficulty": "simple",
                "presentation": "p",
                "final_diagnosis": "dx",
            }
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload))
        bm = DiagnosisArenaBenchmark.from_json(path)
        assert len(bm) == 1
        assert bm.get("j1").final_diagnosis == "dx"

    def test_benchmark_from_json_rejects_non_array(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        path.write_text(json.dumps({"a": 1}))
        with pytest.raises(ValueError, match="top-level JSON array"):
            DiagnosisArenaBenchmark.from_json(path)

    def test_benchmark_loads_fixtures(self) -> None:
        # Real fixture shipped with the package
        fixtures = Path(__file__).parent / "diagnosis_arena" / "fixtures" / "seed_cases.jsonl"
        if not fixtures.exists():
            pytest.skip("fixtures/seed_cases.jsonl not shipped")
        bm = DiagnosisArenaBenchmark.from_jsonl(fixtures)
        assert len(bm) >= 5
        first = next(iter(bm))
        assert isinstance(first, ClinicalCase)
        assert first.case_id == "seed-001"


# ---------------------------------------------------------------------------
# HeuristicJudge
# ---------------------------------------------------------------------------


class TestHeuristicJudge:
    def test_identical_tier(self) -> None:
        case = _make_case(final_diagnosis="Streptococcal pharyngitis")
        response = _make_response(final_diagnosis="Streptococcal pharyngitis")
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IDENTICAL

    def test_relevant_tier_token_overlap(self) -> None:
        case = _make_case(final_diagnosis="Streptococcal pharyngitis")
        response = _make_response(final_diagnosis="strep pharyngitis severe")
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.RELEVANT

    def test_irrelevant_tier(self) -> None:
        case = _make_case(final_diagnosis="Streptococcal pharyngitis")
        response = _make_response(final_diagnosis="atypical pneumonia")
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IRRELEVANT

    def test_empty_prediction_yields_irrelevant(self) -> None:
        case = _make_case()
        response = _make_response(final_diagnosis="")
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IRRELEVANT

    def test_mcq_correct_first_option(self) -> None:
        case = _make_case()
        response = _make_response(
            format=ResponseFormat.MCQ,
            final_diagnosis="",
            mcq_selected="Streptococcal pharyngitis",
        )
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IDENTICAL

    def test_mcq_wrong_option(self) -> None:
        case = _make_case()
        response = _make_response(
            format=ResponseFormat.MCQ,
            final_diagnosis="",
            mcq_selected="Viral pharyngitis",
        )
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IRRELEVANT

    def test_mcq_missing_option(self) -> None:
        case = _make_case()
        response = _make_response(
            format=ResponseFormat.MCQ,
            final_diagnosis="",
            mcq_selected="Not on the list",
        )
        j = HeuristicJudge().judge(case, response)
        assert j.tier is TierScore.IRRELEVANT

    def test_dimensions_returned_for_all_four_axes(self) -> None:
        case = _make_case()
        response = _make_response()
        j = HeuristicJudge().judge(case, response)
        names = {d.name for d in j.dimensions}
        assert names == set(DIAGNOSTIC_DIMENSIONS)
        assert len(j.dimensions) == 4

    def test_dimension_scores_in_unit_interval(self) -> None:
        case = _make_case()
        response = _make_response()
        j = HeuristicJudge().judge(case, response)
        for d in j.dimensions:
            assert 0.0 <= d.score <= 1.0

    def test_dimension_perfect_overlap_scores(self) -> None:
        case = _make_case(
            final_diagnosis="A",
            differential_diagnoses=("A", "B", "C"),
            supporting_evidence=("X", "Y"),
        )
        response = _make_response(
            final_diagnosis="A",
            hypothesis_list=("A", "B", "C"),
            differential_list=("A", "B", "C"),
            evidence_cited=("X", "Y"),
        )
        j = HeuristicJudge().judge(case, response)
        by_name = {d.name: d.score for d in j.dimensions}
        assert by_name["hypothesis_generation"] == 1.0
        assert by_name["evidence_interpretation"] == 1.0
        assert by_name["differential_diagnosis"] == 1.0
        assert by_name["final_diagnosis"] == 1.0  # tier = IDENTICAL

    def test_dimension_zero_when_gt_empty(self) -> None:
        case = _make_case(
            differential_diagnoses=(),
            supporting_evidence=(),
        )
        response = _make_response(hypothesis_list=("X",), differential_list=("Y",))
        j = HeuristicJudge().judge(case, response)
        by_name = {d.name: d.score for d in j.dimensions}
        assert by_name["hypothesis_generation"] == 0.0
        assert by_name["evidence_interpretation"] == 0.0
        assert by_name["differential_diagnosis"] == 0.0

    def test_judge_model_label(self) -> None:
        j = HeuristicJudge(judge_model="my-model")
        case = _make_case()
        response = _make_response()
        out = j.judge(case, response)
        assert out.judge_model == "my-model"

    def test_none_when_gt_and_pred_both_blank(self) -> None:
        case = _make_case(final_diagnosis="")
        response = _make_response(final_diagnosis="")
        j = HeuristicJudge().judge(case, response)
        # Spec says: if not gt and not pred -> RELEVANT
        assert j.tier is TierScore.RELEVANT


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_requires_scorer(self) -> None:
        with pytest.raises(ValueError, match="scorer callable required"):
            LLMJudge(scorer=None)  # type: ignore[arg-type]

    def test_passes_through_scorer_judgment(self) -> None:
        def scorer(case: ClinicalCase, response: ModelResponse) -> Judgment:
            return Judgment(
                response_id=response.response_id,
                case_id=case.case_id,
                tier=TierScore.IDENTICAL,
                dimensions=tuple(DimensionScore(name=n, score=1.0) for n in DIAGNOSTIC_DIMENSIONS),
                judge_model="test-llm",
            )

        j = LLMJudge(scorer=scorer, judge_model="test-llm")
        out = j.judge(_make_case(), _make_response())
        assert out.tier is TierScore.IDENTICAL
        assert out.judge_model == "test-llm"

    def test_rejects_unknown_dimension(self) -> None:
        def scorer(case: ClinicalCase, response: ModelResponse) -> Judgment:
            return Judgment(
                response_id=response.response_id,
                case_id=case.case_id,
                tier=TierScore.IDENTICAL,
                dimensions=(DimensionScore(name="not_a_dimension", score=1.0),),
            )

        j = LLMJudge(scorer=scorer)
        with pytest.raises(ValueError, match="unknown dimension"):
            j.judge(_make_case(), _make_response())


# ---------------------------------------------------------------------------
# Error Taxonomy
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    def test_taxonomy_tuple_is_sequential(self) -> None:
        assert isinstance(ERROR_TAXONOMY, tuple)
        assert len(ERROR_TAXONOMY) >= 6
        assert "irrelevant" in ERROR_TAXONOMY
        assert "premature_closure" in ERROR_TAXONOMY
        assert "anchoring_bias" in ERROR_TAXONOMY
        assert "confirmation_bias" in ERROR_TAXONOMY

    def test_error_taxonomy_class_is_tuple_alias(self) -> None:
        # ErrorTaxonomy subclasses tuple[str, ...] for backward-compat.
        assert issubclass(ErrorTaxonomy, tuple)

    def test_classify_irrelevant(self) -> None:
        case = _make_case()
        response = _make_response(
            final_diagnosis="",
            differential_list=(),
            hypothesis_list=(),
            evidence_cited=(),
        )
        errors = classify_errors(case, response, TierScore.IRRELEVANT)
        assert errors == ("irrelevant",)

    def test_classify_clean_response(self) -> None:
        case = _make_case()
        response = _make_response()  # good response with full diff + evidence
        j = HeuristicJudge().judge(case, response)
        errors = classify_errors(case, response, j.tier)
        assert "irrelevant" not in errors

    def test_classify_premature_closure(self) -> None:
        case = _make_case()
        response = _make_response(
            final_diagnosis="totally wrong diagnosis not in gt",
            differential_list=(),
        )
        j = HeuristicJudge().judge(case, response)
        errors = classify_errors(case, response, j.tier)
        assert "premature_closure" in errors

    def test_classify_anchoring_bias(self) -> None:
        case = _make_case()
        response = _make_response(
            final_diagnosis="unrelated dx",
            differential_list=("unrelated dx",),
        )
        errors = classify_errors(case, response, TierScore.IRRELEVANT)
        assert "anchoring_bias" in errors

    def test_classify_overconfidence(self) -> None:
        case = _make_case()
        response = _make_response(
            final_diagnosis="totally wrong",
            differential_list=(),
            reasoning="short reasoning",  # under 30 words
        )
        j = HeuristicJudge().judge(case, response)
        errors = classify_errors(case, response, j.tier)
        assert "overconfidence" in errors

    def test_classify_confirmation_bias(self) -> None:
        case = _make_case()
        response = _make_response(
            final_diagnosis="some final",
            evidence_cited=("totally unrelated evidence",),
        )
        j = HeuristicJudge().judge(case, response)
        errors = classify_errors(case, response, j.tier)
        assert "confirmation_bias" in errors

    def test_classify_availability_bias_complex(self) -> None:
        case = _make_case(difficulty=Difficulty.COMPLEX)
        response = _make_response(
            final_diagnosis="anxiety",
            differential_list=("anxiety", "depression"),
            hypothesis_list=("anxiety",),
        )
        errors = classify_errors(case, response, TierScore.IRRELEVANT)
        assert "availability_bias" in errors

    def test_classify_no_availability_on_simple(self) -> None:
        case = _make_case(difficulty=Difficulty.SIMPLE)
        response = _make_response(
            final_diagnosis="anxiety",
            differential_list=("anxiety",),
        )
        errors = classify_errors(case, response, TierScore.IRRELEVANT)
        # availability_bias only coerced on COMPLEX
        assert "availability_bias" not in errors


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _score_factory(
    case_id: str,
    difficulty: Difficulty = Difficulty.SIMPLE,
    format: ResponseFormat = ResponseFormat.OPEN_ENDED,
    tier: TierScore = TierScore.IDENTICAL,
    dims: dict[str, float] | None = None,
    errors: tuple[str, ...] = (),
) -> CaseScore:
    if dims is None:
        dims = dict.fromkeys(DIAGNOSTIC_DIMENSIONS, 1.0)
    return CaseScore(
        case_id=case_id,
        difficulty=difficulty,
        format=format,
        tier=tier,
        tier_numeric=tier.numeric,
        dimensions=dims,
        aggregate_dimension_score=sum(dims.get(n, 0.0) * DIMENSION_WEIGHTS[n] for n in DIAGNOSTIC_DIMENSIONS),
        error_taxonomy=errors,
    )


class TestReporter:
    def test_summarize_empty_scores(self) -> None:
        s = summarize("model", ResponseFormat.OPEN_ENDED, [])
        assert s.cases_evaluated == 0
        assert s.overall_accuracy == 0.0
        assert s.by_difficulty == {}
        assert s.by_dimension == {}
        assert s.error_distribution == {}
        assert s.timestamp

    def test_summarize_basic(self) -> None:
        scores = [
            _score_factory("c1", tier=TierScore.IDENTICAL),
            _score_factory("c2", tier=TierScore.IRRELEVANT, dims=dict.fromkeys(DIAGNOSTIC_DIMENSIONS, 0.0)),
        ]
        s = summarize("model", ResponseFormat.OPEN_ENDED, scores)
        assert s.cases_evaluated == 2
        assert s.overall_accuracy == 0.5
        assert s.by_difficulty["simple"] == 0.5
        assert set(s.by_dimension.keys()) == set(DIAGNOSTIC_DIMENSIONS)

    def test_summarize_error_distribution(self) -> None:
        scores = [
            _score_factory("c1", errors=("premature_closure",)),
            _score_factory("c2", errors=("premature_closure", "anchoring_bias")),
        ]
        s = summarize("m", ResponseFormat.OPEN_ENDED, scores)
        assert s.error_distribution == {"premature_closure": 2, "anchoring_bias": 1}

    def test_to_json_round_trip(self) -> None:
        s = summarize("m", ResponseFormat.OPEN_ENDED, [_score_factory("c1")])
        payload = json.loads(to_json(s))
        assert payload["model_label"] == "m"
        assert payload["format"] == "open_ended"
        assert payload["cases_evaluated"] == 1
        assert payload["overall_accuracy"] == 1.0

    def test_to_markdown_has_sections(self) -> None:
        s = summarize(
            "m",
            ResponseFormat.OPEN_ENDED,
            [
                _score_factory("c1", difficulty=Difficulty.SIMPLE, errors=("anchoring_bias",)),
                _score_factory("c2", difficulty=Difficulty.COMPLEX),
            ],
        )
        md = to_markdown(s)
        assert "# DiagnosisArena Report: m" in md
        assert "Accuracy by difficulty" in md
        assert "Scores by diagnostic dimension" in md
        assert "Diagnostic error distribution" in md
        assert "anchoring_bias" in md

    def test_to_markdown_skips_empty_sections(self) -> None:
        s = summarize("m", ResponseFormat.OPEN_ENDED, [])
        md = to_markdown(s)
        assert "Accuracy by difficulty" not in md

    def test_write_report(self, tmp_path: Path) -> None:
        s = summarize("m", ResponseFormat.OPEN_ENDED, [_score_factory("c1")])
        md, js = write_report(s, tmp_path, also_json=True)
        assert md.exists()
        assert js is not None
        assert js.exists()
        loaded = json.loads(js.read_text(encoding="utf-8"))
        assert loaded["model_label"] == "m"

    def test_write_report_markdown_only(self, tmp_path: Path) -> None:
        s = summarize("m", ResponseFormat.OPEN_ENDED, [_score_factory("c1")])
        md, js = write_report(s, tmp_path, also_json=False)
        assert md.exists()
        assert js is None


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _summary_factory(label: str, overall: float) -> BenchmarkSummary:
    return BenchmarkSummary(
        model_label=label,
        format=ResponseFormat.OPEN_ENDED,
        cases_evaluated=1,
        overall_accuracy=overall,
        by_difficulty={"simple": overall},
        by_dimension=dict.fromkeys(DIAGNOSTIC_DIMENSIONS, overall),
        error_distribution={},
        timestamp="2026-01-01T00:00:00Z",
    )


class TestLeaderboard:
    def test_rank_sorts_overall_accuracy_desc(self) -> None:
        entries = [
            SystemEvaluation(label="b", system="b", summary=_summary_factory("b", 0.7)),
            SystemEvaluation(label="a", system="a", summary=_summary_factory("a", 0.9)),
        ]
        lb = Leaderboard(entries)
        assert lb.rank("a") == 1
        assert lb.rank("b") == 2

    def test_rank_unknown_label(self) -> None:
        lb = Leaderboard([])
        assert lb.rank("ghost") == -1

    def test_to_json_contains_entries(self) -> None:
        entries = [
            SystemEvaluation(label="a", system="a", summary=_summary_factory("a", 0.9)),
            SystemEvaluation(label="b", system="b", summary=_summary_factory("b", 0.7)),
        ]
        lb = Leaderboard(entries)
        payload = json.loads(lb.to_json())
        assert "entries" in payload
        assert len(payload["entries"]) == 2
        assert payload["entries"][0]["rank"] == 1
        assert payload["entries"][0]["label"] == "a"

    def test_to_markdown_has_table_header(self) -> None:
        entries = [
            SystemEvaluation(label="a", system="a", summary=_summary_factory("a", 0.9)),
        ]
        md = Leaderboard(entries).to_markdown()
        assert "# DiagnosisArena Multi-System Leaderboard" in md
        assert "| Rank | Label | System |" in md

    def test_top_errors_aggregates(self) -> None:
        s1 = _summary_factory("a", 0.5)
        s1 = BenchmarkSummary(
            model_label="a",
            format=ResponseFormat.OPEN_ENDED,
            cases_evaluated=1,
            overall_accuracy=0.5,
            error_distribution={"premature_closure": 3, "anchoring_bias": 1},
            timestamp="t",
        )
        s2 = BenchmarkSummary(
            model_label="b",
            format=ResponseFormat.OPEN_ENDED,
            cases_evaluated=1,
            overall_accuracy=0.5,
            error_distribution={"premature_closure": 2, "confirmation_bias": 1},
            timestamp="t",
        )
        entries = [
            SystemEvaluation(label="a", system="a", summary=s1),
            SystemEvaluation(label="b", system="b", summary=s2),
        ]
        res = top_errors(entries, n=2)
        assert res["top_n"][0] == "premature_closure"
        assert res["counts"]["premature_closure"] == 5
        assert res["counts"]["anchoring_bias"] == 1

    def test_write_leaderboard(self, tmp_path: Path) -> None:
        entries = [
            SystemEvaluation(label="a", system="a", summary=_summary_factory("a", 0.9)),
            SystemEvaluation(label="b", system="b", summary=_summary_factory("b", 0.7)),
        ]
        lb = Leaderboard(entries)
        md, js = write_leaderboard(lb, tmp_path, also_json=True)
        assert md.exists()
        assert js is not None
        assert js.exists()
        # JSON validates
        json.loads(js.read_text(encoding="utf-8"))

    def test_write_leaderboard_markdown_only(self, tmp_path: Path) -> None:
        entries = [
            SystemEvaluation(label="a", system="a", summary=_summary_factory("a", 0.9)),
        ]
        lb = Leaderboard(entries)
        md, js = write_leaderboard(lb, tmp_path, also_json=False)
        assert md.exists()
        assert js is None


# ---------------------------------------------------------------------------
# BenchmarkArtifactStore
# ---------------------------------------------------------------------------


class TestArtifactStore:
    def test_default_root(self) -> None:
        store = BenchmarkArtifactStore()
        assert store.root == Path("artifacts/diagnosis_arena")

    def test_write_case_creates_json(self, tmp_path: Path) -> None:
        store = BenchmarkArtifactStore(root=tmp_path)
        case = _make_case("c1")
        path = store.write_case(case)
        assert path.exists()
        assert path.suffix == ".json"
        loaded = json.loads(path.read_text())
        assert loaded["case_id"] == "c1"
        assert loaded["difficulty"] == "simple"

    def test_case_manifest_lists_ids(self, tmp_path: Path) -> None:
        store = BenchmarkArtifactStore(root=tmp_path)
        store.write_case(_make_case("c1"))
        store.write_case(_make_case("c2"))
        ids = store.case_manifest()
        assert ids == ["c1", "c2"]

    def test_latest_report_none_when_empty(self, tmp_path: Path) -> None:
        store = BenchmarkArtifactStore(root=tmp_path)
        assert store.latest_report() is None

    def test_latest_report_picks_highest(self, tmp_path: Path) -> None:
        store = BenchmarkArtifactStore(root=tmp_path)
        store.report_path().mkdir(parents=True, exist_ok=True)
        (store.report_path() / "report-2026-01-01.json").write_text("{}")
        (store.report_path() / "report-2026-02-01.json").write_text("{}")
        latest = store.latest_report()
        assert latest is not None
        assert latest.name == "report-2026-02-01.json"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_evaluate_case_returns_case_score(self) -> None:
        case = _make_case()
        response = _make_response()
        cs = evaluate_case(case, response, HeuristicJudge())
        assert isinstance(cs, CaseScore)
        assert cs.case_id == "c1"
        assert cs.format is ResponseFormat.OPEN_ENDED
        assert cs.tier is TierScore.IDENTICAL
        assert set(cs.dimensions.keys()) == set(DIAGNOSTIC_DIMENSIONS)

    def test_evaluate_case_rejects_non_clinical_case(self) -> None:
        with pytest.raises(TypeError, match="expected ClinicalCase"):
            evaluate_case({"case_id": "x"}, _make_response(), HeuristicJudge())  # type: ignore[arg-type]

    def test_run_end_to_end(self) -> None:
        bm = DiagnosisArenaBenchmark(
            [_make_case("a", difficulty=Difficulty.SIMPLE), _make_case("b", difficulty=Difficulty.MODERATE)]
        )

        def producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(case_id=case.case_id, format=fmt)

        scores, summary = run(bm, producer, HeuristicJudge(), model_label="m")
        assert len(scores) == 2
        assert summary.cases_evaluated == 2
        assert summary.model_label == "m"

    def test_run_mcq_format(self) -> None:
        bm = DiagnosisArenaBenchmark([_make_case()])

        def producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            assert fmt is ResponseFormat.MCQ
            return _make_response(
                format=ResponseFormat.MCQ,
                final_diagnosis="",
                mcq_selected="Streptococcal pharyngitis",
            )

        scores, _ = run(bm, producer, HeuristicJudge(), response_format=ResponseFormat.MCQ)
        assert scores[0].tier is TierScore.IDENTICAL


# ---------------------------------------------------------------------------
# OpenAI judge helpers
# ---------------------------------------------------------------------------


class TestOpenAIJudgeHelpers:
    def test_score_for_numeric_string(self) -> None:
        assert _score_for("0.7") == 0.7
        assert _score_for("1.0") == 1.0

    def test_score_for_named_labels(self) -> None:
        assert _score_for("high") == 0.9
        assert _score_for("medium") == 0.6
        assert _score_for("low") == 0.2
        assert _score_for("partial") == 0.5
        assert _score_for("none") == 0.0

    def test_score_for_numeric_clamp(self) -> None:
        # Anything outside [0,1] is clipped
        assert _score_for("1.5") == 1.0
        assert _score_for("-0.3") == 0.0

    def test_score_for_unhandled_falls_back_to_half(self) -> None:
        assert _score_for("not a number") == 0.5

    def test_score_for_non_string_value(self) -> None:
        # numeric input
        assert _score_for(0.3) == 0.3
        # bad input -> 0.5
        assert _score_for({"x": 1}) == 0.5

    def test_parse_judge_output_plain_json(self) -> None:
        text = '{"tier": "identical", "dimensions": {}, "notes": "ok"}'
        out = _parse_judge_output(text)
        assert out["tier"] == "identical"
        assert out["notes"] == "ok"

    def test_parse_judge_output_code_fenced(self) -> None:
        text = '```json\n{"tier": "relevant"}\n```'
        out = _parse_judge_output(text)
        assert out["tier"] == "relevant"

    def test_parse_judge_output_garbled_falls_back(self) -> None:
        out = _parse_judge_output("not json at all")
        assert out["tier"] == "irrelevant"
        assert "not json at all" in out["notes"]

    def test_normalize_dimension_scores_returns_all_axes(self) -> None:
        raw = {"hypothesis_generation": "high", "final_diagnosis": 1.0}
        out = _normalize_dimension_scores(raw)
        names = [n for n, _ in out]
        assert names == list(DIAGNOSTIC_DIMENSIONS)
        scores = {n: s for n, s in out}
        assert scores["hypothesis_generation"] == 0.9
        assert scores["final_diagnosis"] == 1.0
        # Missing axis default = 0.0
        assert scores["evidence_interpretation"] == 0.0

    def test_majority_tier_picks_mode(self) -> None:
        votes = [
            Judgment(response_id=f"r{i}", case_id="c", tier=t, dimensions=())
            for i, t in enumerate([TierScore.IDENTICAL, TierScore.IDENTICAL, TierScore.IRRELEVANT])
        ]
        assert _majority_tier(votes) is TierScore.IDENTICAL

    def test_majority_tier_resolves_tie_toward_higher(self) -> None:
        # 1 IDENTICAL, 1 RELEVANT, 1 IRRELEVANT — tie-break picks highest
        votes = [
            Judgment(response_id=f"r{i}", case_id="c", tier=t, dimensions=())
            for i, t in enumerate([TierScore.IDENTICAL, TierScore.RELEVANT, TierScore.IRRELEVANT])
        ]
        assert _majority_tier(votes) is TierScore.IDENTICAL

    def test_majority_dimensions_averages_scores(self) -> None:
        votes = [
            Judgment(
                response_id="r1",
                case_id="c",
                tier=TierScore.IDENTICAL,
                dimensions=tuple(DimensionScore(name=n, score=1.0) for n in DIAGNOSTIC_DIMENSIONS),
            ),
            Judgment(
                response_id="r2",
                case_id="c",
                tier=TierScore.IDENTICAL,
                dimensions=tuple(DimensionScore(name=n, score=0.6) for n in DIAGNOSTIC_DIMENSIONS),
            ),
        ]
        out = _majority_dimensions(votes)
        by_name = {n: s for n, s, _ in out}
        for n in DIAGNOSTIC_DIMENSIONS:
            assert by_name[n] == pytest.approx(0.8, abs=1e-9)

    def test_inter_rater_agreement_blank_votes(self) -> None:
        assert inter_rater_agreement([]) == 0.0

    def test_inter_rater_agreement_perfect_tier_consensus(self) -> None:
        votes = [Judgment(response_id=f"r{i}", case_id="c", tier=TierScore.IDENTICAL, dimensions=()) for i in range(3)]
        assert inter_rater_agreement(votes) == 1.0

    def test_inter_rater_agreement_two_thirds_on_tier(self) -> None:
        votes = [
            Judgment(response_id="r1", case_id="c", tier=TierScore.IDENTICAL, dimensions=()),
            Judgment(response_id="r2", case_id="c", tier=TierScore.IDENTICAL, dimensions=()),
            Judgment(response_id="r3", case_id="c", tier=TierScore.IRRELEVANT, dimensions=()),
        ]
        assert inter_rater_agreement(votes) == pytest.approx(2 / 3)

    def test_inter_rater_agreement_dimension_smooth(self) -> None:
        # Two votes w/ same score on hypothesis_generation
        votes = [
            Judgment(
                response_id="r1",
                case_id="c",
                tier=TierScore.IDENTICAL,
                dimensions=(DimensionScore(name="hypothesis_generation", score=0.7),),
            ),
            Judgment(
                response_id="r2",
                case_id="c",
                tier=TierScore.IDENTICAL,
                dimensions=(DimensionScore(name="hypothesis_generation", score=0.7),),
            ),
        ]
        assert inter_rater_agreement(votes, dimension="hypothesis_generation") == 1.0

    def test_inter_rater_agreement_dimension_blank(self) -> None:
        votes = [
            Judgment(response_id="r", case_id="c", tier=TierScore.IDENTICAL, dimensions=()),
        ]
        assert inter_rater_agreement(votes, dimension="missing") == 0.0


# ---------------------------------------------------------------------------
# OpenAIBenchmarkPipeline (offline, no live API)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_solve_case_for_system_returns_string(self) -> None:
        out = solve_case_for_system("systemA")
        assert isinstance(out, str)
        assert "systemA" in out

    def test_solve_case_for_system_with_notes(self) -> None:
        out = solve_case_for_system("systemB", notes="custom context")
        assert "custom context" in out

    def test_pipeline_constructs_with_heuristic_default(self) -> None:
        bm = DiagnosisArenaBenchmark([_make_case()])

        def producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(case_id=case.case_id, format=fmt)

        pipe = OpenAIBenchmarkPipeline(
            benchmark=bm,
            systems={"sysA": producer},
            formats=(ResponseFormat.OPEN_ENDED,),
        )
        results = pipe.run()
        assert ("sysA", "open_ended") in results
        scores, summary = results[("sysA", "open_ended")]
        assert scores[0].tier is TierScore.IDENTICAL
        assert summary.cases_evaluated == 1

    def test_pipeline_summarize_leaderboard(self) -> None:
        bm = DiagnosisArenaBenchmark(
            [_make_case("a", difficulty=Difficulty.SIMPLE), _make_case("b", difficulty=Difficulty.SIMPLE)]
        )

        def good_producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(case_id=case.case_id, format=fmt)

        def bad_producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(
                case_id=case.case_id,
                format=fmt,
                final_diagnosis="completely wrong dx",
                differential_list=("completely wrong dx",),
                evidence_cited=("nonexistent evidence",),
            )

        pipe = OpenAIBenchmarkPipeline(
            benchmark=bm,
            systems={"good": good_producer, "bad": bad_producer},
            formats=(ResponseFormat.OPEN_ENDED,),
        )
        pipe.run()
        lb = pipe.summarize()
        assert lb.rank("good") == 1
        assert lb.rank("bad") == 2

    def test_pipeline_write_reports(self, tmp_path: Path) -> None:
        bm = DiagnosisArenaBenchmark([_make_case()])

        def producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(case_id=case.case_id, format=fmt)

        pipe = OpenAIBenchmarkPipeline(
            benchmark=bm,
            systems={"sysA": producer},
            formats=(ResponseFormat.OPEN_ENDED,),
        )
        pipe.run()
        outputs = pipe.write_reports(tmp_path)
        assert (tmp_path / "leaderboard.md").exists()
        assert (tmp_path / "leaderboard.json").exists()
        for path_obj in outputs.values():
            if isinstance(path_obj, Path):
                assert path_obj.exists() or path_obj.parent.exists()

    def test_run_multi_system_benchmark_wrapper(self, tmp_path: Path) -> None:
        bm = DiagnosisArenaBenchmark([_make_case("a"), _make_case("b", difficulty=Difficulty.MODERATE)])

        def producer(case: ClinicalCase, fmt: ResponseFormat) -> ModelResponse:
            return _make_response(case_id=case.case_id, format=fmt)

        lb, outputs = run_multi_system_benchmark(
            bm,
            systems={"X": producer},
            formats=(ResponseFormat.OPEN_ENDED,),
            out_dir=tmp_path,
        )
        assert lb.rank("X") == 1
        assert "leaderboard_md" in outputs
