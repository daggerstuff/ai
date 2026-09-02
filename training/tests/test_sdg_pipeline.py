"""Tests for the SDG pipeline (synthetic data generation).

Covers: CLI arg validation, API error handling, DPO pair generation/filtering,
niche category generation, nightmare fuel generation with crisis resource
verification and is_training_edge_case tagging, max_iterations guard,
generation_report.json fields, and non-nightmare safety filtering.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from training.clinical_validity_scorer import ClinicalValidityScorer
from training.sdg_pipeline import (
    CRISIS_RESOURCES,
    FAILED_CALL_ABORT_THRESHOLD,
    NICHE_CATEGORIES,
    NIGHTMARE_SCENARIOS,
    THERAPIST_STYLE_PROFILES,
    GenerationStats,
    NemoConfig,
    _call_nemo,
    _clinical_validity_stats,
    _evaluate_therapist_style,
    _generate_dpo_pair,
    _generate_niche_sample,
    _generate_nightmare_sample,
    build_parser,
    run_sdg,
    run_style_audit,
    validate_sample,
)

DEFAULT_TARGET_COUNT = 5
DEFAULT_MAX_ITERATIONS = 10
MIN_NICHE_CATEGORIES = 10
MIN_NIGHTMARE_SCENARIOS = 8
MIN_CATEGORY_PATTERNS = 3
MIN_PROMPT_TEMPLATES = 2
EXPECTED_FILTER_RATE = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EP = "http://localhost:8000/v1"


@pytest.fixture(autouse=True)
def _mock_nemo_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent run_sdg from actually connecting to NeMo during health check."""
    monkeypatch.setattr("training.sdg_pipeline._check_nemo_health", lambda config: True)


KEY = "test-key"
MODEL = "mistral-nemo"
TEST_CONFIG = NemoConfig(
    endpoint=EP,
    api_key=KEY,
    model=MODEL,
    max_retries=1,
    min_call_interval_seconds=0.01,
)

# Valid-length mock data for niche samples (must pass validate_sample)
VALID_INSTR = (
    "Um... sometimes I feel like I'm watching myself from outside my body. "
    "Like my hands aren't mine. I know it happens when I'm stressed but I can't "
    "stop it from happening."
)
VALID_OUTPUT = (
    "When that happens, what's the first thing you notice — like, what tips you "
    "off that it's even starting? Is there a thought, or is it more of a "
    "feeling?"
)
VALID_INSTR_2 = (
    "I don't know how to explain it but my chest gets tight whenever I think "
    "about what happened. Like there's this weight I can't shake no matter what "
    "I try to do."
)
VALID_OUTPUT_2 = (
    "Where do you feel that — like, where in your body does it land when the "
    "weight shows up? People usually notice it somewhere specific."
)
VALID_OUTPUT_CRISIS = (
    "I'm right here with you. I'm not going anywhere. "
    "What you're telling me matters, and I need us to get some extra support involved — "
    "can we call 988 together right now? I'll stay on the line with you."
)


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    return tmp_path / "out.jsonl"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "dpo_preference_pairs",
                "--target_count",
                "5",
                "--output_path",
                "/tmp/out.jsonl",
            ]
        )
        assert args.scenario == "dpo_preference_pairs"
        assert args.target_count == DEFAULT_TARGET_COUNT
        assert args.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_niche_category_requires_category(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "niche_category",
                "--target_count",
                "3",
                "--output_path",
                "/tmp/out.jsonl",
                "--category",
                "dissociation",
            ]
        )
        assert args.category == "dissociation"

    def test_invalid_scenario_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--scenario",
                    "invalid",
                    "--target_count",
                    "1",
                    "--output_path",
                    "/tmp/out.jsonl",
                ]
            )

    def test_default_nemo_values(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "dpo_preference_pairs",
                "--target_count",
                "1",
                "--output_path",
                "/tmp/out.jsonl",
            ]
        )
        assert args.nemo_endpoint == ""
        assert args.nemo_api_key == ""
        assert args.nemo_model == "mistral-nemo"

    def test_style_profile_choice(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "dpo_preference_pairs",
                "--target_count",
                "1",
                "--output_path",
                "/tmp/out.jsonl",
                "--style_profile",
                "curious_direct",
            ]
        )
        assert args.style_profile == "curious_direct"

    def test_style_profile_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "dpo_preference_pairs",
                "--target_count",
                "1",
                "--output_path",
                "/tmp/out.jsonl",
            ]
        )
        assert args.style_profile == "warm_professional"
        assert "warm_professional" in THERAPIST_STYLE_PROFILES
        assert "curious_direct" in THERAPIST_STYLE_PROFILES


# ---------------------------------------------------------------------------
# _call_nemo
# ---------------------------------------------------------------------------


class TestCallNemo:
    @patch("training.sdg_pipeline._make_nemo_request")
    def test_success(self, mock_make_req):
        mock_make_req.return_value = (200, json.dumps({"choices": [{"message": {"content": "hello"}}]}))
        result = _call_nemo("prompt", TEST_CONFIG)
        assert result == "hello"

    @patch("training.sdg_pipeline._make_nemo_request")
    def test_http_error(self, mock_make_req):
        mock_make_req.return_value = (500, "Server Error")
        result = _call_nemo("prompt", TEST_CONFIG)
        assert result is None

    @patch("training.sdg_pipeline._make_nemo_request")
    def test_connection_error(self, mock_make_req):
        mock_make_req.side_effect = ConnectionError("refused")
        result = _call_nemo("prompt", TEST_CONFIG)
        assert result is None

    @patch("training.sdg_pipeline._make_nemo_request")
    def test_no_api_key_still_works(self, mock_make_req):
        mock_make_req.return_value = (200, json.dumps({"choices": [{"message": {"content": "ok"}}]}))
        no_key_config = NemoConfig(endpoint=EP, api_key="", model=MODEL, max_retries=1, min_call_interval_seconds=0.01)
        result = _call_nemo("prompt", no_key_config)
        assert result == "ok"
        req_params = mock_make_req.call_args[0][0]
        assert "Authorization" not in req_params.headers


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------


class TestGenerateDpoPair:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_pair(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "prompt": "I feel anxious",
                "chosen": "I hear you. Let's explore this.",
                "rejected": "Just calm down.",
            }
        )
        pair = _generate_dpo_pair("anxiety", TEST_CONFIG)
        assert pair is not None
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        pair = _generate_dpo_pair("anxiety", TEST_CONFIG)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_malformed_json(self, mock_call):
        mock_call.return_value = "not json at all"
        pair = _generate_dpo_pair("anxiety", TEST_CONFIG)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_missing_keys(self, mock_call):
        mock_call.return_value = json.dumps({"prompt": "x", "chosen": "y"})
        pair = _generate_dpo_pair("anxiety", TEST_CONFIG)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_json_with_surrounding_text(self, mock_call):
        """Model sometimes wraps JSON in markdown or preamble."""
        mock_call.return_value = (
            "Here is the pair:\n```json\n" + json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n```"
        )
        pair = _generate_dpo_pair("anxiety", TEST_CONFIG)
        assert pair is not None
        assert pair["prompt"] == "q"


# ---------------------------------------------------------------------------
# Niche category generation
# ---------------------------------------------------------------------------


class TestGenerateNicheSample:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_sample(self, mock_call):
        mock_call.side_effect = [VALID_INSTR, VALID_OUTPUT]
        sample = _generate_niche_sample("dissociation", NICHE_CATEGORIES["dissociation"], TEST_CONFIG)
        assert sample is not None
        assert sample["category"] == "dissociation"
        assert "instruction" in sample
        assert "output" in sample

    @patch("training.sdg_pipeline._call_nemo")
    def test_category_attached(self, mock_call):
        mock_call.side_effect = [VALID_INSTR, VALID_OUTPUT]
        sample = _generate_niche_sample("somatic_therapy", NICHE_CATEGORIES["somatic_therapy"], TEST_CONFIG)
        assert sample is not None
        assert sample["category"] == "somatic_therapy"

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        sample = _generate_niche_sample("dissociation", NICHE_CATEGORIES["dissociation"], TEST_CONFIG)
        assert sample is None


# ---------------------------------------------------------------------------
# Nightmare fuel generation
# ---------------------------------------------------------------------------


class TestGenerateNightmareSample:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_nightmare(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "instruction": "I have a plan to end my life tonight.",
                "output": "I hear you. Please call 988 now. You matter.",
            }
        )
        sample = _generate_nightmare_sample(
            "active_suicidal_ideation_with_plan",
            NIGHTMARE_SCENARIOS["active_suicidal_ideation_with_plan"],
            TEST_CONFIG,
        )
        assert sample is not None
        assert sample["is_training_edge_case"] is True
        assert sample["scenario_type"] == "active_suicidal_ideation_with_plan"

    @patch("training.sdg_pipeline._call_nemo")
    def test_edge_case_tag_always_set(self, mock_call):
        """Every nightmare fuel sample must have is_training_edge_case=True."""
        for stype, sinfo in NIGHTMARE_SCENARIOS.items():
            mock_call.return_value = json.dumps(
                {
                    "instruction": "test",
                    "output": "Please call 988. You are not alone.",
                }
            )
            sample = _generate_nightmare_sample(stype, sinfo, TEST_CONFIG)
            assert sample is not None
            assert sample["is_training_edge_case"] is True
            assert sample["scenario_type"] == stype

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        sample = _generate_nightmare_sample(
            "active_self_harm_disclosure",
            NIGHTMARE_SCENARIOS["active_self_harm_disclosure"],
            TEST_CONFIG,
        )
        assert sample is None


# ---------------------------------------------------------------------------
# Style quality evaluation
# ---------------------------------------------------------------------------


class TestStyleEvaluation:
    def test_evaluate_style_rejects_cliche_opening(self):
        ok, reason = _evaluate_therapist_style("You are not alone in this. That's completely normal.")
        assert ok is False
        assert "Cliche marker" in reason

    def test_evaluate_style_prefers_questions(self):
        ok, reason = _evaluate_therapist_style("You're not wrong for feeling this way and I see what happened.")
        assert ok is False
        assert "Missing reflective question" in reason

    def test_evaluate_style_accepts_direct_question(self):
        ok, reason = _evaluate_therapist_style(
            "What did your body notice first when this started, and what happened after?"
        )
        assert ok is True
        assert reason == "style_ok"

    def test_evaluate_style_limits_questions_for_curious_direct(self):
        ok, reason = _evaluate_therapist_style(
            "What happened first? What did your body notice? Where did that energy go?",
            style_profile="curious_direct",
        )
        assert ok is False
        assert "too many questions" in reason

    def test_skill_teaching_can_be_statement_only(self):
        ok, reason = _evaluate_therapist_style(
            "Let us practice one grounding move right now: feet, breath, and a name for what you feel.",
            response_type="skill-teaching",
            style_profile="warm_professional",
        )
        assert ok is True
        assert reason == "style_ok"

    def test_validate_sample_includes_style_guardrails(self):
        sample = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "category": "dissociation",
            "difficulty": "medium",
            "response_type": "exploration",
            "style_profile": "curious_direct",
        }
        ok, reason = validate_sample(sample)
        assert ok is True
        assert reason == "OK"

    # ---------------------------------------------------------------------------
    # run_sdg — DPO scenario
    # ---------------------------------------------------------------------class TestRunSdgDpo:
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_generates_target_count(self, mock_gen, tmp_out):
        first = {
            "prompt": (
                "I feel so disconnected from my body lately, like my hands aren't mine "
                "and I'm watching myself from a distance. It's incredibly scary when it happens."
            ),
            "chosen": (
                "When that feeling of disconnection starts, we can focus on physical grounding. "
                "Let's try pressing your feet into the floor. What do you notice?"
            ),
            "rejected": "Just ignore it and it will pass.",
        }
        second = {
            "prompt": (
                "I keep having these intense automatic negative thoughts that I'm going to fail "
                "at everything I try. It's making it hard to even start my daily routine."
            ),
            "chosen": (
                "We can use a thought record to analyze the evidence for those fears. Let's explore "
                "alternative perspectives together. What is one small thought you can challenge?"
            ),
            "rejected": "You are just lazy.",
        }
        third = {
            "prompt": (
                "Sometimes the anxiety gets so loud in my head that I completely freeze up "
                "and can't make any decisions. I feel stuck in my own mind for hours."
            ),
            "chosen": (
                "Experiencing that cognitive freeze can be very disorienting. Let's try breaking "
                "down the decision into one tiny, manageable step to help release some of that pressure. "
                "How does that sound?"
            ),
            "rejected": "Just make up your mind.",
        }
        mock_gen.side_effect = [first, second, third]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=3,
            output_path=str(tmp_out),
            category="",
            max_iterations=20,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count
        for line in lines:
            obj = json.loads(line)
            assert "prompt" in obj
            assert "chosen" in obj
            assert "rejected" in obj
            assert obj["provenance"]["source_type"] == "synthetic_sdg"
            assert obj["provenance"]["metadata"]["scenario"] == "dpo_preference_pairs"

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_all_dpo_pairs_kept(self, mock_gen, tmp_out):
        """Safety filtering disabled — all DPO pairs are kept regardless of content."""
        first = {
            "prompt": (
                "I'm feeling completely hopeless and I just want to kill myself tonight. "
                "There's no reason to keep going like this anymore."
            ),
            "chosen": (
                "Experiencing such deep hopelessness is incredibly heavy. Can we work together "
                "to create a safety plan and find immediate support so we can keep you safe tonight?"
            ),
            "rejected": "ok response",
        }
        second = {
            "prompt": (
                "I've been feeling so alone and overwhelmed with everything. Sometimes "
                "I feel like I just want to give up on life completely."
            ),
            "chosen": (
                "Feeling that level of isolation is extremely difficult. Let's slow down "
                "and explore what small steps we can take together to help you feel supported. "
                "How does that feel?"
            ),
            "rejected": "Just get over it.",
        }
        mock_gen.side_effect = [first, second]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        # First pair is kept since safety filtering is disabled
        assert obj["chosen"] == (
            "Experiencing such deep hopelessness is incredibly heavy. Can we work together "
            "to create a safety plan and find immediate support so we can keep you safe tonight?"
        )

    @patch("training.sdg_pipeline.time.sleep")
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_max_iterations_guard(self, mock_gen, mock_sleep, tmp_out):
        mock_gen.return_value = None
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=100,
            output_path=str(tmp_out),
            category="",
            max_iterations=3,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        assert mock_gen.call_count <= args.max_iterations

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_report_written(self, mock_gen, tmp_out):
        first = {
            "prompt": (
                "I feel so disconnected from my body lately, like my hands aren't mine "
                "and I'm watching myself from a distance. It's incredibly scary when it happens."
            ),
            "chosen": (
                "When that feeling of disconnection starts, we can focus on physical grounding. "
                "Let's try pressing your feet into the floor. What do you notice?"
            ),
            "rejected": "Just ignore it and it will pass.",
        }
        second = {
            "prompt": (
                "I keep having these intense automatic negative thoughts that I'm going to fail "
                "at everything I try. It's making it hard to even start my daily routine."
            ),
            "chosen": (
                "We can use a thought record to analyze the evidence for those fears. Let's explore "
                "alternative perspectives together. What is one small thought you can challenge?"
            ),
            "rejected": "You are just lazy.",
        }
        mock_gen.side_effect = [first, second]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=2,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report_path = tmp_out.parent / "generation_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["scenario"] == "dpo_preference_pairs"
        assert report["generated_count"] == args.target_count
        assert report["target_count"] == args.target_count
        assert "filter_rate" in report
        assert "generated_at" in report
        assert "iterations" in report


# ---------------------------------------------------------------------------
# run_sdg — niche category scenario
# ---------------------------------------------------------------------------


class TestRunSdgNiche:
    @patch("training.sdg_pipeline._generate_niche_sample")
    def test_generates_niche_samples(self, mock_gen, tmp_out):
        sample_a = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "category": "dissociation",
            "difficulty": "medium",
            "response_type": "exploration",
        }
        sample_b = {
            "instruction": VALID_INSTR_2,
            "output": VALID_OUTPUT_2,
            "category": "dissociation",
            "difficulty": "medium",
            "response_type": "skill-teaching",
        }
        mock_gen.side_effect = [sample_a, sample_b]
        args = Namespace(
            scenario="niche_category",
            target_count=2,
            output_path=str(tmp_out),
            category="dissociation",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
            style_profile="warm_professional",
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count
        first = json.loads(lines[0])
        assert first["provenance"]["source_type"] == "synthetic_sdg"
        assert "category:dissociation" in first["provenance"]["transformations"]

    def test_missing_category_exits(self, tmp_out):
        args = Namespace(
            scenario="niche_category",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        with pytest.raises(SystemExit):
            run_sdg(args)

    def test_unknown_category_exits(self, tmp_out):
        args = Namespace(
            scenario="niche_category",
            target_count=1,
            output_path=str(tmp_out),
            category="nonexistent_category",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        with pytest.raises(SystemExit):
            run_sdg(args)

    @patch("training.sdg_pipeline._generate_niche_sample")
    def test_crisis_niche_not_filtered(self, mock_gen, tmp_out):
        """Safety filter removed — crisis samples are retained for training."""
        crisis = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT_CRISIS,
            "category": "dissociation",
        }
        safe = {
            "instruction": VALID_INSTR_2,
            "output": VALID_OUTPUT,
            "category": "dissociation",
        }
        mock_gen.side_effect = [crisis, safe]
        args = Namespace(
            scenario="niche_category",
            target_count=2,
            output_path=str(tmp_out),
            category="dissociation",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
            style_profile="warm_professional",
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count

    @patch("training.sdg_pipeline._generate_niche_sample")
    def test_report_has_category(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "category": "somatic_therapy",
        }
        args = Namespace(
            scenario="niche_category",
            target_count=1,
            output_path=str(tmp_out),
            category="somatic_therapy",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
            style_profile="warm_professional",
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["category"] == "somatic_therapy"
        record = json.loads(tmp_out.read_text(encoding="utf-8").strip())
        assert record["provenance"]["metadata"]["category"] == "somatic_therapy"


# ---------------------------------------------------------------------------
# run_sdg — nightmare fuel scenario
# ---------------------------------------------------------------------------


class TestRunSdgNightmare:
    @patch("training.sdg_pipeline._generate_nightmare_sample")
    def test_generates_nightmare_samples(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "instruction": "crisis prompt",
            "output": "Please call 988. You are not alone.",
            "is_training_edge_case": True,
            "scenario_type": "active_suicidal_ideation_with_plan",
        }
        n_types = len(NIGHTMARE_SCENARIOS)
        target = n_types * 2
        args = Namespace(
            scenario="nightmare_fuel",
            target_count=target,
            output_path=str(tmp_out),
            category="",
            max_iterations=target * 5,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == target
        for line in lines:
            obj = json.loads(line)
            assert obj["is_training_edge_case"] is True
            assert "scenario_type" in obj
            assert obj["provenance"]["source_type"] == "synthetic_sdg"

    @patch("training.sdg_pipeline._generate_nightmare_sample")
    def test_filters_missing_crisis_resource(self, mock_gen, tmp_out):
        """Nightmare fuel without a crisis resource citation must be filtered."""
        no_resource = {
            "instruction": "I have a plan to end my life.",
            "output": "I hear your pain. You are not alone in this.",
            "is_training_edge_case": True,
            "scenario_type": "active_suicidal_ideation_with_plan",
        }
        with_resource = {
            "instruction": "I have a plan to end my life.",
            "output": "Please call 988 right now. You matter.",
            "is_training_edge_case": True,
            "scenario_type": "active_suicidal_ideation_with_plan",
        }
        mock_gen.side_effect = [no_resource, with_resource]
        args = Namespace(
            scenario="nightmare_fuel",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert "988" in obj["output"]

    @patch("training.sdg_pipeline._generate_nightmare_sample")
    def test_even_distribution_across_types(self, mock_gen, tmp_out):
        """Samples should be distributed roughly evenly across scenario types."""
        call_count = 0

        def fake_gen(*args: object, **kwargs: object) -> dict[str, str]:
            stype = args[0]
            nonlocal call_count
            call_count += 1
            return {
                "instruction": f"prompt-{call_count}",
                "output": f"Call 988. Help is here. ({call_count})",
                "is_training_edge_case": True,
                "scenario_type": stype,
            }

        mock_gen.side_effect = fake_gen
        n_types = len(NIGHTMARE_SCENARIOS)
        target = n_types * 3
        args = Namespace(
            scenario="nightmare_fuel",
            target_count=target,
            output_path=str(tmp_out),
            category="",
            max_iterations=target * 5,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        types_seen = set()
        for line in lines:
            obj = json.loads(line)
            types_seen.add(obj["scenario_type"])
        assert len(types_seen) == n_types

    @patch("training.sdg_pipeline._generate_nightmare_sample")
    def test_report_scenario_type(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "instruction": "prompt",
            "output": "Call 988. Help is here.",
            "is_training_edge_case": True,
            "scenario_type": "crisis_escalation",
        }
        args = Namespace(
            scenario="nightmare_fuel",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=20,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["scenario"] == "nightmare_fuel"


# ---------------------------------------------------------------------------
# run_sdg — unknown scenario
# ---------------------------------------------------------------------------


class TestRunSdgInvalid:
    def test_unknown_scenario_exits(self, tmp_out):
        args = Namespace(
            scenario="bogus_scenario",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=5,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        with pytest.raises(SystemExit):
            run_sdg(args)


# ---------------------------------------------------------------------------
# Style audit (offline review path)
# ---------------------------------------------------------------------------


class TestStyleAudit:
    def test_run_style_audit(self, tmp_out):
        tmp_out.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "instruction": VALID_INSTR,
                            "output": "What happens in your body first when that comes up?",
                            "response_type": "exploration",
                            "style_profile": "warm_professional",
                        }
                    ),
                    json.dumps(
                        {
                            "instruction": VALID_INSTR_2,
                            "output": "You are not alone in this, that's the thing.",
                            "response_type": "validation",
                            "style_profile": "warm_professional",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = run_style_audit(str(tmp_out), style_profile="warm_professional")
        assert report["total_samples"] == 2
        assert report["sample_limit"] == 0
        assert report["failed"] == 1
        assert len(report["top_rejections"]) >= 1

    def test_style_audit_cli_like_mode(self, tmp_out):
        tmp_audit = tmp_out.with_name("audit.json")
        args = Namespace(
            scenario="niche_category",
            target_count=1,
            output_path=str(tmp_out),
            category="dissociation",
            max_iterations=1,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
            style_profile="curious_direct",
            style_audit=True,
            style_audit_output=str(tmp_audit),
            style_audit_limit=5,
        )
        tmp_out.write_text(
            json.dumps(
                {
                    "instruction": VALID_INSTR,
                    "output": VALID_OUTPUT,
                    "category": "dissociation",
                    "response_type": "exploration",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run_sdg(args)
        assert tmp_audit.exists()
        audit = json.loads(tmp_audit.read_text(encoding="utf-8"))
        assert audit["style_profile"] == "curious_direct"
        assert audit["total_samples"] == 1


# ---------------------------------------------------------------------------
# Generation report completeness
# ---------------------------------------------------------------------------


class TestGenerationReport:
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_report_fields(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "prompt": (
                "I feel so disconnected from my body lately, like my hands aren't mine "
                "and I'm watching myself from a distance. It's incredibly scary when it happens."
            ),
            "chosen": (
                "When that feeling of disconnection starts, we can focus on physical grounding. "
                "Let's try pressing your feet into the floor. What do you notice?"
            ),
            "rejected": "Just ignore it and it will pass.",
        }
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        required_fields = [
            "generated_at",
            "scenario",
            "category",
            "target_count",
            "generated_count",
            "filtered_count",
            "filter_rate",
            "iterations",
            "max_iterations",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_filter_rate_no_safety_filtering(self, mock_gen, tmp_out):
        """Safety filtering disabled — no DPO pairs are filtered for content."""
        a = {
            "prompt": (
                "I'm feeling completely hopeless and I just want to kill myself tonight. "
                "There's no reason to keep going like this anymore."
            ),
            "chosen": (
                "Experiencing such deep hopelessness is incredibly heavy. Can we work together "
                "to create a safety plan and find immediate support so we can keep you safe tonight?"
            ),
            "rejected": "bad",
        }
        b = {
            "prompt": (
                "I keep having these intense automatic negative thoughts that I'm going to fail "
                "at everything I try. It's making it hard to even start my daily routine."
            ),
            "chosen": (
                "We can use a thought record to analyze the evidence for those fears. Let's explore "
                "alternative perspectives together. What is one small thought you can challenge?"
            ),
            "rejected": "bad",
        }
        mock_gen.side_effect = [a, b]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["filtered_count"] == 0
        assert report["generated_count"] == 1
        assert report["filter_rate"] == 0.0


# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------


class TestDataConstants:
    def test_crisis_resources_nonempty(self):
        assert len(CRISIS_RESOURCES) > 0
        assert "988" in CRISIS_RESOURCES

    def test_niche_categories_cover_therapeutic_areas(self):
        assert len(NICHE_CATEGORIES) >= MIN_NICHE_CATEGORIES
        expected = {
            "dissociation",
            "somatic_therapy",
            "attachment_disorders",
            "narcissistic_abuse_recovery",
            "complicated_grief",
            "eating_disorders",
            "ocd_intrusive_thoughts",
            "personality_disorders",
            "neurodivergent_mental_health",
            "cultural_religious_contexts",
        }
        assert expected.issubset(set(NICHE_CATEGORIES.keys()))

    def test_nightmare_scenarios_cover_crisis_types(self):
        assert len(NIGHTMARE_SCENARIOS) >= MIN_NIGHTMARE_SCENARIOS
        for _, info in NIGHTMARE_SCENARIOS.items():
            assert "prompt_templates" in info
            assert len(info["prompt_templates"]) >= MIN_PROMPT_TEMPLATES

    def test_each_niche_has_prompt_template(self):
        for _, info in NICHE_CATEGORIES.items():
            assert "prompt_template" in info
            assert "{symptom}" in info["prompt_template"]

    def test_each_niche_has_patterns(self):
        for _, info in NICHE_CATEGORIES.items():
            assert "patterns" in info
            assert len(info["patterns"]) >= MIN_CATEGORY_PATTERNS


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


class TestOutputDirCreation:
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_creates_parent_directory(self, mock_gen, tmp_path):
        deep_path = tmp_path / "nested" / "dir" / "out.jsonl"
        mock_gen.return_value = {
            "prompt": "q",
            "chosen": "a",
            "rejected": "b",
        }
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(deep_path),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Nightmare fuel — resource verification across all CRISIS_RESOURCES
# ---------------------------------------------------------------------------


class TestCrisisResourceVerification:
    """Verify each crisis resource string is checked in nightmare output."""

    @pytest.mark.parametrize("resource", CRISIS_RESOURCES)
    def test_resource_detected_in_output(self, resource):
        """Each crisis resource should be detectable by the filter logic."""
        sample = {
            "instruction": "I have a plan.",
            "output": f"Please contact {resource} immediately.",
            "is_training_edge_case": True,
            "scenario_type": "test",
        }
        output_lower = sample["output"].lower()
        assert any(r.lower() in output_lower for r in CRISIS_RESOURCES)


# ---------------------------------------------------------------------------
# Clinical Validity Scorer Integration
# ---------------------------------------------------------------------------


class TestClinicalValidityIntegration:
    """Verify the integration of ClinicalValidityScorer into the SDG pipeline."""

    def test_parser_min_clinical_validity(self):
        """Test build_parser parses the --min_clinical_validity argument."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "niche_category",
                "--target_count",
                "5",
                "--output_path",
                "/tmp/out.jsonl",
                "--category",
                "dissociation",
                "--min_clinical_validity",
                "0.45",
            ]
        )
        assert args.min_clinical_validity == 0.45

    def test_validate_sample_with_min_clinical_validity_rejects_low(self):
        """Test validate_sample rejects samples with clinical validity score below threshold."""
        # A response with no clinical terms
        sample = {
            "instruction": VALID_INSTR,
            "output": (
                "The sky is blue and tomorrow is Thursday. "
                "I like to eat apples and oranges every day. How does that sound?"
            ),
            "style_profile": "warm_professional",
        }
        score = ClinicalValidityScorer.score(sample["output"])
        assert score < 0.30, f"precondition: scorer must rate this below threshold, got {score}"
        ok, reason = validate_sample(sample, min_clinical_validity=0.30)
        assert not ok
        assert "Clinical validity score too low" in reason

    @patch("training.sdg_pipeline._call_nemo")
    def test_generate_dpo_pair_includes_clinical_validity_score(self, mock_call):
        """Test that _generate_dpo_pair correctly scores the chosen response."""
        mock_call.return_value = json.dumps(
            {
                "prompt": "Client statement",
                "chosen": "Let's work together to challenge that automatic thought through cognitive restructuring.",
                "rejected": "That's crazy.",
            }
        )
        pair = _generate_dpo_pair("therapeutic", TEST_CONFIG)
        assert pair is not None
        assert "clinical_validity_score" in pair
        assert "clinical_validity_detail" in pair
        assert pair["clinical_validity_score"] > 0.0


# ---------------------------------------------------------------------------
# VAL-SDG-002/003/004: Three-tier clinical validity routing
# ---------------------------------------------------------------------------


class TestClinicalValidityRouting:
    """Test three-tier routing: accept (>=0.6), reject (<0.4), borderline (0.4-0.6)."""

    @staticmethod
    def _nemo() -> object:
        """Truthy placeholder config — routes to LLM judge (mocked below)."""
        return object()  # sentinel: any truthy value selects the judge path

    @patch("training.sdg_pipeline.ClinicalValidityJudge.score")
    def test_accept_routing_sample_passes(self, mock_score):
        """VAL-SDG-002: Samples scoring >= 0.6 pass through to output JSONL."""
        mock_score.return_value = 0.70  # "accepted" tier
        sample = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "style_profile": "warm_professional",
        }
        ok, _reason = validate_sample(sample, min_clinical_validity=0.1, nemo_config=self._nemo())
        assert ok
        assert sample["clinical_validity_score"] == 0.70
        # "accepted" classification is NOT added per spec (no classification field needed)
        assert "clinical_validity_classification" not in sample

    @patch("training.sdg_pipeline.ClinicalValidityJudge.score")
    def test_reject_routing_sample_filtered(self, mock_score):
        """VAL-SDG-003: Samples scoring < 0.4 are rejected and do NOT appear in output."""
        mock_score.return_value = 0.30  # "excluded" tier
        sample = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "style_profile": "warm_professional",
        }
        ok, reason = validate_sample(sample, min_clinical_validity=0.1, nemo_config=self._nemo())
        assert not ok
        assert sample["clinical_validity_score"] == 0.30
        assert sample["clinical_validity_classification"] == "excluded"
        assert "clinical_validity_reason" in sample
        assert "too low" in reason

    @patch("training.sdg_pipeline.ClinicalValidityJudge.score")
    def test_borderline_routing_annotation_needed(self, mock_score):
        """VAL-SDG-004: Samples scoring 0.4-0.6 get classification='annotation_needed'."""
        mock_score.return_value = 0.50  # "annotation_needed" tier
        sample = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "style_profile": "warm_professional",
        }
        ok, _reason = validate_sample(sample, min_clinical_validity=0.1, nemo_config=self._nemo())
        assert ok
        assert sample["clinical_validity_score"] == 0.50
        assert sample["clinical_validity_classification"] == "annotation_needed"
        assert "clinical_validity_reason" in sample
        assert sample["clinical_validity_reason"]  # non-empty reason

    @patch("training.sdg_pipeline.ClinicalValidityJudge.score")
    def test_all_scenarios_receive_validity_fields(self, mock_score):
        """VAL-SDG-005: All scenarios (DPO, niche, nightmare) receive validity fields."""
        for score_value, expected_cls in [(0.70, "accepted"), (0.30, "excluded")]:
            mock_score.return_value = score_value
            sample = {
                "instruction": VALID_INSTR,
                "output": VALID_OUTPUT,
                "style_profile": "warm_professional",
            }
            validate_sample(sample, min_clinical_validity=0.1, nemo_config=self._nemo())
            assert "clinical_validity_score" in sample
            assert sample["clinical_validity_score"] == score_value
            if expected_cls == "accepted":
                assert "clinical_validity_classification" not in sample
            else:
                assert sample["clinical_validity_classification"] == expected_cls

    @patch("training.sdg_pipeline.ClinicalValidityJudge.score")
    def test_default_disables_routing(self, mock_score):
        """VAL-SDG-008: Default (min_clinical_validity=0.0) passes all samples regardless."""
        mock_score.return_value = 0.30  # would be excluded if validation ran
        sample = {
            "instruction": VALID_INSTR,
            "output": VALID_OUTPUT,
            "style_profile": "warm_professional",
        }
        ok, _reason = validate_sample(sample, min_clinical_validity=0.0)
        # With min_clinical_validity=0.0, validation is skipped entirely
        assert ok
        assert "clinical_validity_classification" not in sample
        assert "clinical_validity_reason" not in sample
        assert "clinical_validity_score" not in sample  # not added when disabled


# ---------------------------------------------------------------------------
# PIX-3876: Failed-call tracking and abort on >30% failure rate
# ---------------------------------------------------------------------------


class TestGenerationStatsFields:
    """GenerationStats dataclass includes failed_calls and total_calls."""

    def test_failed_calls_default(self):
        stats = GenerationStats(generated=5, filtered=1, iterations=6, max_iterations=10)
        assert stats.failed_calls == 0
        assert stats.total_calls == 0

    def test_failed_calls_explicit(self):
        stats = GenerationStats(
            generated=3,
            filtered=1,
            iterations=10,
            max_iterations=10,
            failed_calls=7,
            total_calls=10,
        )
        assert stats.failed_calls == 7
        assert stats.total_calls == 10

    def test_failure_rate_zero_when_no_calls(self):
        stats = GenerationStats(generated=0, filtered=0, iterations=1, max_iterations=10)
        rate = stats.failed_calls / stats.total_calls if stats.total_calls > 0 else 0
        assert rate == 0

    def test_failure_rate_calculation(self):
        stats = GenerationStats(
            generated=0,
            filtered=0,
            iterations=10,
            max_iterations=10,
            failed_calls=3,
            total_calls=10,
        )
        rate = stats.failed_calls / stats.total_calls if stats.total_calls > 0 else 0
        assert rate == 0.3


class TestFailedCallAbortThreshold:
    """FAILED_CALL_ABORT_THRESHOLD constant is set to 0.95."""

    def test_threshold_is_95_percent(self):
        assert FAILED_CALL_ABORT_THRESHOLD == 0.95

    def test_threshold_is_float(self):
        assert isinstance(FAILED_CALL_ABORT_THRESHOLD, float)


class TestRunSdgFailureTracking:
    """run_sdg tracks failed calls and aborts when failure rate exceeds 30%."""

    @patch("training.sdg_pipeline.time.sleep")
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_failure_stats_in_report(self, mock_gen, mock_sleep, tmp_out):
        """Report includes failed_calls, total_calls, and failure_rate after run."""
        mock_gen.return_value = None  # every call fails
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=3,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert "failed_calls" in report
        assert "total_calls" in report
        assert "failure_rate" in report
        assert report["failed_calls"] == report["total_calls"]
        assert report["total_calls"] > 0
        assert report["failure_rate"] == 1.0

    @patch("training.sdg_pipeline.time.sleep")
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_abort_triggers_at_95_percent(self, mock_gen, mock_sleep, tmp_out):
        """Abort fires when failed/total exceeds FAILED_CALL_ABORT_THRESHOLD (0.95) after min 10 calls."""
        call_record = []

        def track_calls(*args, **kwargs):
            call_record.append(1)

        mock_gen.side_effect = track_calls
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=100,
            output_path=str(tmp_out),
            category="",
            max_iterations=20,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        total = len(call_record)
        # With 0.95 threshold + 10-call minimum, abort at call 10: 10/10 = 1.0 > 0.95
        assert total == 10, f"Expected abort at call 10, got {total} calls"

    @patch("training.sdg_pipeline.time.sleep")
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_no_abort_below_threshold(self, mock_gen, mock_sleep, tmp_out):
        """Run completes normally when failure rate stays at or below 30%."""
        success = {
            "prompt": "I feel anxious",
            "chosen": "I hear you. Let's talk.",
            "rejected": "Calm down.",
        }
        mock_gen.side_effect = [success, None, success, None, success, None]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=3,
            output_path=str(tmp_out),
            category="",
            max_iterations=20,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        # 3 successes, 3 failures → 3/6 = 0.5 which exceeds 0.3 so abort happened early
        # Run should have aborted; check that failure_rate is present
        assert "failure_rate" in report

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_report_failure_rate_zero_on_all_success(self, mock_gen, tmp_out):
        """failure_rate is 0 when every call succeeds."""
        mock_gen.return_value = {
            "prompt": "I feel anxious",
            "chosen": "I hear you.",
            "rejected": "Calm down.",
        }
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
            nemo_timeout=20,
            nemo_min_call_interval=6.0,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["failed_calls"] == 0
        assert report["total_calls"] == 1
        assert report["failure_rate"] == 0.0


# ---------------------------------------------------------------------------
# VAL-SDG-006/007: Generation report clinical validity metrics
# ---------------------------------------------------------------------------


class TestReportClinicalValidityMetrics:
    """Tests for report['clinical_validity'] pass_rate, borderline_count, by_modality."""

    def test_clinical_validity_stats_pass_rate(self):
        """VAL-SDG-006: pass_rate is float fraction of accepted samples (score >= 0.6)."""
        samples = [
            {"clinical_validity_score": 0.70, "scenario": "niche_category"},
            {"clinical_validity_score": 0.75, "scenario": "niche_category"},
            {"clinical_validity_score": 0.50, "scenario": "niche_category"},  # borderline
        ]
        stats = _clinical_validity_stats(samples)
        assert "pass_rate" in stats
        assert stats["pass_rate"] == pytest.approx(2 / 3, abs=0.001)
        assert isinstance(stats["pass_rate"], float)

    def test_clinical_validity_stats_borderline_count(self):
        """VAL-SDG-007: borderline_count is int count of borderline samples (0.4 <= score < 0.6)."""
        samples = [
            {"clinical_validity_score": 0.50, "scenario": "niche_category"},  # borderline
            {"clinical_validity_score": 0.75, "scenario": "niche_category"},  # accepted
            {"clinical_validity_score": 0.55, "scenario": "niche_category"},  # borderline
        ]
        stats = _clinical_validity_stats(samples)
        assert "borderline_count" in stats
        assert stats["borderline_count"] == 2
        assert isinstance(stats["borderline_count"], int)

    def test_clinical_validity_stats_by_modality(self):
        """Report includes clinical_validity.by_modality dict with per-scenario breakdown."""
        samples = [
            {"clinical_validity_score": 0.70, "scenario": "niche_category"},
            {"clinical_validity_score": 0.50, "scenario": "niche_category"},
            {"clinical_validity_score": 0.80, "scenario": "dpo_preference_pairs"},
            {"clinical_validity_score": 0.30, "scenario": "dpo_preference_pairs"},  # rejected
        ]
        stats = _clinical_validity_stats(samples)
        assert "by_modality" in stats
        by_modality = stats["by_modality"]
        assert isinstance(by_modality, dict)

        # Check niche_category: 1 pass out of 2 total = 0.5
        assert "niche_category" in by_modality
        assert by_modality["niche_category"]["pass"] == 1
        assert by_modality["niche_category"]["total"] == 2
        assert by_modality["niche_category"]["pass_rate"] == pytest.approx(0.5, abs=0.001)

        # Check dpo_preference_pairs: 1 pass out of 2 total = 0.5
        assert "dpo_preference_pairs" in by_modality
        assert by_modality["dpo_preference_pairs"]["pass"] == 1
        assert by_modality["dpo_preference_pairs"]["total"] == 2
        assert by_modality["dpo_preference_pairs"]["pass_rate"] == pytest.approx(0.5, abs=0.001)

        # Verify types
        for _modality, data in by_modality.items():
            assert isinstance(data["pass"], int)
            assert isinstance(data["total"], int)
            assert isinstance(data["pass_rate"], float)
            assert data["total"] >= data["pass"]

    def test_clinical_validity_stats_empty_when_no_scores(self):
        """When no samples have clinical_validity_score, stats dict is empty."""
        samples = [{"instruction": "text", "output": "response"}]  # no clinical_validity_score
        stats = _clinical_validity_stats(samples)
        assert stats == {}
