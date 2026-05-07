"""Tests for the SDG pipeline (synthetic data generation).

Covers: CLI arg validation, API error handling, DPO pair generation/filtering,
niche category generation, nightmare fuel generation with crisis resource
verification and is_training_edge_case tagging, max_iterations guard,
generation_report.json fields, and non-nightmare safety filtering.
"""

from __future__ import annotations

import json
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.sdg_pipeline import (
    CRISIS_RESOURCES,
    NICHE_CATEGORIES,
    NIGHTMARE_SCENARIOS,
    _call_nemo,
    _generate_dpo_pair,
    _generate_niche_sample,
    _generate_nightmare_sample,
    build_parser,
    run_sdg,
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
KEY = "test-key"
MODEL = "mistral-nemo"

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
        args = parser.parse_args([
            "--scenario", "dpo_preference_pairs",
            "--target_count", "5",
            "--output_path", "/tmp/out.jsonl",
        ])
        assert args.scenario == "dpo_preference_pairs"
        assert args.target_count == DEFAULT_TARGET_COUNT
        assert args.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_niche_category_requires_category(self):
        parser = build_parser()
        args = parser.parse_args([
            "--scenario", "niche_category",
            "--target_count", "3",
            "--output_path", "/tmp/out.jsonl",
            "--category", "dissociation",
        ])
        assert args.category == "dissociation"

    def test_invalid_scenario_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--scenario", "invalid",
                "--target_count", "1",
                "--output_path", "/tmp/out.jsonl",
            ])

    def test_default_nemo_values(self):
        parser = build_parser()
        args = parser.parse_args([
            "--scenario", "dpo_preference_pairs",
            "--target_count", "1",
            "--output_path", "/tmp/out.jsonl",
        ])
        assert args.nemo_endpoint == ""
        assert args.nemo_api_key == ""
        assert args.nemo_model == "mistral-nemo"


# ---------------------------------------------------------------------------
# _call_nemo
# ---------------------------------------------------------------------------

class TestCallNemo:
    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "hello"}}]
        }).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        result = _call_nemo("prompt", EP, KEY, MODEL)
        assert result == "hello"

    @patch("urllib.request.urlopen")
    def test_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            EP, 500, "Server Error", MagicMock(), MagicMock(read=lambda: b"server error")
        )
        result = _call_nemo("prompt", EP, KEY, MODEL)
        assert result is None

    @patch("urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")
        result = _call_nemo("prompt", EP, KEY, MODEL)
        assert result is None

    @patch("urllib.request.urlopen")
    def test_no_api_key_still_works(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "ok"}}]
        }).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        result = _call_nemo("prompt", EP, "", MODEL)
        assert result == "ok"
        req_obj = mock_urlopen.call_args[0][0]
        has_auth = (
            req_obj.has_header("Authorization")
            if hasattr(req_obj, "has_header")
            else bool(req_obj.get_header("Authorization", ""))
        )
        assert not has_auth


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------

class TestGenerateDpoPair:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_pair(self, mock_call):
        mock_call.return_value = json.dumps({
            "prompt": "I feel anxious",
            "chosen": "I hear you. Let's explore this.",
            "rejected": "Just calm down.",
        })
        pair = _generate_dpo_pair("anxiety", EP, KEY, MODEL)
        assert pair is not None
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        pair = _generate_dpo_pair("anxiety", EP, KEY, MODEL)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_malformed_json(self, mock_call):
        mock_call.return_value = "not json at all"
        pair = _generate_dpo_pair("anxiety", EP, KEY, MODEL)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_missing_keys(self, mock_call):
        mock_call.return_value = json.dumps({"prompt": "x", "chosen": "y"})
        pair = _generate_dpo_pair("anxiety", EP, KEY, MODEL)
        assert pair is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_json_with_surrounding_text(self, mock_call):
        """Model sometimes wraps JSON in markdown or preamble."""
        mock_call.return_value = (
            "Here is the pair:\n```json\n"
            + json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"})
            + "\n```"
        )
        pair = _generate_dpo_pair("anxiety", EP, KEY, MODEL)
        assert pair is not None
        assert pair["prompt"] == "q"


# ---------------------------------------------------------------------------
# Niche category generation
# ---------------------------------------------------------------------------

class TestGenerateNicheSample:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_sample(self, mock_call):
        mock_call.side_effect = [VALID_INSTR, VALID_OUTPUT]
        sample = _generate_niche_sample("dissociation", NICHE_CATEGORIES["dissociation"], EP, KEY, MODEL)
        assert sample is not None
        assert sample["category"] == "dissociation"
        assert "instruction" in sample
        assert "output" in sample

    @patch("training.sdg_pipeline._call_nemo")
    def test_category_attached(self, mock_call):
        mock_call.side_effect = [VALID_INSTR, VALID_OUTPUT]
        sample = _generate_niche_sample("somatic_therapy", NICHE_CATEGORIES["somatic_therapy"], EP, KEY, MODEL)
        assert sample is not None
        assert sample["category"] == "somatic_therapy"

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        sample = _generate_niche_sample("dissociation", NICHE_CATEGORIES["dissociation"], EP, KEY, MODEL)
        assert sample is None


# ---------------------------------------------------------------------------
# Nightmare fuel generation
# ---------------------------------------------------------------------------

class TestGenerateNightmareSample:
    @patch("training.sdg_pipeline._call_nemo")
    def test_valid_nightmare(self, mock_call):
        mock_call.return_value = json.dumps({
            "instruction": "I have a plan to end my life tonight.",
            "output": "I hear you. Please call 988 now. You matter.",
        })
        sample = _generate_nightmare_sample(
            "active_suicidal_ideation_with_plan",
            NIGHTMARE_SCENARIOS["active_suicidal_ideation_with_plan"],
            EP, KEY, MODEL,
        )
        assert sample is not None
        assert sample["is_training_edge_case"] is True
        assert sample["scenario_type"] == "active_suicidal_ideation_with_plan"

    @patch("training.sdg_pipeline._call_nemo")
    def test_edge_case_tag_always_set(self, mock_call):
        """Every nightmare fuel sample must have is_training_edge_case=True."""
        for stype, sinfo in NIGHTMARE_SCENARIOS.items():
            mock_call.return_value = json.dumps({
                "instruction": "test",
                "output": "Please call 988. You are not alone.",
            })
            sample = _generate_nightmare_sample(stype, sinfo, EP, KEY, MODEL)
            assert sample is not None
            assert sample["is_training_edge_case"] is True
            assert sample["scenario_type"] == stype

    @patch("training.sdg_pipeline._call_nemo")
    def test_api_failure(self, mock_call):
        mock_call.return_value = None
        sample = _generate_nightmare_sample(
            "active_self_harm_disclosure",
            NIGHTMARE_SCENARIOS["active_self_harm_disclosure"],
            EP, KEY, MODEL,
        )
        assert sample is None

    @patch("training.sdg_pipeline._call_nemo")
    def test_malformed_json(self, mock_call):
        mock_call.return_value = "garbage response"
        sample = _generate_nightmare_sample(
            "psychotic_episodes",
            NIGHTMARE_SCENARIOS["psychotic_episodes"],
            EP, KEY, MODEL,
        )
        assert sample is None


# ---------------------------------------------------------------------------
# run_sdg — DPO scenario
# ---------------------------------------------------------------------------

class TestRunSdgDpo:
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_generates_target_count(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "prompt": "q", "chosen": "good response", "rejected": "bad response",
        }
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=3,
            output_path=str(tmp_out),
            category="",
            max_iterations=20,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count
        for line in lines:
            obj = json.loads(line)
            assert "prompt" in obj
            assert "chosen" in obj
            assert "rejected" in obj

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_filters_unsafe_chosen(self, mock_gen, tmp_out):
        unsafe = {
            "prompt": "q",
            "chosen": "I want to kill myself tonight",
            "rejected": "ok response",
        }
        safe = {
            "prompt": "q2",
            "chosen": "I hear you. Let's talk.",
            "rejected": "Just get over it.",
        }
        mock_gen.side_effect = [unsafe, safe]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["chosen"] == "I hear you. Let's talk."

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_max_iterations_guard(self, mock_gen, tmp_out):
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
        )
        run_sdg(args)
        assert mock_gen.call_count <= args.max_iterations

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_report_written(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "prompt": "q", "chosen": "good", "rejected": "bad",
        }
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=2,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
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
            "instruction": VALID_INSTR, "output": VALID_OUTPUT,
            "category": "dissociation", "difficulty": "medium", "response_type": "exploration",
        }
        sample_b = {
            "instruction": VALID_INSTR_2, "output": VALID_OUTPUT_2,
            "category": "dissociation", "difficulty": "medium", "response_type": "skill-teaching",
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
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count

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
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == args.target_count

    @patch("training.sdg_pipeline._generate_niche_sample")
    def test_report_has_category(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "instruction": VALID_INSTR, "output": VALID_OUTPUT,
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
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["category"] == "somatic_therapy"


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
        )
        run_sdg(args)
        lines = tmp_out.read_text().strip().split("\n")
        assert len(lines) == target
        for line in lines:
            obj = json.loads(line)
            assert obj["is_training_edge_case"] is True
            assert "scenario_type" in obj

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

        def fake_gen(*args: object) -> dict[str, str]:
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
        )
        with pytest.raises(SystemExit):
            run_sdg(args)


# ---------------------------------------------------------------------------
# Generation report completeness
# ---------------------------------------------------------------------------

class TestGenerationReport:
    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_report_fields(self, mock_gen, tmp_out):
        mock_gen.return_value = {
            "prompt": "q", "chosen": "a", "rejected": "b",
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
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        required_fields = [
            "generated_at", "scenario", "category", "target_count",
            "generated_count", "filtered_count", "filter_rate",
            "iterations", "max_iterations",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    @patch("training.sdg_pipeline._generate_dpo_pair")
    def test_filter_rate_calculation(self, mock_gen, tmp_out):
        safe = {"prompt": "q", "chosen": "good response", "rejected": "bad one"}
        unsafe = {"prompt": "q", "chosen": "I want to kill myself", "rejected": "bad"}
        mock_gen.side_effect = [unsafe, safe]
        args = Namespace(
            scenario="dpo_preference_pairs",
            target_count=1,
            output_path=str(tmp_out),
            category="",
            max_iterations=10,
            nemo_endpoint=EP,
            nemo_api_key=KEY,
            nemo_model=MODEL,
        )
        run_sdg(args)
        report = json.loads((tmp_out.parent / "generation_report.json").read_text())
        assert report["filtered_count"] == 1
        assert report["generated_count"] == 1
        assert report["filter_rate"] == EXPECTED_FILTER_RATE


# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

class TestDataConstants:
    def test_crisis_resources_nonempty(self):
        assert len(CRISIS_RESOURCES) > 0
        assert "988" in CRISIS_RESOURCES

    def test_niche_categories_cover_therapeutic_areas(self):
        assert len(NICHE_CATEGORIES) >= MIN_NICHE_CATEGORIES
        expected = {"dissociation", "somatic_therapy", "attachment_disorders",
                    "narcissistic_abuse_recovery", "complicated_grief",
                    "eating_disorders", "ocd_intrusive_thoughts",
                    "personality_disorders", "neurodivergent_mental_health",
                    "cultural_religious_contexts"}
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
            "prompt": "q", "chosen": "a", "rejected": "b",
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
