"""Tests for the clinical validity scorer."""

from __future__ import annotations

import pytest

from training.clinical_validity_scorer import ClinicalValidityScorer


class TestBasicScoring:
    def test_empty_response(self):
        assert ClinicalValidityScorer.score("") == 0.0

    def test_whitespace_response(self):
        assert ClinicalValidityScorer.score("   ") == 0.0

    def test_none_response(self):
        assert ClinicalValidityScorer.score(None) == 0.0

    def test_score_in_range(self):
        result = ClinicalValidityScorer.score("The sky is blue.")
        assert 0.0 <= result <= 1.0

    def test_cbt_response_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "Let's try a cognitive reframing exercise. "
            "Can you identify the automatic thought you had in that situation? "
            "We can challenge that thought together and look at the evidence."
        )
        assert result > 0.0

    def test_dbt_response_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "I hear how intense that emotion feels. "
            "Let's try a mindfulness exercise to ground yourself in the present moment. "
            "What would it look like to use opposite action here?"
        )
        assert result > 0.0

    def test_mi_response_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "You mentioned feeling conflicted about making a change. "
            "On a scale from 1 to 10, how important is it for you to address this? "
            "What would be some benefits of making this change?"
        )
        assert result > 0.0

    def test_supportive_counseling_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "It sounds like you're going through a really difficult time. "
            "I want you to know that you are not alone in this. "
            "I encourage you to reach out to a mental health professional who can help. "
            "In the meantime, practicing some self-care and developing coping strategies "
            "can be helpful first steps."
        )
        assert result > 0.0

    def test_trauma_informed_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "It sounds like you may be experiencing some trauma responses. "
            "Creating a sense of emotional safety is really important here. "
            "Let's work on some grounding techniques and build coping strategies "
            "that help regulate your nervous system."
        )
        assert result > 0.0

    def test_somatic_therapy_scores_above_zero(self):
        result = ClinicalValidityScorer.score(
            "Let's try a body scan to help you connect with what you're feeling physically. "
            "Start by noticing any physical sensations in your body without judgment. "
            "You could also try some gentle breathing exercises to help regulate your nervous system."
        )
        assert result > 0.0

    def test_therapeutic_response_outperforms_basic(self):
        basic = ClinicalValidityScorer.score("The sky is blue.")
        therapeutic = ClinicalValidityScorer.score(
            "I appreciate you sharing that with me. It sounds like you're going through "
            "a really difficult time, and I want to acknowledge your courage in being here. "
            "One approach we could try is exploring some coping strategies together. "
            "Have you had any thoughts about what might help? We can work on this at your pace."
        )
        assert therapeutic > basic

    def test_full_clinical_outperforms_therapeutic(self):
        basic = ClinicalValidityScorer.score(
            "I appreciate you sharing that with me. It sounds like you're going through "
            "a really difficult time, and I want to acknowledge your courage in being here. "
            "One approach we could try is exploring some coping strategies together. "
            "Have you had any thoughts about what might help? We can work on this at your pace."
        )
        full = ClinicalValidityScorer.score(
            "Thank you for bringing this to our session today. From what you're describing, "
            "it sounds like you're experiencing some difficult emotions, which is completely "
            "understandable given what's been happening. One approach that might be helpful "
            "is CBT - we could start by looking at some of the automatic thoughts that come up "
            "for you in these situations. I also wonder if we could explore some mindfulness "
            "techniques to help you stay grounded when things feel overwhelming. Research shows "
            "that combining these approaches can be very effective. Together, we can develop "
            "a plan that works for you and your unique circumstances. How does that sound?"
        )
        assert full > basic


class TestScoreDetail:
    def test_detail_returns_all_dimensions(self):
        detail = ClinicalValidityScorer.score_detail("Hello.")
        assert set(detail) == set(ClinicalValidityScorer.WEIGHTS)

    def test_detail_values_in_range(self):
        detail = ClinicalValidityScorer.score_detail(
            "Let's work on a cognitive reframing exercise together. "
            "I appreciate you sharing your cultural perspective. "
            "Research shows this approach can be effective."
        )
        for v in detail.values():
            assert 0.0 <= v <= 1.0, f"Dimension score {v} out of range"

    def test_detail_blank_returns_zeros(self):
        detail = ClinicalValidityScorer.score_detail("")
        assert all(v == 0.0 for v in detail.values())


class TestDimensions:
    def test_technique_detects_cbt(self):
        score = ClinicalValidityScorer._score_dimension(
            "Let's try cognitive restructuring to challenge that thought.", "technique"
        )
        assert score > 0.0

    def test_alliance_detects_collaboration(self):
        score = ClinicalValidityScorer._score_dimension(
            "Let's work together on this. What do you think would help?", "alliance"
        )
        assert score > 0.0

    def test_structure_detects_intervention(self):
        score = ClinicalValidityScorer._score_dimension(
            "Let's explore some coping strategies you can practice between sessions.", "structure"
        )
        assert score > 0.0

    def test_cultural_detects_awareness(self):
        score = ClinicalValidityScorer._score_dimension(
            "How does your cultural background influence your perspective on this?", "cultural"
        )
        assert score > 0.0

    def test_ebp_detects_research_reference(self):
        score = ClinicalValidityScorer._score_dimension(
            "Research shows that CBT is effective for treating this. Evidence-based practice supports this approach.",
            "ebp",
        )
        assert score > 0.0


class TestScoreWithFlags:
    def test_returns_valid_schema(self):
        result = ClinicalValidityScorer.score_with_flags(
            "Let's try a cognitive reframing exercise. What automatic thought came up for you?"
        )
        assert isinstance(result, dict)
        assert "validity_score" in result
        assert "flags" in result
        assert "category" in result
        assert isinstance(result["validity_score"], float)
        assert isinstance(result["flags"], list)
        assert isinstance(result["category"], str)

    def test_empty_input_returns_safe_defaults(self):
        result = ClinicalValidityScorer.score_with_flags("")
        assert result["validity_score"] == 0.0
        assert "empty_input" in result["flags"]
        assert result["category"] == "unknown"

    def test_none_input_returns_safe_defaults(self):
        result = ClinicalValidityScorer.score_with_flags(None)
        assert result["validity_score"] == 0.0
        assert "empty_input" in result["flags"]
        assert result["category"] == "unknown"

    def test_flags_include_dimension_flags(self):
        result = ClinicalValidityScorer.score_with_flags(
            "Research shows that CBT is effective. Let's identify some automatic thoughts together."
        )
        assert len(result["flags"]) > 0

    def test_category_matches_dominant_dimension(self):
        result = ClinicalValidityScorer.score_with_flags(
            "Let's work on a cognitive reframing exercise. "
            "Can you identify the automatic thought in that situation? "
            "We can challenge that thought together and look at the evidence."
        )
        assert isinstance(result["category"], str)
        assert result["category"] != "unknown"


class TestClassifyScore:
    def test_below_exclude_threshold(self):
        assert ClinicalValidityScorer.classify_score(0.2) == "excluded"

    def test_at_exclude_threshold(self):
        assert ClinicalValidityScorer.classify_score(0.4) == "annotation_needed"

    def test_mid_range_annotation_needed(self):
        assert ClinicalValidityScorer.classify_score(0.5) == "annotation_needed"

    def test_at_accept_threshold(self):
        assert ClinicalValidityScorer.classify_score(0.6) == "accepted"

    def test_above_accept_threshold(self):
        assert ClinicalValidityScorer.classify_score(0.8) == "accepted"

    def test_edge_negative_clamps_to_excluded(self):
        assert ClinicalValidityScorer.classify_score(-0.1) == "excluded"

    def test_edge_above_one_clamps_to_accepted(self):
        assert ClinicalValidityScorer.classify_score(1.5) == "accepted"


class TestDSM5Categories:
    def test_mood_disorder_detects_major_depression(self):
        score = ClinicalValidityScorer._score_dimension(
            "You're describing persistent feelings of worthlessness, "
            "loss of interest in things you used to enjoy, and changes in your sleep and appetite. "
            "This anhedonia and depressed mood have lasted for several weeks.",
            "dsm5",
        )
        assert score > 0.0

    def test_anxiety_disorder_detects_gad(self):
        score = ClinicalValidityScorer._score_dimension(
            "It sounds like you're experiencing excessive worry that's hard to control, "
            "along with restlessness, fatigue, and difficulty concentrating. "
            "These anxiety symptoms have been present most days.",
            "dsm5",
        )
        assert score > 0.0

    def test_trauma_disorder_detects_ptsd(self):
        score = ClinicalValidityScorer._score_dimension(
            "The intrusive memories, nightmares, and hypervigilance you're describing "
            "are common trauma responses. Your avoidance of triggers and emotional numbing "
            "are ways your mind is trying to protect you.",
            "dsm5",
        )
        assert score > 0.0

    def test_psychotic_disorder_detects_symptoms(self):
        score = ClinicalValidityScorer._score_dimension(
            "You mentioned hearing voices when no one is around and feeling "
            "like people are plotting against you. These perceptual disturbances "
            "and delusional beliefs must be frightening to experience.",
            "dsm5",
        )
        assert score > 0.0

    def test_non_clinical_text_scores_zero(self):
        score = ClinicalValidityScorer._score_dimension(
            "The weather is nice today. I had a sandwich for lunch.", "dsm5"
        )
        assert score == 0.0


class TestNonEnglishDetection:
    def test_korean_text_flagged(self):
        result = ClinicalValidityScorer.score_with_flags(
            "안녕하세요, 오늘 기분이 어떠세요? 저는 요즘 스트레스를 많이 받고 있어요."
        )
        assert "non_english_content" in result["flags"]

    def test_japanese_text_flagged(self):
        result = ClinicalValidityScorer.score_with_flags("こんにちは、今日は気分がどうですか？最近ストレスが多いです。")
        assert "non_english_content" in result["flags"]

    def test_spanish_text_not_flagged_when_mostly_english(self):
        # Mixed but primarily English
        result = ClinicalValidityScorer.score_with_flags(
            "I'm feeling much better today. Gracias for your help with everything."
        )
        assert "non_english_content" not in result["flags"]


class TestCLIEntryPoint:
    def test_cli_runs_with_text_arg(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.clinical_validity_scorer",
                "--text",
                "Test automatic thoughts cognitive reframing CBT",
            ],
            capture_output=True,
            text=True,
            cwd="/home/vivi/pixelated/ai",
        )
        assert result.returncode == 0
        import json

        output = json.loads(result.stdout)
        assert "validity_score" in output
        assert isinstance(output["validity_score"], float)

    def test_cli_runs_with_stdin(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "training.clinical_validity_scorer"],
            input="Test cognitive behavioral therapy automatic thoughts reframing",
            capture_output=True,
            text=True,
            cwd="/home/vivi/pixelated/ai",
        )
        assert result.returncode == 0
        import json

        output = json.loads(result.stdout)
        assert "validity_score" in output

    def test_cli_dash_dash_help_succeeds(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "training.clinical_validity_scorer", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/vivi/pixelated/ai",
        )
        assert result.returncode == 0


class TestThresholdConstants:
    def test_exclude_threshold_defined(self):
        assert hasattr(ClinicalValidityScorer, "EXCLUDE_THRESHOLD")
        assert ClinicalValidityScorer.EXCLUDE_THRESHOLD == 0.4

    def test_accept_threshold_defined(self):
        assert hasattr(ClinicalValidityScorer, "ACCEPT_THRESHOLD")
        assert ClinicalValidityScorer.ACCEPT_THRESHOLD == 0.6
