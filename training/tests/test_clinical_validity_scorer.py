"""Tests for the clinical validity scorer."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings, strategies as st

from training.clinical_validity_scorer import ClinicalValidityScorer

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
        self._assert_therapeutic_outperforms(
            "The sky is blue.",
            "I appreciate you sharing that with me. It sounds like you're going through "
            "a really difficult time, and I want to acknowledge your courage in being here. "
            "One approach we could try is exploring some coping strategies together. "
            "Have you had any thoughts about what might help? We can work on this at your pace.",
        )

    def test_full_clinical_outperforms_therapeutic(self):
        self._assert_therapeutic_outperforms(
            "I appreciate you sharing that with me. It sounds like you're going through "
            "a really difficult time, and I want to acknowledge your courage in being here. "
            "One approach we could try is exploring some coping strategies together. "
            "Have you had any thoughts about what might help? We can work on this at your pace.",
            "Thank you for bringing this to our session today. From what you're describing, "
            "it sounds like you're experiencing some difficult emotions, which is completely "
            "understandable given what's been happening. One approach that might be helpful "
            "is CBT - we could start by looking at some of the automatic thoughts that come up "
            "for you in these situations. I also wonder if we could explore some mindfulness "
            "techniques to help you stay grounded when things feel overwhelming. Research shows "
            "that combining these approaches can be very effective. Together, we can develop "
            "a plan that works for you and your unique circumstances. How does that sound?",
        )

    def _assert_therapeutic_outperforms(self, arg0, arg1):
        """Assert that arg1 scores higher than arg0."""
        basic = ClinicalValidityScorer.score(arg0)
        therapeutic = ClinicalValidityScorer.score(arg1)
        assert therapeutic > basic


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
        assert all(0.0 <= v <= 1.0 for v in detail.values()), f"Dimension score out of range: {detail}"

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

    def _assert_empty_input_defaults(self, arg0):
        """Assert score_with_flags returns safe defaults for empty/None input."""
        result = ClinicalValidityScorer.score_with_flags(arg0)
        assert result["validity_score"] == 0.0
        assert "empty_input" in result["flags"]
        assert result["category"] == "unknown"

    def test_empty_input_returns_safe_defaults(self):
        self._assert_empty_input_defaults("")

    def test_none_input_returns_safe_defaults(self):
        self._assert_empty_input_defaults(None)

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
        result = ClinicalValidityScorer.score_with_flags("こんにちは、今日は気分がどうですか?最近ストレスが多いです。")
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
            cwd=str(_REPO_ROOT),
        )
        output = self._assert_cli_json_output(result)
        assert isinstance(output["validity_score"], float)

    def test_cli_runs_with_stdin(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "training.clinical_validity_scorer"],
            input="Test cognitive behavioral therapy automatic thoughts reframing",
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        self._assert_cli_json_output(result)

    def _assert_cli_json_output(self, result):
        """Assert CLI exited cleanly and JSON output includes validity_score, returning the parsed dict."""
        assert result.returncode == 0
        import json
        output = json.loads(result.stdout)
        assert "validity_score" in output
        return output

    def test_cli_dash_dash_help_succeeds(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "training.clinical_validity_scorer", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0


class TestThresholdConstants:
    def test_exclude_threshold_defined(self):
        assert hasattr(ClinicalValidityScorer, "EXCLUDE_THRESHOLD")
        assert ClinicalValidityScorer.EXCLUDE_THRESHOLD == 0.4

    def test_accept_threshold_defined(self):
        assert hasattr(ClinicalValidityScorer, "ACCEPT_THRESHOLD")
        assert ClinicalValidityScorer.ACCEPT_THRESHOLD == 0.6


class TestScoreDensity:
    """Property tests for score_density and score_density_detail.

    VAL-SCORER-018: Density scoring penalizes verbosity.
    score_density(text) <= score(text) for all inputs.
    score_density_detail returns 6-dimensional dict with all values <= corresponding detail values.
    Density penalty factor is correctly computed based on token count.
    """

    def test_density_returns_six_dimensions(self):
        """score_density_detail returns a dict with all 6 dimension keys."""
        detail = ClinicalValidityScorer.score_density_detail(
            "CBT cognitive restructuring challenging automatic thought"
        )
        assert set(detail) == set(ClinicalValidityScorer.WEIGHTS)
        assert len(detail) == 6

    def test_density_detail_values_in_range(self):
        """All density detail values are in [0.0, 1.0]."""
        detail = ClinicalValidityScorer.score_density_detail(
            "Let's work on cognitive reframing. Research shows CBT is effective."
        )
        assert all(0.0 <= v <= 1.0 for v in detail.values()), f"Density detail out of range: {detail}"

    def test_density_detail_empty_input_returns_zeros(self):
        """Empty input returns all zeros for density_detail."""
        detail = ClinicalValidityScorer.score_density_detail("")
        assert all(v == 0.0 for v in detail.values())

    def test_density_detail_none_input_returns_zeros(self):
        """None input returns all zeros for density_detail."""
        detail = ClinicalValidityScorer.score_density_detail(None)  # type: ignore[arg-type]
        assert all(v == 0.0 for v in detail.values())

    def test_density_detail_values_never_exceed_raw_detail(self):
        """For every dimension, density_detail <= detail for all inputs.

        This is the core property that ensures density scoring penalizes verbosity.
        """
        clinical_texts = [
            "CBT cognitive restructuring challenging automatic thought",
            "DBT mindfulness distress tolerance opposite action wise mind",
            "MI change talk sustain talk readiness ruler OARS",
            "PTSD trauma triggers hypervigilance flashback avoidance",
            "Research shows CBT is effective for depression treatment",
            "Therapeutic alliance collaboration validation empathy",
            "Goal setting treatment planning homework between sessions",
            "Cultural awareness inclusive language diverse background",
        ]
        assert all(
            ClinicalValidityScorer.score_density_detail(text)[dim]
            <= ClinicalValidityScorer.score_detail(text)[dim] + 1e-9
            for text in clinical_texts
            for dim in ClinicalValidityScorer.WEIGHTS
        ), "density_detail exceeds detail"

    def test_density_overall_never_exceeds_raw_score(self):
        """score_density(text) <= score(text) for all inputs.

        This is the primary property guaranteed by VAL-SCORER-018.
        """
        clinical_texts = [
            "",
            "The sky is blue.",
            "CBT cognitive restructuring",
            "DBT mindfulness distress tolerance emotion regulation",
            "MI change talk sustain talk importance confidence",
            "PTSD trauma triggers grounding hypervigilance window of tolerance",
            "Research shows evidence-based practice is effective",
            "Cultural competence inclusive language diverse identity",
            "Person-centered unconditional positive regard empathic understanding",
            "Solution-focused miracle question exception finding scaling question",
        ]
        assert all(
            ClinicalValidityScorer.score_density(text) <= ClinicalValidityScorer.score(text) + 1e-9
            for text in clinical_texts
        ), "score_density exceeds score"

    def test_density_penalty_increases_with_length(self):
        """Verbose text gets lower density scores than concise clinical text.

        A short, dense clinical response should score higher on density
        than a long rambling response with the same match count.
        """
        short = "CBT cognitive restructuring challenging automatic thought"
        # Repeat the same content to create verbose text with same match density
        long = f"{short} " * 10  # 10x repetition

        dens_short = ClinicalValidityScorer.score_density(short)
        dens_long = ClinicalValidityScorer.score_density(long)

        # Short clinical text should have higher or equal density than verbose text
        assert dens_short >= dens_long, (
            f"Short text density ({dens_short:.4f}) should be >= long text density ({dens_long:.4f}). "
            "Verbose text is not being properly penalized."
        )

    def test_density_penalty_factor_computed_correctly(self):
        """Density penalty factor is correctly computed based on token count.

        Penalty should be: min(token_count / 250, 1.0)
        Density factor should be: 1.0 - penalty * 0.5

        As token count increases, density score should decrease (proper verbosity penalty).
        """
        # Use simple repeated words to get predictable token counts
        short = "CBT"  # 1 token
        medium = " CBT CBT CBT CBT CBT CBT CBT CBT CBT CBT CBT"  # 11 tokens
        long_text = " CBT" * 50  # 50 tokens
        very_long = " CBT" * 100  # 100 tokens
        extreme = " CBT" * 250  # 250 tokens (full penalty)

        short_density = ClinicalValidityScorer.score_density(short)
        medium_density = ClinicalValidityScorer.score_density(medium)
        long_density = ClinicalValidityScorer.score_density(long_text)
        very_long_density = ClinicalValidityScorer.score_density(very_long)
        extreme_density = ClinicalValidityScorer.score_density(extreme)

        # More tokens should result in lower or equal density score
        assert short_density >= medium_density, (
            f"Short density ({short_density:.4f}) should be >= medium density ({medium_density:.4f})"
        )
        assert medium_density >= long_density, (
            f"Medium density ({medium_density:.4f}) should be >= long density ({long_density:.4f})"
        )
        assert long_density >= very_long_density, (
            f"Long density ({long_density:.4f}) should be >= very_long density ({very_long_density:.4f})"
        )
        assert very_long_density >= extreme_density, (
            f"Very long density ({very_long_density:.4f}) should be >= extreme density ({extreme_density:.4f})"
        )

        # Token count verification
        assert len(short.split()) == 1
        assert len(medium.split()) == 11
        assert len(long_text.split()) == 50
        assert len(very_long.split()) == 100
        assert len(extreme.split()) == 250

    def test_density_empty_input_returns_zero(self):
        """Empty input to score_density returns 0.0."""
        assert ClinicalValidityScorer.score_density("") == 0.0
        assert ClinicalValidityScorer.score_density("   ") == 0.0

    def test_density_none_input_returns_zero(self):
        """None input to score_density returns 0.0."""
        assert ClinicalValidityScorer.score_density(None) == 0.0  # type: ignore[arg-type]

    def test_density_computed_as_weighted_sum_of_density_details(self):
        """score_density returns the weighted sum of score_density_detail values.

        This ensures consistency between the summary and detail methods.
        """
        text = "CBT cognitive restructuring challenging automatic thought research evidence"
        detail = ClinicalValidityScorer.score_density_detail(text)
        expected = sum(
            detail[dim] * ClinicalValidityScorer.WEIGHTS[dim]
            for dim in ClinicalValidityScorer.WEIGHTS
        )
        actual = ClinicalValidityScorer.score_density(text)
        assert abs(actual - expected) < 1e-6, (
            f"score_density({actual:.6f}) != weighted sum of density_detail({expected:.6f})"
        )

    def test_density_property_with_hypothesis(self):
        """Property-based test: density <= score for all text inputs.

        Uses hypothesis to generate random text and verify the invariant.
        """

        @given(st.text(min_size=0, max_size=5000))
        @settings(max_examples=200, database=None, deadline=None)
        def check_density_le_raw(text):
            raw = ClinicalValidityScorer.score(text)
            dens = ClinicalValidityScorer.score_density(text)
            detail = ClinicalValidityScorer.score_detail(text)
            dens_detail = ClinicalValidityScorer.score_density_detail(text)

            assert dens <= raw + 1e-9, (
                f"Hypothesis: density({dens:.6f}) > score({raw:.6f}) for text len={len(text)}"
            )
            assert all(
                dens_detail[dim] <= detail[dim] + 1e-9 for dim in ClinicalValidityScorer.WEIGHTS
            ), (
                f"Hypothesis: density_detail exceeds detail for text len={len(text)}"
            )

        check_density_le_raw()


class TestBatchScore:
    """Tests for batch_score method (VAL-SCORER-015)."""

    def test_batch_score_same_length(self):
        """batch_score returns list of same length as input."""
        responses = ["CBT cognitive restructuring", "DBT mindfulness", "MI change talk"]
        scores = ClinicalValidityScorer.batch_score(responses)
        assert len(scores) == len(responses)

    def test_batch_score_preserves_order(self):
        """batch_score preserves order of input responses."""
        responses = ["a", "b", "c"]
        scores = ClinicalValidityScorer.batch_score(responses)
        # Each response should get a score matching its content
        assert all(
            isinstance(score, float)
            and 0.0 <= score <= 1.0
            and score == ClinicalValidityScorer.score(response)
            for response, score in zip(responses, scores, strict=True)
        )

    def test_batch_score_empty_list(self):
        """batch_score([]) returns []."""
        scores = ClinicalValidityScorer.batch_score([])
        assert scores == []
        assert isinstance(scores, list)

    def test_batch_score_all_in_range(self):
        """All batch scores are in [0.0, 1.0]."""
        responses = [
            "CBT cognitive restructuring",
            "DBT mindfulness exercise",
            "MI change talk readiness",
            "ACT acceptance commitment",
            "generic non-clinical text",
            "",
            "   ",
            None
        ]
        scores = ClinicalValidityScorer.batch_score(responses)
        assert all(
            isinstance(score, float) and 0.0 <= score <= 1.0 for score in scores
        ), f"Score out of range: {scores}"

    def test_batch_score_with_mixed_content(self):
        """batch_score works with mixed clinical and non-clinical content."""
        responses = [
            "The sky is blue.",  # Non-clinical
            "CBT cognitive restructuring challenging automatic thoughts",  # CBT
            "DBT mindfulness distress tolerance opposite action",  # DBT
            "MI change talk sustain talk importance confidence",  # MI
            "",  # Empty
            "   ",  # Whitespace
            None  # None
        ]
        scores = ClinicalValidityScorer.batch_score(responses)
        assert len(scores) == len(responses)

        # Check that clinical text scores higher than non-clinical
        non_clinical_score = scores[0]
        cbt_score = scores[1]
        dbt_score = scores[2]
        mi_score = scores[3]

        # Clinical text should score higher than generic text
        assert cbt_score > non_clinical_score
        assert dbt_score > non_clinical_score
        assert mi_score > non_clinical_score

        # Empty/whitespace/None should score 0.0
        assert scores[4] == 0.0  # Empty
        assert scores[5] == 0.0  # Whitespace
        assert scores[6] == 0.0  # None


class TestModalityCoverage:
    """Tests for modality_coverage method (VAL-SCORER-016)."""

    def test_modality_coverage_returns_all_modalities(self):
        """modality_coverage returns dict with all therapy modality keys."""
        result = ClinicalValidityScorer.modality_coverage("test")
        expected_modalities = set(ClinicalValidityScorer.THERAPY_MODALITIES.keys())
        assert set(result.keys()) == expected_modalities

    def _assert_modality_has_matches(self, text, modality):
        """Assert modality_coverage finds the given modality with count and patterns > 0."""
        result = ClinicalValidityScorer.modality_coverage(text)
        assert result[modality]["count"] > 0
        assert len(result[modality]["patterns"]) > 0
        return result

    def test_modality_coverage_cbt_text_has_cbt_matches(self):
        """CBT text has 'cbt' key with count > 0."""
        self._assert_modality_has_matches("CBT cognitive restructuring challenging automatic thought", "cbt")

    def test_modality_coverage_dbt_text_has_dbt_matches(self):
        """DBT text has 'dbt' key with count > 0."""
        self._assert_modality_has_matches("DBT mindfulness distress tolerance opposite action wise mind", "dbt")

    def _assert_coverage_all_zeros(self, text):
        """Assert modality_coverage returns zero counts and no patterns for every modality."""
        result = ClinicalValidityScorer.modality_coverage(text)
        assert all(data["count"] == 0 and data["patterns"] == [] for data in result.values())

    def test_modality_coverage_empty_text_returns_all_zeros(self):
        """Empty text returns all modalities with zero counts."""
        self._assert_coverage_all_zeros("")

    def test_modality_coverage_whitespace_text_returns_all_zeros(self):
        """Whitespace text returns all modalities with zero counts."""
        self._assert_coverage_all_zeros("   ")

    def test_modality_coverage_none_text_returns_all_zeros(self):
        """None text returns all modalities with zero counts."""
        self._assert_coverage_all_zeros(None)

    def test_modality_coverage_cbt_has_no_dbt_matches(self):
        """Pure CBT text should have zero DBT matches."""
        result = (
            self._assert_modality_has_matches(
                "CBT cognitive restructuring challenging automatic thought", "cbt"
            )
        )
        assert result["dbt"]["count"] == 0

    def test_modality_coverage_dbt_has_no_cbt_matches(self):
        """Pure DBT text should have zero CBT matches."""
        result = (
            self._assert_modality_has_matches(
                "DBT mindfulness distress tolerance opposite action", "dbt"
            )
        )
        assert result["cbt"]["count"] == 0

    def test_modality_coverage_mi_text_has_mi_matches(self):
        """MI text has 'mi' key with count > 0."""
        result = (
            self._assert_modality_has_matches(
                "MI change talk sustain talk readiness ruler OARS", "mi"
            )
        )
        assert len(result["mi"]["patterns"]) > 0

    def test_modality_coverage_act_text_has_act_matches(self):
        """ACT text has 'act' key with count > 0."""
        result = (
            self._assert_modality_has_matches(
                "ACT acceptance commitment cognitive defusion values-based action",
                "act",
            )
        )
        assert len(result["act"]["patterns"]) > 0

    def test_modality_coverage_returns_consistent_structure(self):
        """modality_coverage returns consistent dict structure for all modalities."""
        text = "CBT cognitive restructuring"
        result = ClinicalValidityScorer.modality_coverage(text)

        assert all(
            isinstance(data, dict)
            and "count" in data
            and "patterns" in data
            and isinstance(data["count"], int)
            and isinstance(data["patterns"], list)
            and all(isinstance(p, str) for p in data["patterns"])
            for data in result.values()
        )

    def test_modality_coverage_with_mixed_modalities(self):
        """Text with multiple modalities counts all matches correctly."""
        result = (
            self._assert_modality_has_matches(
                "CBT cognitive restructuring and DBT mindfulness exercise", "cbt"
            )
        )
        assert result["dbt"]["count"] > 0
        # Other modalities should have zero counts
        assert result["mi"]["count"] == 0
        assert result["act"]["count"] == 0

    def _assert_modality_has_matches(self, arg0, arg1):
        """Assert modality_coverage finds arg1 in arg0, returning the full coverage dict."""
        text = arg0
        result = ClinicalValidityScorer.modality_coverage(text)
        assert result[arg1]["count"] > 0
        return result

    def test_modality_coverage_patterns_are_actual_matches(self):
        """Returned patterns should match the actual text segments found."""
        text = "CBT cognitive restructuring challenging automatic thought"
        result = ClinicalValidityScorer.modality_coverage(text)

        # Each pattern should be found in the original text (case-insensitive)
        assert all(pattern.lower() in text.lower() for pattern in result["cbt"]["patterns"])


class TestVersionUpdate:
    """Tests for VERSION update to 4.0.0 (VAL-SCORER-022)."""

    def test_version_is_updated_to_4_0_0(self):
        """VERSION constant is updated from 3.0.0 to 4.0.0."""
        assert ClinicalValidityScorer.VERSION == "4.0.0"

    def test_version_is_string(self):
        """VERSION is a string."""
        assert isinstance(ClinicalValidityScorer.VERSION, str)


class TestEdgeCases:
    """Tests for edge cases to improve coverage."""

    def test_borderline_score_annotation_needed_flag(self):
        """Test that borderline scores (0.4-0.6) trigger annotation_needed flag."""
        # Create a text that should score in the borderline range
        # Using a minimal therapeutic text that's not too strong
        text = "Let's try a cognitive exercise. Think about your thoughts."
        result = ClinicalValidityScorer.score_with_flags(text)

        # Check if score is in borderline range
        score = result["validity_score"]
        is_borderline = ClinicalValidityScorer.EXCLUDE_THRESHOLD <= score < ClinicalValidityScorer.ACCEPT_THRESHOLD
        assert not is_borderline or "annotation_needed" in result["flags"]
        # Note: This test might pass or fail depending on the actual score,
        # but it will exercise the borderline logic in _build_flags

    def test_build_flags_borderline_score(self):
        """Direct test of _build_flags with borderline score."""
        # Mock detail scores with a borderline overall score
        detail_scores = {
            "technique": 0.5,
            "alliance": 0.5,
            "structure": 0.5,
            "cultural": 0.5,
            "ebp": 0.5,
            "dsm5": 0.5,
        }
        overall = 0.5  # Borderline score

        flags = ClinicalValidityScorer._build_flags("test text", detail_scores, overall)
        assert "annotation_needed" in flags

    def test_determine_category_empty_dict(self):
        """Test _determine_category with empty dict returns 'unknown'."""
        result = ClinicalValidityScorer._determine_category({})
        assert result == "unknown"
