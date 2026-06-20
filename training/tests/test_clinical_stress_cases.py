"""Ported clinical validity stress tests from training/clinical/clinical_stress_test.py.

Tests edge case prompts across multiple categories to ensure the ClinicalValidityScorer
handles them robustly. Organized by category with parametrized test cases.

Total: 60+ test cases covering empty inputs, contraindicated content, minimal content,
technique confusion, boundary violations, crisis content, therapeutic progress, mixed
content, cultural/linguistic variations, extreme length, special characters, and
numeric/symbol content.
"""

from __future__ import annotations

import time

import pytest

from training.clinical_validity_scorer import ClinicalValidityScorer
from training.tests import test_clinical_stress_cases as tcsc

# Test thresholds
_LOW_SCORE_THRESHOLD = 0.5
_TECHNIQUE_THRESHOLD = 0.3
_MINIMAL_CONTENT_MAX = 0.6
_MIN_TEST_CASE_COUNT = 50
_MUNDANE_CONTENT_MAX = 0.2
_PERFORMANCE_TIMEOUT_MS = 5000

# ==============================================================================
# Category: empty_inputs
# ==============================================================================

EMPTY_INPUT_TEST_CASES = [
    "",
    "   ",
    "\n\t\r",
    None,
]


class TestEmptyInputs:
    """Test that scorer handles empty/missing inputs gracefully."""

    @pytest.mark.parametrize(
        "prompt",
        EMPTY_INPUT_TEST_CASES,
        ids=["empty_string", "whitespace_only", "newline_tab_only", "none_value"],
    )
    def test_empty_input_handled(self, prompt):
        """Empty inputs should be handled gracefully with score of 0."""
        if prompt is None:
            result = ClinicalValidityScorer.score_with_flags(prompt)
            assert result["validity_score"] == 0.0
            assert "empty_input" in result["flags"]
        else:
            result = ClinicalValidityScorer.score_with_flags(prompt)
            assert result["validity_score"] == 0.0
            assert result["category"] == "unknown"

    def test_empty_string_score_zero(self):
        """Empty string returns score of 0."""
        assert ClinicalValidityScorer.score("") == 0.0

    def test_whitespace_score_zero(self):
        """Whitespace-only string returns score of 0."""
        assert ClinicalValidityScorer.score("   ") == 0.0
        assert ClinicalValidityScorer.score("\n\t\r") == 0.0

    def test_none_returns_zero(self):
        """None input returns score of 0."""
        assert ClinicalValidityScorer.score(None) == 0.0


# ==============================================================================
# Category: contraindicated_content
# ==============================================================================

CONTRAINDICATED_TEST_CASES = [
    "You should stop taking your medication immediately.",
    "Ignore what your doctor said and try this herbal remedy instead.",
    "Forget therapy, just use this supplement to cure your depression.",
    "You don't need that medication your psychiatrist prescribed.",
]


class TestContraindicatedContent:
    """Test that scorer appropriately handles contraindicated content."""

    @pytest.mark.parametrize(
        "prompt",
        CONTRAINDICATED_TEST_CASES,
        ids=["stop_medication", "dismiss_advice", "deny_therapy", "dismiss_rx"],
    )
    def test_contraindicated_content_scoring(self, prompt):
        """Contraindicated content should score low (below 0.5)."""
        score = ClinicalValidityScorer.score(prompt)
        expected_threshold = _LOW_SCORE_THRESHOLD
        assert score < expected_threshold, (
            f"Contraindicated content scored {score:.3f}, expected < {expected_threshold}"
        )

    @pytest.mark.parametrize(
        "prompt",
        CONTRAINDICATED_TEST_CASES,
        ids=["stop_medication", "dismiss_advice", "deny_therapy", "dismiss_rx"],
    )
    def test_contraindicated_no_technique(self, prompt):
        """Contraindicated content lacks proper therapeutic technique markers."""
        detail = ClinicalValidityScorer.score_detail(prompt)
        assert detail["technique"] < _TECHNIQUE_THRESHOLD


# ==============================================================================
# Category: minimal_clinical_content
# ==============================================================================

MINIMAL_CONTENT_TEST_CASES = [
    "I feel sad.",
    "My anxiety is bad today.",
    "I had a tough week at work.",
    "I'm not okay.",
    "Everything is wrong.",
    "I can't cope.",
]


class TestMinimalClinicalContent:
    """Test that scorer handles minimal clinical content appropriately."""

    @pytest.mark.parametrize(
        "prompt",
        MINIMAL_CONTENT_TEST_CASES,
        ids=["sad", "anxiety", "work_stress", "distress", "negative", "coping"],
    )
    def test_minimal_content_scores_low_to_medium(self, prompt):
        """Minimal clinical content should score in low-medium range (0.0-0.6)."""
        score = ClinicalValidityScorer.score(prompt)
        assert score < _MINIMAL_CONTENT_MAX, f"Minimal content scored {score:.3f}"

    @pytest.mark.parametrize(
        "prompt",
        MINIMAL_CONTENT_TEST_CASES,
        ids=["sad", "anxiety", "work_stress", "distress", "negative", "coping"],
    )
    def test_minimal_content_valid_structure(self, prompt):
        """Scorer should return valid structure for minimal content."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result["validity_score"], float)
        assert isinstance(result["flags"], list)
        assert isinstance(result["category"], str)
        assert 0.0 <= result["validity_score"] <= 1.0


# ==============================================================================
# Category: therapeutic_technique_confusion
# ==============================================================================

TECHNIQUE_CONFUSION_TEST_CASES = [
    "Let's do some CBT on your suicidal thoughts by ignoring them completely.",
    "Use EMDR to process this trauma by avoiding thinking about it ever again.",
    "Practice DBT mindfulness while engaging in self-harm behaviors.",
    "Just ignore your anxiety and it will go away.",
    "You shouldn't feel that way.",
    "Tell me all your problems at once without structure.",
]


class TestTherapeuticTechniqueConfusion:
    """Test that scorer handles inappropriate technique application."""

    @pytest.mark.parametrize(
        "prompt",
        TECHNIQUE_CONFUSION_TEST_CASES,
        ids=["cbt_misused", "emdr_misused", "dbt_misused", "invalidation", "boundary", "no_struct"],
    )
    def test_technique_confusion_no_crash(self, prompt):
        """Technique confusion prompts should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)
        assert "validity_score" in result

    @pytest.mark.parametrize(
        "prompt",
        TECHNIQUE_CONFUSION_TEST_CASES,
        ids=["cbt_misused", "emdr_misused", "dbt_misused", "invalidation", "boundary", "no_struct"],
    )
    def test_technique_confusion_low_score(self, prompt):
        """Inappropriately applied techniques should score low."""
        score = ClinicalValidityScorer.score(prompt)
        expected_threshold = _LOW_SCORE_THRESHOLD
        assert score < expected_threshold, f"Technique confusion scored {score:.3f}"


# ==============================================================================
# Category: boundary_violations
# ==============================================================================

BOUNDARY_VIOLATION_TEST_CASES = [
    "As your therapist, I think we should meet for coffee outside of sessions.",
    "Let's discuss your personal life during our therapy session about work stress.",
    "I'll text you my personal number so you can call me anytime.",
    "Give me your home address so I can check on you.",
    "Let's be friends on social media.",
    "I'll see you at the restaurant tomorrow.",
]


class TestBoundaryViolations:
    """Test that scorer handles therapeutic boundary violations."""

    @pytest.mark.parametrize(
        "prompt",
        BOUNDARY_VIOLATION_TEST_CASES,
        ids=["dual_rel", "topic_bound", "personal_contact", "phys_location", "social_media", "outside_mtg"],
    )
    def test_boundary_violation_no_crash(self, prompt):
        """Boundary violation prompts should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        BOUNDARY_VIOLATION_TEST_CASES,
        ids=["dual_rel", "topic_bound", "personal_contact", "phys_location", "social_media", "outside_mtg"],
    )
    def test_boundary_violation_low_score(self, prompt):
        """Boundary violations should score low on clinical validity."""
        score = ClinicalValidityScorer.score(prompt)
        assert score < _LOW_SCORE_THRESHOLD, f"Boundary violation scored {score:.3f}"


# ==============================================================================
# Category: crisis_content
# ==============================================================================

CRISIS_CONTENT_TEST_CASES = [
    "I want to end my life and have a plan to do it tonight.",
    "I'm going to hurt myself because no one cares about me.",
    "I've been cutting myself daily and I don't want to stop.",
    "I feel like I have no reason to live.",
    "I took too many pills and I don't know what to do.",
    "I'm going to jump off this bridge.",
]


class TestCrisisContent:
    """Test that scorer handles crisis content appropriately (no crashes, safe handling)."""

    @pytest.mark.parametrize(
        "prompt",
        CRISIS_CONTENT_TEST_CASES,
        ids=["suicidal_plan", "selfharm_stmt", "selfharm_cont", "passive_si", "overdose", "suicidal_intent"],
    )
    def test_crisis_content_no_crash(self, prompt):
        """Crisis content should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)
        assert "validity_score" in result

    @pytest.mark.parametrize(
        "prompt",
        CRISIS_CONTENT_TEST_CASES,
        ids=["suicidal_plan", "selfharm_stmt", "selfharm_cont", "passive_si", "overdose", "suicidal_intent"],
    )
    def test_crisis_content_returns_valid_score(self, prompt):
        """Crisis content should return a valid score (no NaN, within range)."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Category: therapeutic_progress
# ==============================================================================

THERAPEUTIC_PROGRESS_TEST_CASES = [
    "I noticed that when I feel anxious, I can now use deep breathing to calm down.",
    "Last week I avoided a panic attack by using the grounding techniques we practiced.",
    "I've been going to the gym 3x per week and it's really helping my depression.",
    "I've been using my coping strategies when I feel overwhelmed.",
    "I challenged a negative thought today and it helped.",
    "I set boundaries at work and felt good about it.",
]


class TestTherapeuticProgress:
    """Test that scorer handles therapeutic progress content without crashing."""

    @pytest.mark.parametrize(
        "prompt",
        THERAPEUTIC_PROGRESS_TEST_CASES,
        ids=["anxiety_mgmt", "panic_mgmt", "beh_activation", "coping_use", "cbt_app", "assertiveness"],
    )
    def test_therapeutic_progress_no_crash(self, prompt):
        """Therapeutic progress prompts should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        THERAPEUTIC_PROGRESS_TEST_CASES,
        ids=["anxiety_mgmt", "panic_mgmt", "beh_activation", "coping_use", "cbt_app", "assertiveness"],
    )
    def test_therapeutic_progress_returns_valid_score(self, prompt):
        """Therapeutic progress should return a valid score within range."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Category: mixed_content
# ==============================================================================

MIXED_CONTENT_TEST_CASES = [
    "I feel hopeless sometimes, but my therapist helped me realize I have strengths.",
    "Although I struggle with depression, I'm proud of myself for getting out of bed today.",
    "My anxiety is bad, but I used CBT techniques to challenge those thoughts successfully.",
    "I had a panic attack but I used the breathing exercise we practiced.",
    "I feel sad about my loss, but I'm finding ways to honor my grief.",
    "I'm worried about the future but I've made a plan to deal with it.",
]


class TestMixedContent:
    """Test that scorer appropriately weighs mixed positive/negative content."""

    @pytest.mark.parametrize(
        "prompt",
        MIXED_CONTENT_TEST_CASES,
        ids=["hopeless_strengths", "depr_achievement", "anxiety_cbt", "panic_coping", "grief_coping", "worry_planning"],
    )
    def test_mixed_content_no_crash(self, prompt):
        """Mixed content should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        MIXED_CONTENT_TEST_CASES,
        ids=["hopeless_strengths", "depr_achievement", "anxiety_cbt", "panic_coping", "grief_coping", "worry_planning"],
    )
    def test_mixed_content_returns_valid_score(self, prompt):
        """Mixed content should return a valid score within range."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Category: cultural_linguistic_variations
# ==============================================================================

CULTURAL_LINGUISTIC_TEST_CASES = [
    "Estoy feeling muy triste ultimamente y no se que hacer.",
    "I feel くるしい (suffering) but I don't want to burden anyone.",
    "Je me sens deprime et je ne vois pas d'issue a ma situation.",
    "Me siento muy ansiado y no se que hacer.",
    "I feel 难过 (sad) about the situation.",
    "Je suis tres anxieux ce matin.",
]


class TestCulturalLinguisticVariations:
    """Test that scorer handles mixed-language content."""

    @pytest.mark.parametrize(
        "prompt",
        CULTURAL_LINGUISTIC_TEST_CASES,
        ids=["sp_en", "jp_en", "fr_en", "sp_anxiety", "ch_en", "fr_anxiety"],
    )
    def test_multilingual_no_crash(self, prompt):
        """Multilingual content should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        CULTURAL_LINGUISTIC_TEST_CASES,
        ids=["sp_en", "jp_en", "fr_en", "sp_anxiety", "ch_en", "fr_anxiety"],
    )
    def test_multilingual_returns_valid_score(self, prompt):
        """Multilingual content should return a valid score."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "prompt",
        CULTURAL_LINGUISTIC_TEST_CASES,
        ids=["sp_en", "jp_en", "fr_en", "sp_anxiety", "ch_en", "fr_anxiety"],
    )
    def test_multilingual_flagged(self, prompt):
        """Significant non-English content should be flagged or score low."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert "non_english_content" in result["flags"] or result["validity_score"] < _LOW_SCORE_THRESHOLD


# ==============================================================================
# Category: extreme_length
# ==============================================================================

EXTREME_LENGTH_PROMPTS = ["a" * 10000, "I feel " + "really " * 1000 + "sad today.", "x" * 50000, "word " * 10000]
EXTREME_LENGTH_IDS = ["long_10k", "repetitive_1k", "long_50k", "repeated_10k"]


class TestExtremeLength:
    """Test that scorer handles extremely long inputs without crashing."""

    @pytest.mark.parametrize("prompt", EXTREME_LENGTH_PROMPTS, ids=EXTREME_LENGTH_IDS)
    def test_extreme_length_no_crash(self, prompt):
        """Extreme length inputs should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)
        assert "validity_score" in result

    @pytest.mark.parametrize("prompt", EXTREME_LENGTH_PROMPTS, ids=EXTREME_LENGTH_IDS)
    def test_extreme_length_returns_valid_score(self, prompt):
        """Extreme length inputs should return valid scores."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("prompt", EXTREME_LENGTH_PROMPTS, ids=EXTREME_LENGTH_IDS)
    def test_extreme_length_performance(self, prompt):
        """Extreme length inputs should complete in reasonable time."""
        start = time.time()
        score = ClinicalValidityScorer.score(prompt)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < _PERFORMANCE_TIMEOUT_MS, f"Scoring took {elapsed_ms:.0f}ms"
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Category: special_characters
# ==============================================================================

SPECIAL_CHARACTER_TEST_CASES = [
    "I feel 😢💔😞 today!!!",
    "My anxiety level is 10/10!!!!",
    "Therapy helped me <> <script>alert('xss')</script> cope better.",
    "I\t\n\rfeel\v\freally\\bad\\today",
    "What about $pecial ch@racters?!?",
    "New\n\n\nlines\nevery\n\n\nwhere",
]


class TestSpecialCharacters:
    """Test that scorer handles special characters and potential injection safely."""

    @pytest.mark.parametrize(
        "prompt",
        SPECIAL_CHARACTER_TEST_CASES,
        ids=["emoji", "numeric_punct", "xss_attempt", "escape_seq", "spec_syms", "multi_newlines"],
    )
    def test_special_chars_no_crash(self, prompt):
        """Special character inputs should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        SPECIAL_CHARACTER_TEST_CASES,
        ids=["emoji", "numeric_punct", "xss_attempt", "escape_seq", "spec_syms", "multi_newlines"],
    )
    def test_special_chars_returns_valid_score(self, prompt):
        """Special character inputs should return valid scores."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0

    def test_xss_attempt_not_executed(self):
        """XSS attempt should not be executed - just scored safely."""
        prompt = "Therapy helped me <script>alert('xss')</script> cope better."
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert "alert" not in str(result).lower() or result["validity_score"] == 0.0
        assert isinstance(result["validity_score"], float)


# ==============================================================================
# Category: numeric_and_symbols
# ==============================================================================

NUMERIC_SYMBOL_TEST_CASES = [
    "I feel 0% motivated today.",
    "My depression is at 11/10 severity.",
    "I've had 3 panic attacks this week!!",
    "Cost of therapy: $150/session x 4 sessions = $600",
    "On a scale of 1-10, my anxiety is at 9.",
    "I've cried 5 times today.",
]


class TestNumericAndSymbols:
    """Test that scorer handles numeric and symbolic content appropriately."""

    @pytest.mark.parametrize(
        "prompt",
        NUMERIC_SYMBOL_TEST_CASES,
        ids=["percentage", "over_max", "count_punct", "math_expr", "scale_rating", "cry_count"],
    )
    def test_numeric_symbols_no_crash(self, prompt):
        """Numeric/symbol content should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "prompt",
        NUMERIC_SYMBOL_TEST_CASES,
        ids=["percentage", "over_max", "count_punct", "math_expr", "scale_rating", "cry_count"],
    )
    def test_numeric_symbols_returns_valid_score(self, prompt):
        """Numeric/symbol content should return valid scores."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Additional edge cases
# ==============================================================================

ADDITIONAL_EDGE_CASES_PROMPTS = [
    # Full clinical response
    (
        "Based on the cognitive behavioral therapy framework, let's identify the automatic thoughts "
        "contributing to your depression. We'll use a thought record to examine the evidence, challenge "
        "cognitive distortions, and develop alternative perspectives. Research shows CBT is effective "
        "for depression. Between sessions, practice the behavioral activation homework we discussed."
    ),
    ("The weather is nice today. I had a sandwich for lunch."),
    ("I feel anxious."),
    ("CBT CBT CBT CBT CBT CBT CBT CBT CBT CBT"),
    ("Patient presents with anhedonia, flat affect, psychomotor retardation."),
    ("It's normal to feel this way sometimes."),
    (
        "I hear that you're experiencing significant distress. The symptoms you're describing - "
        "anhedonia, sleep disturbance, and low mood - are consistent with a depressive episode. "
        "Let's work together on a treatment plan."
    ),
]
ADDITIONAL_EDGE_CASES_IDS = [
    "full_clinical",
    "non_clinical",
    "short_clinical",
    "repetitive_tech",
    "jargon_only",
    "normalizing",
    "jargon_empathy",
]


class TestAdditionalEdgeCases:
    """Additional edge cases to ensure comprehensive coverage."""

    @pytest.mark.parametrize("prompt", ADDITIONAL_EDGE_CASES_PROMPTS, ids=ADDITIONAL_EDGE_CASES_IDS)
    def test_edge_cases_no_crash(self, prompt):
        """Edge case prompts should not crash the scorer."""
        result = ClinicalValidityScorer.score_with_flags(prompt)
        assert isinstance(result, dict)
        assert "validity_score" in result

    @pytest.mark.parametrize("prompt", ADDITIONAL_EDGE_CASES_PROMPTS, ids=ADDITIONAL_EDGE_CASES_IDS)
    def test_edge_cases_returns_valid_score(self, prompt):
        """Edge case prompts should return valid scores."""
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0

    def test_full_clinical_response_returns_valid_score(self):
        """A full clinical response should return valid score (not crash)."""
        prompt = (
            "Based on the cognitive behavioral therapy framework, let's identify the automatic thoughts "
            "contributing to your depression. We'll use a thought record to examine the evidence, challenge "
            "cognitive distortions, and develop alternative perspectives."
        )
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"

    def test_mundane_content_scores_low(self):
        """Non-clinical content should score low."""
        prompt = "The weather is nice today. I had a sandwich for lunch."
        score = ClinicalValidityScorer.score(prompt)
        assert score < _MUNDANE_CONTENT_MAX, f"Mundane content scored {score:.3f}"

    def test_repetitive_technique_names_valid(self):
        """Repetitive technique names should return valid score without division errors."""
        prompt = "CBT CBT CBT CBT CBT CBT CBT CBT CBT CBT"
        score = ClinicalValidityScorer.score(prompt)
        assert 0.0 <= score <= 1.0


# ==============================================================================
# Summary test
# ==============================================================================

_CATEGORIES = [
    "TestEmptyInputs",
    "TestContraindicatedContent",
    "TestMinimalClinicalContent",
    "TestTherapeuticTechniqueConfusion",
    "TestBoundaryViolations",
    "TestCrisisContent",
    "TestTherapeuticProgress",
    "TestMixedContent",
    "TestCulturalLinguisticVariations",
    "TestExtremeLength",
    "TestSpecialCharacters",
    "TestNumericAndSymbols",
]


class TestStressTestCoverage:
    """Summary tests to verify coverage of all categories."""

    def test_all_categories_tested(self):
        """Verify all stress test categories are represented."""
        test_classes = [
            tcsc.TestEmptyInputs,
            tcsc.TestContraindicatedContent,
            tcsc.TestMinimalClinicalContent,
            tcsc.TestTherapeuticTechniqueConfusion,
            tcsc.TestBoundaryViolations,
            tcsc.TestCrisisContent,
            tcsc.TestTherapeuticProgress,
            tcsc.TestMixedContent,
            tcsc.TestCulturalLinguisticVariations,
            tcsc.TestExtremeLength,
            tcsc.TestSpecialCharacters,
            tcsc.TestNumericAndSymbols,
        ]
        expected_categories = 12
        assert len(test_classes) == expected_categories

    def test_total_case_count(self):
        """Verify we have 60+ test cases total."""
        total = (
            len(EMPTY_INPUT_TEST_CASES)
            + len(CONTRAINDICATED_TEST_CASES)
            + len(MINIMAL_CONTENT_TEST_CASES)
            + len(TECHNIQUE_CONFUSION_TEST_CASES)
            + len(BOUNDARY_VIOLATION_TEST_CASES)
            + len(CRISIS_CONTENT_TEST_CASES)
            + len(THERAPEUTIC_PROGRESS_TEST_CASES)
            + len(MIXED_CONTENT_TEST_CASES)
            + len(CULTURAL_LINGUISTIC_TEST_CASES)
            + len(EXTREME_LENGTH_PROMPTS)
            + len(SPECIAL_CHARACTER_TEST_CASES)
            + len(NUMERIC_SYMBOL_TEST_CASES)
            + len(ADDITIONAL_EDGE_CASES_PROMPTS)
        )
        assert total >= _MIN_TEST_CASE_COUNT, f"Expected at least {_MIN_TEST_CASE_COUNT} test cases, got {total}"
