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
            "Let's explore some coping strategies you can practice between sessions.",
            "structure"
        )
        assert score > 0.0

    def test_cultural_detects_awareness(self):
        score = ClinicalValidityScorer._score_dimension(
            "How does your cultural background influence your perspective on this?",
            "cultural"
        )
        assert score > 0.0

    def test_ebp_detects_research_reference(self):
        score = ClinicalValidityScorer._score_dimension(
            "Research shows that CBT is effective for treating this. "
            "Evidence-based practice supports this approach.",
            "ebp"
        )
        assert score > 0.0
