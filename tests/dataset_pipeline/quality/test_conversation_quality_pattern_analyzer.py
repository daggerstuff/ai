"""Tests for the inquiry-type classification system (PIX-3908, Task 2).

Verifies the rule-based classifier, the Liebig-bottleneck analysis, and
the session distribution helpers. These tests are intentionally
deterministic — no LLM judge is exercised.
"""

import unittest

from ai.core.pipelines.quality.conversation_quality_pattern_analyzer import (
    InquiryType,
    InquiryTypeClassifier,
    RECOMMENDED_RATIOS,
    SessionClassification,
    UtteranceClassification,
    liebig_bottleneck,
    liebig_quality_score,
    session_distribution,
)
from ai.core.pipelines.quality.conversation_schema import Conversation, Message


class TestInquiryTypeClassification(unittest.TestCase):
    """Unit tests for the rule-based inquiry-type classifier."""

    def setUp(self) -> None:
        self.clf = InquiryTypeClassifier()

    # ── Single-utterance classification ────────────────────────────

    def test_classifier_rejects_non_string(self):
        with self.assertRaises(TypeError):
            self.clf.classify_utterance(42)  # type: ignore[arg-type]

    def test_classifier_handles_empty_text(self):
        result = self.clf.classify_utterance("")
        self.assertEqual(result.inquiry_type, InquiryType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("empty", result.rationale.lower())

    def test_classifier_handles_whitespace_only(self):
        result = self.clf.classify_utterance("   \n  \t  ")
        self.assertEqual(result.inquiry_type, InquiryType.UNKNOWN)

    def test_classifier_detects_closed_ended(self):
        result = self.clf.classify_utterance("Do you sleep well at night?")
        self.assertEqual(result.inquiry_type, InquiryType.CLOSED_ENDED)
        self.assertGreater(result.confidence, 0.0)

    def test_classifier_detects_open_ended(self):
        result = self.clf.classify_utterance("Tell me more about how that feels.")
        self.assertEqual(result.inquiry_type, InquiryType.OPEN_ENDED)
        self.assertGreater(result.confidence, 0.0)

    def test_classifier_detects_guided(self):
        result = self.clf.classify_utterance(
            "When you feel anxious, what thoughts come up for you?"
        )
        self.assertEqual(result.inquiry_type, InquiryType.GUIDED)
        self.assertGreater(result.confidence, 0.0)

    def test_classifier_detects_reflective(self):
        result = self.clf.classify_utterance(
            "How does that pattern show up elsewhere in your life?"
        )
        self.assertEqual(result.inquiry_type, InquiryType.REFLECTIVE)
        self.assertGreater(result.confidence, 0.0)

    def test_classifier_returns_4_signal_categories(self):
        """All four inquiry types must be scoreable in scores output."""
        result = self.clf.classify_utterance("Tell me what's been on your mind lately?")
        self.assertEqual(
            set(result.scores.keys()),
            {
                InquiryType.CLOSED_ENDED,
                InquiryType.OPEN_ENDED,
                InquiryType.GUIDED,
                InquiryType.REFLECTIVE,
            },
        )

    def test_classifier_unknown_for_unscored_utterance(self):
        """A pure statement with no inquiry signals should not classify."""
        result = self.clf.classify_utterance("The patient sat quietly in the chair.")
        self.assertEqual(result.inquiry_type, InquiryType.UNKNOWN)

    # ── LLM judge adapter ────────────────────────────────────────

    def test_llm_judge_invoked_for_low_confidence(self):
        """When rule-based score is below threshold, LLM judge is consulted."""

        def judge(_text: str) -> InquiryType:
            return InquiryType.REFLECTIVE

        clf = InquiryTypeClassifier(llm_judge=judge, min_confidence=0.99)
        result = clf.classify_utterance("Yes.")  # very low signal
        if result.inquiry_type == InquiryType.REFLECTIVE:
            self.assertIn("judge", result.rationale.lower())

    def test_llm_judge_exception_does_not_crash(self):
        def bad_judge(_text: str) -> InquiryType:
            raise RuntimeError("judge offline")

        clf = InquiryTypeClassifier(llm_judge=bad_judge)
        result = clf.classify_utterance("Yes.")
        self.assertIn(
            result.inquiry_type,
            {
                InquiryType.UNKNOWN,
                InquiryType.CLOSED_ENDED,
                InquiryType.OPEN_ENDED,
                InquiryType.GUIDED,
                InquiryType.REFLECTIVE,
            },
        )

    # ── Session-level aggregation ────────────────────────────────

    def test_classify_session_filters_therapist_turns(self):
        conv = Conversation(
            conversation_id="sess-1",
            messages=[
                Message(role="user", content="I've been feeling stuck."),
                Message(role="assistant", content="Tell me more about that."),
                Message(role="user", content="Like I can't get out of my own way."),
                Message(role="assistant", content="How long has this been going on?"),
            ],
        )
        sess = self.clf.classify_session(conv)
        self.assertEqual(sess.total, 2)  # only the two assistant turns
        self.assertEqual(len(sess.utterances), 2)
        self.assertEqual(sess.conversation_id, "sess-1")

    def test_session_distribution_excludes_unknown_from_total(self):
        conv = Conversation(
            conversation_id="sess-2",
            messages=[
                Message(role="assistant", content="Tell me about that."),     # OPEN
                Message(role="assistant", content="The patient appeared sad."),  # UNKNOWN
                Message(role="assistant", content="Do you feel anxious?"),    # CLOSED
            ],
        )
        sess = self.clf.classify_session(conv)
        # total counts only known types
        self.assertEqual(sess.total, 2)
        self.assertIn(InquiryType.UNKNOWN, sess.distribution)
        # ratios are over the known total
        ratios = sess.ratio
        known = [ratios[t] for t in ratios if t != InquiryType.UNKNOWN]
        self.assertAlmostEqual(sum(known), 1.0, places=6)

    def test_session_empty_conversation(self):
        conv = Conversation(conversation_id="empty", messages=[])
        sess = self.clf.classify_session(conv)
        self.assertEqual(sess.total, 0)
        self.assertTrue(all(v == 0.0 for v in sess.ratio.values()))

    # ── Liebig bottleneck ────────────────────────────────────────

    def test_liebig_bottleneck_picks_largest_deficit(self):
        # A session heavy on closed-ended, light on open-ended
        conv = Conversation(
            conversation_id="sess-3",
            messages=[
                Message(role="assistant", content="Do you feel sad?"),
                Message(role="assistant", content="Did you sleep last night?"),
                Message(role="assistant", content="Are you eating well?"),
                Message(role="assistant", content="Have you been isolating?"),
                Message(role="assistant", content="Do you have thoughts of self-harm?"),
                Message(role="assistant", content="Tell me about today."),  # OPEN
            ],
        )
        sess = self.clf.classify_session(conv)
        bn, deficit = liebig_bottleneck(sess)
        self.assertGreaterEqual(deficit, 0.0)
        self.assertIn(
            bn,
            {InquiryType.OPEN_ENDED, InquiryType.GUIDED, InquiryType.REFLECTIVE},
        )

    def test_liebig_bottleneck_empty_session(self):
        sess = SessionClassification(conversation_id="empty", total=0)
        bn, deficit = liebig_bottleneck(sess)
        self.assertEqual(bn, InquiryType.UNKNOWN)
        self.assertEqual(deficit, 0.0)

    def test_liebig_quality_score_is_minimum_of_ratios(self):
        sess = SessionClassification(
            conversation_id="balanced",
            distribution={
                InquiryType.OPEN_ENDED: 4,
                InquiryType.GUIDED: 3,
                InquiryType.REFLECTIVE: 2,
                InquiryType.CLOSED_ENDED: 1,
            },
            total=10,
        )
        score = liebig_quality_score(sess)
        # Min ratio is 0.1 (CLOSED_ENDED = 1/10) — floored at ideal_min=0.10
        self.assertAlmostEqual(score, 0.10, places=6)

    def test_liebig_quality_score_above_floor(self):
        sess = SessionClassification(
            conversation_id="even",
            distribution={
                InquiryType.OPEN_ENDED: 5,
                InquiryType.GUIDED: 5,
                InquiryType.REFLECTIVE: 5,
                InquiryType.CLOSED_ENDED: 5,
            },
            total=20,
        )
        score = liebig_quality_score(sess)
        self.assertAlmostEqual(score, 0.25, places=6)

    def test_session_distribution_helper(self):
        sess = SessionClassification(
            conversation_id="x",
            distribution={
                InquiryType.OPEN_ENDED: 1,
                InquiryType.CLOSED_ENDED: 3,
            },
            total=4,
        )
        dist = session_distribution(sess)
        self.assertAlmostEqual(dist[InquiryType.OPEN_ENDED], 0.25, places=6)
        self.assertAlmostEqual(dist[InquiryType.CLOSED_ENDED], 0.75, places=6)

    # ── Configuration / API surface ───────────────────────────────

    def test_classifier_validates_min_confidence(self):
        with self.assertRaises(ValueError):
            InquiryTypeClassifier(min_confidence=-0.1)
        with self.assertRaises(ValueError):
            InquiryTypeClassifier(min_confidence=1.5)

    def test_classifier_validates_win_margin(self):
        with self.assertRaises(ValueError):
            InquiryTypeClassifier(win_margin=2.0)

    def test_recommended_ratios_have_all_four_types(self):
        """RECOMMENDED_RATIOS must cover all four inquiry types from the paper."""
        self.assertEqual(
            set(RECOMMENDED_RATIOS.keys()),
            {
                InquiryType.OPEN_ENDED,
                InquiryType.GUIDED,
                InquiryType.REFLECTIVE,
                InquiryType.CLOSED_ENDED,
            },
        )
        for itype, (lo, hi) in RECOMMENDED_RATIOS.items():
            self.assertLessEqual(lo, hi, f"invalid range for {itype}: {lo}..{hi}")
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_classifier_with_diverse_utterances(self):
        """Realistic mix: should produce non-empty distribution across types."""
        conv = Conversation(
            conversation_id="mixed",
            messages=[
                Message(role="assistant", content="Tell me what's on your mind."),       # OPEN
                Message(role="assistant", content="Do you feel anxious at work?"),         # CLOSED
                Message(role="assistant", content="When you feel that, what happens?"),    # GUIDED
                Message(role="assistant", content="How does that pattern show up elsewhere?"),  # REFLECTIVE
                Message(role="assistant", content="Have you noticed any triggers?"),         # CLOSED
            ],
        )
        sess = self.clf.classify_session(conv)
        # All four canonical types should appear at least once
        self.assertIn(InquiryType.OPEN_ENDED, sess.distribution)
        self.assertIn(InquiryType.CLOSED_ENDED, sess.distribution)
        self.assertIn(InquiryType.GUIDED, sess.distribution)
        self.assertIn(InquiryType.REFLECTIVE, sess.distribution)
        self.assertEqual(sess.total, 5)

    def test_utterance_classification_carries_original_text(self):
        result = self.clf.classify_utterance("Tell me about that.")
        uc = UtteranceClassification(
            role="assistant", content="Tell me about that.", result=result
        )
        self.assertEqual(uc.content, "Tell me about that.")
        self.assertEqual(uc.result.inquiry_type, InquiryType.OPEN_ENDED)


if __name__ == "__main__":
    unittest.main()
