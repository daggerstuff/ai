"""Tests for the inquiry-type classification system (PIX-3908, Task 2).

Verifies the rule-based classifier, the Liebig-bottleneck analysis, and
the session distribution helpers. These tests are intentionally
deterministic — no LLM judge is exercised.
"""

import unittest

from ai.tools.utilities.core.pipelines.quality.conversation_quality_pattern_analyzer import (
    RECOMMENDED_RATIOS,
    HallucinationDetector,
    HallucinationFinding,
    HallucinationReport,
    HallucinationSeverity,
    InquiryType,
    InquiryTypeClassifier,
    SessionClassification,
    UtteranceClassification,
    liebig_bottleneck,
    liebig_quality_score,
    session_distribution,
)
from ai.tools.utilities.core.pipelines.schemas.conversation_schema import Conversation, Message


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


# ===========================================================================
# Tests for the Hallucination Detection System (PIX-3908, Task 6)
# ===========================================================================
#
# A realistic MDD (Major Depressive Disorder) CCD case profile used
# across most tests. Mirrors the structure produced by
# ``ClinicalProfile.ccd_config`` and ``PatientCCD.to_dict()``.


MDD_CASE: dict = {
    "description": "Major depressive disorder with persistent sad mood",
    "diagnoses": ["F32.2", "F33.2"],
    "typical_symptoms": [
        "persistent sad mood",
        "fatigue",
        "feelings of worthlessness",
        "loss of interest",
        "sleep disturbance",
    ],
    "core_beliefs": [
        {"content": "I am fundamentally worthless", "domain": "self", "conviction": 0.92},
        {"content": "The world is empty and meaningless", "domain": "world", "conviction": 0.88},
        {"content": "There is no hope for improvement", "domain": "future", "conviction": 0.85},
    ],
    "intermediate_beliefs": [
        {"content": "If I am not perfect, I am a failure", "rule_type": "rule", "conviction": 0.85},
    ],
    "emotional_responses": [
        {"emotion": "sadness", "intensity": 0.9, "valence": "negative"},
        {"emotion": "hopelessness", "intensity": 0.85, "valence": "negative"},
        {"emotion": "worthlessness", "intensity": 0.88, "valence": "negative"},
    ],
    "behavioral_responses": [
        {"behavior": "social withdrawal", "triggered_by": "low mood"},
        {"behavior": "rumination about past failures", "triggered_by": "negative events"},
    ],
    "coping_strategies": [
        {"content": "Social withdrawal", "strategy_type": "avoidance", "effectiveness": 0.2},
        {"content": "Oversleeping to escape distress", "strategy_type": "avoidance", "effectiveness": 0.3},
    ],
    "situation_interpretations": [
        {
            "situation": "Receiving feedback at work",
            "interpretation": "Criticism confirms my worthlessness",
            "distortion_type": "personalization",
        },
    ],
}


class TestHallucinationSeverity(unittest.TestCase):
    """Test the severity enum and its numeric weights."""

    def test_severity_values(self) -> None:
        self.assertEqual(HallucinationSeverity.LOW.value, "low")
        self.assertEqual(HallucinationSeverity.MEDIUM.value, "medium")
        self.assertEqual(HallucinationSeverity.HIGH.value, "high")
        self.assertEqual(HallucinationSeverity.CRITICAL.value, "critical")

    def test_severity_numeric_weights(self) -> None:
        self.assertEqual(HallucinationSeverity.LOW.numeric, 0.25)
        self.assertEqual(HallucinationSeverity.MEDIUM.numeric, 0.5)
        self.assertEqual(HallucinationSeverity.HIGH.numeric, 0.75)
        self.assertEqual(HallucinationSeverity.CRITICAL.numeric, 1.0)

    def test_severity_ordering(self) -> None:
        self.assertLess(
            HallucinationSeverity.LOW.numeric,
            HallucinationSeverity.MEDIUM.numeric,
        )
        self.assertLess(
            HallucinationSeverity.MEDIUM.numeric,
            HallucinationSeverity.HIGH.numeric,
        )
        self.assertLess(
            HallucinationSeverity.HIGH.numeric,
            HallucinationSeverity.CRITICAL.numeric,
        )


class TestHallucinationFinding(unittest.TestCase):
    """Test the finding dataclass."""

    def test_finding_creation(self) -> None:
        f = HallucinationFinding(
            detection_type="scope_compliance",
            severity=HallucinationSeverity.HIGH,
            description="Test finding",
            evidence="flashback",
            expected="symptom from profile",
        )
        self.assertEqual(f.detection_type, "scope_compliance")
        self.assertEqual(f.severity, HallucinationSeverity.HIGH)
        self.assertEqual(f.description, "Test finding")
        self.assertEqual(f.evidence, "flashback")
        self.assertEqual(f.expected, "symptom from profile")

    def test_finding_is_critical(self) -> None:
        f = HallucinationFinding(
            detection_type="scope_compliance",
            severity=HallucinationSeverity.CRITICAL,
            description="Critical",
        )
        self.assertTrue(f.is_critical)

        f2 = HallucinationFinding(
            detection_type="scope_compliance",
            severity=HallucinationSeverity.HIGH,
            description="High",
        )
        self.assertFalse(f2.is_critical)

    def test_finding_is_frozen(self) -> None:
        f = HallucinationFinding(
            detection_type="scope_compliance",
            severity=HallucinationSeverity.HIGH,
            description="Test",
        )
        with self.assertRaises(Exception):
            f.description = "Modified"  # type: ignore[misc]


class TestHallucinationReport(unittest.TestCase):
    """Test the report aggregation logic."""

    def test_empty_report(self) -> None:
        report = HallucinationReport(response="hello")
        self.assertIsNone(report.overall_severity)
        self.assertEqual(report.hallucination_rate, 0.0)
        self.assertFalse(report.is_hallucinated)
        self.assertEqual(report.findings, [])

    def test_report_with_low_severity_only(self) -> None:
        report = HallucinationReport(
            response="test",
            findings=[
                HallucinationFinding(
                    detection_type="numerical_accuracy",
                    severity=HallucinationSeverity.LOW,
                    description="Low",
                ),
            ],
        )
        self.assertEqual(report.overall_severity, HallucinationSeverity.LOW)
        # LOW (0.25) is below the 0.5 threshold for being a hallucination
        self.assertEqual(report.hallucination_rate, 0.0)
        self.assertFalse(report.is_hallucinated)

    def test_report_with_critical_severity(self) -> None:
        report = HallucinationReport(
            response="test",
            findings=[
                HallucinationFinding(
                    detection_type="scope_compliance",
                    severity=HallucinationSeverity.CRITICAL,
                    description="Critical",
                ),
            ],
        )
        self.assertEqual(report.overall_severity, HallucinationSeverity.CRITICAL)
        self.assertEqual(report.hallucination_rate, 1.0)
        self.assertTrue(report.is_hallucinated)

    def test_report_overall_severity_is_max(self) -> None:
        report = HallucinationReport(
            response="test",
            findings=[
                HallucinationFinding(
                    detection_type="a", severity=HallucinationSeverity.LOW, description="low"
                ),
                HallucinationFinding(
                    detection_type="b", severity=HallucinationSeverity.HIGH, description="high"
                ),
                HallucinationFinding(
                    detection_type="c", severity=HallucinationSeverity.MEDIUM, description="medium"
                ),
            ],
        )
        self.assertEqual(report.overall_severity, HallucinationSeverity.HIGH)

    def test_report_to_dict(self) -> None:
        report = HallucinationReport(
            response="test",
            findings=[
                HallucinationFinding(
                    detection_type="scope_compliance",
                    severity=HallucinationSeverity.HIGH,
                    description="Test",
                    evidence="flashback",
                    expected="from profile",
                ),
            ],
        )
        d = report.to_dict()
        self.assertEqual(d["response"], "test")
        self.assertEqual(d["overall_severity"], "high")
        self.assertEqual(d["hallucination_rate"], 1.0)
        self.assertTrue(d["is_hallucinated"])
        self.assertEqual(len(d["findings"]), 1)
        self.assertEqual(d["findings"][0]["detection_type"], "scope_compliance")
        self.assertEqual(d["findings"][0]["severity"], "high")


class TestHallucinationDetectorDetect(unittest.TestCase):
    """Test the integrated detect() method."""

    def test_detect_clean_response(self) -> None:
        """A response consistent with the case data should not be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        report = detector.detect(
            "I have been feeling very sad and worthless. I don't have much energy."
        )
        self.assertFalse(
            report.is_hallucinated,
            f"Clean response was incorrectly flagged: {report.findings}",
        )

    def test_detect_response_with_synonyms(self) -> None:
        """Synonyms (tired, exhausted) should not be flagged when the
        profile contains the base word (fatigue)."""
        detector = HallucinationDetector(MDD_CASE)
        report = detector.detect("I have been feeling exhausted and tired lately.")
        self.assertFalse(
            report.is_hallucinated,
            f"Synonym response was incorrectly flagged: {report.findings}",
        )

    def test_detect_response_with_new_symptoms(self) -> None:
        """A response mentioning symptoms not in the case data should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        report = detector.detect(
            "I have been having flashbacks and panic attacks constantly."
        )
        self.assertTrue(report.is_hallucinated)
        scope_findings = [
            f for f in report.findings if f.detection_type == "scope_compliance"
        ]
        self.assertGreater(len(scope_findings), 0)

    def test_detect_response_with_new_diagnosis(self) -> None:
        """A response mentioning a diagnosis not in the profile should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        report = detector.detect("I think I have schizophrenia.")
        self.assertTrue(report.is_hallucinated)
        diag_findings = [
            f for f in report.findings
            if f.detection_type == "scope_compliance"
            and "schizophrenia" in f.description
        ]
        self.assertGreater(len(diag_findings), 0)

    def test_detect_empty_response(self) -> None:
        """An empty response should not crash."""
        detector = HallucinationDetector(MDD_CASE)
        report = detector.detect("")
        self.assertEqual(report.findings, [])
        self.assertFalse(report.is_hallucinated)

    def test_detect_no_case_data(self) -> None:
        """Without case data, the detector should flag everything as critical."""
        detector = HallucinationDetector()
        report = detector.detect("I feel sad.")
        self.assertTrue(report.is_hallucinated)
        self.assertEqual(
            report.overall_severity, HallucinationSeverity.CRITICAL
        )


class TestFactualConsistency(unittest.TestCase):
    """Test the check_factual_consistency() method."""

    def test_no_findings_for_consistent_response(self) -> None:
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_factual_consistency(
            "I feel worthless and the world feels empty."
        )
        self.assertEqual(findings, [])

    def test_detects_negated_profile_fact(self) -> None:
        """Negating a fact from the profile should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_factual_consistency(
            "I am not worthless at all."
        )
        neg_findings = [
            f for f in findings
            if f.detection_type == "factual_consistency"
            and f.severity == HallucinationSeverity.HIGH
        ]
        self.assertGreater(
            len(neg_findings), 0,
            f"Expected negation finding, got: {findings}",
        )

    def test_detects_novel_symptom(self) -> None:
        """A response mentioning a symptom not in the profile should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_factual_consistency(
            "I've been having flashbacks and obsessing over things."
        )
        symptom_findings = [
            f for f in findings if "symptom" in f.description.lower()
        ]
        self.assertGreater(
            len(symptom_findings), 0,
            f"Expected symptom finding, got: {findings}",
        )


class TestTemporalConsistency(unittest.TestCase):
    """Test the check_temporal_consistency() method."""

    def test_no_history_no_check(self) -> None:
        """Without session history, temporal check returns no findings."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_temporal_consistency(
            "As I said before, I feel sad.", session_history=[]
        )
        self.assertEqual(findings, [])

    def test_contradicts_history(self) -> None:
        """Claiming 'I never said that' when history shows they did."""
        detector = HallucinationDetector(MDD_CASE)
        history = ["I told you I'm feeling worthless."]
        findings = detector.check_temporal_consistency(
            "I never said that I'm worthless.",
            session_history=history,
        )
        contradiction_findings = [
            f for f in findings
            if f.detection_type == "temporal_consistency"
            and f.severity == HallucinationSeverity.HIGH
        ]
        self.assertGreater(
            len(contradiction_findings), 0,
            f"Expected contradiction finding, got: {findings}",
        )

    def test_consistent_with_history(self) -> None:
        """A response that doesn't contradict history should be fine."""
        detector = HallucinationDetector(MDD_CASE)
        history = ["I told you I'm feeling sad."]
        findings = detector.check_temporal_consistency(
            "Yes, I do feel very sad today.",
            session_history=history,
        )
        contradiction_findings = [
            f for f in findings
            if f.severity == HallucinationSeverity.HIGH
        ]
        self.assertEqual(contradiction_findings, [])


class TestNumericalAccuracy(unittest.TestCase):
    """Test the check_numerical_accuracy() method."""

    def test_no_findings_for_unrelated_numbers(self) -> None:
        """Numbers not tied to profile metrics should not be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_numerical_accuracy("I ate 3 meals today.")
        self.assertEqual(findings, [])

    def test_detects_inconsistent_conviction(self) -> None:
        """A conviction value contradicting the profile should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_numerical_accuracy(
            "My conviction is 0.1 for being worthless."
        )
        num_findings = [
            f for f in findings if f.detection_type == "numerical_accuracy"
        ]
        self.assertGreater(
            len(num_findings), 0,
            f"Expected numerical finding, got: {findings}",
        )

    def test_detects_unbacked_percentage(self) -> None:
        """A specific percentage claim without backing should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_numerical_accuracy("I'm 90% better now.")
        pct_findings = [
            f for f in findings
            if f.detection_type == "numerical_accuracy"
            and f.severity == HallucinationSeverity.LOW
        ]
        self.assertGreater(len(pct_findings), 0)


class TestScopeCompliance(unittest.TestCase):
    """Test the check_scope_compliance() method."""

    def test_no_case_data_flags_critical(self) -> None:
        """Without case data, everything is a critical scope violation."""
        detector = HallucinationDetector()
        findings = detector.check_scope_compliance("I feel sad.")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, HallucinationSeverity.CRITICAL)

    def test_in_scope_response_no_findings(self) -> None:
        """A response using profile vocabulary should not be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_scope_compliance(
            "I feel sad and worthless. I have no energy."
        )
        self.assertEqual(findings, [])

    def test_out_of_scope_symptom(self) -> None:
        """A response introducing a new symptom should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_scope_compliance(
            "I've been having flashbacks of my trauma."
        )
        symptom_findings = [
            f for f in findings
            if f.detection_type == "scope_compliance"
            and "symptom" in f.description.lower()
            and f.severity == HallucinationSeverity.HIGH
        ]
        self.assertGreater(len(symptom_findings), 0)

    def test_out_of_scope_diagnosis(self) -> None:
        """A response mentioning a new diagnosis should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_scope_compliance("I think I have PTSD.")
        diag_findings = [
            f for f in findings
            if f.detection_type == "scope_compliance"
            and "ptsd" in f.description.lower()
        ]
        self.assertGreater(len(diag_findings), 0)

    def test_out_of_scope_emotion(self) -> None:
        """A response mentioning new emotions should be flagged."""
        detector = HallucinationDetector(MDD_CASE)
        findings = detector.check_scope_compliance(
            "I feel so guilty all the time."  # guilt not in emotional_responses
        )
        emotion_findings = [
            f for f in findings
            if f.detection_type == "scope_compliance"
            and "emotional" in f.description.lower()
        ]
        self.assertGreater(len(emotion_findings), 0)


class TestVerifySession(unittest.TestCase):
    """Test the post-hoc session verification."""

    def test_verify_clean_session(self) -> None:
        """A clean session should have 0% hallucination rate."""
        detector = HallucinationDetector(MDD_CASE)
        turns = [
            ("How are you feeling?", "I feel very sad and worthless."),
            ("Tell me more.", "I have no energy to do anything anymore."),
        ]
        result = detector.verify_session(turns)
        self.assertEqual(result["total_responses"], 2)
        self.assertEqual(result["hallucinated_count"], 0)
        self.assertEqual(result["hallucination_rate"], 0.0)
        self.assertEqual(len(result["reports"]), 2)

    def test_verify_hallucinated_session(self) -> None:
        """A session with hallucinations should be detected."""
        detector = HallucinationDetector(MDD_CASE)
        turns = [
            ("How are you feeling?", "I have flashbacks and panic attacks daily."),
            ("Tell me more.", "I think I might have schizophrenia."),
        ]
        result = detector.verify_session(turns)
        self.assertEqual(result["total_responses"], 2)
        self.assertGreater(result["hallucinated_count"], 0)
        self.assertGreater(result["hallucination_rate"], 0.0)

    def test_verify_mixed_session(self) -> None:
        """A session with some clean and some hallucinated turns."""
        detector = HallucinationDetector(MDD_CASE)
        turns = [
            ("How are you feeling?", "I feel very sad and worthless."),
            ("Tell me more.", "I also have flashbacks and panic attacks."),
            ("What helps?", "Sometimes rest and talking to friends helps."),
        ]
        result = detector.verify_session(turns)
        self.assertEqual(result["total_responses"], 3)
        self.assertGreaterEqual(result["hallucinated_count"], 1)
        self.assertGreater(result["hallucination_rate"], 0.0)
        self.assertLess(result["hallucination_rate"], 1.0)

    def test_verify_empty_session(self) -> None:
        """An empty session should return 0% hallucination rate."""
        detector = HallucinationDetector(MDD_CASE)
        result = detector.verify_session([])
        self.assertEqual(result["total_responses"], 0)
        self.assertEqual(result["hallucination_rate"], 0.0)
        self.assertEqual(result["reports"], [])

    def test_session_history_accumulates(self) -> None:
        """Session history should accumulate across turns for temporal checking."""
        detector = HallucinationDetector(MDD_CASE)
        turns = [
            ("How are you?", "I feel sad and worthless."),
            ("Can you tell me more?", "I said earlier I have no energy."),
        ]
        result = detector.verify_session(turns)
        self.assertEqual(result["total_responses"], 2)


class TestIntegrationWithPatientPsi(unittest.TestCase):
    """Test the detector with a realistic Patient-Ψ CCD profile."""

    def test_mdd_profile_clean_response(self) -> None:
        """A response that uses the MDD profile's vocabulary should be clean."""
        mdd_profile = {
            "description": "Persistent low mood, anhedonia, worthlessness beliefs",
            "diagnoses": ["F32.2"],
            "typical_symptoms": [
                "persistent sad mood",
                "fatigue",
                "feelings of worthlessness",
                "loss of interest",
            ],
            "core_beliefs": [
                {"content": "I am worthless", "domain": "self", "conviction": 0.9},
            ],
            "emotional_responses": [
                {"emotion": "sadness", "intensity": 0.9, "valence": "negative"},
                {"emotion": "hopelessness", "intensity": 0.85, "valence": "negative"},
            ],
            "behavioral_responses": [
                {"behavior": "social withdrawal", "triggered_by": "low mood"},
            ],
        }
        detector = HallucinationDetector(mdd_profile)
        report = detector.detect(
            "I feel sad and hopeless. I've been withdrawing from my friends."
        )
        self.assertFalse(
            report.is_hallucinated,
            f"Clean MDD response was flagged: {report.findings}",
        )

    def test_gad_profile_anxiety_response(self) -> None:
        """A response consistent with a GAD profile should be clean."""
        gad_profile = {
            "description": "Generalized anxiety with excessive worry",
            "diagnoses": ["F41.1"],
            "typical_symptoms": [
                "excessive worry",
                "restlessness",
                "fatigue",
                "difficulty concentrating",
                "irritability",
            ],
            "core_beliefs": [
                {"content": "Something bad will happen", "domain": "future", "conviction": 0.85},
            ],
            "emotional_responses": [
                {"emotion": "anxiety", "intensity": 0.9, "valence": "negative"},
            ],
            "behavioral_responses": [
                {"behavior": "avoidance of feared situations", "triggered_by": "anxiety"},
            ],
        }
        detector = HallucinationDetector(gad_profile)
        # Clean response using profile vocabulary + synonyms
        report = detector.detect(
            "I feel anxious and worried all the time. I can't concentrate."
        )
        self.assertFalse(
            report.is_hallucinated,
            f"Clean GAD response was flagged: {report.findings}",
        )

        # Hallucinated response introducing new symptoms
        report2 = detector.detect("I have flashbacks and hear voices.")
        self.assertTrue(report2.is_hallucinated)


if __name__ == "__main__":
    unittest.main()
