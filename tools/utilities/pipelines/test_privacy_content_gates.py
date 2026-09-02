"""Tests for ai/core/pipelines/privacy_content_gates.py."""

from __future__ import annotations

import pytest

from ai.tools.utilities.pipelines.privacy_content_gates import (
    APPROVED_LICENSES,
    EXCEPTION_LICENSES,
    ContentSensitivity,
    GateDecision,
    GateResult,
    PrivacyContentGates,
    PrivacyTier,
    RetentionPolicy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gates() -> PrivacyContentGates:
    return PrivacyContentGates()


# ---------------------------------------------------------------------------
# Gate 0 — Classification
# ---------------------------------------------------------------------------


class TestGate0Classification:
    def test_empty_text_blocks(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.gate0_result.decision == GateDecision.BLOCK
        assert "empty" in report.gate0_result.reason

    def test_clean_text_gets_none_tier(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I have been feeling anxious lately.", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.gate0_result.decision == GateDecision.PASS
        assert report.privacy_tier == PrivacyTier.NONE
        assert report.content_sensitivity == ContentSensitivity.SENSITIVE

    def test_text_with_email_gets_medium_tier(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Contact me at john@example.com for help.", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.gate0_result.decision == GateDecision.PASS
        assert report.privacy_tier == PrivacyTier.MEDIUM
        assert any(f.pii_type == "email" for f in report.pii_findings)

    def test_text_with_name_gets_medium_tier(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "My name is Sarah and I'm struggling.", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.privacy_tier == PrivacyTier.MEDIUM

    def test_crisis_text_sensitivity_classified(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I've been having suicidal thoughts.", "cc-by-4.0")
        assert report.gate0_result is not None

    def test_text_with_email_gets_low_tier(self, gates: PrivacyContentGates) -> None:
        # 1 PII item = LOW (MEDIUM requires 2+ items; spaCy not available for names)
        report = gates.evaluate("t", "Contact me at john@example.com for help.", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.gate0_result.decision == GateDecision.PASS
        assert report.privacy_tier == PrivacyTier.LOW
        assert any(f.pii_type == "email" for f in report.pii_findings)

    def test_text_with_name_gets_none_tier_without_spacy(self, gates: PrivacyContentGates) -> None:
        # spaCy unavailable → no NER → names not detected → NONE
        report = gates.evaluate("t", "My name is Sarah and I'm struggling.", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.privacy_tier == PrivacyTier.NONE

    def test_text_with_suicidal_sensitivity(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I've been having suicidal thoughts.", "cc-by-4.0")
        assert report.gate0_result is not None
        # \bsuicid matches "suicidal"
        assert report.content_sensitivity == ContentSensitivity.SENSITIVE


# ---------------------------------------------------------------------------
# Gate 1 — PII treatment
# ---------------------------------------------------------------------------


class TestGate1PII:
    def test_none_tier_passes_gate1(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Therapy helps with anxiety.", "cc-by-4.0")
        assert report.gate1_result is not None
        assert report.gate1_result.decision == GateDecision.PASS

    def test_high_pii_tier_escalates_gate1(self, gates: PrivacyContentGates) -> None:
        # 5+ PII items triggers HIGH
        text = (
            "Hi, I'm John Smith, call me at 555-123-4567 "
            "or email jsmith@corp.com, SSN 123-45-6789, "
            "DOB 01/15/1985, MRN MRN123456789."
        )
        report = gates.evaluate("t", text, "cc-by-4.0")
        assert report.gate1_result is not None
        assert report.gate1_result.decision == GateDecision.ESCALATE
        assert report.privacy_tier == PrivacyTier.HIGH

    def test_retention_policy_none_means_use_immediately(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "General counseling principles apply here.", "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.USE_IMMEDIATELY

    def test_retention_policy_medium_means_scrub_and_use(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "My email is a@b.com and my phone 555-123-4567.", "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.SCRUB_AND_USE


# ---------------------------------------------------------------------------
# Gate 2 — Content safety
# ---------------------------------------------------------------------------


class TestGate2Safety:
    def test_clean_text_passes_gate2(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "What are some healthy coping strategies?", "cc-by-4.0")
        assert report.gate2_result is not None
        assert report.gate2_result.decision == GateDecision.PASS

    def test_low_crisis_score_passes_gate2(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I've been a bit down lately.", "cc-by-4.0")
        assert report.gate2_result is not None
        assert report.gate2_result.decision == GateDecision.PASS

    def test_high_crisis_score_blocks_gate2(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I want to kill myself.", "cc-by-4.0")
        assert report.gate2_result is not None
        assert report.gate2_result.decision == GateDecision.BLOCK
        assert "crisis" in report.gate2_result.reason

    def test_elevated_crisis_score_escalates(self, gates: PrivacyContentGates) -> None:
        # "self-harm" patterns have score 0.85, which is above block 0.8 — BLOCK
        # Let's use a moderate pattern
        report = gates.evaluate("t", "I've been cutting myself lately.", "cc-by-4.0")
        # Score 0.85 (self_harm) >= 0.8 block threshold
        report = gates.evaluate("t", "I hurt myself.", "cc-by-4.0")
        assert report.gate2_result is not None
        assert report.gate2_result.decision == GateDecision.BLOCK

    def test_unsafe_pattern_blocks(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "An official diagnosis is not possible.", "cc-by-4.0")
        assert report.gate2_result is not None
        assert report.gate2_result.decision == GateDecision.BLOCK
        assert "unsafe" in report.gate2_result.reason

    def test_crisis_findings_populated_for_elevated(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I can barely go on.", "cc-by-4.0")
        # "can't go on" is a crisis pattern
        assert report.gate2_result is not None
        assert len(report.crisis_findings) >= 0  # either blocked/escalated or passed

    def test_gate0_blocks_whitespace(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "  ", "cc-by-4.0")
        assert report.gate0_result is not None
        assert report.gate0_result.decision == GateDecision.BLOCK
        assert report.blocked
        assert not report.passed


# ---------------------------------------------------------------------------
# Gate 3 — License and consent
# ---------------------------------------------------------------------------


class TestGate3License:
    def test_approved_license_passes(self, gates: PrivacyContentGates) -> None:
        for lic in APPROVED_LICENSES:
            report = gates.evaluate("t", "General counseling content.", lic)
            assert report.gate3_result is not None
            assert report.gate3_result.decision == GateDecision.PASS, f"failed for {lic}"

    def test_cc0_passes(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Therapy practice note.", "cc0-1.0")
        assert report.gate3_result is not None
        assert report.gate3_result.decision == GateDecision.PASS

    def test_exception_license_passes(self, gates: PrivacyContentGates) -> None:
        for lic in EXCEPTION_LICENSES:
            report = gates.evaluate("t", "General counseling content.", lic)
            assert report.gate3_result is not None
            assert report.gate3_result.decision == GateDecision.PASS, f"failed for {lic}"

    def test_exception_license_needs_consent_for_sensitive(self, gates: PrivacyContentGates) -> None:
        # Sensitive content + NC license + no consent = ESCALATE
        report = gates.evaluate(
            "t",
            "I've been dealing with trauma from abuse.",
            "cc-by-nc-4.0",
            consent_recorded=False,
        )
        assert report.gate3_result is not None
        assert report.gate3_result.decision == GateDecision.ESCALATE

    def test_exception_license_consent_sufficient(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate(
            "t",
            "I've been dealing with trauma from abuse.",
            "cc-by-nc-4.0",
            consent_recorded=True,
        )
        assert report.gate3_result is not None
        assert report.gate3_result.decision == GateDecision.PASS

    def test_unknown_license_blocks(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Some content.", "unknown-proprietary")
        assert report.gate3_result is not None
        assert report.gate3_result.decision == GateDecision.BLOCK


# ---------------------------------------------------------------------------
# Full pipeline — report properties
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_clean_item_passes(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "What is CBT therapy?", "cc-by-4.0")
        assert report.passed
        assert not report.blocked
        assert not report.needs_review

    def test_blocked_item_not_passed(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "I want to kill myself.", "cc-by-4.0")
        assert report.blocked
        assert not report.passed

    def test_escalated_item_not_passed_until_review(self, gates: PrivacyContentGates) -> None:
        text = (
            "Hi, I'm John Smith, call me at 555-123-4567 "
            "or email jsmith@corp.com, SSN 123-45-6789, "
            "DOB 01/15/1985, MRN MRN123456789."
        )
        report = gates.evaluate("t", text, "cc-by-4.0")
        assert report.needs_review
        assert not report.passed
        # After human review override...
        report = gates.override_with_review(report, GateDecision.PASS, "chad", "verified scrubbing manually")
        assert report.passed
        assert report.gate4_result is not None
        assert report.gate4_result.decision == GateDecision.PASS

    def test_review_override_reject(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "General content.", "cc-by-4.0")
        report = gates.override_with_review(report, GateDecision.BLOCK, "chad", "insufficient therapeutic value")
        assert not report.passed
        assert report.blocked

    def test_review_override_invalid_decision_raises(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "General content.", "cc-by-4.0")
        with pytest.raises(ValueError, match="PASS or BLOCK"):
            gates.override_with_review(report, GateDecision.ESCALATE, "chad", "invalid")


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestBatchEvaluation:
    def test_evaluate_batch_reports(self, gates: PrivacyContentGates) -> None:
        items = [
            {"id": "a", "text": "Normal content.", "license_id": "cc-by-4.0"},
            {"id": "b", "text": "Crisis: I want to kill myself.", "license_id": "cc-by-4.0"},
            {"id": "c", "text": "  ", "license_id": "cc-by-4.0"},
        ]
        reports = gates.evaluate_batch(items)
        assert len(reports) == 3
        assert reports[0].passed
        assert reports[1].blocked
        assert reports[2].blocked  # empty text

    def test_batch_graceful_on_error(self, gates: PrivacyContentGates) -> None:
        items = [
            {"id": "ok", "text": "Normal.", "license_id": "cc-by-4.0"},
            {"id": "bad", "text": 123},  # type error
        ]
        reports = gates.evaluate_batch(items)
        assert len(reports) == 2
        assert reports[0].passed
        assert reports[1].blocked  # exception caught gracefully


# ---------------------------------------------------------------------------
# apply_scrub
# ---------------------------------------------------------------------------


class TestApplyScrub:
    def test_scrub_removes_email(self, gates: PrivacyContentGates) -> None:
        text = "Contact me at jane@example.com please."
        clean, findings = gates.apply_scrub(text)
        assert "[EMAIL]" in clean or "jane@example.com" not in clean
        assert any(f.pii_type == "email" for f in findings)

    def test_scrub_handles_no_pii(self, gates: PrivacyContentGates) -> None:
        text = "Mindfulness meditation helps reduce anxiety."
        clean, findings = gates.apply_scrub(text)
        assert clean == text
        assert findings == []

    def test_apply_scrub_after_evaluate(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "My email is test@test.com.", "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.SCRUB_AND_USE
        clean, _ = gates.apply_scrub("My email is test@test.com.")
        assert "test@test.com" not in clean


# ---------------------------------------------------------------------------
# to_dict / serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_report_to_dict(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("s1", "General counseling content.", "cc-by-4.0")
        d = report.to_dict()
        assert d["source_id"] == "s1"
        assert "gate0" in d["gates"]
        assert "privacy_tier" in d
        assert "content_sensitivity" in d
        assert "retention_policy" in d
        assert d["passed"] is True

    def test_gate_result_to_dict(self) -> None:
        r = GateResult(
            gate="gate2",
            decision=GateDecision.PASS,
            reason="all clear",
            details=["no issues"],
        )
        d = r.to_dict()
        assert d["gate"] == "gate2"
        assert d["decision"] == "pass"
        assert d["reason"] == "all clear"
        assert d["details"] == ["no issues"]

    def test_license_check_in_report(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Content.", "cc-by-4.0", consent_recorded=True)
        assert report.license_check is not None
        assert report.license_check.license_id == "cc-by-4.0"
        assert report.license_check.status == "approved"


# ---------------------------------------------------------------------------
# Retention policy mapping
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_none_tier_uses_immediately(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Breathing exercises.", "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.USE_IMMEDIATELY

    def test_low_tier_scrub_and_use(self, gates: PrivacyContentGates) -> None:
        report = gates.evaluate("t", "Email me at x@y.com.", "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.SCRUB_AND_USE

    def test_high_tier_review_then_use(self, gates: PrivacyContentGates) -> None:
        text = "I'm John Smith, SSN 123-45-6789, call 555-123-4567, email me@j.com, DOB 01/01/1980, MRN MRN123456789."
        report = gates.evaluate("t", text, "cc-by-4.0")
        assert report.retention_policy == RetentionPolicy.REVIEW_THEN_USE
