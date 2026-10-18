"""Unit tests for InputGuard integration with the ingestion gate pipeline.

Tests that InputGuard runs as the first stage in evaluate_all, performs
PHI scrubbing before downstream gates, and surfaces intent metadata.
"""

import importlib.util
import unittest
from pathlib import Path

from ai.memory.gates import GatingReport
from ai.orchestration.safety.guards import InputGuard, SafetyGuardResult, _apply_phi_scrubbing

_GATE_PATH = Path(__file__).resolve().parent.parent / "services" / "ingestion" / "gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("gate", str(_GATE_PATH))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInputGuard(unittest.TestCase):
    def test_phi_scrubbing_ssn(self):
        result = _apply_phi_scrubbing("My SSN is 123-45-6789")
        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_phi_scrubbing_email(self):
        result = _apply_phi_scrubbing("Contact me at test@example.com")
        assert "[EMAIL]" in result
        assert "test@example.com" not in result

    def test_phi_scrubbing_phone(self):
        result = _apply_phi_scrubbing("Call 555-123-4567 please")
        assert "[PHONE]" in result
        assert "555-123-4567" not in result

    def test_phi_scrubbing_no_phi(self):
        original = "I feel anxious today"
        result = _apply_phi_scrubbing(original)
        assert result == original

    def test_input_guard_passes_clean_content(self):
        guard = InputGuard()
        result = guard.run("I feel anxious about work")
        assert isinstance(result, SafetyGuardResult)
        assert result.passed
        assert result.sanitized_text == "I feel anxious about work"

    def test_input_guard_scrubs_phi(self):
        guard = InputGuard()
        result = guard.run("My SSN is 123-45-6789")
        assert result.passed
        assert "123-45-6789" not in result.sanitized_text
        assert "[SSN]" in result.sanitized_text

    def test_input_guard_detects_history_intent(self):
        guard = InputGuard()
        result = guard.run("Tell me about my past medication history")
        assert result.metadata.get("intent") == "ask_history"

    def test_input_guard_detects_pain_intent(self):
        guard = InputGuard()
        result = guard.run("I am in pain and it hurts")
        assert result.metadata.get("intent") == "address_pain"

    def test_input_guard_detects_exam_intent(self):
        guard = InputGuard()
        result = guard.run("Let me check your physical exam results")
        assert result.metadata.get("intent") == "perform_exam"

    def test_input_guard_detects_intervention_intent(self):
        guard = InputGuard()
        result = guard.run("Give medication at the recommended dose")
        assert result.metadata.get("intent") == "perform_intervention"

    def test_input_guard_default_intent(self):
        guard = InputGuard()
        result = guard.run("Hello, how are you?")
        assert result.metadata.get("intent") == "general_talk"


class TestInputGuardGateIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate_mod = _load_gate_module()
        cls.protocol = cls.gate_mod.protocol

    def test_evaluate_all_populates_safety_intent(self):
        report = self.protocol.evaluate_all(
            "Tell me about my past medication history",
            source_id="test-intent",
            user_id=None,
        )
        assert isinstance(report, GatingReport)
        assert report.safety_intent == "ask_history"

    def test_evaluate_all_phi_scrubbed_before_pii_gate(self):
        report = self.protocol.evaluate_all(
            "My SSN is 123-45-6789 and I need help",
            source_id="test-phi",
            user_id=None,
        )
        assert "123-45-6789" not in report.content
        assert "[SSN]" in report.content

    def test_evaluate_all_crisis_blocked(self):
        report = self.protocol.evaluate_all(
            "I want to kill myself right now",
            source_id="test-crisis",
            user_id=None,
        )
        assert report.blocked
        assert report.crisis_tier != "none"

    def test_evaluate_all_report_has_safety_intent_in_dict(self):
        report = self.protocol.evaluate_all(
            "I am in pain today",
            source_id="test-dict",
            user_id=None,
        )
        d = report.to_dict()
        assert "safety_intent" in d
        assert d["safety_intent"] == "address_pain"

    def test_evaluate_all_general_talk_intent(self):
        report = self.protocol.evaluate_all(
            "I had a good day.",
            source_id="test-general",
            user_id=None,
        )
        assert report.safety_intent == "general_talk"


if __name__ == "__main__":
    unittest.main()
