"""Unit tests for InputGuard integration with the ingestion gate pipeline.

Tests that InputGuard runs as the first stage in evaluate_all, performs
PHI scrubbing before downstream gates, and surfaces intent metadata.
"""

import importlib.util
import unittest
from pathlib import Path

from ai.orchestration.safety.guards import InputGuard, SafetyGuardResult, _apply_phi_scrubbing
from ai.memory.gates import GateDecision, GateResult, GatingReport


_GATE_PATH = Path(__file__).resolve().parent.parent.parent / "ai-services" / "ingestion" / "gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("gate", str(_GATE_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInputGuard(unittest.TestCase):
    def test_phi_scrubbing_ssn(self):
        result = _apply_phi_scrubbing("My SSN is 123-45-6789")
        self.assertIn("[SSN]", result)
        self.assertNotIn("123-45-6789", result)

    def test_phi_scrubbing_email(self):
        result = _apply_phi_scrubbing("Contact me at test@example.com")
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("test@example.com", result)

    def test_phi_scrubbing_phone(self):
        result = _apply_phi_scrubbing("Call 555-123-4567 please")
        self.assertIn("[PHONE]", result)
        self.assertNotIn("555-123-4567", result)

    def test_phi_scrubbing_no_phi(self):
        original = "I feel anxious today"
        result = _apply_phi_scrubbing(original)
        self.assertEqual(result, original)

    def test_input_guard_passes_clean_content(self):
        guard = InputGuard()
        result = guard.run("I feel anxious about work")
        self.assertIsInstance(result, SafetyGuardResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.sanitized_text, "I feel anxious about work")

    def test_input_guard_scrubs_phi(self):
        guard = InputGuard()
        result = guard.run("My SSN is 123-45-6789")
        self.assertTrue(result.passed)
        self.assertNotIn("123-45-6789", result.sanitized_text)
        self.assertIn("[SSN]", result.sanitized_text)

    def test_input_guard_detects_history_intent(self):
        guard = InputGuard()
        result = guard.run("Tell me about my past medication history")
        self.assertEqual(result.metadata.get("intent"), "ask_history")

    def test_input_guard_detects_pain_intent(self):
        guard = InputGuard()
        result = guard.run("I am in pain and it hurts")
        self.assertEqual(result.metadata.get("intent"), "address_pain")

    def test_input_guard_detects_exam_intent(self):
        guard = InputGuard()
        result = guard.run("Let me check your physical exam results")
        self.assertEqual(result.metadata.get("intent"), "perform_exam")

    def test_input_guard_detects_intervention_intent(self):
        guard = InputGuard()
        result = guard.run("Give medication at the recommended dose")
        self.assertEqual(result.metadata.get("intent"), "perform_intervention")

    def test_input_guard_default_intent(self):
        guard = InputGuard()
        result = guard.run("Hello, how are you?")
        self.assertEqual(result.metadata.get("intent"), "general_talk")


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
        self.assertIsInstance(report, GatingReport)
        self.assertEqual(report.safety_intent, "ask_history")

    def test_evaluate_all_phi_scrubbed_before_pii_gate(self):
        report = self.protocol.evaluate_all(
            "My SSN is 123-45-6789 and I need help",
            source_id="test-phi",
            user_id=None,
        )
        self.assertNotIn("123-45-6789", report.content)
        self.assertIn("[SSN]", report.content)

    def test_evaluate_all_crisis_blocked(self):
        report = self.protocol.evaluate_all(
            "I want to kill myself right now",
            source_id="test-crisis",
            user_id=None,
        )
        self.assertTrue(report.blocked)
        self.assertNotEqual(report.crisis_tier, "none")

    def test_evaluate_all_report_has_safety_intent_in_dict(self):
        report = self.protocol.evaluate_all(
            "I am in pain today",
            source_id="test-dict",
            user_id=None,
        )
        d = report.to_dict()
        self.assertIn("safety_intent", d)
        self.assertEqual(d["safety_intent"], "address_pain")

    def test_evaluate_all_general_talk_intent(self):
        report = self.protocol.evaluate_all(
            "I had a good day.",
            source_id="test-general",
            user_id=None,
        )
        self.assertEqual(report.safety_intent, "general_talk")


if __name__ == "__main__":
    unittest.main()
