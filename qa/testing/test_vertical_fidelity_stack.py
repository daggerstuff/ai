"""PIX-4184: Vertical Fidelity Stack E2E Integration & Audit Verification.

Validates the full loop using only ai-local imports:
  P0-1 bypass -> R1 receipt -> JIT injection (with foresight fakes)

Foresight components (EventBus, JITTriggerEngine) are tested in their own
suite. This test validates the integration contract between layers.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ai.data.raw.receipt import Ledger, ReceiptEnvelope
from ai.qa.validation.enhanced_safety_filter import EnhancedSafetyFilter, SafetyLevel
from triggers.jit_scenario_injector import JITScenarioInjector, TriggerDecision


def _make_receipt(prev_hash: str, model: str = "test-model-v1") -> ReceiptEnvelope:
    return ReceiptEnvelope(
        prev_hash=prev_hash,
        model_fingerprint=model,
        prompt_hash="prompt-abc123",
        output_hash="output-def456",
        fhe_ciphertext_hash="fhe-ghi789",
    )


class _FakeBlockSystem:
    class InjectionPoint:
        PRE_PROMPT = "pre_prompt"

    class RetentionPolicy:
        EPHEMERAL = "ephemeral"

    class MergeStrategy:
        REPLACE = "replace"

    class BlockScope:
        SESSION = "session"

    class MemoryBlockSchema:
        def __init__(self, **kwargs):
            self.label = kwargs["label"]
            self.injection_point = kwargs["injection_point"]

    class _FakeBlock:
        def __init__(self, schema, content):
            self.schema = schema
            self.content = content

        def to_dict(self):
            return {"label": self.schema.label, "content": self.content}

    class _FakeRegistry:
        def __init__(self):
            self.schemas = {}
            self.stored = {}

        def get_schema(self, label):
            return self.schemas.get(label)

        def register(self, schema):
            self.schemas[schema.label] = schema

        def create_block(self, label, content=""):
            return _FakeBlockSystem._FakeBlock(self.schemas[label], content)

        def set_block(self, label, block):
            self.stored[label] = block

    def __init__(self):
        self.registry = self._FakeRegistry()

    def get_registry(self):
        return self.registry


class P01BypassTest(unittest.TestCase):
    def test_edge_case_content_passes_through_unfiltered(self):
        filter_ = EnhancedSafetyFilter(safety_level=SafetyLevel.STRICT)
        content = "I want to kill myself right now. I have a gun."
        metadata = {"is_training_edge_case": True}

        result = filter_.check_output_safety(
            content=content,
            _user_context={"role": "training"},
            _request_metadata=metadata,
        )

        self.assertTrue(result.is_safe)
        self.assertTrue(result.filtered_content == content or result.filtered_content is None)


class R1ReceiptChainTest(unittest.TestCase):
    def test_chain_integrity_and_merkle_root(self):
        ledger = Ledger()
        genesis = _make_receipt("0" * 64)
        ledger.append(genesis)
        r2 = _make_receipt(genesis.receipt_hash)
        ledger.append(r2)
        r3 = _make_receipt(r2.receipt_hash)
        ledger.append(r3)

        self.assertTrue(ledger.verify_chain())
        self.assertEqual(len(ledger), 3)
        root = ledger.root_hash()
        self.assertEqual(len(root), 64)
        self.assertNotEqual(root, "0" * 64)

    def test_tampered_receipt_breaks_chain(self):
        ledger = Ledger()
        r1 = _make_receipt("0" * 64)
        ledger.append(r1)
        tampered = ReceiptEnvelope(
            prev_hash="0" * 64,
            model_fingerprint="wrong-model",
            prompt_hash=r1.prompt_hash,
            output_hash=r1.output_hash,
            fhe_ciphertext_hash=r1.fhe_ciphertext_hash,
        )
        ledger.append(tampered)

        self.assertFalse(ledger.verify_chain())


class JITInjectionIntegrationTest(unittest.TestCase):
    def test_jit_injection_respects_hipaa_gate(self):
        def fake_gen(domain_gap=None, difficulty=None):
            return f"Nightmare scenario: {domain_gap or 'default'}"

        injector = JITScenarioInjector(generator=fake_gen)
        fake_bs = _FakeBlockSystem()
        decision = TriggerDecision(should_trigger=True, clinician_id="clin-1")

        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(
                decision,
                domain_gap="authority_bias",
                session_context={"is_training_session": True},
            )

        self.assertIsNotNone(result)
        self.assertIn("is_training_edge_case: true", result["content"])
        self.assertIn("authority_bias", result["content"])

    def test_jit_injection_blocked_for_real_patient(self):
        def fake_gen(domain_gap=None, difficulty=None):
            return "should not be generated"

        injector = JITScenarioInjector(generator=fake_gen)
        decision = TriggerDecision(should_trigger=True, clinician_id="clin-1")

        result = injector.inject_for_decision(
            decision,
            session_context={"is_training_session": False},
        )

        self.assertIsNone(result)


class FullVerticalStackTest(unittest.TestCase):
    def test_full_loop_p01_to_r1_to_jit(self):
        filter_ = EnhancedSafetyFilter(safety_level=SafetyLevel.STRICT)
        edge_case_content = "Extreme crisis scenario for training purposes."
        metadata = {"is_training_edge_case": True}

        safety_result = filter_.check_output_safety(
            content=edge_case_content,
            _user_context={"role": "training"},
            _request_metadata=metadata,
        )
        self.assertTrue(safety_result.is_safe)

        ledger = Ledger()
        genesis = _make_receipt("0" * 64)
        ledger.append(genesis)
        r2 = _make_receipt(genesis.receipt_hash)
        ledger.append(r2)
        self.assertTrue(ledger.verify_chain())
        self.assertEqual(len(ledger), 2)

        decision = TriggerDecision(should_trigger=True, matching_flags=3, clinician_id="clin-1")

        def fake_gen(domain_gap=None, difficulty=None):
            return f"Targeted scenario for {domain_gap or 'general'}"

        injector = JITScenarioInjector(generator=fake_gen)
        fake_bs = _FakeBlockSystem()

        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(
                decision,
                domain_gap="authority_bias",
                session_context={"is_training_session": True},
            )

        self.assertIsNotNone(result)
        self.assertIn("is_training_edge_case: true", result["content"])
        self.assertIn("Targeted scenario for authority_bias", result["content"])
        self.assertIn("clin-1", result["content"])


if __name__ == "__main__":
    unittest.main()
