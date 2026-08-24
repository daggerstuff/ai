"""Tests for the JIT scenario injector (IMPLEMENTATION_PLAN.md work #3)."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from ai.configs.jit_scenario_injector import (
    DEFAULT_DIFFICULTY,
    DEFAULT_DOMAIN_GAP,
    JIT_SCENARIO_BLOCK_LABEL,
    JITScenarioInjector,
    TriggerDecision,
    inject_for_decision,
    subscribe_to_event_bus,
)


class _FakeInjectionPoint:
    PRE_PROMPT = "pre_prompt"


class _FakeRetentionPolicy:
    EPHEMERAL = "ephemeral"


class _FakeMergeStrategy:
    REPLACE = "replace"


class _FakeBlockScope:
    SESSION = "session"


class _FakeSchema:
    def __init__(self, **kwargs):
        self.label = kwargs["label"]
        self.injection_point = kwargs["injection_point"]
        self.kwargs = kwargs


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
        if label not in self.schemas:
            raise ValueError(f"schema missing: {label}")
        return _FakeBlock(self.schemas[label], content)

    def set_block(self, label, block):
        self.stored[label] = block


class _FakeBlockSystem:
    InjectionPoint = _FakeInjectionPoint
    RetentionPolicy = _FakeRetentionPolicy
    MergeStrategy = _FakeMergeStrategy
    BlockScope = _FakeBlockScope
    MemoryBlockSchema = _FakeSchema

    def __init__(self):
        self.registry = _FakeRegistry()

    def get_registry(self):
        return self.registry


class _FakeGenerator:
    def __init__(self):
        self.calls = []
        self.scenario = "Generated nightmare scenario text."
        self.raise_error = False

    def __call__(self, **kwargs):
        if self.raise_error:
            raise RuntimeError("generator down")
        self.calls.append(kwargs)
        return self.scenario


class JITScenarioInjectorTest(unittest.TestCase):
    def _injector_with_fakes(self):
        fake_gen = _FakeGenerator()
        fake_bs = _FakeBlockSystem()
        injector = JITScenarioInjector(generator=fake_gen)
        return injector, fake_gen, fake_bs

    def test_no_trigger_returns_none_without_calling_generator(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=False, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(decision)
        self.assertIsNone(result)
        self.assertEqual(fake_gen.calls, [])

    def test_trigger_generates_and_injects_block(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, matching_flags=3, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(decision, domain_gap="gap X", difficulty="extreme")
        self.assertEqual(
            fake_gen.calls,
            [{"domain_gap": "gap X", "difficulty": "extreme"}],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("c1", result["content"])
        self.assertIn("Generated nightmare scenario text.", result["content"])
        schema = fake_bs.registry.schemas[JIT_SCENARIO_BLOCK_LABEL]
        self.assertEqual(schema.injection_point, "pre_prompt")
        self.assertEqual(
            fake_bs.registry.stored[JIT_SCENARIO_BLOCK_LABEL].content,
            result["content"],
        )

    def test_default_targeting_when_params_omitted(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            injector.inject_for_decision(decision)
        self.assertEqual(
            fake_gen.calls,
            [{"domain_gap": DEFAULT_DOMAIN_GAP, "difficulty": DEFAULT_DIFFICULTY}],
        )

    def test_schema_registered_once(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            injector.inject_for_decision(decision)
            injector.inject_for_decision(decision)
        self.assertEqual(len(fake_bs.registry.schemas), 1)

    def test_missing_foresight_returns_none(self):
        injector, fake_gen, _ = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=None,
        ):
            result = injector.inject_for_decision(decision)
        self.assertIsNone(result)

    def test_generator_error_returns_none(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        fake_gen.raise_error = True
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(decision)
        self.assertIsNone(result)

    def test_module_level_convenience_delegates_to_singleton(self):
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch("triggers.jit_scenario_injector._injector") as mock_injector:
            inject_for_decision(decision, domain_gap="g")
            mock_injector.inject_for_decision.assert_called_once_with(
                decision, domain_gap="g", difficulty=None, session_context=None
            )

    def test_is_training_edge_case_tag_in_block_content(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(decision)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("is_training_edge_case: true", result["content"])

    def test_hipaa_gate_blocks_non_training_session(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(
                decision, session_context={"is_training_session": False, "session_id": "s1"}
            )
        self.assertIsNone(result)
        self.assertEqual(fake_gen.calls, [])

    def test_hipaa_gate_allows_training_session(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(
                decision, session_context={"is_training_session": True, "session_id": "s2"}
            )
        self.assertIsNotNone(result)

    def test_hipaa_gate_no_context_allows(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        decision = TriggerDecision(should_trigger=True, clinician_id="c1")
        with patch(
            "triggers.jit_scenario_injector._load_foresight_block_system",
            return_value=fake_bs,
        ):
            result = injector.inject_for_decision(decision, session_context=None)
        self.assertIsNotNone(result)


@dataclass
class FakeEvent:
    event_type: str
    payload: dict


class FakeEventType:
    BIAS_THRESHOLD_EXCEEDED = "bias_threshold_exceeded"
    CRISIS_THRESHOLD_EXCEEDED = "crisis_threshold_exceeded"


class FakeEventBus:
    def __init__(self):
        self.subscriptions = {}

    def subscribe(self, event_type, handler):
        self.subscriptions.setdefault(event_type, []).append(handler)

    def emit(self, event):
        for handler in self.subscriptions.get(event.event_type, []):
            handler(event)


class EventBusIntegrationTest(unittest.TestCase):
    def _injector_with_fakes(self):
        fake_gen = _FakeGenerator()
        fake_bs = _FakeBlockSystem()
        injector = JITScenarioInjector(generator=fake_gen)
        return injector, fake_gen, fake_bs

    def test_subscribe_returns_true_with_event_bus(self):
        injector, _, _ = self._injector_with_fakes()
        bus = FakeEventBus()
        with patch(
            "triggers.jit_scenario_injector._load_foresight_event_bus",
            return_value=MagicMock(EventType=FakeEventType),
        ):
            result = injector.subscribe_to_event_bus(bus)
        self.assertTrue(result)
        self.assertTrue(injector._event_bus_subscribed)

    def test_subscribe_returns_false_without_event_bus(self):
        injector, _, _ = self._injector_with_fakes()
        with patch(
            "triggers.jit_scenario_injector._load_foresight_event_bus",
            return_value=None,
        ):
            result = injector.subscribe_to_event_bus()
        self.assertFalse(result)

    def test_bias_event_triggers_injection(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        bus = FakeEventBus()
        with (
            patch(
                "triggers.jit_scenario_injector._load_foresight_event_bus",
                return_value=MagicMock(EventType=FakeEventType),
            ),
            patch(
                "triggers.jit_scenario_injector._load_foresight_block_system",
                return_value=fake_bs,
            ),
        ):
            injector.subscribe_to_event_bus(bus)
            bus.emit(
                FakeEvent(
                    event_type=FakeEventType.BIAS_THRESHOLD_EXCEEDED,
                    payload={
                        "clinician_id": "c42",
                        "overall_bias_score": 0.8,
                        "detected_biases": ["authority_bias"],
                    },
                )
            )
        self.assertEqual(len(fake_gen.calls), 1)
        self.assertEqual(fake_gen.calls[0]["domain_gap"], "authority_bias")
        self.assertEqual(fake_gen.calls[0]["difficulty"], "critical")
        self.assertIn("c42", fake_bs.registry.stored[JIT_SCENARIO_BLOCK_LABEL].content)

    def test_crisis_event_triggers_injection(self):
        injector, fake_gen, fake_bs = self._injector_with_fakes()
        bus = FakeEventBus()
        with (
            patch(
                "triggers.jit_scenario_injector._load_foresight_event_bus",
                return_value=MagicMock(EventType=FakeEventType),
            ),
            patch(
                "triggers.jit_scenario_injector._load_foresight_block_system",
                return_value=fake_bs,
            ),
        ):
            injector.subscribe_to_event_bus(bus)
            bus.emit(
                FakeEvent(
                    event_type=FakeEventType.CRISIS_THRESHOLD_EXCEEDED,
                    payload={
                        "clinician_id": "c99",
                        "overall_bias_score": 0.5,
                    },
                )
            )
        self.assertEqual(len(fake_gen.calls), 1)
        self.assertEqual(fake_gen.calls[0]["difficulty"], "high")

    def test_module_level_subscribe_wrapper(self):
        with patch("triggers.jit_scenario_injector._injector") as mock_injector:
            mock_injector.subscribe_to_event_bus.return_value = True
            result = subscribe_to_event_bus(event_bus="fake_bus")
            mock_injector.subscribe_to_event_bus.assert_called_once_with("fake_bus")
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
