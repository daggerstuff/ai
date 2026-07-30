"""Tests for the PATIENT-Ψ conversation state machine."""

from __future__ import annotations

import pytest

from ai.pkg_mera.platform.patient_psi.profiles import ProfileRegistry
from ai.pkg_mera.platform.patient_psi.state_machine import (
    ConversationPhase,
    ConversationState,
    StateMachine,
    StateTransition,
)


class TestConversationPhase:
    """ConversationPhase enum behaviour."""

    def test_has_expected_phases(self) -> None:
        phases = list(ConversationPhase)
        expected = [
            ConversationPhase.INITIAL,
            ConversationPhase.ENGAGING,
            ConversationPhase.RESISTANT,
            ConversationPhase.REFLECTIVE,
            ConversationPhase.DISTRESSED,
            ConversationPhase.INSIGHT,
            ConversationPhase.CLOSURE,
        ]
        assert phases == expected

    def test_str_values(self) -> None:
        assert str(ConversationPhase.INITIAL) == "initial"
        assert str(ConversationPhase.INSIGHT) == "insight"


class TestConversationState:
    """ConversationState defaults and validation."""

    def test_default_phase_is_initial(self) -> None:
        state = ConversationState()
        assert state.phase == ConversationPhase.INITIAL

    def test_default_metrics_in_range(self) -> None:
        state = ConversationState()
        for attr in ("engagement_level", "therapeutic_alliance"):
            assert getattr(state, attr) == 0.5

    def test_fields_clamped_to_0_1(self) -> None:
        with pytest.raises(ValueError):
            ConversationState(engagement_level=1.5)  # type: ignore[arg-type]


class TestStateMachine:
    """StateMachine lifecycle and transitions."""

    def setup_method(self) -> None:
        registry = ProfileRegistry()
        self.profile = registry.get_default_profile()

    def test_initial_state_initial(self) -> None:
        sm = StateMachine(self.profile)
        assert sm.state.phase == ConversationPhase.INITIAL
        assert sm.state.turn_count == 0

    def test_transition_from_initial(self) -> None:
        sm = StateMachine(self.profile)
        new_phase = sm.transition("therapist_greeting", seed=42)
        assert new_phase in list(ConversationPhase)
        assert sm.state.turn_count == 1

    def test_transition_from_initial_deterministic_seed(self) -> None:
        sm1 = StateMachine(self.profile)
        sm2 = StateMachine(self.profile)
        p1 = sm1.transition("therapist_greeting", seed=42)
        p2 = sm2.transition("therapist_greeting", seed=42)
        assert p1 == p2

    def test_increments_turn_count(self) -> None:
        sm = StateMachine(self.profile)
        for seed in range(5):
            sm.transition("neutral_probe", seed=seed)
        assert sm.state.turn_count == 5

    def test_unknown_trigger_falls_back_to_engaging(self) -> None:
        sm = StateMachine(self.profile)
        phase = sm.transition("foobar")
        assert phase == ConversationPhase.ENGAGING

    def test_possible_transitions_from_initial(self) -> None:
        sm = StateMachine(self.profile)
        ts = sm.get_possible_transitions()
        assert all(t.from_state == ConversationPhase.INITIAL for t in ts)

    def test_possible_transitions_not_empty(self) -> None:
        sm = StateMachine(self.profile)
        ts = sm.get_possible_transitions()
        assert len(ts) > 0

    def test_reset_restores_defaults(self) -> None:
        sm = StateMachine(self.profile)
        sm.transition("therapist_greeting", seed=42)
        sm.reset()
        assert sm.state.phase == ConversationPhase.INITIAL
        assert sm.state.turn_count == 0

    def test_all_triggers_covered_from_each_state(self) -> None:
        sm = StateMachine(self.profile)
        phases_with_rules: set[ConversationPhase] = {t.from_state for t in sm._transitions}
        for phase in ConversationPhase:
            assert phase in phases_with_rules, f"{phase} has no outgoing transitions"

    def test_resistance_baseline_in_bounds(self) -> None:
        registry = ProfileRegistry()
        for name in registry.list_profiles():
            sm = StateMachine(registry.get_profile(name))
            baseline = sm._profile_resistance_baseline()
            assert baseline >= 0.0
            assert baseline <= 1.0

    def test_profile_resistance_baseline_differs(self) -> None:
        registry = ProfileRegistry()
        hostile = registry.get_profile("paranoid_schizophrenia")
        friendly = registry.get_profile("narcissistic_personality")

        sm_h = StateMachine(hostile)
        sm_f = StateMachine(friendly)

        assert sm_h._profile_resistance_baseline() > sm_f._profile_resistance_baseline()

    def test_engagement_increases_on_successful_engaging(self) -> None:
        sm = StateMachine(self.profile)
        sm.state.engagement_level = 0.3
        phase = sm.transition("therapist_greeting", seed=1)
        assert phase == ConversationPhase.ENGAGING
        assert sm.state.engagement_level > 0.3


class TestStateTransition:
    """StateTransition validation."""

    def test_probability_clamped(self) -> None:
        with pytest.raises(ValueError):
            StateTransition(
                from_state=ConversationPhase.INITIAL,
                to_state=ConversationPhase.ENGAGING,
                trigger="test",
                probability=1.5,
            )

    def test_valid_transition(self) -> None:
        t = StateTransition(
            from_state=ConversationPhase.INITIAL,
            to_state=ConversationPhase.ENGAGING,
            trigger="greeting",
            probability=0.8,
        )
        assert t.probability == 0.8
        assert t.trigger == "greeting"
