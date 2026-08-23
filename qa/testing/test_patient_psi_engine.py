"""Tests for the PATIENT-Ψ simulation engine."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from ai.pkg_mera.platform.patient_psi.coherence import CoherenceScore
from ai.pkg_mera.platform.patient_psi.engine import (
    PatientPsiEngine,
    SimulationConfig,
    SimulationStatus,
)
from ai.pkg_mera.platform.patient_psi.profiles import ProfileRegistry
from ai.pkg_mera.platform.patient_psi.state_machine import ConversationPhase
from ai.pkg_mera.platform.patient_psi.styles import ConversationalStyle


class TestSimulationConfig:
    """SimulationConfig validation."""

    def test_defaults(self) -> None:
        config = SimulationConfig(profile_name="generalized_anxiety")
        assert config.profile_name == "generalized_anxiety"
        assert config.style is None
        assert config.difficulty == 0.5
        assert config.max_turns == 50
        assert config.patient_name == "Client"

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig(profile_name="generalized_anxiety", difficulty=1.5)

    def test_invalid_max_turns_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig(profile_name="generalized_anxiety", max_turns=0)


class TestPatientPsiEngine:
    """Engine lifecycle and orchestration."""

    @pytest.fixture
    def engine(self) -> PatientPsiEngine:
        return PatientPsiEngine()

    @pytest.fixture
    def config(self) -> SimulationConfig:
        return SimulationConfig(profile_name="generalized_anxiety", max_turns=10)

    # ── Session Creation ────────────────────────────────────────────────

    def test_create_session_returns_uuid(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_create_session_stores_state(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        session = engine.get_session(session_id)
        assert session is not None
        assert session.config.profile_name == "generalized_anxiety"
        assert session.status == SimulationStatus.ACTIVE
        assert session.state_machine.state.phase == ConversationPhase.INITIAL

    def test_create_session_unknown_profile_raises(self, engine: PatientPsiEngine) -> None:
        config = SimulationConfig(profile_name="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            engine.create_session(config)

    def test_create_session_infers_style_from_profile(self, engine: PatientPsiEngine) -> None:
        config = SimulationConfig(profile_name="major_depressive_disorder")  # MDD uses MELANCHOLIC
        session_id = engine.create_session(config)
        session = engine.get_session(session_id)
        assert session is not None
        assert session.style == ConversationalStyle.MELANCHOLIC

    def test_create_session_respects_explicit_style(self, engine: PatientPsiEngine) -> None:
        config = SimulationConfig(profile_name="major_depressive_disorder", style=ConversationalStyle.FRIENDLY)
        session_id = engine.create_session(config)
        session = engine.get_session(session_id)
        assert session is not None
        assert session.style == ConversationalStyle.FRIENDLY

    # ── Interact ────────────────────────────────────────────────────────

    def test_interact_returns_turn(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        turn = engine.interact(session_id, "Hello, how are you today?")
        assert turn.turn_number == 1
        assert turn.phase is not None
        assert turn.trigger == "therapist_greeting"
        assert turn.therapist_utterance == "Hello, how are you today?"
        assert isinstance(turn.patient_utterance, str)
        assert len(turn.patient_utterance) > 0

    def test_interact_multiple_turns_increment(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        engine.interact(session_id, "Hello")
        engine.interact(session_id, "How do you feel?")
        turn3 = engine.interact(session_id, "Tell me more about that")
        assert turn3.turn_number == 3
        session = engine.get_session(session_id)
        assert session is not None
        assert len(session.turns) == 3

    def test_interact_includes_coherence_score(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        turn = engine.interact(session_id, "Hello")
        assert turn.coherence_score is not None
        assert isinstance(turn.coherence_score, CoherenceScore)
        assert 0.0 <= turn.coherence_score.overall <= 1.0

    def test_interact_unknown_session_raises(self, engine: PatientPsiEngine) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            engine.interact("nonexistent", "Hello")

    def test_interact_terminated_session_raises(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        engine.terminate_session(session_id)
        with pytest.raises(KeyError, match="is not active"):
            engine.interact(session_id, "Hello")

    # ── Trigger Detection ───────────────────────────────────────────────

    @pytest.mark.parametrize(
        ("utterance", "expected_trigger"),
        [
            ("Hello!", "therapist_greeting"),
            ("Hi there", "therapist_greeting"),
            ("How are you feeling?", "therapist_insight_question"),
            ("What do you think about that?", "therapist_insight_question"),
            ("That sounds difficult", "challenging_topic"),
            ("I hear you", "therapist_validation"),
            ("Tell me about the trauma", "trauma_disclosure"),
            ("Goodbye for now", "session_end"),
            ("Let's stop here", "session_end_early"),
            ("Tell me more", "elaboration"),
            ("What color is the sky?", "neutral_probe"),
        ],
    )
    def test_detect_trigger(
        self, engine: PatientPsiEngine, config: SimulationConfig, utterance: str, expected_trigger: str
    ) -> None:
        session_id = engine.create_session(config)
        turn = engine.interact(session_id, utterance)
        assert turn.trigger == expected_trigger, f"{utterance!r} → {turn.trigger}, expected {expected_trigger}"

    # ── Deterministic via seed ───────────────────────────────────────────

    def test_interact_deterministic_phase_with_seed(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        """Seed controls state machine phase transitions deterministically."""
        session_a = engine.create_session(config)
        session_b = engine.create_session(config)
        turn_a = engine.interact(session_a, "Hello", seed=42)
        turn_b = engine.interact(session_b, "Hello", seed=42)
        assert turn_a.phase == turn_b.phase
        assert turn_a.trigger == turn_b.trigger

    # ── Session Lifecycle ───────────────────────────────────────────────

    def test_completed_when_max_turns_reached(self, engine: PatientPsiEngine) -> None:
        config = SimulationConfig(profile_name="generalized_anxiety", max_turns=2)
        session_id = engine.create_session(config)
        engine.interact(session_id, "Hello")
        engine.interact(session_id, "How do you feel?")
        session = engine.get_session(session_id)
        assert session is not None
        assert session.status == SimulationStatus.COMPLETED

    def test_terminate_session_returns_true(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        assert engine.terminate_session(session_id) is True
        session = engine.get_session(session_id)
        assert session is not None
        assert session.status == SimulationStatus.TERMINATED

    def test_terminate_nonexistent_returns_false(self, engine: PatientPsiEngine) -> None:
        assert engine.terminate_session("nonexistent") is False

    def test_get_session_nonexistent_returns_none(self, engine: PatientPsiEngine) -> None:
        assert engine.get_session("nonexistent") is None

    # ── List Active Sessions ────────────────────────────────────────────

    def test_list_active_sessions_empty(self, engine: PatientPsiEngine) -> None:
        assert engine.list_active_sessions() == []

    def test_list_active_sessions(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        sid1 = engine.create_session(config)
        sid2 = engine.create_session(config)
        engine.terminate_session(sid2)
        active = engine.list_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == sid1
        assert active[0].status == SimulationStatus.ACTIVE

    def test_list_active_sessions_info_fields(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        active = engine.list_active_sessions()
        assert len(active) == 1
        info = active[0]
        assert info.session_id == session_id
        assert info.config.profile_name == "generalized_anxiety"
        assert info.state is not None
        assert info.turn_count == 0
        assert info.created_at is not None
        assert info.updated_at is not None

    # ── Custom Profile Registry ─────────────────────────────────────────

    def test_custom_profile_registry(self) -> None:
        registry = ProfileRegistry()
        engine = PatientPsiEngine(profile_registry=registry)
        session_id = engine.create_session(SimulationConfig(profile_name="generalized_anxiety"))
        session = engine.get_session(session_id)
        assert session is not None

    # ── Timing ──────────────────────────────────────────────────────────

    def test_updated_at_changes_after_interact(self, engine: PatientPsiEngine, config: SimulationConfig) -> None:
        session_id = engine.create_session(config)
        original = engine.get_session(session_id)
        assert original is not None
        original_time = original.updated_at
        time.sleep(0.01)  # Ensure time advances
        engine.interact(session_id, "Hello")
        updated = engine.get_session(session_id)
        assert updated is not None
        assert updated.updated_at > original_time
