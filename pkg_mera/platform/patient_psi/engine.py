"""PATIENT-Ψ simulation engine.

Orchestrates StateMachine, StyleRegistry, CoherenceModel, and ProfileRegistry
to run cognitive patient simulation sessions.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ai.platform.patient_psi.coherence import CoherenceModel, CoherenceScore
from ai.platform.patient_psi.profiles import ClinicalProfile, ProfileRegistry
from ai.platform.patient_psi.state_machine import (
    ConversationPhase,
    ConversationState,
    StateMachine,
)
from ai.platform.patient_psi.styles import ConversationalStyle, StyleRegistry


class SimulationConfig(BaseModel):
    """Configuration for a PATIENT-Ψ simulation session."""

    profile_name: str
    style: ConversationalStyle | None = None
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    max_turns: int = Field(default=50, ge=1)
    patient_name: str = "Client"


class SimulationTurn(BaseModel):
    """A single turn in a simulation session."""

    turn_number: int
    phase: ConversationPhase
    trigger: str
    therapist_utterance: str
    patient_utterance: str
    coherence_score: CoherenceScore | None = None


class SimulationStatus(StrEnum):
    """Lifecycle status of a simulation session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class SessionInfo(BaseModel):
    """Public metadata for a simulation session."""

    session_id: str
    config: SimulationConfig
    state: ConversationState
    turn_count: int
    status: SimulationStatus
    created_at: datetime
    updated_at: datetime


@dataclass
class SimulationSession:
    """Internal runtime state for a simulation session."""

    session_id: str
    config: SimulationConfig
    profile: ClinicalProfile
    state_machine: StateMachine
    coherence_model: CoherenceModel
    style: ConversationalStyle
    turns: list[SimulationTurn] = field(default_factory=list)
    status: SimulationStatus = SimulationStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionNotFoundError(KeyError):
    """Raised when a session ID does not map to any session."""


class SessionNotActiveError(KeyError):
    """Raised when a session exists but is not in ACTIVE status."""

    def __init__(self, session_id: str, status: str) -> None:
        self.session_id = session_id
        self.status = status
        super().__init__(f"Session {session_id!r} is not active (status: {status})")


class PatientPsiEngine:
    """Orchestrates PATIENT-Ψ cognitive patient simulation sessions."""

    def __init__(
        self,
        profile_registry: ProfileRegistry | None = None,
        style_registry: StyleRegistry | None = None,
    ) -> None:
        self._profile_registry = profile_registry or ProfileRegistry()
        self._style_registry = style_registry or StyleRegistry()
        self._sessions: dict[str, SimulationSession] = {}

        self._lock = threading.Lock()

    def create_session(self, config: SimulationConfig) -> str:
        """Create a new simulation session.

        Args:
            config: Simulation configuration including profile name.

        Returns:
            The UUID of the newly created session.

        Raises:
            KeyError: If the profile_name is not found in the registry.
        """
        profile = self._profile_registry.get_profile(config.profile_name)
        state_machine = StateMachine(profile)
        coherence_model = CoherenceModel(profile)
        style = config.style or profile.default_style

        session_id = str(uuid.uuid4())
        session = SimulationSession(
            session_id=session_id,
            config=config,
            profile=profile,
            state_machine=state_machine,
            coherence_model=coherence_model,
            style=style,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def interact(
        self,
        session_id: str,
        therapist_utterance: str,
        *,
        seed: int | None = None,
    ) -> SimulationTurn:
        """Process a therapist utterance and return the patient response.

        Args:
            session_id: The session UUID.
            therapist_utterance: The therapist's input text.
            seed: Optional RNG seed for deterministic behavior.

        Returns:
            A SimulationTurn capturing the interaction.

        Raises:
            KeyError: If the session is not found or not active.
        """
        session = self._get_active_session(session_id)

        trigger = self._detect_trigger(therapist_utterance)
        new_phase = session.state_machine.transition(trigger, seed=seed)
        utterance_type = self._phase_to_utterance_type(new_phase)

        context: dict[str, object] = {
            "phase": new_phase.value,
            "profile_name": session.profile.name,
            "session_id": session.session_id,
            "patient_name": session.config.patient_name,
        }
        patient_utterance = self._style_registry.get_utterance(
            session.style,
            utterance_type,
            context,
        )

        coherence_score = session.coherence_model.evaluate(
            patient_utterance,
            context={"phase": new_phase.value},
        )

        turn = SimulationTurn(
            turn_number=session.state_machine.state.turn_count,
            phase=new_phase,
            trigger=trigger,
            therapist_utterance=therapist_utterance,
            patient_utterance=patient_utterance,
            coherence_score=coherence_score,
        )
        session.turns.append(turn)
        session.updated_at = datetime.now(UTC)

        if session.state_machine.state.turn_count >= session.config.max_turns or new_phase == ConversationPhase.CLOSURE:
            session.status = SimulationStatus.COMPLETED

        return turn

    def get_session(self, session_id: str) -> SimulationSession | None:
        """Return a session by ID, or None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def terminate_session(self, session_id: str) -> bool:
        """Mark a session as terminated.

        Args:
            session_id: The session UUID.

        Returns:
            True if the session was found and terminated, False otherwise.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.status = SimulationStatus.TERMINATED
            session.updated_at = datetime.now(UTC)
            return True

    def list_active_sessions(self) -> list[SessionInfo]:
        """Return metadata for all currently active sessions."""
        active: list[SessionInfo] = []
        with self._lock:
            for session in self._sessions.values():
                if session.status == SimulationStatus.ACTIVE:
                    active.append(
                        SessionInfo(
                            session_id=session.session_id,
                            config=session.config,
                            state=session.state_machine.state,
                            turn_count=session.state_machine.state.turn_count,
                            status=session.status,
                            created_at=session.created_at,
                            updated_at=session.updated_at,
                        )
                    )
        return active

    def _get_active_session(self, session_id: str) -> SimulationSession:
        """Retrieve a session, raising a typed error if not found or not active."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            if session.status != SimulationStatus.ACTIVE:
                raise SessionNotActiveError(session_id, session.status.value)
            return session

    def _detect_trigger(self, therapist_utterance: str) -> str:  # noqa: PLR0911
        """Map therapist input to a state machine trigger keyword.

        Uses case-insensitive keyword matching in priority order.
        """
        lower = therapist_utterance.lower()

        def _has(*words):
            return any(re.search(rf"\b{re.escape(w)}\b", lower) for w in words)

        if _has("sorry", "trauma"):
            return "trauma_disclosure"
        if _has("feel", "feeling", "emotion", "how does that make"):
            return "therapist_insight_question"
        if _has("hello", "hi", "welcome", "good to see", "how are you"):
            return "therapist_greeting"
        if _has("difficult", "hard", "challenging", "tough"):
            return "challenging_topic"
        if _has("understand", "hear you", "see that", "makes sense"):
            return "therapist_validation"
        if _has("think about", "why do you", "what if"):
            return "therapist_insight_question"
        if _has("goodbye", "see you", "next time", "we're done"):
            return "session_end"
        if _has("stop", "enough", "let's end"):
            return "session_end_early"
        if _has("tell me more", "elaborate", "go on", "continue"):
            return "elaboration"

        return "neutral_probe"

    def _phase_to_utterance_type(self, phase: ConversationPhase) -> str:
        """Map a conversation phase to an utterance type for style generation."""
        if phase == ConversationPhase.INITIAL:
            return "greeting"
        if phase == ConversationPhase.CLOSURE:
            return "closure"
        if phase == ConversationPhase.RESISTANT:
            return "counter_question"
        return "response"
