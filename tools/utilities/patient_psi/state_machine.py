"""PATIENT-Ψ conversation state machine.

Models clinical conversation state transitions driven by
therapist interaction patterns and patient profile characteristics.
Provides deterministic and stochastic transition logic grounded
in the CCD profile's severity, style, and belief structure.
"""

from __future__ import annotations

import random
from enum import StrEnum

from pydantic import BaseModel, Field

from ai.tools.utilities.platform.patient_psi.profiles import ClinicalProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# After this many turns, resistance transitions are less likely (alliance builds)
_ALLIANCE_BUILD_TURN = 10


class ConversationPhase(StrEnum):
    """Phases of a therapy conversation session."""

    INITIAL = "initial"
    ENGAGING = "engaging"
    RESISTANT = "resistant"
    REFLECTIVE = "reflective"
    DISTRESSED = "distressed"
    INSIGHT = "insight"
    CLOSURE = "closure"


class StateTransition(BaseModel):
    """A single state transition rule.

    Attributes:
        from_state: The current phase to transition from.
        to_state: The target phase to transition to.
        trigger: Label describing what kind of interaction prompts this move.
        probability: Base likelihood (0-1) of this transition when eligible.
    """

    from_state: ConversationPhase
    to_state: ConversationPhase
    trigger: str
    probability: float = Field(default=0.5, ge=0.0, le=1.0)


class ConversationState(BaseModel):
    """Live state of a therapy conversation.

    Attributes:
        phase: Current conversation phase.
        turn_count: Number of conversational turns so far.
        engagement_level: How engaged the patient is (0 = disengaged, 1 = fully).
        resistance_level: Level of resistance / defensiveness (0 = open, 1 = fully resistant).
        distress_level: Current emotional distress (0 = calm, 1 = maximum distress).
        insight_level: Demonstrated psychological insight (0 = none, 1 = profound).
        therapeutic_alliance: Quality of the therapist-patient bond (0 = ruptured, 1 = strong).
    """

    phase: ConversationPhase = ConversationPhase.INITIAL
    turn_count: int = 0
    engagement_level: float = Field(default=0.5, ge=0.0, le=1.0)
    resistance_level: float = Field(default=0.0, ge=0.0, le=1.0)
    distress_level: float = Field(default=0.0, ge=0.0, le=1.0)
    insight_level: float = Field(default=0.0, ge=0.0, le=1.0)
    therapeutic_alliance: float = Field(default=0.5, ge=0.0, le=1.0)


class StateMachine:
    """Manages conversation state transitions for a specific patient profile.

    The state machine loads default transition probabilities and modulates
    them at runtime based on the patient's profile (severity, style, profile-typical
    resistance patterns) and the ongoing conversation context.
    """

    def __init__(self, profile: ClinicalProfile) -> None:
        self.profile = profile
        self.state = ConversationState()
        self._transitions: list[StateTransition] = []
        self._build_default_transitions()

    # ------------------------------------------------------------------
    # Default transition table
    # ------------------------------------------------------------------

    def _build_default_transitions(self) -> None:
        """Populate the base transition rules between conversation phases."""
        self._transitions = [
            # INITIAL → early session options
            StateTransition(
                from_state=ConversationPhase.INITIAL,
                to_state=ConversationPhase.ENGAGING,
                trigger="therapist_greeting",
                probability=0.6,
            ),
            StateTransition(
                from_state=ConversationPhase.INITIAL,
                to_state=ConversationPhase.RESISTANT,
                trigger="therapist_greeting",
                probability=0.25,
            ),
            StateTransition(
                from_state=ConversationPhase.INITIAL,
                to_state=ConversationPhase.DISTRESSED,
                trigger="sensitive_topic",
                probability=0.15,
            ),
            # ENGAGING
            StateTransition(
                from_state=ConversationPhase.ENGAGING,
                to_state=ConversationPhase.REFLECTIVE,
                trigger="therapist_insight_question",
                probability=0.4,
            ),
            StateTransition(
                from_state=ConversationPhase.ENGAGING,
                to_state=ConversationPhase.RESISTANT,
                trigger="challenging_topic",
                probability=0.25,
            ),
            StateTransition(
                from_state=ConversationPhase.ENGAGING,
                to_state=ConversationPhase.DISTRESSED,
                trigger="trauma_disclosure",
                probability=0.15,
            ),
            StateTransition(
                from_state=ConversationPhase.ENGAGING,
                to_state=ConversationPhase.CLOSURE,
                trigger="session_end",
                probability=0.05,
            ),
            StateTransition(
                from_state=ConversationPhase.ENGAGING,
                to_state=ConversationPhase.ENGAGING,
                trigger="neutral_probe",
                probability=0.15,
            ),
            # RESISTANT
            StateTransition(
                from_state=ConversationPhase.RESISTANT,
                to_state=ConversationPhase.ENGAGING,
                trigger="therapist_validation",
                probability=0.3,
            ),
            StateTransition(
                from_state=ConversationPhase.RESISTANT,
                to_state=ConversationPhase.REFLECTIVE,
                trigger="skillful_reframe",
                probability=0.2,
            ),
            StateTransition(
                from_state=ConversationPhase.RESISTANT,
                to_state=ConversationPhase.DISTRESSED,
                trigger="persistent_probe",
                probability=0.25,
            ),
            StateTransition(
                from_state=ConversationPhase.RESISTANT,
                to_state=ConversationPhase.RESISTANT,
                trigger="therapist_directive",
                probability=0.25,
            ),
            # REFLECTIVE
            StateTransition(
                from_state=ConversationPhase.REFLECTIVE,
                to_state=ConversationPhase.INSIGHT,
                trigger="interpretation",
                probability=0.35,
            ),
            StateTransition(
                from_state=ConversationPhase.REFLECTIVE,
                to_state=ConversationPhase.DISTRESSED,
                trigger="painful_realization",
                probability=0.2,
            ),
            StateTransition(
                from_state=ConversationPhase.REFLECTIVE,
                to_state=ConversationPhase.ENGAGING,
                trigger="shift_topic",
                probability=0.25,
            ),
            StateTransition(
                from_state=ConversationPhase.REFLECTIVE,
                to_state=ConversationPhase.CLOSURE,
                trigger="session_end",
                probability=0.1,
            ),
            StateTransition(
                from_state=ConversationPhase.REFLECTIVE,
                to_state=ConversationPhase.REFLECTIVE,
                trigger="elaboration",
                probability=0.1,
            ),
            # DISTRESSED
            StateTransition(
                from_state=ConversationPhase.DISTRESSED,
                to_state=ConversationPhase.ENGAGING,
                trigger="therapist_grounding",
                probability=0.25,
            ),
            StateTransition(
                from_state=ConversationPhase.DISTRESSED,
                to_state=ConversationPhase.REFLECTIVE,
                trigger="safe_processing",
                probability=0.3,
            ),
            StateTransition(
                from_state=ConversationPhase.DISTRESSED,
                to_state=ConversationPhase.RESISTANT,
                trigger="deepening_exploration",
                probability=0.2,
            ),
            StateTransition(
                from_state=ConversationPhase.DISTRESSED,
                to_state=ConversationPhase.CLOSURE,
                trigger="session_end_early",
                probability=0.15,
            ),
            StateTransition(
                from_state=ConversationPhase.DISTRESSED,
                to_state=ConversationPhase.DISTRESSED,
                trigger="continuous_disclosure",
                probability=0.1,
            ),
            # INSIGHT
            StateTransition(
                from_state=ConversationPhase.INSIGHT,
                to_state=ConversationPhase.REFLECTIVE,
                trigger="integration_question",
                probability=0.35,
            ),
            StateTransition(
                from_state=ConversationPhase.INSIGHT,
                to_state=ConversationPhase.DISTRESSED,
                trigger="unexpected_emotion",
                probability=0.15,
            ),
            StateTransition(
                from_state=ConversationPhase.INSIGHT,
                to_state=ConversationPhase.CLOSURE,
                trigger="session_end",
                probability=0.3,
            ),
            StateTransition(
                from_state=ConversationPhase.INSIGHT,
                to_state=ConversationPhase.ENGAGING,
                trigger="new_topic",
                probability=0.2,
            ),
            # CLOSURE (absorbing state for the session)
            StateTransition(
                from_state=ConversationPhase.CLOSURE,
                to_state=ConversationPhase.CLOSURE,
                trigger="session_complete",
                probability=1.0,
            ),
        ]

    # ------------------------------------------------------------------
    # Profile-modulated resistance baseline
    # ------------------------------------------------------------------

    def _profile_resistance_baseline(self) -> float:
        """Return an estimated resistance level derived from profile traits.

        Higher values indicate a profile more prone to therapeutic resistance.
        """
        style = self.profile.default_style
        severity = sum(self.profile.severity_range) / 2.0

        # Styles with known high resistance
        if str(style) in ("hostile",):
            base = 0.7
        elif str(style) in ("anxious", "melancholic"):
            base = 0.5
        elif str(style) in ("manic", "neutral"):
            base = 0.35
        else:
            base = 0.3  # friendly

        # Higher severity patients tend to be more guarded
        return min(base + severity * 0.2, 1.0)

    # ------------------------------------------------------------------
    # State metrics
    # ------------------------------------------------------------------

    def _decay_resistance(self) -> None:
        """Gradually decay resistance when in supportive phases."""
        if self.state.phase in (ConversationPhase.REFLECTIVE, ConversationPhase.INSIGHT):
            self.state.resistance_level = max(0.0, self.state.resistance_level - 0.08)
        elif self.state.phase == ConversationPhase.ENGAGING:
            self.state.resistance_level = max(0.0, self.state.resistance_level - 0.04)


    def _modulate_probabilities(
        self,
        candidates: list[StateTransition],
        resistance_base: float,
    ) -> tuple[list[tuple[ConversationPhase, float]], float]:
        """Apply profile-based modulation to transition probabilities."""
        total_weight = 0.0
        weighted: list[tuple[ConversationPhase, float]] = []
        for t in candidates:
            prob = t.probability

            # Boost resistance transitions for profiles with high resistance baseline
            if t.to_state == ConversationPhase.RESISTANT:
                prob += resistance_base * 0.15

            # Reduce resistance transitions after alliance-build turn
            if t.to_state == ConversationPhase.RESISTANT and self.state.turn_count > _ALLIANCE_BUILD_TURN:
                prob = max(0.0, prob - 0.15)

            # Boost insight transitions for reflective patients
            if t.to_state == ConversationPhase.INSIGHT and self.state.phase == ConversationPhase.REFLECTIVE:
                prob += self.state.insight_level * 0.15

            # Reduce hostile transitions for friendly-style profiles
            if (
                t.to_state in (ConversationPhase.RESISTANT, ConversationPhase.DISTRESSED)
                and str(self.profile.default_style) == "friendly"
            ):
                prob = max(0.0, prob - 0.1)

            total_weight += prob
            weighted.append((t.to_state, prob))

        return weighted, total_weight

    def _update_metrics(self, chosen: ConversationPhase) -> None:
        """Update auxiliary state metrics based on the chosen phase."""
        if chosen == ConversationPhase.INSIGHT:
            self.state.insight_level = min(1.0, self.state.insight_level + 0.1)
            self.state.engagement_level = min(1.0, self.state.engagement_level + 0.05)
        elif chosen == ConversationPhase.DISTRESSED:
            self.state.distress_level = min(1.0, self.state.distress_level + 0.12)
            self.state.therapeutic_alliance = max(0.0, self.state.therapeutic_alliance - 0.03)
        elif chosen == ConversationPhase.RESISTANT:
            self.state.resistance_level = min(1.0, self.state.resistance_level + 0.1)
        elif chosen == ConversationPhase.ENGAGING:
            self.state.engagement_level = min(1.0, self.state.engagement_level + 0.06)
            self.state.therapeutic_alliance = min(1.0, self.state.therapeutic_alliance + 0.03)
        elif chosen == ConversationPhase.REFLECTIVE:
            self.state.insight_level = min(1.0, self.state.insight_level + 0.05)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def transition(self, trigger: str, *, seed: int | None = None) -> ConversationPhase:
        """Advance the state machine given a *trigger* and return the new phase.

        Args:
            trigger: A label describing the therapist action or conversation event.
            seed: Optional RNG seed for deterministic testing.

        Returns:
            The resulting ConversationPhase after the transition.
        """
        rng = random.Random(seed) if seed is not None else random

        self.state.turn_count += 1

        candidates = [t for t in self._transitions if t.from_state == self.state.phase and t.trigger == trigger]

        if not candidates:
            # No explicit rule — natural drift toward engaging
            if self.state.phase != ConversationPhase.CLOSURE:
                self.state.phase = ConversationPhase.ENGAGING
            self._decay_resistance()
            return self.state.phase

        # Modulate probabilities by profile
        resistance_base = self._profile_resistance_baseline()
        weighted, total_weight = self._modulate_probabilities(candidates, resistance_base)

        if total_weight <= 0:
            if self.state.phase != ConversationPhase.CLOSURE:
                self.state.phase = ConversationPhase.ENGAGING
            self._decay_resistance()
            return self.state.phase

        # Normalise and sample
        r = rng.random() * total_weight
        cumulative = 0.0
        chosen: ConversationPhase = ConversationPhase.ENGAGING
        for phase, prob in weighted:
            cumulative += prob
            if r <= cumulative:
                chosen = phase
                break

        self.state.phase = chosen

        # Update auxiliary metrics
        self._decay_resistance()
        self._update_metrics(chosen)

        return chosen

    def get_possible_transitions(self) -> list[StateTransition]:
        """Return all transitions available from the current phase."""
        return [t for t in self._transitions if t.from_state == self.state.phase]

    def reset(self) -> None:
        """Reset the conversation state to initial values."""
        self.state = ConversationState()
