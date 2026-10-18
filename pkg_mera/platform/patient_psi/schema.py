"""PATIENT-Ψ 8-component CCD data model.

Based on the 8-component cognitive conceptualization model
(arXiv 2405.19660 §3) for cognitive patient simulation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CognitiveTriad(BaseModel):
    """Beck's negative cognitive triad: views of self, world, and future."""

    self_views: float = Field(..., ge=0.0, le=1.0)
    world_views: float = Field(..., ge=0.0, le=1.0)
    future_views: float = Field(..., ge=0.0, le=1.0)

    @field_validator("self_views", "world_views", "future_views")
    @classmethod
    def _check_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"Value must be between 0.0 and 1.0, got {value}"
            raise ValueError(msg)
        return value


class CoreBelief(BaseModel):
    """Deeply held core belief about self, others, world, or future."""

    content: str
    domain: Literal["self", "others", "world", "future"]
    conviction: float = Field(..., ge=0.0, le=1.0)

    @field_validator("conviction")
    @classmethod
    def _check_conviction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"Conviction must be between 0.0 and 1.0, got {value}"
            raise ValueError(msg)
        return value


class IntermediateBelief(BaseModel):
    """Intermediate belief: rule, attitude, or assumption."""

    content: str
    rule_type: Literal["rule", "attitude", "assumption"]
    conviction: float = Field(..., ge=0.0, le=1.0)

    @field_validator("conviction")
    @classmethod
    def _check_conviction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"Conviction must be between 0.0 and 1.0, got {value}"
            raise ValueError(msg)
        return value


class CopingStrategy(BaseModel):
    """Coping strategy with effectiveness rating."""

    content: str
    strategy_type: Literal["avoidance", "compensation", "overcompensation"]
    effectiveness: float = Field(..., ge=0.0, le=1.0)

    @field_validator("effectiveness")
    @classmethod
    def _check_effectiveness(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"Effectiveness must be between 0.0 and 1.0, got {value}"
            raise ValueError(msg)
        return value


class CompensatoryStrategy(BaseModel):
    """Compensatory behavior used to offset perceived deficits."""

    content: str
    behavior: str
    overcompensation_for: str | None = None


class SituationInterpretation(BaseModel):
    """Interpretation of a specific situation, possibly with cognitive distortion."""

    situation: str
    interpretation: str
    distortion_type: str | None = None


class EmotionalResponse(BaseModel):
    """Emotional response with intensity and valence."""

    emotion: str
    intensity: float = Field(..., ge=0.0, le=1.0)
    valence: Literal["positive", "negative", "mixed"]

    @field_validator("intensity")
    @classmethod
    def _check_intensity(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"Intensity must be between 0.0 and 1.0, got {value}"
            raise ValueError(msg)
        return value


class BehavioralResponse(BaseModel):
    """Behavioral response triggered by a specific stimulus."""

    behavior: str
    triggered_by: str
    consequence: str | None = None


class PatientCCD(BaseModel):
    """Complete CCD profile for a patient (PATIENT-Ψ 8-component model)."""

    client_id: str
    triads: CognitiveTriad | None = None
    core_beliefs: list[CoreBelief] = []
    intermediate_beliefs: list[IntermediateBelief] = []
    coping_strategies: list[CopingStrategy] = []
    compensatory_strategies: list[CompensatoryStrategy] = []
    situation_interpretations: list[SituationInterpretation] = []
    emotional_responses: list[EmotionalResponse] = []
    behavioral_responses: list[BehavioralResponse] = []

    def add_core_belief(
        self,
        content: str,
        domain: Literal["self", "others", "world", "future"],
        conviction: float = 0.8,
    ) -> CoreBelief:
        """Append and return a new CoreBelief."""
        belief = CoreBelief(content=content, domain=domain, conviction=conviction)
        self.core_beliefs.append(belief)
        return belief

    def add_emotional_response(
        self,
        emotion: str,
        intensity: float,
        valence: Literal["positive", "negative", "mixed"],
    ) -> EmotionalResponse:
        """Append and return a new EmotionalResponse."""
        response = EmotionalResponse(emotion=emotion, intensity=intensity, valence=valence)
        self.emotional_responses.append(response)
        return response

    def add_behavioral_response(
        self,
        behavior: str,
        triggered_by: str,
        consequence: str | None = None,
    ) -> BehavioralResponse:
        """Append and return a new BehavioralResponse."""
        response = BehavioralResponse(behavior=behavior, triggered_by=triggered_by, consequence=consequence)
        self.behavioral_responses.append(response)
        return response

    def add_coping_strategy(
        self,
        content: str,
        strategy_type: Literal["avoidance", "compensation", "overcompensation"],
        effectiveness: float = 0.5,
    ) -> CopingStrategy:
        """Append and return a new CopingStrategy."""
        strategy = CopingStrategy(content=content, strategy_type=strategy_type, effectiveness=effectiveness)
        self.coping_strategies.append(strategy)
        return strategy

    def get_negative_triad_score(self) -> float:
        """Average of (1 - view) across triad dimensions; 0.5 default if no triads."""
        if self.triads is None:
            return 0.5
        return (
            (1.0 - self.triads.self_views) + (1.0 - self.triads.world_views) + (1.0 - self.triads.future_views)
        ) / 3.0

    def to_dict(self) -> dict:
        """Serialize to dictionary via model_dump."""
        return self.model_dump()

    def get_high_conviction_beliefs(self, threshold: float = 0.7) -> list[CoreBelief]:
        """Return core beliefs with conviction >= threshold."""
        return [belief for belief in self.core_beliefs if belief.conviction >= threshold]
