"""Adversarial benchmarking schemas for the Nightmare Fuel Generator.

Models failure modes from clinician burnout narratives (Domain 3)
and adversarial evaluation seeds (CounselBench-ADV).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class FailureModeCategory(str, Enum):
    """6 failure mode categories extracted from clinician burnout narratives."""

    MORAL_INJURY = "moral_injury"
    SECONDARY_TRAUMA = "secondary_trauma"
    ETHICAL_DEAD_END = "ethical_dead_end"
    PHYSIOLOGICAL_BURNOUT = "physiological_burnout"
    CASELOAD_OVERWHELM = "caseload_overwhelm"
    IDENTITY_COLLAPSE = "identity_collapse"


class AdversarialTrigger(str, Enum):
    """Trigger types from CounselBench-ADV taxonomy."""

    SAFETY_MISS = "safety_miss"  # LLM fails to detect suicide/self-harm risk
    MISINFORMATION = "misinformation"  # Clinically incorrect advice
    BOUNDARY_VIOLATION = "boundary_violation"  # Therapist boundary crossing
    COUNTERTRANSFERENCE = "countertransference"  # Unmanaged emotional reactivity
    PREMATURE_CLOSURE = "premature_closure"  # Closing topic before resolution
    INVALIDATION = "invalidation"  # Dismissing patient experience
    OVER_IDENTIFICATION = "over_identification"  # Excessive self-disclosure
    RIGID_ADHERENCE = "rigid_adherence"  # Protocol over patient needs
    CULTURAL_BLINDNESS = "cultural_blindness"  # Ignoring cultural context
    SYSTEM_FAILURE = "system_failure"  # Institutional/systemic collapse


class SeverityLevel(str, Enum):
    """Simulated scenario severity."""

    MODERATE = "moderate"  # Discomfort, recoverable
    SEVERE = "severe"  # Significant distress, career impact
    CATASTROPHIC = "catastrophic"  # Career termination, patient harm, suicide


class BurnoutNarrative(BaseModel):
    """Firsthand clinician burnout/trauma account."""

    narrative_id: str
    source_url: str
    source_type: str = Field(..., description="substack, blog, medium, reddit, etc.")
    author_role: str = Field(..., description="counselor, therapist, nurse, etc.")
    years_experience: Optional[int] = None
    clinical_setting: Optional[str] = Field(
        None, description="FQHC, VA, private practice, etc."
    )
    failure_modes: list[FailureModeCategory] = Field(default_factory=list)
    key_dynamics: list[str] = Field(
        default_factory=list,
        description="Extracted relational/systemic dynamics",
    )
    trauma_exposure: Optional[str] = Field(
        None, description="Type of trauma the clinician was exposed to"
    )
    outcome: str = Field(
        ..., description="quit, fired, disabled, career_change, suicidal_crisis"
    )
    outcome_severity: SeverityLevel
    extractable_patterns: list[str] = Field(
        default_factory=list,
        description="Recurring patterns extractable for simulation seeds",
    )
    raw_text: Optional[str] = None  # Full narrative text if available


class AdversarialScenario(BaseModel):
    """A single Nightmare Fuel Generator scenario."""

    scenario_id: str
    title: str
    description: str = Field(
        ..., description="Detailed scenario description for simulation"
    )
    failure_mode: FailureModeCategory
    triggers: list[AdversarialTrigger]
    severity: SeverityLevel
    # Patient configuration
    patient_profile: dict = Field(
        default_factory=dict,
        description="Patient demographics, diagnoses, presentation",
    )
    clinician_profile: dict = Field(
        default_factory=dict,
        description="Clinician experience level, setting, current state",
    )
    # Context that makes this adversarial
    context_pressures: list[str] = Field(
        default_factory=list,
        description="Systemic, institutional, personal pressures",
    )
    ethical_tensions: list[str] = Field(
        default_factory=list,
        description="Conflicting ethical principles in play",
    )
    # Expected failure behaviors
    expected_failure_behaviors: list[str] = Field(
        default_factory=list,
        description="What a model doing this wrong would do/say",
    )
    expected_safe_behaviors: list[str] = Field(
        default_factory=list,
        description="What correct handling looks like",
    )
    # Evaluation
    evaluation_criteria: list[str] = Field(
        default_factory=list,
        description="How to score model response",
    )
    source_narratives: list[str] = Field(
        default_factory=list,
        description="BurnoutNarrative IDs this scenario derives from",
    )
    source_adversarial_seeds: list[str] = Field(
        default_factory=list,
        description="CounselBench-ADV question IDs this builds on",
    )


class AdversarialEvaluationResult(BaseModel):
    """Result of running a model against an adversarial scenario."""

    scenario_id: str
    model_id: str
    response_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Scoring
    detected_safety_risk: bool = False
    safety_risk_details: Optional[str] = None
    failure_behaviors_exhibited: list[str] = Field(default_factory=list)
    safe_behaviors_exhibited: list[str] = Field(default_factory=list)
    # Dimension scores (from CounselBench 6 clinical dimensions)
    empathy_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    clinical_accuracy_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    therapeutic_approach_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    safety_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    communication_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    boundary_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    overall_pass: bool = False
    judge_model: Optional[str] = None
    judge_notes: Optional[str] = None


class NightmareFuelBatch(BaseModel):
    """A batch of generated adversarial scenarios."""

    batch_id: str
    scenarios: list[AdversarialScenario]
    generation_params: dict = Field(
        default_factory=dict,
        description="Parameters used to generate this batch",
    )
    source_narrative_count: int = 0
    source_adversarial_seed_count: int = 0
    coverage: dict = Field(
        default_factory=dict,
        description="Coverage map: failure_mode -> count, trigger -> count",
    )
