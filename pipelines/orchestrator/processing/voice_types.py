"""
Voice Processing Type Definitions

Shared type definitions for voice processing components to avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PersonalityTrait(Enum):
    """Personality trait categories."""

    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class PersonalityProfile:
    """Personality profile for voice optimization."""

    trait_scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    confidence_scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    dominant_traits: list[PersonalityTrait] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def get_trait_score(self, trait: PersonalityTrait) -> float:
        """Get score for a specific trait."""
        return self.trait_scores.get(trait, 0.0)

    def get_confidence(self, trait: PersonalityTrait) -> float:
        """Get confidence for a specific trait."""
        return self.confidence_scores.get(trait, 0.0)


@dataclass
class OptimizationResult:
    """Result of voice training optimization."""

    conversation_id: str
    original_quality_score: float
    optimized_quality_score: float
    improvement_percentage: float
    optimization_techniques: list[str] = field(default_factory=list)
    personality_adjustments: dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AuthenticityProfile:
    """Authenticity assessment profile."""

    overall_authenticity_score: float
    consistency_score: float
    naturalness_score: float
    emotional_alignment_score: float
    authenticity_factors: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    assessment_timestamp: datetime = field(default_factory=datetime.now)
