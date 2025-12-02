#!/usr/bin/env python3
"""
Edge Category and Profile Types for Mental Health Training Datasets
Defines nightmare fuel edge case categories, intensity levels, and edge profiles
as described in the expanded project brief.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class EdgeCategory(Enum):
    """
    Edge crisis and trauma categories for nightmare fuel training scenarios.
    Based on the 25+ high-difficulty categories from the expanded project brief.
    """
    # Core crisis categories
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    HOMICIDAL_IDEATION = "homicidal_ideation"
    PSYCHOTIC_EPISODES = "psychotic_episodes"

    # Abuse and violence categories
    CHILD_ABUSE_REPORTING = "child_abuse_reporting"
    DOMESTIC_VIOLENCE = "domestic_violence"
    INTIMATE_PARTNER_VIOLENCE = "intimate_partner_violence"
    SEXUAL_ASSAULT = "sexual_assault"

    # Mental health crisis categories
    EATING_DISORDERS = "eating_disorders"
    SUBSTANCE_ABUSE = "substance_abuse"
    ADDICTION_CRISIS = "addiction_crisis"
    OVERDOSE_SCENARIOS = "overdose_scenarios"

    # Personality and boundary testing
    BORDERLINE_PERSONALITY_CRISIS = "borderline_personality_crisis"
    BOUNDARY_TESTING_SCENARIOS = "boundary_testing_scenarios"
    EMOTIONAL_DYSREGULATION = "emotional_dysregulation"
    MANIPULATION_PATTERNS = "manipulation_patterns"
    SPLITTING_BEHAVIORS = "splitting_behaviors"

    # Trauma and severe scenarios
    SEVERE_TRAUMA = "severe_trauma"
    TRAUMA_FLASHBACKS = "trauma_flashbacks"
    COMPLEX_TRAUMA = "complex_trauma"
    PTSD_CRISIS = "ptsd_crisis"

    # Additional high-intensity categories
    PSYCHIATRIC_EMERGENCIES = "psychiatric_emergencies"
    HOSTILITY_AND_AGGRESSION = "hostility_and_aggression"
    CRISIS_ESCALATION = "crisis_escalation"
    EMERGENCY_INTERVENTION = "emergency_intervention"

    # Cultural and minority mental health edge cases
    MINORITY_MENTAL_HEALTH_CRISIS = "minority_mental_health_crisis"
    CULTURAL_TRAUMA_SCENARIOS = "cultural_trauma_scenarios"


class IntensityLevel(Enum):
    """
    Crisis intensity levels for edge scenarios.
    Matches the brief's "Very high, high, moderate" levels with additional granularity.
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class EdgeProfile:
    """
    Complete edge profile metadata for a training example.
    Combines category, intensity, tone, stage, and additional metadata.
    """
    profile_id: str
    category: EdgeCategory
    intensity: IntensityLevel
    tone: str  # Will be refined in Task 4 (less-chipper module)
    stage: int  # Training stage (1-4) - typically Stage 3 for edge cases

    # Additional metadata
    scenario_type: Optional[str] = None
    challenge_type: Optional[str] = None
    requires_supervision: bool = True
    crisis_language_preserved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "category": self.category.value,
            "intensity": self.intensity.value,
            "tone": self.tone,
            "stage": self.stage,
            "scenario_type": self.scenario_type,
            "challenge_type": self.challenge_type,
            "requires_supervision": self.requires_supervision,
            "crisis_language_preserved": self.crisis_language_preserved,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeProfile":
        """Deserialize edge profile from dictionary."""
        return cls(
            profile_id=data["profile_id"],
            category=EdgeCategory(data["category"]),
            intensity=IntensityLevel(data["intensity"]),
            tone=data["tone"],
            stage=data["stage"],
            scenario_type=data.get("scenario_type"),
            challenge_type=data.get("challenge_type"),
            requires_supervision=data.get("requires_supervision", True),
            crisis_language_preserved=data.get("crisis_language_preserved", True),
            metadata=data.get("metadata", {}),
        )


def get_all_edge_categories() -> List[EdgeCategory]:
    """Get all available edge categories."""
    return list(EdgeCategory)


def get_all_intensity_levels() -> List[IntensityLevel]:
    """Get all available intensity levels."""
    return list(IntensityLevel)


def get_categories_by_intensity(intensity: IntensityLevel) -> List[EdgeCategory]:
    """
    Get edge categories that are typically associated with a given intensity level.
    This is a helper for filtering/querying, not a strict requirement.
    """
    if intensity in [IntensityLevel.VERY_HIGH, IntensityLevel.EXTREME]:
        # Very high and extreme intensity categories
        return [
            EdgeCategory.SUICIDAL_IDEATION,
            EdgeCategory.HOMICIDAL_IDEATION,
            EdgeCategory.PSYCHOTIC_EPISODES,
            EdgeCategory.CHILD_ABUSE_REPORTING,
            EdgeCategory.OVERDOSE_SCENARIOS,
            EdgeCategory.EMERGENCY_INTERVENTION,
        ]
    elif intensity == IntensityLevel.HIGH:
        # High intensity categories
        return [
            EdgeCategory.DOMESTIC_VIOLENCE,
            EdgeCategory.SELF_HARM,
            EdgeCategory.ADDICTION_CRISIS,
            EdgeCategory.BORDERLINE_PERSONALITY_CRISIS,
            EdgeCategory.SEVERE_TRAUMA,
            EdgeCategory.PSYCHIATRIC_EMERGENCIES,
        ]
    elif intensity == IntensityLevel.MODERATE:
        # Moderate intensity categories
        return [
            EdgeCategory.EATING_DISORDERS,
            EdgeCategory.EMOTIONAL_DYSREGULATION,
            EdgeCategory.BOUNDARY_TESTING_SCENARIOS,
            EdgeCategory.CULTURAL_TRAUMA_SCENARIOS,
        ]
    else:
        return []  # LOW intensity typically not edge categories


def validate_edge_profile(profile: EdgeProfile) -> tuple[bool, Optional[str]]:
    """
    Validate an edge profile for consistency.
    Returns (is_valid, error_message).
    """
    if profile.stage < 1 or profile.stage > 4:
        return False, f"Stage must be 1-4, got {profile.stage}"

    if not profile.profile_id:
        return False, "profile_id is required"

    return (True, None) if profile.tone else (False, "tone is required")

