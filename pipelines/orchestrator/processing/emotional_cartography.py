"""
Emotional Cartography Module
----------------------------
Implements Plutchik's Wheel of Emotions and advanced emotional mapping logic.
This module provides the "Emotional Cartography" necessary for the AI to understand
complex, mixed, and evolving emotional states in therapeutic contexts.

Key Concepts:
- 8 Primary Emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation)
- Intensities (Hypo, Basic, Hyper)
- Dyads (Primary, Secondary, Tertiary combinations)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PlutchikEmotion(Enum):
    """The 8 primary dimensions of Plutchik's wheel."""

    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class Intensity(Enum):
    """Intensity levels for emotions."""

    SERENITY = "serenity"  # Low intensity Joy
    JOY = "joy"  # Medium intensity Joy
    ECSTASY = "ecstasy"  # High intensity Joy

    ACCEPTANCE = "acceptance"  # Low intensity Trust
    TRUST = "trust"  # Medium intensity Trust
    ADMIRATION = "admiration"  # High intensity Trust

    APPREHENSION = "apprehension"  # Low intensity Fear
    FEAR = "fear"  # Medium intensity Fear
    TERROR = "terror"  # High intensity Fear

    DISTRACTION = "distraction"  # Low intensity Surprise
    SURPRISE = "surprise"  # Medium intensity Surprise
    AMAZEMENT = "amazement"  # High intensity Surprise

    PENSIVENESS = "pensiveness"  # Low intensity Sadness
    SADNESS = "sadness"  # Medium intensity Sadness
    GRIEF = "grief"  # High intensity Sadness

    BOREDOM = "boredom"  # Low intensity Disgust
    DISGUST = "disgust"  # Medium intensity Disgust
    LOATHING = "loathing"  # High intensity Disgust

    ANNOYANCE = "annoyance"  # Low intensity Anger
    ANGER = "anger"  # Medium intensity Anger
    RAGE = "rage"  # High intensity Anger

    INTEREST = "interest"  # Low intensity Anticipation
    ANTICIPATION = "anticipation"  # Medium intensity Anticipation
    VIGILANCE = "vigilance"  # High intensity Anticipation


@dataclass
class EmotionalState:
    """Represents a specific point on the emotional map."""

    primary_emotion: PlutchikEmotion
    intensity_score: float = 0.5  # 0.0 to 1.0
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self._determine_display_name()

    def _determine_display_name(self) -> str:
        # Simplified mapping logic for display names based on intensity
        if self.intensity_score < 0.33:
            return f"Light {self.primary_emotion.value}"
        elif self.intensity_score > 0.66:
            return f"Intense {self.primary_emotion.value}"
        return self.primary_emotion.value


class EmotionalCartographer:
    """
    Maps emotional currents and identifies complex emotional states (dyads).
    """

    def __init__(self):
        self.dyad_map = self._build_dyad_map()

    def _build_dyad_map(self) -> Dict[Tuple[PlutchikEmotion, PlutchikEmotion], str]:
        """
        Builds the map of emotional combinations (Dyads).
        Order of tuple keys does not matter in usage (we sort them).
        """
        # Primary Dyads (Adjacent)
        m = {}

        # Primary (Adjacent)
        self._add_dyad(m, PlutchikEmotion.JOY, PlutchikEmotion.TRUST, "Love")
        self._add_dyad(m, PlutchikEmotion.TRUST, PlutchikEmotion.FEAR, "Submission")
        self._add_dyad(m, PlutchikEmotion.FEAR, PlutchikEmotion.SURPRISE, "Awe")
        self._add_dyad(m, PlutchikEmotion.SURPRISE, PlutchikEmotion.SADNESS, "Disapproval")
        self._add_dyad(m, PlutchikEmotion.SADNESS, PlutchikEmotion.DISGUST, "Remorse")
        self._add_dyad(m, PlutchikEmotion.DISGUST, PlutchikEmotion.ANGER, "Contempt")
        self._add_dyad(m, PlutchikEmotion.ANGER, PlutchikEmotion.ANTICIPATION, "Aggressiveness")
        self._add_dyad(m, PlutchikEmotion.ANTICIPATION, PlutchikEmotion.JOY, "Optimism")

        # Secondary Dyads (One apart)
        self._add_dyad(m, PlutchikEmotion.JOY, PlutchikEmotion.FEAR, "Guilt")
        self._add_dyad(m, PlutchikEmotion.TRUST, PlutchikEmotion.SURPRISE, "Curiosity")
        self._add_dyad(m, PlutchikEmotion.FEAR, PlutchikEmotion.SADNESS, "Despair")
        self._add_dyad(
            m, PlutchikEmotion.SURPRISE, PlutchikEmotion.DISGUST, "Unbelief"
        )  # or Horror/Shock
        self._add_dyad(m, PlutchikEmotion.SADNESS, PlutchikEmotion.ANGER, "Envy")  # Sullenness/Envy
        self._add_dyad(m, PlutchikEmotion.DISGUST, PlutchikEmotion.ANTICIPATION, "Cynicism")
        self._add_dyad(m, PlutchikEmotion.ANGER, PlutchikEmotion.JOY, "Pride")
        self._add_dyad(m, PlutchikEmotion.ANTICIPATION, PlutchikEmotion.TRUST, "Fatalism")

        # Tertiary Dyads (Two apart - Opposites logic varies, usually these are conflicting)
        self._add_dyad(m, PlutchikEmotion.JOY, PlutchikEmotion.SURPRISE, "Delight")
        self._add_dyad(m, PlutchikEmotion.TRUST, PlutchikEmotion.SADNESS, "Sentimentality")
        self._add_dyad(m, PlutchikEmotion.FEAR, PlutchikEmotion.DISGUST, "Shame")
        self._add_dyad(m, PlutchikEmotion.SURPRISE, PlutchikEmotion.ANGER, "Outrage")
        self._add_dyad(m, PlutchikEmotion.SADNESS, PlutchikEmotion.ANTICIPATION, "Pessimism")
        self._add_dyad(m, PlutchikEmotion.DISGUST, PlutchikEmotion.JOY, "Morbidness")
        self._add_dyad(m, PlutchikEmotion.ANGER, PlutchikEmotion.TRUST, "Dominance")
        self._add_dyad(m, PlutchikEmotion.ANTICIPATION, PlutchikEmotion.FEAR, "Anxiety")

        return m

    def _add_dyad(self, map_obj, e1, e2, name):
        # Store sorted tuple to ensure order independence
        key = tuple(sorted([e1.value, e2.value]))
        map_obj[key] = name

    def identify_dyad(self, emotion1: PlutchikEmotion, emotion2: PlutchikEmotion) -> Optional[str]:
        """Identifies the complex emotion formed by two primary emotions."""
        key = tuple(sorted([emotion1.value, emotion2.value]))
        return self.dyad_map.get(key)

    def map_complex_state(self, emotions: List[EmotionalState]) -> Dict[str, Any]:
        """
        Analyzes a list of detected atomic emotions and returns a cartographic report.
        """
        if not emotions:
            return {"primary_state": "Neutral", "complex_states": []}

        # simple logic: take top 2 intense emotions
        sorted_emotions = sorted(emotions, key=lambda x: x.intensity_score, reverse=True)
        top_emotions = sorted_emotions[:2]

        result = {
            "primary_state": top_emotions[0].primary_emotion.value,
            "intensity": top_emotions[0].intensity_score,
            "complex_states": [],
        }

        if len(top_emotions) >= 2:
            dyad = self.identify_dyad(
                top_emotions[0].primary_emotion, top_emotions[1].primary_emotion
            )
            if dyad:
                result["complex_states"].append(dyad)

        return result
