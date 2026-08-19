"""Emotion mapping utilities inspired by Plutchik's wheel of emotions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PlutchikEmotion(StrEnum):
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


@dataclass(frozen=True)
class EmotionalState:
    emotion: PlutchikEmotion
    intensity: float


@dataclass
class EmotionalMap:
    primary_state: str
    complex_states: list[str]
    intensities: dict[str, float]


class EmotionalCartographer:
    def __init__(self) -> None:
        self._dyad_map = {
            frozenset({PlutchikEmotion.JOY, PlutchikEmotion.TRUST}): "Love",
            frozenset({PlutchikEmotion.ANGER, PlutchikEmotion.DISGUST}): "Contempt",
            frozenset({PlutchikEmotion.JOY, PlutchikEmotion.FEAR}): "Guilt",
            frozenset({PlutchikEmotion.ANTICIPATION, PlutchikEmotion.TRUST}): "Fatalism",
        }

    def identify_dyad(self, first: PlutchikEmotion, second: PlutchikEmotion) -> str:
        return self._dyad_map.get(frozenset((first, second)), "")

    def map_complex_state(self, states: list[EmotionalState]) -> dict[str, object]:
        if not states:
            return {"primary_state": "Neutral", "complex_states": [], "intensities": {}}

        ordered = sorted(states, key=lambda item: item.intensity, reverse=True)
        primary = ordered[0].emotion.value

        intensities = {state.emotion.value: state.intensity for state in states}
        complex_states: list[str] = []
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                dyad = self.identify_dyad(first.emotion, second.emotion)
                if dyad:
                    complex_states.append(dyad)

        return {
            "primary_state": primary,
            "complex_states": complex_states,
            "intensities": intensities,
        }


__all__ = ["EmotionalCartographer", "EmotionalMap", "EmotionalState", "PlutchikEmotion"]
