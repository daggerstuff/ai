"""PATIENT-Ψ rational-emotive coherence model.

Evaluates the internal consistency and cognitive coherence
of patient responses against the CCD profile's belief structure.
Uses a rule-based approach grounded in cognitive behavioural theory
(Beck, 1964; Ellis, 1962) to measure belief alignment, emotional
congruence, narrative coherence, and cognitive dissonance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from ai.tools.utilities.platform.patient_psi.profiles import ClinicalProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Narrative coherence word-count thresholds
_MIN_COHERENT_WORDS = 3
_MODERATE_COHERENT_WORDS = 15

# Composite score weights
_WEIGHT_BELIEF = 0.30
_WEIGHT_EMOTION = 0.25
_WEIGHT_NARRATIVE = 0.25
_WEIGHT_DISSONANCE = 0.20

# Phase modulation deltas
_DISTRESS_EMOTIONAL_BOOST = 0.1
_DISTRESS_NARRATIVE_PENALTY = 0.15
_DISTRESS_DISSONANCE_BOOST = 0.1
_RESISTANT_BELIEF_PENALTY = 0.15
_RESISTANT_NARRATIVE_PENALTY = 0.1
_INSIGHT_NARRATIVE_BOOST = 0.15
_INSIGHT_BELIEF_BOOST = 0.1

# Dissonance amplification factor for unstable profiles
_BPD_DISSONANCE_FACTOR = 1.25

# Fragment penalty scaling
_FRAGMENT_PENALTY_FACTOR = 15

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CoherenceScore(BaseModel):
    """Multi-dimensional coherence measurement for a patient response.

    Attributes:
        overall: Weighted composite coherence score (0 = fully incoherent,
            1 = fully coherent with profile).
        belief_consistency: How well the response aligns with the profile's
            core / intermediate beliefs (0 = contradicts core beliefs,
            1 = perfectly consistent).
        emotional_congruence: Whether expressed emotion matches the profile's
            expected emotional responses (0 = incongruent, 1 = fully congruent).
        narrative_coherence: Internal logical consistency of the utterance
            itself (0 = fragmented/jumbled, 1 = clear and connected).
        cognitive_dissonance: Signal of internal conflict, contradictory
            statements, or struggling between incompatible beliefs
            (0 = no dissonance, 1 = maximum dissonance).
    """

    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    belief_consistency: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_congruence: float = Field(default=0.5, ge=0.0, le=1.0)
    narrative_coherence: float = Field(default=0.5, ge=0.0, le=1.0)
    cognitive_dissonance: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Keyword lexicons for rule-based coherence detection
# ---------------------------------------------------------------------------

# Markers of internal contradiction / cognitive dissonance
_DISSONANCE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(but|however|yet|although|even though)\b", 0.3),
    (r"\b(on the one hand|on the other hand)\b", 0.4),
    (r"\b(i don'?t know\b.*\bbut|i\'?m not sure\b.*\bbut)", 0.35),
    (r"\b(part of me|another part of me|at the same time)\b", 0.35),
    (r"\b(i should|i ought to)\b", 0.2),
    (r"\b(contradictory|conflicted|confused|torn between)\b", 0.4),
    (r"\b(it doesn'?t make sense|it makes no sense)\b", 0.45),
    (r"\b(both.*and.*at the same time)\b", 0.2),
]

# Markers of emotionally-laden language (positive *and* negative)
_EMOTIONAL_WORDS: dict[str, Literal["negative", "positive", "mixed"]] = {
    "angry": "negative",
    "sad": "negative",
    "depressed": "negative",
    "anxious": "negative",
    "terrified": "negative",
    "scared": "negative",
    "ashamed": "negative",
    "guilty": "negative",
    "hopeless": "negative",
    "worthless": "negative",
    "empty": "negative",
    "frustrated": "negative",
    "hurts": "negative",
    "pain": "negative",
    "hurt": "negative",
    "afraid": "negative",
    "hate": "negative",
    "fear": "negative",
    "numb": "negative",
    "panicked": "negative",
    "happy": "positive",
    "hopeful": "positive",
    "grateful": "positive",
    "thankful": "positive",
    "relieved": "positive",
    "calm": "positive",
    "content": "positive",
    "excited": "positive",
    "optimistic": "positive",
    "confident": "positive",
    "loved": "positive",
    "safe": "positive",
    "proud": "positive",
    "bittersweet": "mixed",
    "conflicted": "mixed",
    "ambivalent": "mixed",
    "uncertain": "mixed",
}

# Markers of fragmented / low-narrative-coherence speech
_FRAGMENT_MARKERS: list[str] = [
    "i don't know",
    "i don't understand",
    "it's confusing",
    "everything",
    "nothing",
    "maybe",
    "sort of",
    "kind of",
    "i guess",
    "whatever",
]

_ABSOLUTIST_WORDS: list[str] = [
    "always",
    "never",
    "everyone",
    "nobody",
    "everything",
    "nothing",
    "completely",
    "totally",
    "absolutely",
    "impossible",
    "must",
    "cannot",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_patterns(text: str, patterns: list[str]) -> int:
    """Count occurrences of literal pattern words (case-insensitive word search)."""
    lower = text.lower()
    count = 0
    for pat in patterns:
        count += len(re.findall(rf"\b{re.escape(pat)}\b", lower))
    return count


def _count_regex_patterns(text: str, patterns: list[tuple[str, float]]) -> float:
    """Score dissonance-related regex patterns in text.

    Returns a cumulative score weighted by pattern severity.
    """
    total = 0.0
    for pat, weight in patterns:
        matches = re.findall(pat, text.lower())
        total += len(matches) * weight
    return min(total, 1.0)


def _extract_emotions(text: str) -> dict[Literal["negative", "positive", "mixed"], int]:
    """Count emotion words by valence in *text*."""
    lower = text.lower()
    counts: dict[Literal["negative", "positive", "mixed"], int] = {
        "negative": 0,
        "positive": 0,
        "mixed": 0,
    }
    for word, valence in _EMOTIONAL_WORDS.items():
        matches = re.findall(rf"\b{re.escape(word)}\b", lower)
        counts[valence] += len(matches)
    return counts


# ---------------------------------------------------------------------------
# Emotional congruence
# ---------------------------------------------------------------------------


def _compute_emotional_congruence(response: str, profile: ClinicalProfile) -> float:
    """Score how well the response's emotional tone matches the profile's expected style."""
    emotion_counts = _extract_emotions(response)
    profile_style = str(profile.default_style)

    if profile_style in ("melancholic", "hostile"):
        neg_ratio = emotion_counts["negative"] / max(emotion_counts["negative"] + emotion_counts["positive"], 1)
        return 0.3 + 0.7 * neg_ratio

    if profile_style in ("anxious",):
        neg_ratio = emotion_counts["negative"] / max(emotion_counts["negative"] + emotion_counts["positive"], 1)
        return 0.2 + 0.8 * neg_ratio

    if profile_style in ("manic",):
        pos_ratio = emotion_counts["positive"] / max(emotion_counts["negative"] + emotion_counts["positive"], 1)
        return 0.2 + 0.8 * pos_ratio

    if profile_style in ("friendly",):
        pos_ratio = emotion_counts["positive"] / max(emotion_counts["negative"] + emotion_counts["positive"], 1)
        return 0.3 + 0.7 * pos_ratio

    # neutral — balanced expectation
    if emotion_counts["negative"] + emotion_counts["positive"] == 0:
        return 0.5
    bal = emotion_counts["positive"] / max(emotion_counts["negative"] + emotion_counts["positive"], 1)
    return 0.3 + 0.4 * bal


# ---------------------------------------------------------------------------
# Coherence Model
# ---------------------------------------------------------------------------


@dataclass
class _BeliefMatch:
    """Internal helper: tracks which profile beliefs a response touches."""

    core_belief_hits: int = 0
    intermediate_belief_hits: int = 0
    total_core: int = 0
    total_intermediate: int = 0


def _match_beliefs(text: str, profile: ClinicalProfile) -> _BeliefMatch:
    """Count how many profile beliefs are referenced or contradicted."""
    lower = text.lower()
    response_tokens = set(re.findall(r"\w+", lower))
    match = _BeliefMatch()

    # Core beliefs
    config = profile.ccd_config
    match.total_core = len(config.get("core_beliefs", []))
    for cb in config.get("core_beliefs", []):
        if isinstance(cb, dict):
            content = cb.get("content", "")
            if any(word in response_tokens for word in re.findall(r"\w{4,}", content.lower())):
                match.core_belief_hits += 1

    # Intermediate beliefs
    match.total_intermediate = len(config.get("intermediate_beliefs", []))
    for ib in config.get("intermediate_beliefs", []):
        if isinstance(ib, dict):
            content = ib.get("content", "")
            if any(word in response_tokens for word in re.findall(r"\w{4,}", content.lower())):
                match.intermediate_belief_hits += 1

    return match


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CoherenceModel:
    """Evaluates response coherence against a clinical profile.

    The model combines rule-based belief matching, emotional congruence
    checks, narrative flow analysis, and cognitive dissonance detection
    into a multi-dimensional CoherenceScore.
    """

    def __init__(self, profile: ClinicalProfile) -> None:
        """Initialise with the simulated patient's clinical profile."""
        self.profile = profile

    def evaluate(self, response: str, context: dict | None = None) -> CoherenceScore:
        """Score the coherence of a patient utterance.

        Args:
            response: The patient's utterance as plain text.
            context: Optional dict with additional signals
                (e.g. ``{"phase": "distressed"}``).

        Returns:
            A CoherenceScore with all five dimensions populated.
        """
        if not response.strip():
            return CoherenceScore(overall=0.5)

        phase = (context or {}).get("phase", "engaging")

        # --- Belief consistency ---
        belief = _match_beliefs(response, self.profile)
        if belief.total_core + belief.total_intermediate > 0:
            total_possible = belief.total_core + belief.total_intermediate
            total_hits = belief.core_belief_hits + belief.intermediate_belief_hits
            belief_consistency = min(total_hits / max(total_possible, 1), 1.0)
        else:
            belief_consistency = 0.5

        # --- Emotional congruence ---
        emotional_congruence = _compute_emotional_congruence(response, self.profile)

        # --- Narrative coherence ---
        words = response.split()
        word_count = len(words)
        if word_count < _MIN_COHERENT_WORDS:
            narrative_coherence = 0.3
        elif word_count < _MODERATE_COHERENT_WORDS:
            narrative_coherence = 0.5
        else:
            frag_count = _count_patterns(response, _FRAGMENT_MARKERS)
            frag_rate = frag_count / max(word_count, 1)
            narrative_coherence = max(0.1, 1.0 - frag_rate * _FRAGMENT_PENALTY_FACTOR)

        # --- Cognitive dissonance ---
        dissonance = _count_regex_patterns(response, _DISSONANCE_PATTERNS)
        # Amplify dissonance in profiles with unstable core beliefs (BPD, Bipolar)
        if self.profile.name in ("borderline_personality", "bipolar_i", "bipolar_ii"):
            dissonance = min(1.0, dissonance * _BPD_DISSONANCE_FACTOR)

        # --- Phase modulation ---
        if phase == "distressed":
            emotional_congruence = min(1.0, emotional_congruence + _DISTRESS_EMOTIONAL_BOOST)
            narrative_coherence = max(0.1, narrative_coherence - _DISTRESS_NARRATIVE_PENALTY)
            dissonance = min(1.0, dissonance + _DISTRESS_DISSONANCE_BOOST)
        elif phase == "resistant":
            belief_consistency = max(0.0, belief_consistency - _RESISTANT_BELIEF_PENALTY)
            narrative_coherence = max(0.1, narrative_coherence - _RESISTANT_NARRATIVE_PENALTY)
        elif phase == "insight":
            narrative_coherence = min(1.0, narrative_coherence + _INSIGHT_NARRATIVE_BOOST)
            belief_consistency = min(1.0, belief_consistency + _INSIGHT_BELIEF_BOOST)

        # --- Composite ---
        overall = (
            _WEIGHT_BELIEF * belief_consistency
            + _WEIGHT_EMOTION * emotional_congruence
            + _WEIGHT_NARRATIVE * narrative_coherence
            + _WEIGHT_DISSONANCE * (1.0 - dissonance)
        )

        return CoherenceScore(
            overall=round(overall, 4),
            belief_consistency=round(belief_consistency, 4),
            emotional_congruence=round(emotional_congruence, 4),
            narrative_coherence=round(narrative_coherence, 4),
            cognitive_dissonance=round(dissonance, 4),
        )

    def predict_coherence_range(self) -> tuple[float, float]:
        """Return the expected coherence range for this profile's typical responses.

        Derived from the profile's severity range and default style.
        """
        severity_low, severity_high = self.profile.severity_range
        style = str(self.profile.default_style)

        # Profile-level base coherence
        style_base: dict[str, float] = {
            "neutral": 0.65,
            "friendly": 0.7,
            "hostile": 0.4,
            "anxious": 0.5,
            "melancholic": 0.45,
            "manic": 0.35,
        }
        base = style_base.get(style, 0.5)

        low = max(0.1, base - severity_high * 0.25)
        high = min(1.0, base + (1.0 - severity_low) * 0.2)
        return (round(low, 4), round(high, 4))
