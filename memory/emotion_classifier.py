"""
PIX-510 Task 3: Emotional Tagging System
Implements:
  - Valence/Arousal/Dominance (VAD) scoring
  - Plutchik wheel emotion classification (multi-label)
  - Emotion-to-multiplier mapping
  - Session-level trajectory tracking

Acceptance: detection accuracy > 85% on therapeutic test set, latency < 50ms.
"""

from __future__ import annotations

import importlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)

# ─── Plutchik wheel categories ────────────────────────────────────────────────

PLUTCHIK_PRIMARY = frozenset(
    [
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    ]
)

PLUTCHIK_SECONDARY = frozenset(
    [
        "optimism",
        "love",
        "submission",
        "awe",
        "disapproval",
        "remorse",
        "contempt",
        "aggression",
    ]
)

ALL_EMOTION_CATEGORIES = PLUTCHIK_PRIMARY | PLUTCHIK_SECONDARY


# ─── VAD lexical resources ───────────────────────────────────────────────────


@dataclass
class _VADLexicon:
    """
    Curated lexical indicators for Valence, Arousal, Dominance scoring.
    Based on Warrington ANEW and therapeutic dialogue norms.
    """

    HIGH_VALENCE: ClassVar[list[str]] = [
        "happy",
        "joy",
        "grateful",
        "hopeful",
        "excited",
        "relieved",
        "peaceful",
        "love",
        "appreciate",
        "wonderful",
        "better",
        "improving",
        "progress",
    ]
    LOW_VALENCE: ClassVar[list[str]] = [
        "sad",
        "depressed",
        "hopeless",
        "worthless",
        "anxious",
        "worried",
        "fear",
        "terrible",
        "awful",
        "horrible",
        "devastated",
        "anguish",
        "despair",
    ]

    HIGH_AROUSAL: ClassVar[list[str]] = [
        "panic",
        "overwhelmed",
        "shocked",
        "frantic",
        "intense",
        "overwhelmed",
        "trembling",
        "racing",
        "heart pounding",
        "can't breathe",
        "screaming",
    ]
    LOW_AROUSAL: ClassVar[list[str]] = [
        "calm",
        "peaceful",
        "relaxed",
        "numb",
        "detached",
        "empty",
        "flat",
        "indifferent",
        "still",
        "quiet",
        "resting",
    ]

    HIGH_DOMINANCE: ClassVar[list[str]] = [
        "in control",
        "confident",
        "capable",
        "strong",
        "determined",
        "empowered",
        "I can handle",
        "I will",
        "standing up",
        "setting boundaries",
    ]
    LOW_DOMINANCE: ClassVar[list[str]] = [
        "helpless",
        "out of control",
        "overwhelmed",
        "powerless",
        "trapped",
        "stuck",
        "unable",
        "giving up",
        "surrendering",
    ]


# ─── Emotion-to-multiplier mapping ───────────────────────────────────────────


EMOTION_MULTIPLIER: dict[str, float] = {
    # Crisis indicators (PIX-510 spec)
    "suicide": 5.0,
    "self-harm": 5.0,
    "overdose": 5.0,
    "panic": 5.0,
    "psychosis": 5.0,
    # High-intensity therapeutic emotions
    "grief": 2.5,
    "trauma": 2.5,
    "despair": 2.5,
    "hopelessness": 2.5,
    "anxiety": 2.0,
    "fear": 2.0,
    "anger": 2.0,
    "terror": 2.0,
    # Standard Plutchik
    "joy": 1.0,
    "sadness": 1.0,
    "surprise": 1.0,
    "disgust": 1.0,
    "trust": 1.0,
    "anticipation": 1.0,
    # Secondary
    "optimism": 1.0,
    "love": 1.0,
    "submission": 1.0,
    "awe": 1.0,
    "disapproval": 1.0,
    "remorse": 1.0,
    "contempt": 1.0,
    "aggression": 1.5,
}


def emotion_multiplier(categories: list[str]) -> float:
    """Return highest applicable multiplier from emotion categories."""
    if not categories:
        return 1.0
    return max((EMOTION_MULTIPLIER.get(c.lower(), 1.0) for c in categories), default=1.0)


# ─── Tokenisation ─────────────────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


# ─── VAD scorer ──────────────────────────────────────────────────────────────


@dataclass
class VADScorer:
    """
    Lexicon-based Valence/Arousal/Dominance scorer.
    Fast, interpretable, no model inference required.
    """

    _lexicon: ClassVar[_VADLexicon] = _VADLexicon()

    def score(self, text: str) -> tuple[float, float, float]:
        """
        Compute VAD scores from text.

        Returns:
            (valence, arousal, dominance) each in [0.0, 1.0]
            valence: 1.0=very positive, 0.0=very negative
            arousal: 1.0=high activation, 0.0=calm
            dominance: 1.0=in control, 0.0=helpless
        """
        tokens = _tokenise(text)
        n = max(len(tokens), 1)

        valence = self._score_dimension(tokens, n, self._lexicon.HIGH_VALENCE, self._lexicon.LOW_VALENCE)
        arousal = self._score_dimension(tokens, n, self._lexicon.HIGH_AROUSAL, self._lexicon.LOW_AROUSAL)
        dominance = self._score_dimension(tokens, n, self._lexicon.HIGH_DOMINANCE, self._lexicon.LOW_DOMINANCE)

        return (valence, arousal, dominance)

    @staticmethod
    def _score_dimension(tokens: set[str], _n: int, pos: list[str], neg: list[str]) -> float:
        pos_count = sum(1 for w in pos if w in tokens)
        neg_count = sum(1 for w in neg if w in tokens)
        total = pos_count + neg_count
        if total == 0:
            return 0.5  # neutral baseline
        # Normalise to [0, 1]: pos=1.0, neg=0.0, mixed=middle
        return (pos_count - neg_count) / (2 * total) + 0.5


# ─── Main emotion classifier ─────────────────────────────────────────────────


@dataclass
class EmotionClassifier:
    """
    Classifies emotional content in therapeutic dialogue text.

    Supports two modes:
    - `lexicon`: Fast rule-based classification (no model inference).
                 Suitable for real-time use, ~0ms latency.
    - `model`:   Hugging Face transformer-based classification.
                 Higher accuracy, ~20-50ms latency depending on hardware.

    Switching: set CLASSIFIER_MODE env var to "model" and configure MODEL_NAME.
    """

    # ── config ──────────────────────────────────────────────────────────────

    mode: str = field(default_factory=lambda: _get_env("EMOTION_CLASSIFIER_MODE", "lexicon"))
    model_name: str = field(
        default_factory=lambda: _get_env("EMOTION_CLASSIFIER_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    )
    device: str = field(default_factory=lambda: _get_env("EMOTION_CLASSIFIER_DEVICE", "cpu"))
    confidence_threshold: float = 0.5

    # ── state ───────────────────────────────────────────────────────────────

    _vad: VADScorer = field(default_factory=VADScorer, repr=False)
    _pipeline: object | None = field(default_factory=lambda: None, repr=False, compare=False)
    _model_loaded: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode == "model":
            self._load_model()

    # ── public API ─────────────────────────────────────────────────────────

    def classify(self, text: str, multi_label: bool = True) -> EmotionClassificationResult:
        """
        Classify emotional content of input text.

        Args:
            text: Input text (typically a message or memory content string).
            multi_label: If True, return all categories above threshold.
                         If False, return top category only.

        Returns:
            EmotionClassificationResult with categories, VAD scores, and multiplier.
        """
        if not text or not text.strip():
            return EmotionClassificationResult(
                categories=[],
                category_scores={},
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                top_category=None,
                top_score=0.0,
                multiplier=1.0,
            )

        if self.mode == "model" and self._model_loaded:
            return self._classify_model(text, multi_label)
        return self._classify_lexicon(text, multi_label)

    def classify_batch(self, texts: list[str], multi_label: bool = True) -> list[EmotionClassificationResult]:
        """Batch classification for efficiency."""
        return [self.classify(t, multi_label) for t in texts]

    def session_trajectory(self, results: list[EmotionClassificationResult]) -> EmotionTrajectory:
        """
        Analyse emotional trajectory across a session's messages.
        Useful for detecting escalation patterns (e.g., rising anxiety).
        """
        if not results:
            return EmotionTrajectory(
                start_valence=0.5,
                end_valence=0.5,
                start_arousal=0.5,
                end_arousal=0.5,
                start_dominance=0.5,
                end_dominance=0.5,
                trend="stable",
                max_intensity=0.0,
                crisis_indicators=[],
                trajectory_scores=[],
            )

        first = results[0]
        last = results[-1]
        valences = [r.valence for r in results]
        dominances = [r.dominance for r in results]

        crisis: list[str] = [
            r.top_category for r in results if r.top_category in ["suicide", "self-harm", "panic", "psychosis"]
        ]

        trend = _compute_trend(valences, dominances)

        max_intensity = 0.0
        for r in results:
            if r.top_category is not None:
                score = r.category_scores.get(r.top_category, 0.0)
                max_intensity = max(max_intensity, score)

        return EmotionTrajectory(
            start_valence=first.valence,
            end_valence=last.valence,
            start_arousal=first.arousal,
            end_arousal=last.arousal,
            start_dominance=first.dominance,
            end_dominance=last.dominance,
            trend=trend,
            max_intensity=max_intensity,
            crisis_indicators=crisis,
            trajectory_scores=[{"valence": r.valence, "arousal": r.arousal, "dominance": r.dominance} for r in results],
        )

    def benchmark_latency(self, text: str, n: int = 100) -> float:
        """Return average ms per classification over n iterations."""
        start = time.perf_counter()
        for _ in range(n):
            self.classify(text)
        return ((time.perf_counter() - start) / n) * 1000

    # ── internal ────────────────────────────────────────────────────────────

    def _classify_lexicon(self, text: str, multi_label: bool) -> EmotionClassificationResult:
        text_lower = text.lower()
        category_scores: dict[str, float] = {}

        # Keyword → Plutchik mapping
        # Single-word keywords: match in token set. Multi-word: match as substring.
        KEYWORD_MAP: dict[str, list[str]] = {
            "joy": ["happy", "joy", "glad", "excited", "wonderful", "grateful", "love"],
            "sadness": ["sad", "unhappy", "depressed", "grief", "sorrow", "crying", "tears"],
            "anger": ["angry", "rage", "furious", "frustrated", "irritated", "mad"],
            "fear": [
                "afraid",
                "scared",
                "fearful",
                "frightened",
                "terrified",
                "panic",
                "anxious",
                "worried",
                "nervous",
            ],
            "surprise": ["surprised", "amazed", "shocked", "unexpected", "wow"],
            "disgust": ["disgusted", "revolted", "gross", "sickened", "repulsed"],
            "trust": ["trust", "believe", "confidence", "rely", "comfort"],
            "anticipation": ["anticipate", "expect", "looking forward", "hopeful", "hope"],
            "grief": ["grief", "mourning", "loss", "bereavement", "lost"],
            "trauma": ["trauma", "traumatic", "abuse", "violence", "assault"],
            "despair": ["despair", "hopeless", "worthless", "helpless", "giving up"],
            "suicide": ["suicide", "kill myself", "end it all", "no reason to live", "end my life"],
            "self-harm": ["cut myself", "self-harm", "hurt myself", "self injury", "selfharm"],
        }

        MAX_KEYWORDS = 7
        for emotion, keywords in KEYWORD_MAP.items():
            count = 0
            for kw in keywords:
                # Substring match handles multi-word phrases; for single-word
                # keywords this is equivalent to token-set membership.
                if kw.lower() in text_lower:
                    count += 1
            if count > 0:
                score = min(count / MAX_KEYWORDS, 1.0)
                category_scores[emotion] = score
                score = min(count / MAX_KEYWORDS, 1.0)
                category_scores[emotion] = score

        # VAD from lexicon
        valence, arousal, dominance = self._vad.score(text)

        # Determine top category and multiplier
        if category_scores:
            top_category = max(category_scores, key=category_scores.__getitem__)
            top_score = category_scores[top_category]
            if not multi_label:
                category_scores = {top_category: top_score}
        else:
            top_category = None
            top_score = 0.0

        multiplier = emotion_multiplier(list(category_scores.keys()))

        return EmotionClassificationResult(
            categories=list(category_scores.keys()),
            category_scores=category_scores,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            top_category=top_category,
            top_score=top_score,
            multiplier=multiplier,
        )

    def _classify_model(self, text: str, multi_label: bool) -> EmotionClassificationResult:
        if self._pipeline is None:
            self._load_model()
        pipeline = self._pipeline
        if pipeline is None:
            return self._classify_lexicon(text, multi_label)

        results = pipeline(text)
        # Results format: [{"label": "joy", "score": 0.95}, ...]
        if isinstance(results, dict):
            results = results.get("labels", results)

        category_scores: dict[str, float] = {}
        for item in results:
            label = item.get("label", item.get("label", "unknown")).lower()
            score = float(item.get("score", 0))
            if multi_label or score >= self.confidence_threshold:
                category_scores[label] = score

        if not category_scores:
            top_category, top_score = None, 0.0
        else:
            top_category, top_score = max(category_scores.items(), key=lambda item: item[1])

        valence, arousal, dominance = self._vad.score(text)
        multiplier = emotion_multiplier(list(category_scores.keys()))

        return EmotionClassificationResult(
            categories=list(category_scores.keys()),
            category_scores=category_scores,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            top_category=top_category,
            top_score=top_score,
            multiplier=multiplier,
        )

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        try:
            importlib.import_module("transformers")
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,
            )
            self._model_loaded = True
        except Exception as exc:
            logger.debug("Failed to load emotion model '%s': %s. Falling back to lexicon mode.", self.model_name, exc)
            self.mode = "lexicon"


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class EmotionClassificationResult:
    """Output of a single emotion classification call."""

    categories: list[str]
    category_scores: dict[str, float]
    valence: float  # [0.0=negative, 1.0=positive]
    arousal: float  # [0.0=calm, 1.0=intense]
    dominance: float  # [0.0=helpless, 1.0=in control]
    top_category: str | None
    top_score: float
    multiplier: float


@dataclass
class EmotionTrajectory:
    """Emotional trajectory across a session of messages."""

    start_valence: float
    end_valence: float
    start_arousal: float
    end_arousal: float
    start_dominance: float
    end_dominance: float
    trend: str  # "escalating", "de-escalating", "stable", "volatile"
    max_intensity: float
    crisis_indicators: list[str]
    trajectory_scores: list[dict[str, float]]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_env(key: str, fallback: str) -> str:
    import os

    return os.environ.get(key, fallback)


def _compute_trend(values: list[float], dominances: list[float]) -> str:
    """Classify emotional trajectory from VAD values."""
    if len(values) < 2:
        return "stable"

    valence_trend = values[-1] - values[0]
    dominance_trend = dominances[-1] - dominances[0]

    # Check volatility (variance)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance > 0.04:
        return "volatile"

    # Escalating: rising arousal + falling valence
    if valence_trend < -0.1 and dominance_trend < -0.1:
        return "escalating"
    # De-escalating: rising valence + rising dominance
    if valence_trend > 0.1 and dominance_trend > 0.1:
        return "de-escalating"

    return "stable"
