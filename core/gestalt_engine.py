"""
PIX-149: Gestalt Fusion Engine

Unifies the three core emotional intelligence models of Pixelated Empathy
into a single, real-time inference call:

  1. PsyDefDetect  — DMRS Defense Mechanism Classifier (DeBERTa)
  2. Plutchik      — 8-emotion wheel scoring (passed in externally)
  3. OCEAN         — Big Five personality trait scoring (passed in externally)

The fused ``GestaltState`` dataclass powers:
  - PIX-147: WebSocket "Live X-Ray" resistance monitor
  - PIX-148: Adversarial Persona Injection (dynamic defense-aware prompts)
  - PIX-150: Empathy PQ metric validation (trainee scoring)
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from transformers import AutoTokenizer

from ai.training.defense_mechanisms.constants import DEFENSE_LABELS
from ai.training.defense_mechanisms.dataset import format_dialogue
from ai.training.defense_mechanisms.model import DefenseClassifier, DefensePrediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------

PLUTCHIK_EMOTIONS = frozenset(
    {
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    }
)

OCEAN_TRAITS = frozenset(
    {
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    }
)


class CrisisLevel(str, Enum):
    """
    Behavioral risk level derived from the fused Gestalt state.
    """

    NONE = "none"
    ELEVATED = "elevated"
    HIGH = "high"
    ACUTE = "acute"


# ---------------------------------------------------------------------------
# Fused output dataclass
# ---------------------------------------------------------------------------


@dataclass
class GestaltState:
    """
    Unified emotional-psychological state for a single dialogue turn.
    """

    # --- Defense Mechanism (PsyDefDetect) ---
    defense_label: int
    defense_label_name: str
    defense_confidence: float
    defense_maturity: Optional[float]
    defense_probabilities: dict[str, float]

    # --- Emotion (Plutchik) ---
    plutchik_scores: dict[str, float]
    dominant_emotion: str
    dominant_emotion_intensity: float

    # --- Personality (OCEAN) ---
    ocean_scores: dict[str, float]

    # --- Fused behavioral outputs ---
    crisis_level: CrisisLevel
    behavioral_prediction: str
    persona_directive: str
    breakthrough_score: float
    behavioral_pattern: str
    behavioral_pattern_confidence: float

    raw_metadata: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Crisis & behavioral prediction logic
# ---------------------------------------------------------------------------

_ACTION_DEFENSE_LABEL = 1
_HIGH_ADAPTIVE_LABEL = 7
_DISAVOWAL_LABEL = 3
_MAJOR_IMAGE_DISTORTING_LABEL = 2

_CRISIS_AMPLIFYING_EMOTIONS = frozenset({"sadness", "fear", "anger", "disgust"})
_INJECTION_MATURITY_THRESHOLD = 0.43
_BREAKTHROUGH_MATURITY_THRESHOLD = 0.71

_PLUTCHIK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anger": ("angry", "mad", "frustrated", "rage", "furious"),
    "anticipation": ("hope", "expect", "anticipate", "plan", "goal"),
    "disgust": ("disgust", "gross", "hate", "repulsed"),
    "fear": ("afraid", "scared", "panic", "anxious", "worry", "threat"),
    "joy": ("happy", "relieved", "content", "glad", "excited"),
    "sadness": ("sad", "hurt", "grief", "depressed", "empty", "overwhelmed"),
    "surprise": ("surprised", "unexpected", "shocked", "wow"),
    "trust": ("understand", "trust", "safe", "heard", "believe"),
}

_OCEAN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "openness": ("curious", "explore", "imagine", "creative", "different"),
    "conscientiousness": ("organized", "plan", "structure", "follow", "through"),
    "extraversion": ("talk", "share", "group", "connect", "social"),
    "agreeableness": ("kind", "calm", "appreciate", "thank", "supportive"),
    "neuroticism": ("worried", "anxious", "confused", "guilt", "shaky", "nervous"),
}


def _dominant_emotion(plutchik: dict[str, float]) -> tuple[str, float]:
    """Return the emotion with the highest intensity score."""
    if not plutchik:
        return "unknown", 0.0
    dominant = max(plutchik, key=lambda k: plutchik[k])
    return dominant, plutchik[dominant]


def _compute_crisis_level(
    defense_label: int,
    defense_maturity: Optional[float],
    dominant_emotion: str,
    dominant_intensity: float,
    ocean_neuroticism: float,
) -> CrisisLevel:
    """Determine crisis level from the fused signals."""
    is_action = defense_label == _ACTION_DEFENSE_LABEL
    is_major_distorting = defense_label == _MAJOR_IMAGE_DISTORTING_LABEL
    emotion_is_crisis = dominant_emotion in _CRISIS_AMPLIFYING_EMOTIONS

    if is_action:
        return (
            CrisisLevel.ACUTE
            if emotion_is_crisis
            and dominant_intensity > 0.6
            or ocean_neuroticism > 0.75
            else CrisisLevel.HIGH
        )

    if is_major_distorting and emotion_is_crisis:
        return CrisisLevel.HIGH

    maturity = defense_maturity if defense_maturity is not None else 0.5
    if maturity < 0.3 and emotion_is_crisis:
        return CrisisLevel.ELEVATED

    return CrisisLevel.NONE


def _behavioral_prediction(
    defense_label_name: str,
    dominant_emotion: str,
    crisis_level: CrisisLevel,
    defense_maturity: Optional[float],
) -> str:
    """Produce a concise human-readable behavioral prediction string."""
    maturity_str = (
        f"maturity={defense_maturity:.2f}" if defense_maturity is not None else "N/A"
    )
    base = f"{dominant_emotion.capitalize()} + {defense_label_name} ({maturity_str})"

    if crisis_level == CrisisLevel.ACUTE:
        return f"{base} → acute distress risk. Therapist must de-escalate immediately."
    if crisis_level == CrisisLevel.HIGH:
        return (
            f"{base} → high-risk defensive pattern. "
            "Therapist should slow pace and validate."
        )
    if crisis_level == CrisisLevel.ELEVATED:
        return (
            f"{base} → elevated defensiveness. "
            "Therapist should use reflective listening."
        )

    if (
        defense_maturity is not None
        and defense_maturity >= _BREAKTHROUGH_MATURITY_THRESHOLD
    ):
        return (
            f"{base} → adaptive coping detected. "
            "Therapist can introduced deeper reframing."
        )
    return f"{base} → neutral or intermediate defense. Continue with exploration."


def _persona_directive(
    defense_label: int,
    defense_label_name: str,
    defense_maturity: Optional[float],
) -> str:
    """Return a system-prompt injection clause for PIX-148."""
    if defense_maturity is None or defense_maturity >= _INJECTION_MATURITY_THRESHOLD:
        return ""

    directives: dict[int, str] = {
        _ACTION_DEFENSE_LABEL: (
            "[System: You are in an Action Defense state. "
            "Act out, change subject. Do NOT acknowledge emotions directly.]"
        ),
        _MAJOR_IMAGE_DISTORTING_LABEL: (
            "[System: You are in a Major Image-Distorting state. "
            "Split people into all-good or all-bad.]"
        ),
        _DISAVOWAL_LABEL: (
            "[System: You are in a Disavowal state. Deny emotional reality.]"
        ),
    }
    return directives.get(
        defense_label,
        f"[System: Maintain your '{defense_label_name}' defense mechanism.]",
    )


def _breakthrough_score(
    defense_maturity: Optional[float],
    previous_maturity: Optional[float],
) -> float:
    """Score the magnitude of a positive maturity shift."""
    if defense_maturity is None or previous_maturity is None:
        return 0.0
    delta = defense_maturity - previous_maturity
    return max(0.0, min(1.0, delta))


def _validate_scores(
    scores: dict[str, float],
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, float]:
    """Validate normalized scores."""
    for key, value in scores.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{label}['{key}'] must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label}['{key}'] score {value} out of range [0, 1]")
    if unknown := set(scores) - expected_keys:
        logger.warning(f"{label} contains unknown keys {sorted(unknown)}")

    return {k: float(v) for k, v in scores.items()}


def _normalize_score(value: float) -> float:
    """Clamp model outputs to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _infer_plutchik_signature(text: str) -> dict[str, float]:
    """Fallback Plutchik inference from lexical cues."""
    lowered = text.lower()
    raw = {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

    for emotion, keywords in _PLUTCHIK_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        raw[emotion] = 0.2 + 0.15 * min(hits, 4)

    if max(raw.values()) <= 0.2:
        raw["trust"] = 0.34
        raw["anticipation"] = 0.3
        raw["sadness"] = 0.2

    return {k: _normalize_score(v) for k, v in raw.items()}


def _infer_ocean_signature(text: str) -> dict[str, float]:
    """Fallback OCEAN inference from lexical cues."""
    lowered = text.lower()
    scores = {trait: 0.5 for trait in OCEAN_TRAITS}

    for trait, keywords in _OCEAN_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits:
            scores[trait] = _normalize_score(0.5 + 0.08 * hits)

    if "open" in lowered:
        scores["openness"] = _normalize_score(scores["openness"] + 0.1)
    if "avoid" in lowered:
        scores["neuroticism"] = _normalize_score(scores["neuroticism"] + 0.1)
        scores["conscientiousness"] = _normalize_score(max(0.1, scores["conscientiousness"] - 0.1))

    return scores


# ---------------------------------------------------------------------------
# GestaltEngine
# ---------------------------------------------------------------------------


class GestaltEngine:
    """
    Real-time Gestalt Fusion Engine for the Empathy Gym™.
    """

    def __init__(self) -> None:
        self._defense_model = None
        self._defense_tokenizer = None
        self._previous_maturity: Optional[float] = None
        self._analysis_executor = ThreadPoolExecutor(max_workers=12)

    def load_defense_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Load the PsyDefDetect model (optional if using NIM)."""
        if not checkpoint_path:
            logger.info("GestaltEngine: No checkpoint provided, using NIM by default.")
            self._defense_model = DefenseClassifier()
            return

        try:
            self._initialize_local_model(checkpoint_path, device)
        except Exception as exc:
            logger.error(
                f"GestaltEngine: Failed to load checkpoint {checkpoint_path}: {exc}"
            )
            logger.info("GestaltEngine: Falling back to NIM-based DefenseClassifier.")
            self._defense_model = DefenseClassifier()

    def _initialize_local_model(self, checkpoint_path: str, device: str) -> None:
        """Initialize the legacy DeBERTa model from a local checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = checkpoint.get("config", {})
        model_name = config.get("base_model", "microsoft/deberta-v3-base")

        model = DefenseClassifier(
            model_name=model_name,
            num_labels=config.get("num_labels", 9),
            r_drop_enabled=False,
        )
        # Handle PyTorch-based DefenseClassifier if checkpoint is provided
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(device)
            model.eval()

        self._defense_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._defense_model = model
        logger.info(f"GestaltEngine: model loaded from {checkpoint_path}")

    @property
    def defense_model_loaded(self) -> bool:
        return self._defense_model is not None

    def reset_session(self) -> None:
        self._previous_maturity = None

    def _classify_defense(
        self,
        dialogue: list[dict[str, str]],
        target_utterance: str,
        max_turns: int = 40,
    ) -> tuple[int, str, float, Optional[float], dict[str, float]]:
        """Run PsyDefDetect inference."""
        if self._defense_model is None:
            raise RuntimeError(
                "GestaltEngine: Defense model not initialized. "
                "Call load_defense_model() first."
            )

        turns = [
            {"speaker": t.get("speaker", "Unknown"), "text": t.get("text", "")}
            for t in dialogue[-max_turns:]
        ]

        # Ensure the target utterance is in the sequence for format_dialogue to mark it
        target_normalized = target_utterance.strip().lower()
        if all(t.get("text", "").strip().lower() != target_normalized for t in turns):
            turns.append({"speaker": "User", "text": target_utterance})

        formatted = format_dialogue(turns, target_utterance, max_turns)

        # Handle the new text-based NIM classifier
        if hasattr(self._defense_model, "nim"):
            pred = self._defense_model.predict([formatted])[0]
        else:
            pred = self._legacy_inference(formatted)

        prob_dict = {
            DEFENSE_LABELS.get(i, str(i)): round(p, 4)
            for i, p in enumerate(pred.probabilities)
        }
        return (
            pred.label,
            pred.label_name,
            pred.confidence,
            pred.maturity_score,
            prob_dict,
        )

    def _classify_plutchik(
        self,
        dialogue: list[dict[str, str]],
        target_utterance: str,
        max_turns: int = 40,
    ) -> dict[str, float]:
        """Fallback Plutchik inference path."""
        _ = dialogue[-max_turns:]
        source = f"{target_utterance} " + " ".join(
            turn.get("text", "") for turn in dialogue[-max_turns:]
        )
        return _infer_plutchik_signature(source)

    def _classify_ocean(
        self,
        dialogue: list[dict[str, str]],
        target_utterance: str,
        max_turns: int = 40,
    ) -> dict[str, float]:
        """Fallback OCEAN inference path."""
        _ = dialogue[-max_turns:]
        source = f"{target_utterance} " + " ".join(
            turn.get("text", "") for turn in dialogue[-max_turns:]
        )
        return _infer_ocean_signature(source)

    def _behavioral_pattern_label(
        self,
        defense_label_name: str,
        defense_label: int,
        dominant_emotion: str,
        defense_maturity: Optional[float],
        dominant_intensity: float,
        crisis_level: CrisisLevel,
    ) -> tuple[str, float]:
        """Map fused signals into a lightweight behavioral pattern descriptor."""
        if defense_label == _ACTION_DEFENSE_LABEL and dominant_emotion == "sadness":
            return "high-risk behavior", 0.95
        if defense_label == _HIGH_ADAPTIVE_LABEL and dominant_emotion == "sadness":
            return "grieving but coping", 0.88
        if defense_label == _MAJOR_IMAGE_DISTORTING_LABEL and dominant_intensity >= 0.6:
            return "splitting / all-or-nothing processing", 0.8
        if defense_label == _DISAVOWAL_LABEL and dominant_emotion in _CRISIS_AMPLIFYING_EMOTIONS:
            return "defensive minimization", 0.76
        if crisis_level == CrisisLevel.HIGH:
            return "elevated defensive dysregulation", 0.7
        if crisis_level == CrisisLevel.ELEVATED:
            return "cautionary emotional reactivity", 0.64
        if defense_maturity is not None and defense_maturity >= _BREAKTHROUGH_MATURITY_THRESHOLD:
            return "adaptive integration", 0.72
        if defense_label_name:
            return f"{defense_label_name} with mixed affect", 0.6
        return "neutral mixed processing", 0.5

    def _legacy_inference(self, formatted_text: str) -> DefensePrediction:
        """Run inference using the local PyTorch DeBERTa model."""
        if self._defense_tokenizer is None:
            raise RuntimeError("GestaltEngine: No tokenizer for PyTorch model.")

        encoding = self._defense_tokenizer(
            formatted_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = next(self._defense_model.parameters()).device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        return self._defense_model.predict(input_ids, attention_mask)[0]

    def analyze_gestalt(
        self,
        dialogue: list[dict[str, str]],
        target_utterance: str,
        plutchik_scores: Optional[dict[str, float]] = None,
        ocean_scores: Optional[dict[str, float]] = None,
        max_turns: int = 40,
    ) -> GestaltState:
        """Fuse signals into a GestaltState."""
        plutchik_future: Optional[Future[dict[str, float]]] = None
        ocean_future: Optional[Future[dict[str, float]]] = None
        if plutchik_scores is None:
            plutchik_future = self._analysis_executor.submit(
                self._classify_plutchik, dialogue, target_utterance, max_turns
            )
        if ocean_scores is None:
            ocean_future = self._analysis_executor.submit(
                self._classify_ocean, dialogue, target_utterance, max_turns
            )
        defense_future = self._analysis_executor.submit(
            self._classify_defense, dialogue, target_utterance, max_turns
        )

        if plutchik_scores is None:
            assert plutchik_future is not None
            plutchik_scores = plutchik_future.result()
        if ocean_scores is None:
            assert ocean_future is not None
            ocean_scores = ocean_future.result()
        def_l, def_n, def_c, def_m, def_p = defense_future.result()

        plutchik = _validate_scores(
            plutchik_scores, PLUTCHIK_EMOTIONS, "plutchik_scores"
        )
        ocean = _validate_scores(ocean_scores, OCEAN_TRAITS, "ocean_scores")

        full_plutchik = {e: plutchik.get(e, 0.0) for e in PLUTCHIK_EMOTIONS}
        full_ocean = {t: ocean.get(t, 0.5) for t in OCEAN_TRAITS}

        dom_e, dom_i = _dominant_emotion(full_plutchik)
        neuro = full_ocean.get("neuroticism", 0.5)

        crisis = _compute_crisis_level(def_l, def_m, dom_e, dom_i, neuro)
        behavioral = _behavioral_prediction(def_n, dom_e, crisis, def_m)
        pattern, pattern_confidence = self._behavioral_pattern_label(
            def_n,
            def_l,
            dom_e,
            def_m,
            dom_i,
            crisis,
        )
        directive = _persona_directive(def_l, def_n, def_m)
        breakthrough = _breakthrough_score(def_m, self._previous_maturity)

        if def_m is not None:
            self._previous_maturity = def_m

        return GestaltState(
            defense_label=def_l,
            defense_label_name=def_n,
            defense_confidence=def_c,
            defense_maturity=def_m,
            defense_probabilities=def_p,
            plutchik_scores=full_plutchik,
            dominant_emotion=dom_e,
            dominant_emotion_intensity=dom_i,
            ocean_scores=full_ocean,
            crisis_level=crisis,
            behavioral_prediction=behavioral,
            persona_directive=directive,
            breakthrough_score=breakthrough,
            behavioral_pattern=pattern,
            behavioral_pattern_confidence=pattern_confidence,
        )
