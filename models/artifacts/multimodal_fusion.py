"""
Multimodal Fusion for Pixel Therapeutic Conversations.

Combines text and audio signals for enhanced emotional understanding.
Synchronizes responses across modalities (text, speech, emotion indicators).

Features:
  - Text + audio emotion fusion with weighted combination
  - Conflict detection between modalities
  - Cross-modal validation and confidence scoring
  - Synchronized multimodal response generation
  - Real-time fusion with streaming audio
  - Integration with Pixel EQ scores

Example:
    >>> fusion = MultimodalFusion()
    >>> result = fusion.fuse_emotions(
    ...     text_emotion={'eq_scores': [0.8, 0.7, 0.6, 0.9, 0.75]},
    ...     audio_emotion={'valence': 0.6, 'arousal': 0.7}
    ... )
    >>> print(f"Fused EQ: {result['fused_eq_scores']}")
"""

import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModalityWeights:
    """Weights for different modalities in fusion."""

    text_weight: float = 0.6  # Text emotion importance
    audio_weight: float = 0.4  # Audio emotion importance
    fusion_confidence_threshold: float = 0.7


@dataclass
class FusedEmotionalState:
    """Fused emotional representation."""

    # Pixel EQ scores (5 dimensions)
    eq_scores: list[float]
    overall_eq: float

    # VAD representation
    valence: float
    arousal: float
    dominance: float

    # Metadata
    text_contribution: float  # How much text influenced fusion
    audio_contribution: float  # How much audio influenced fusion
    conflict_score: float  # 0.0 (aligned) to 1.0 (conflicting)
    confidence: float  # Overall confidence in fusion

    # Detailed breakdown
    text_emotion: dict[str, Any] | None = None
    audio_emotion: dict[str, Any] | None = None


class MultimodalFusion:
    """Fuse text and audio emotions for therapeutic context."""

    def __init__(
        self,
        text_weight: float = 0.6,
        audio_weight: float = 0.4,
        conflict_threshold: float = 0.5,
    ):
        """
        Initialize multimodal fusion.

        Args:
            text_weight: Weight for text emotion (0.0-1.0)
            audio_weight: Weight for audio emotion
            conflict_threshold: Threshold for modality conflict
        """
        self.text_weight = text_weight / (text_weight + audio_weight)
        self.audio_weight = audio_weight / (text_weight + audio_weight)
        self.conflict_threshold = conflict_threshold

    def fuse_emotions(
        self,
        text_emotion: dict[str, Any] | None = None,
        audio_emotion: dict[str, Any] | None = None,
        weights: ModalityWeights | None = None,
    ) -> FusedEmotionalState:
        """
        Fuse text and audio emotions.

        Args:
            text_emotion: Text emotion from Pixel model
            audio_emotion: Audio emotion from speech analysis
            weights: Custom weighting configuration

        Returns:
            FusedEmotionalState with combined representation
        """
        if weights:
            self.text_weight = weights.text_weight / (weights.text_weight + weights.audio_weight)
            self.audio_weight = weights.audio_weight / (weights.text_weight + weights.audio_weight)

        # Default to neutral if missing
        if text_emotion is None:
            text_emotion = self._default_text_emotion()
        if audio_emotion is None:
            audio_emotion = self._default_audio_emotion()

        # Extract EQ scores from text
        text_eq = text_emotion.get("eq_scores", [0.5] * 5)
        text_emotion.get("overall_eq", 0.5)

        # Convert audio VAD to EQ scores
        audio_eq = self._vad_to_eq(
            valence=audio_emotion.get("valence", 0.0),
            arousal=audio_emotion.get("arousal", 0.0),
            dominance=audio_emotion.get("dominance", 0.0),
        )
        np.mean(audio_eq)

        # Fuse EQ scores
        fused_eq = self._fuse_eq_scores(
            text_eq,
            audio_eq,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
        )

        # Calculate VAD from fusion
        valence, arousal, dominance = self._eq_to_vad(fused_eq)

        # Calculate conflict
        conflict = self._calculate_conflict(text_emotion, audio_emotion)

        # Calculate confidence
        text_conf = text_emotion.get("confidence", 0.8)
        audio_conf = audio_emotion.get("confidence", 0.8)
        fused_confidence = text_conf * self.text_weight + audio_conf * self.audio_weight

        return FusedEmotionalState(
            eq_scores=fused_eq,
            overall_eq=float(np.mean(fused_eq)),
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            text_contribution=self.text_weight,
            audio_contribution=self.audio_weight,
            conflict_score=conflict,
            confidence=fused_confidence,
            text_emotion=text_emotion,
            audio_emotion=audio_emotion,
        )

    def _fuse_eq_scores(
        self,
        text_eq: list[float],
        audio_eq: list[float],
        text_weight: float = 0.6,
        audio_weight: float = 0.4,
    ) -> list[float]:
        """
        Fuse EQ scores from text and audio.

        EQ Dimensions:
        0: Self-awareness
        1: Self-regulation
        2: Motivation
        3: Empathy
        4: Social skills
        """

        text_eq = np.array(text_eq)

        audio_eq = np.array(audio_eq)

        # Weighted fusion
        fused = text_eq * text_weight + audio_eq * audio_weight

        return fused.tolist()

    def _vad_to_eq(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> list[float]:
        """
        Convert VAD (Valence-Arousal-Dominance) to EQ scores.

        Mapping:
        - Self-awareness: Related to dominance (how aware of own state)
        - Self-regulation: Inverse of arousal (calm = better regulation)
        - Motivation: Related to arousal (energy = drive)
        - Empathy: Related to valence (positive = more empathetic)
        - Social skills: Combination of valence and dominance
        """
        # Normalize from [-1, 1] to [0, 1]
        v = (valence + 1) / 2
        a = (arousal + 1) / 2
        d = (dominance + 1) / 2

        eq_scores = [
            d,  # Self-awareness: dominance
            1.0 - a,  # Self-regulation: inverse arousal
            a,  # Motivation: arousal/energy
            v,  # Empathy: valence/positivity
            (v + d) / 2,  # Social skills: combined
        ]

        return [float(np.clip(e, 0.0, 1.0)) for e in eq_scores]

    def _eq_to_vad(self, eq_scores: list[float]) -> tuple[float, float, float]:
        """
        Convert EQ scores back to VAD.

        Inverse mapping of _vad_to_eq.
        """
        eq = np.array(eq_scores)

        # Extract VAD from EQ
        d = eq[0]  # dominance from self-awareness
        a = 1.0 - eq[1]  # arousal from inverse self-regulation
        v = (eq[3] + eq[4]) / 2  # valence from empathy and social skills

        # Normalize to [-1, 1]
        valence = v * 2 - 1
        arousal = a * 2 - 1
        dominance = d * 2 - 1

        return float(valence), float(arousal), float(dominance)

    def _calculate_conflict(
        self,
        text_emotion: dict[str, Any],
        audio_emotion: dict[str, Any],
    ) -> float:
        """
        Calculate conflict between modalities.

        Returns:
            Conflict score (0.0 = aligned, 1.0 = conflicting)
        """
        # Extract comparable metrics
        text_eq = np.array(text_emotion.get("eq_scores", [0.5] * 5))

        # Convert audio to EQ
        audio_vad = {
            "valence": audio_emotion.get("valence", 0.0),
            "arousal": audio_emotion.get("arousal", 0.0),
            "dominance": audio_emotion.get("dominance", 0.0),
        }
        audio_eq = np.array(self._vad_to_eq(**audio_vad))

        # Calculate difference
        diff = np.abs(text_eq - audio_eq)
        conflict = float(np.mean(diff))

        return np.clip(conflict, 0.0, 1.0)

    def detect_modality_conflict(
        self,
        fused_state: FusedEmotionalState,
        threshold: float | None = None,
    ) -> bool:
        """
        Detect if modalities are in conflict.

        Args:
            fused_state: Fused emotional state
            threshold: Conflict threshold (uses default if None)

        Returns:
            True if conflict exceeds threshold
        """
        threshold = threshold or self.conflict_threshold
        return fused_state.conflict_score > threshold

    def validate_fusion(
        self,
        fused_state: FusedEmotionalState,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Validate fusion quality.

        Returns:
            True if fusion meets quality thresholds
        """
        # Check confidence
        if fused_state.confidence < confidence_threshold:
            return False

        # Check for extreme conflict
        if fused_state.conflict_score > 0.8:
            logger.warning(f"High modality conflict detected: {fused_state.conflict_score:.2f}")
            return False

        return True

    def _default_text_emotion(self) -> dict[str, Any]:
        """Default neutral text emotion."""
        return {
            "eq_scores": [0.5, 0.5, 0.5, 0.5, 0.5],
            "overall_eq": 0.5,
            "confidence": 0.0,
        }

    def _default_audio_emotion(self) -> dict[str, Any]:
        """Default neutral audio emotion."""
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "confidence": 0.0,
        }


class TextToSpeechGenerator:
    """Generate speech from text with emotional prosody.

    Uses a Hugging Face ``text-to-audio`` pipeline (default
    ``facebook/mms-tts-eng``) on the local device. Override the model via
    the ``TTS_MODEL`` env var; set ``TTS_OUTPUT_DIR`` to write generated
    audio to disk (16-bit PCM WAV).
    """

    def __init__(self, device: str | None = None):
        """Initialize TTS generator."""
        self.device = device or os.environ.get("TTS_DEVICE", "cpu")
        self.model_name = os.environ.get("TTS_MODEL", "facebook/mms-tts-eng")
        self.output_dir = os.environ.get("TTS_OUTPUT_DIR")
        self.model: Any = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazily load the text-to-audio pipeline."""
        if self._model_loaded:
            return
        try:
            from transformers import pipeline  # noqa: PLC0415 - heavy SDK, loaded on first synthesis

            self.model = pipeline(
                "text-to-audio",
                model=self.model_name,
                device=self.device,
            )
            self._model_loaded = True
        except Exception as exc:
            logger.warning("Failed to load TTS model '%s': %s", self.model_name, exc)
            self.model = None

    async def synthesize(
        self,
        text: str,
        session_id: str,
        emotional_state: dict[str, float] | None = None,
        speaker_id: int = 0,
    ) -> dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak
            session_id: Session ID
            emotional_state: Optional VAD hint recorded as prosody metadata
                (MMS-TTS is single-speaker; emotional prosody would need a
                multi-speaker model — accepted here for interface stability
                and surfaced in the response metadata)
            speaker_id: Speaker ID for multi-speaker TTS

        Returns:
            Dict with audio (PCM float array), sampling_rate, duration,
            optional audio_path, and metadata. On model-load failure the
            dict carries ``status: "unavailable"`` plus the reason, so
            callers can degrade to text-only.
        """
        try:
            if not text:
                return {"error": "Empty text"}

            self._load_model()
            if self.model is None:
                return {
                    "status": "unavailable",
                    "session_id": session_id,
                    "text": text,
                    "reason": f"TTS model '{self.model_name}' failed to load",
                }

            generation = self.model(text)
            audio = np.asarray(generation["audio"], dtype=np.float32).reshape(-1)
            sampling_rate = int(generation.get("sampling_rate", 16000))
            audio_duration_s = audio.shape[0] / sampling_rate

            result: dict[str, Any] = {
                "status": "success",
                "session_id": session_id,
                "text": text,
                "speaker_id": speaker_id,
                "audio": audio,
                "sampling_rate": sampling_rate,
                "audio_duration_s": audio_duration_s,
                "model": self.model_name,
                "emotional_prosody": emotional_state or {},
            }

            audio_path = self._write_wav(session_id, audio, sampling_rate)
            if audio_path:
                result["audio_path"] = str(audio_path)

            return result

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e!s}")
            return {"error": str(e)}

    def _write_wav(
        self, session_id: str, audio: np.ndarray, sampling_rate: int
    ) -> Path | None:
        """Persist audio as a 16-bit PCM WAV when TTS_OUTPUT_DIR is set."""
        if not self.output_dir:
            return None
        try:
            output_root = Path(self.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            audio_path = output_root / f"tts-{session_id}.wav"
            pcm = np.clip(audio, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16)
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sampling_rate)
                wav_file.writeframes(pcm.tobytes())
            return audio_path
        except Exception as exc:
            logger.warning("Failed to persist TTS audio: %s", exc)
            return None


class MultimodalResponseGenerator:
    """Generate synchronized multimodal responses."""

    def __init__(self):
        """Initialize response generator."""
        self.fusion = MultimodalFusion()
        self.tts = TextToSpeechGenerator()

    async def generate_multimodal_response(
        self,
        text_response: str,
        fused_emotion: FusedEmotionalState,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Generate multimodal response with text + speech.

        Args:
            text_response: Text response from model
            fused_emotion: Fused emotional state for prosody
            session_id: Session ID

        Returns:
            Dict with text, audio, and metadata
        """
        try:
            # Generate text response (already available)
            result = {
                "text": text_response,
                "emotional_state": {
                    "eq_scores": fused_emotion.eq_scores,
                    "valence": fused_emotion.valence,
                    "arousal": fused_emotion.arousal,
                },
                "modality": "multimodal",
            }

            # Generate speech with emotion
            tts_result = await self.tts.synthesize(
                text=text_response,
                session_id=session_id,
                emotional_state={
                    "valence": fused_emotion.valence,
                    "arousal": fused_emotion.arousal,
                    "dominance": fused_emotion.dominance,
                },
            )

            result["speech"] = tts_result

            return result

        except Exception as e:
            logger.error(f"Multimodal response generation failed: {e!s}")
            return {
                "text": text_response,
                "error": str(e),
                "modality": "text_only",
            }
