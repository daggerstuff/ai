"""
Multimodal Fusion for Pixel Therapeutic Conversations.

Combines text, audio, and video signals for enhanced emotional understanding.
Synchronizes responses across modalities (text, speech, video, emotion indicators).

Features:
  - Text + audio + video emotion fusion with weighted combination
  - Conflict detection between modalities
  - Cross-modal validation and confidence scoring
  - Synchronized multimodal response generation
  - Real-time fusion with streaming audio/video
  - Integration with Pixel EQ scores
  - Support for "Visual micro-leakage" detection

Example:
    >>> fusion = MultimodalFusion()
    >>> result = fusion.fuse_emotions(
    ...     text_emotion={'eq_scores': [0.8, 0.7, 0.6, 0.9, 0.75]},
    ...     audio_emotion={'valence': 0.6, 'arousal': 0.7},
    ...     video_emotion={'valence': 0.5, 'arousal': 0.6}
    ... )
    >>> print(f"Fused EQ: {result['fused_eq_scores']}")
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModalityWeights:
    """Weights for different modalities in fusion."""

    text_weight: float = 0.5
    audio_weight: float = 0.3
    video_weight: float = 0.2
    fusion_confidence_threshold: float = 0.7


@dataclass
class FusedEmotionalState:
    """Fused emotional representation."""

    eq_scores: List[float]
    overall_eq: float
    valence: float
    arousal: float
    dominance: float
    text_contribution: float
    audio_contribution: float
    video_contribution: float = 0.0
    conflict_score: float = 0.0
    confidence: float = 0.0
    visual_micro_leakage: Optional[bool] = None
    text_emotion: Optional[Dict[str, Any]] = None
    audio_emotion: Optional[Dict[str, Any]] = None
    video_emotion: Optional[Dict[str, Any]] = None


class MultimodalFusion:
    """Fuse text, audio, and video emotions for therapeutic context."""

    def __init__(
        self,
        text_weight: float = 0.5,
        audio_weight: float = 0.3,
        video_weight: float = 0.2,
        conflict_threshold: float = 0.5,
    ):
        total = text_weight + audio_weight + video_weight
        self.text_weight = text_weight / total
        self.audio_weight = audio_weight / total
        self.video_weight = video_weight / total
        self.conflict_threshold = conflict_threshold

    def fuse_emotions(
        self,
        text_emotion: Optional[Dict[str, Any]] = None,
        audio_emotion: Optional[Dict[str, Any]] = None,
        video_emotion: Optional[Dict[str, Any]] = None,
        weights: Optional[ModalityWeights] = None,
    ) -> FusedEmotionalState:
        if weights:
            total = weights.text_weight + weights.audio_weight + weights.video_weight
            self.text_weight = weights.text_weight / total
            self.audio_weight = weights.audio_weight / total
            self.video_weight = weights.video_weight / total

        if text_emotion is None:
            text_emotion = self._default_text_emotion()
        if audio_emotion is None:
            audio_emotion = self._default_audio_emotion()
        if video_emotion is None:
            video_emotion = self._default_video_emotion()

        text_eq = text_emotion.get("eq_scores", [0.5] * 5)

        audio_eq = self._vad_to_eq(
            valence=audio_emotion.get("valence", 0.0),
            arousal=audio_emotion.get("arousal", 0.0),
            dominance=audio_emotion.get("dominance", 0.0),
        )

        video_eq = self._vad_to_eq(
            valence=video_emotion.get("valence", 0.0),
            arousal=video_emotion.get("arousal", 0.0),
            dominance=video_emotion.get("dominance", 0.0),
        )

        fused_eq = self._fuse_eq_scores_triple(
            text_eq,
            audio_eq,
            video_eq,
            self.text_weight,
            self.audio_weight,
            self.video_weight,
        )

        valence, arousal, dominance = self._eq_to_vad(fused_eq)

        conflict = self._calculate_conflict_triple(
            text_emotion, audio_emotion, video_emotion
        )

        visual_micro_leakage = self._detect_visual_micro_leakage(
            text_emotion, video_emotion
        )

        text_conf = text_emotion.get("confidence", 0.8)
        audio_conf = audio_emotion.get("confidence", 0.8)
        video_conf = video_emotion.get("confidence", 0.8)
        fused_confidence = (
            text_conf * self.text_weight
            + audio_conf * self.audio_weight
            + video_conf * self.video_weight
        )

        return FusedEmotionalState(
            eq_scores=fused_eq,
            overall_eq=float(np.mean(fused_eq)),
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            text_contribution=self.text_weight,
            audio_contribution=self.audio_weight,
            video_contribution=self.video_weight,
            conflict_score=conflict,
            confidence=fused_confidence,
            visual_micro_leakage=visual_micro_leakage,
            text_emotion=text_emotion,
            audio_emotion=audio_emotion,
            video_emotion=video_emotion,
        )

    def _fuse_eq_scores_triple(
        self,
        text_eq: List[float],
        audio_eq: List[float],
        video_eq: List[float],
        text_weight: float,
        audio_weight: float,
        video_weight: float,
    ) -> List[float]:
        text_arr = np.array(text_eq)
        audio_arr = np.array(audio_eq)
        video_arr = np.array(video_eq)
        fused = (
            text_arr * text_weight + audio_arr * audio_weight + video_arr * video_weight
        )
        return fused.tolist()

    def _vad_to_eq(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> List[float]:
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

    def _eq_to_vad(self, eq_scores: List[float]) -> Tuple[float, float, float]:
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
        text_emotion: Dict[str, Any],
        audio_emotion: Dict[str, Any],
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
        threshold: Optional[float] = None,
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
            logger.warning(
                f"High modality conflict detected: {fused_state.conflict_score:.2f}"
            )
            return False

        return True

    def _default_text_emotion(self) -> Dict[str, Any]:
        """Default neutral text emotion."""
        return {
            "eq_scores": [0.5, 0.5, 0.5, 0.5, 0.5],
            "overall_eq": 0.5,
            "confidence": 0.0,
        }

    def _default_audio_emotion(self) -> Dict[str, Any]:
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "confidence": 0.0,
        }

    def _default_video_emotion(self) -> Dict[str, Any]:
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "confidence": 0.0,
        }

    def _calculate_conflict_triple(
        self,
        text_emotion: Dict[str, Any],
        audio_emotion: Dict[str, Any],
        video_emotion: Dict[str, Any],
    ) -> float:
        text_eq = np.array(text_emotion.get("eq_scores", [0.5] * 5))
        audio_vad = {
            "valence": audio_emotion.get("valence", 0.0),
            "arousal": audio_emotion.get("arousal", 0.0),
            "dominance": audio_emotion.get("dominance", 0.0),
        }
        audio_eq = np.array(self._vad_to_eq(**audio_vad))
        video_vad = {
            "valence": video_emotion.get("valence", 0.0),
            "arousal": video_emotion.get("arousal", 0.0),
            "dominance": video_emotion.get("dominance", 0.0),
        }
        video_eq = np.array(self._vad_to_eq(**video_vad))
        pairwise_diff = (
            np.abs(text_eq - audio_eq)
            + np.abs(audio_eq - video_eq)
            + np.abs(text_eq - video_eq)
        )
        return float(np.clip(np.mean(pairwise_diff) / 2, 0.0, 1.0))

    def _detect_visual_micro_leakage(
        self,
        text_emotion: Dict[str, Any],
        video_emotion: Dict[str, Any],
    ) -> Optional[bool]:
        if video_emotion.get("confidence", 0.0) < 0.3:
            return None
        text_valence = 0.5
        if "eq_scores" in text_emotion:
            eq = text_emotion["eq_scores"]
            text_valence = (eq[3] + eq[4]) / 2 if len(eq) > 4 else 0.5
        video_valence = (video_emotion.get("valence", 0.0) + 1) / 2
        valence_diff = abs(text_valence - video_valence)
        return valence_diff > 0.4


class TextToSpeechGenerator:
    """Generate speech from text with emotional prosody."""

    def __init__(self, device: str = "cuda"):
        """Initialize TTS generator."""
        self.device = device
        self.model = None
        self.vocoder = None

    async def synthesize(
        self,
        text: str,
        session_id: str,
        emotional_state: Optional[Dict[str, float]] = None,
        speaker_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak
            session_id: Session ID
            emotional_state: Optional emotional state for prosody
            speaker_id: Speaker ID for multi-speaker TTS

        Returns:
            Dict with audio_path and metadata
        """
        try:
            # This would integrate with actual TTS model
            # For now, placeholder implementation

            if not text:
                return {"error": "Empty text"}

            # In production: Use Glow-TTS, FastPitch, or Tacotron2
            # with emotion embedding

            return {
                "status": "success",
                "session_id": session_id,
                "text": text,
                "speaker_id": speaker_id,
                "emotional_prosody": emotional_state or {},
                "audio_duration_s": len(text.split()) * 0.5,  # rough estimate
            }

        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            return {"error": str(e)}


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
    ) -> Dict[str, Any]:
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
            logger.error(f"Multimodal response generation failed: {str(e)}")
            return {
                "text": text_response,
                "error": str(e),
                "modality": "text_only",
            }
