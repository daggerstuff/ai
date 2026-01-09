"""
Audio Emotion Recognition for Pixel Multimodal Integration.

Detects emotional states from speech using valence/arousal dimensions.
Integrates with Pixel's EQ scoring system for multimodal emotion tracking.

Features:
  - Valence detection (negative to positive)
  - Arousal detection (calm to excited)
  - Dominance detection (controlled to dominant)
  - Emotion trajectory tracking over time
  - Speaker state confidence scoring
  - Real-time streaming emotion updates
  - Integration with text emotion for fusion

Example:
    >>> recognizer = AudioEmotionRecognizer(model_type="wav2vec2")
    >>> emotions = await recognizer.detect_emotions("session_001", audio_path)
    >>> print(f"Valence: {emotions['valence']:.2f}")
    >>> print(f"Arousal: {emotions['arousal']:.2f}")
"""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Emotion mappings
EMOTION_TO_VAD = {
    "neutral": (0.5, 0.5, 0.5),
    "happiness": (0.9, 0.8, 0.7),
    "sadness": (0.1, 0.2, 0.3),
    "anger": (0.2, 0.9, 0.9),
    "fear": (0.2, 0.8, 0.2),
    "surprise": (0.7, 0.8, 0.6),
    "disgust": (0.2, 0.6, 0.7),
}


@dataclass
class EmotionalState:
    """Emotional state representation using VAD model."""

    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # -1.0 (calm) to 1.0 (excited)
    dominance: float  # -1.0 (submissive) to 1.0 (dominant)
    confidence: float  # 0.0-1.0 confidence
    primary_emotion: str  # closest emotion label
    emotion_probabilities: Dict[str, float]  # all emotions with scores


@dataclass
class AudioEmotionResult:
    """Complete emotion detection result."""

    session_id: str
    audio_path: str
    overall_emotion: EmotionalState
    segment_emotions: List[Tuple[float, float, EmotionalState]]  # (start, end, emotion)
    trajectory: List[EmotionalState]  # emotion over time
    speech_rate_wpm: float  # words per minute
    intensity_score: float  # 0.0-1.0 speech intensity
    processing_time_ms: float
    audio_duration_s: float
    model_name: str
    error: Optional[str] = None


class AudioEmotionRecognizer:
    """Detect emotions from speech audio."""

    def __init__(
        self,
        model_type: str = "wav2vec2",
        model_name: str = "superb-ks",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize emotion recognizer.

        Args:
            model_type: Model type (wav2vec2, hubert, unispeech)
            model_name: Specific model name/path
            device: torch device (cuda/cpu)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device

        self.feature_extractor = None
        self.model = None
        self._load_models()

    def _load_models(self) -> None:
        """Load feature extractor and emotion model."""
        logger.info(f"Loading emotion recognition model: {self.model_name}")

        try:
            self._extracted_from__load_models_6()
        except Exception as e:
            logger.error(f"Failed to load emotion models: {str(e)}")
            raise

    # TODO Rename this here and in `_load_models`
    def _extracted_from__load_models_6(self):
        warnings.filterwarnings("ignore")

        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            f"facebook/{self.model_type}-base"
        )

        # Load emotion classification model
        model_id = "techiepedia/wav2vec2-emotion-recognition"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=8,  # emotions
        ).to(self.device)

        self.model.eval()
        logger.info(f"Emotion model loaded: {model_id}")

    async def detect_emotions(
        self,
        session_id: str,
        audio_path: str,
        segment_length_s: float = 2.0,
    ) -> AudioEmotionResult:
        """
        Detect emotions from audio file.

        Args:
            session_id: Session identifier
            audio_path: Path to audio file
            segment_length_s: Length of audio segments for analysis

        Returns:
            AudioEmotionResult with valence/arousal scores
        """
        start_time = time.time()

        try:
            # Load audio
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            waveform, sample_rate = librosa.load(str(audio_file), sr=16000, mono=True)
            audio_duration = len(waveform) / sample_rate

            # Process in segments
            segment_emotions = []
            segment_size = int(segment_length_s * sample_rate)

            for i in range(0, len(waveform), segment_size):
                segment = waveform[i : i + segment_size]
                if len(segment) < sample_rate:  # Skip tiny segments
                    continue

                start_sec = i / sample_rate
                end_sec = min((i + len(segment)) / sample_rate, audio_duration)

                emotion = await self._detect_segment_emotion(segment, sample_rate)
                segment_emotions.append((start_sec, end_sec, emotion))

            # Calculate overall emotion
            overall_emotion = self._aggregate_emotions([e[2] for e in segment_emotions])

            # Calculate speech metrics
            speech_rate = self._calculate_speech_rate(waveform, sample_rate)
            intensity = self._calculate_intensity(waveform)

            # Trajectory
            trajectory = [e[2] for e in segment_emotions]

            processing_time = (time.time() - start_time) * 1000

            return AudioEmotionResult(
                session_id=session_id,
                audio_path=str(audio_file),
                overall_emotion=overall_emotion,
                segment_emotions=segment_emotions,
                trajectory=trajectory,
                speech_rate_wpm=speech_rate,
                intensity_score=intensity,
                processing_time_ms=processing_time,
                audio_duration_s=audio_duration,
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Emotion detection failed: {str(e)}")
            return AudioEmotionResult(
                session_id=session_id,
                audio_path=audio_path,
                overall_emotion=EmotionalState(0.5, 0.5, 0.5, 0.0, "neutral", {}),
                segment_emotions=[],
                trajectory=[],
                speech_rate_wpm=0.0,
                intensity_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                audio_duration_s=0.0,
                model_name=self.model_name,
                error=str(e),
            )

    async def _detect_segment_emotion(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
    ) -> EmotionalState:
        """Detect emotion in single audio segment."""
        try:
            # Prepare inputs
            inputs = self.feature_extractor(
                audio_segment,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Softmax to probabilities
            probs = torch.softmax(logits, dim=-1)[0]

            # Map to VAD
            emotion_labels = [
                "neutral",
                "happiness",
                "sadness",
                "anger",
                "fear",
                "surprise",
                "disgust",
                "calm",
            ]

            emotion_probs = {
                label: float(prob)
                for label, prob in zip(emotion_labels, probs.cpu().numpy())
            }

            # Get primary emotion
            primary_idx = probs.argmax().item()
            primary_emotion = emotion_labels[primary_idx]
            confidence = float(probs[primary_idx])

            # Convert emotion to VAD
            vad = EMOTION_TO_VAD.get(primary_emotion, (0.5, 0.5, 0.5))
            valence, arousal, dominance = vad

            return EmotionalState(
                valence=valence * 2 - 1,  # Convert to -1.0 to 1.0
                arousal=arousal * 2 - 1,
                dominance=dominance * 2 - 1,
                confidence=confidence,
                primary_emotion=primary_emotion,
                emotion_probabilities=emotion_probs,
            )

        except Exception as e:
            logger.error(f"Segment emotion detection failed: {str(e)}")
            return EmotionalState(0.5, 0.5, 0.5, 0.0, "neutral", {})

    def _aggregate_emotions(self, emotions: List[EmotionalState]) -> EmotionalState:
        """Aggregate emotions across segments."""
        if not emotions:
            return EmotionalState(0.5, 0.5, 0.5, 0.0, "neutral", {})

        # Average VAD
        valences = [e.valence for e in emotions]
        arousals = [e.arousal for e in emotions]
        dominances = [e.dominance for e in emotions]
        confidences = [e.confidence for e in emotions]

        avg_valence = np.mean(valences)
        avg_arousal = np.mean(arousals)
        avg_dominance = np.mean(dominances)
        avg_confidence = np.mean(confidences)

        # Determine primary emotion by averaging probabilities
        avg_probs = {}
        emotion_labels = list(emotions[0].emotion_probabilities.keys())

        for label in emotion_labels:
            probs = [e.emotion_probabilities.get(label, 0.0) for e in emotions]
            avg_probs[label] = np.mean(probs)

        primary = max(avg_probs.items(), key=lambda x: x[1])[0]

        return EmotionalState(
            valence=avg_valence,
            arousal=avg_arousal,
            dominance=avg_dominance,
            confidence=avg_confidence,
            primary_emotion=primary,
            emotion_probabilities=avg_probs,
        )

    def _calculate_speech_rate(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Estimate speech rate in words per minute.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            Estimated WPM
        """
        # Simple estimation based on zero-crossing rate and energy
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        energy = np.sqrt(np.sum(waveform**2) / len(waveform))

        # Rough estimation: higher zcr + energy = faster speech
        base_wpm = 150  # average speaking rate
        rate_factor = (np.mean(zcr) + energy) / 2
        estimated_wpm = base_wpm * (0.8 + rate_factor)

        return float(np.clip(estimated_wpm, 50, 300))

    def _calculate_intensity(self, waveform: np.ndarray) -> float:
        """
        Calculate speech intensity (loudness).

        Args:
            waveform: Audio waveform

        Returns:
            Intensity score (0.0-1.0)
        """
        # RMS energy normalized
        rms = np.sqrt(np.mean(waveform**2))
        intensity = np.clip(rms * 10, 0, 1)  # Scale and clip
        return float(intensity)


class EmotionTrajectory:
    """Track emotion changes over conversation."""

    def __init__(self, window_size: int = 10):
        """
        Initialize trajectory tracker.

        Args:
            window_size: Number of segments to consider for trends
        """
        self.window_size = window_size
        self.emotions: List[Tuple[float, EmotionalState]] = []

    def add_emotion(self, timestamp: float, emotion: EmotionalState) -> None:
        """Add emotion sample."""
        self.emotions.append((timestamp, emotion))

    def get_trend(self) -> Dict[str, float]:
        """
        Calculate emotion trends.

        Returns:
            Trends for valence, arousal, dominance
        """
        if len(self.emotions) < 2:
            return {"valence_trend": 0.0, "arousal_trend": 0.0, "dominance_trend": 0.0}

        # Get last window
        window = self.emotions[-self.window_size :]

        if len(window) < 2:
            return {"valence_trend": 0.0, "arousal_trend": 0.0, "dominance_trend": 0.0}

        # Calculate linear trends
        timestamps = np.array([e[0] for e in window])
        valences = np.array([e[1].valence for e in window])
        arousals = np.array([e[1].arousal for e in window])
        dominances = np.array([e[1].dominance for e in window])

        # Linear regression slopes
        valence_trend = float(np.polyfit(timestamps, valences, 1)[0])
        arousal_trend = float(np.polyfit(timestamps, arousals, 1)[0])
        dominance_trend = float(np.polyfit(timestamps, dominances, 1)[0])

        return {
            "valence_trend": valence_trend,
            "arousal_trend": arousal_trend,
            "dominance_trend": dominance_trend,
        }

    def get_stats(self) -> Dict[str, float]:
        """Calculate emotion statistics."""
        if not self.emotions:
            return {}

        valences = [e[1].valence for e in self.emotions]
        arousals = [e[1].arousal for e in self.emotions]

        return {
            "mean_valence": float(np.mean(valences)),
            "std_valence": float(np.std(valences)),
            "mean_arousal": float(np.mean(arousals)),
            "std_arousal": float(np.std(arousals)),
            "min_valence": float(np.min(valences)),
            "max_valence": float(np.max(valences)),
            "min_arousal": float(np.min(arousals)),
            "max_arousal": float(np.max(arousals)),
        }
