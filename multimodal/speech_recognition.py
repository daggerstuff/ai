"""
Speech Recognition for Pixel Multimodal Integration.

Converts audio input (WAV, MP3, etc.) to text using Whisper or Wav2Vec2.
Handles preprocessing, error recovery, and confidence scoring.

Features:
  - Multi-format audio support (WAV, MP3, FLAC, OGG)
  - Automatic sample rate normalization
  - Language detection and specification
  - Confidence scoring per segment
  - Timestamp alignment with source audio
  - Streaming support for real-time transcription
  - Error handling for degraded audio

Example:
    >>> recognizer = SpeechRecognizer(model_name="base")
    >>> result = await recognizer.transcribe_audio("session_001", audio_path)
    >>> print(f"Text: {result['text']}")
    >>> print(f"Confidence: {result['confidence']:.2f}")
"""

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torchaudio
from faster_whisper import WhisperModel

from ai.utils.torch_proxy import torch

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Represents a single transcription segment."""

    start_time: float  # Seconds
    end_time: float  # Seconds
    text: str
    confidence: float  # 0.0-1.0
    language: str = "en"


@dataclass
class TranscriptionResult:
    """Complete transcription result from speech recognition."""

    session_id: str
    audio_path: str
    full_text: str
    segments: list[TranscriptionSegment]
    language: str
    overall_confidence: float
    processing_time_ms: float
    audio_duration_s: float
    sample_rate: int
    model_name: str
    error: str | None = None


class SpeechRecognizer:
    """Speech-to-text recognition using Whisper or Wav2Vec2."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        language: str | None = None,
        compute_type: str = "float16" if torch.cuda.is_available() else "float32",
    ):
        """
        Initialize speech recognizer.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: torch device (cuda/cpu)
            language: ISO 639-1 language code (auto-detect if None)
            compute_type: float16 or float32 precision
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.compute_type = compute_type

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Faster-Whisper model."""
        logger.info(f"Loading Faster-Whisper {self.model_name} model on {self.device}")

        try:
            # Suppress warnings
            warnings.simplefilter("default")

            # Load model
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info(f"Model loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e!s}")
            raise

    async def transcribe_audio(
        self,
        session_id: str,
        audio_path: str,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            session_id: Unique session identifier
            audio_path: Path to audio file
            language: Language code (overrides default)
            initial_prompt: Context or instruction for transcription

        Returns:
            TranscriptionResult with full text and segments

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If audio format unsupported
            RuntimeError: If transcription fails
        """
        start_time = time.time()

        try:
            # Validate file
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Load audio
            logger.info(f"Loading audio from {audio_path}")
            waveform, sample_rate = self._load_audio(str(audio_file))

            audio_duration = len(waveform) / sample_rate

            # Transcribe
            logger.info(f"Transcribing {audio_duration:.1f}s audio")
            result = await asyncio.to_thread(
                self._transcribe_sync,
                waveform,
                sample_rate,
                language or self.language,
                initial_prompt,
            )

            # Parse result
            segments = self._parse_segments(result)
            full_text = " ".join([seg.text for seg in segments])
            overall_confidence = np.mean([seg.confidence for seg in segments])

            processing_time = (time.time() - start_time) * 1000

            return TranscriptionResult(
                session_id=session_id,
                audio_path=str(audio_file),
                full_text=full_text,
                segments=segments,
                language=result.get("language", "en"),
                overall_confidence=float(overall_confidence),
                processing_time_ms=processing_time,
                audio_duration_s=audio_duration,
                sample_rate=sample_rate,
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e!s}")
            return TranscriptionResult(
                session_id=session_id,
                audio_path=str(audio_path),
                full_text="",
                segments=[],
                language="en",
                overall_confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                audio_duration_s=0.0,
                sample_rate=0,
                model_name=self.model_name,
                error=str(e),
            )

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Try librosa first (handles most formats)
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            return waveform, sample_rate

        except Exception as e:
            logger.warning(f"Librosa failed, trying torchaudio: {e!s}")

            try:
                # Fallback to torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                return waveform.squeeze().numpy(), sample_rate

            except Exception as e2:
                logger.error(f"Both audio loaders failed: {e2!s}")
                raise ValueError(f"Failed to load audio file: {audio_path}") from e2

    def _transcribe_sync(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str | None,
        initial_prompt: str | None,
    ) -> dict[str, Any]:
        """Synchronous transcription (runs in thread)."""
        # Normalize sample rate to 16kHz if needed
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Transcribe
        segments, info = self.model.transcribe(
            waveform,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=5,
        )

        # Convert generator to list for internal result format
        segment_list = list(segments)

        return {
            "segments": segment_list,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def _parse_segments(self, result: dict[str, Any]) -> list[TranscriptionSegment]:
        """Parse Faster-Whisper result into segments."""
        segments = []

        for segment in result.get("segments", []):
            segments.append(
                TranscriptionSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=float(getattr(segment, "avg_logprob", 0.9)),
                    language=result.get("language", "en"),
                )
            )

        return segments

    async def stream_transcribe(
        self,
        session_id: str,
        audio_chunks: list[np.ndarray],
        sample_rate: int,
        _chunk_duration_ms: int = 1000,
    ) -> TranscriptionResult:
        """
        Transcribe streaming audio chunks.

        Args:
            session_id: Session identifier
            audio_chunks: List of audio chunks
            sample_rate: Sample rate of chunks
            chunk_duration_ms: Duration of each chunk

        Returns:
            TranscriptionResult with accumulated text
        """
        try:
            # Concatenate chunks
            full_audio = np.concatenate(audio_chunks)

            # Create temporary file for transcription
            temp_path = f"/tmp/{session_id}_stream.wav"

            torchaudio.save(temp_path, torch.from_numpy(full_audio).unsqueeze(0), sample_rate)

            # Transcribe
            result = await self.transcribe_audio(session_id, temp_path)

            # Clean up
            Path(temp_path).unlink()

            return result

        except Exception as e:
            logger.error(f"Stream transcription failed: {e!s}")
            raise

    def to_dict(self) -> dict[str, Any]:
        """Serialize recognizer configuration."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "compute_type": self.compute_type,
        }


class AudioPreprocessor:
    """Audio preprocessing and feature extraction."""

    @staticmethod
    def normalize_audio(
        waveform: np.ndarray,
        target_sample_rate: int = 16000,
        current_sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Normalize audio waveform.

        Args:
            waveform: Input audio array
            target_sample_rate: Target sample rate
            current_sample_rate: Current sample rate

        Returns:
            Normalized audio
        """
        # Resample if needed
        if current_sample_rate != target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=current_sample_rate,
                target_sr=target_sample_rate,
            )

        # Normalize amplitude
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            waveform = waveform / max_amp * 0.95

        return waveform

    @staticmethod
    def extract_features(
        waveform: np.ndarray,
        sample_rate: int,
    ) -> dict[str, np.ndarray]:
        """
        Extract audio features.

        Args:
            waveform: Input audio
            sample_rate: Sample rate

        Returns:
            Dictionary of features (MFCC, spectral, etc.)
        """
        features = {}

        # MFCC
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        features["mfcc"] = mfcc

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        features["spectral_centroid"] = spectral_centroid

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)
        features["zero_crossing_rate"] = zcr

        # Energy
        energy = np.sqrt(np.sum(waveform**2))
        features["energy"] = energy

        return features

    @staticmethod
    def detect_voice_activity(
        waveform: np.ndarray,
        sample_rate: int,
        threshold: float = 0.02,
    ) -> list[tuple[int, int]]:
        """
        Detect voice activity regions.

        Args:
            waveform: Input audio
            sample_rate: Sample rate
            threshold: Energy threshold

        Returns:
            List of (start, end) sample indices with voice
        """
        # Energy-based VAD
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        energy = np.array([np.sum(waveform[i : i + frame_length] ** 2) for i in range(0, len(waveform), hop_length)])

        # Normalize
        energy = energy / (np.max(energy) + 1e-10)

        # Find active frames
        active = energy > threshold
        regions = []
        start = None

        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i * hop_length
            elif not is_active and start is not None:
                regions.append((start, i * hop_length))
                start = None

        if start is not None:
            regions.append((start, len(waveform)))

        return regions
