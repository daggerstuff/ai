"""
Voice Transcription Pipeline with Whisper Integration and Quality Filtering.

This module provides comprehensive voice transcription capabilities using
Whisper/Faster-Whisper with confidence scoring, quality filtering, and
batch processing for voice training data preparation.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Whisper imports with fallbacks
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from audio_processor import AudioProcessor, AudioQualityMetrics
from logger import setup_logger


@dataclass
class TranscriptionSegment:
    """Represents a transcribed audio segment."""

    start_time: float
    end_time: float
    text: str
    confidence: float
    language: str | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    words: list[dict] | None = None


@dataclass
class TranscriptionResult:
    """Result of transcribing a single audio file."""

    file_path: str
    success: bool
    language: str | None = None
    language_confidence: float | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    full_text: str = ""
    processing_time: float = 0.0
    model_used: str = ""
    quality_score: float = 0.0
    confidence_score: float = 0.0
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchTranscriptionResult:
    """Result of batch transcription processing."""

    total_files: int
    successful_files: int
    failed_files: int
    total_segments: int
    average_confidence: float
    average_quality: float
    total_processing_time: float
    results: list[TranscriptionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class VoiceTranscriber:
    """
    Comprehensive voice transcription pipeline using Whisper models.

    Features:
    - Multiple Whisper model support (OpenAI Whisper, Faster-Whisper)
    - Confidence scoring and quality filtering
    - Language detection and validation
    - Batch processing with progress tracking
    - Detailed transcription metadata
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "float32",
        language: str | None = None,
        min_confidence: float = 0.6,
        min_quality_score: float = 0.5,
        use_faster_whisper: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.compute_type = compute_type
        self.language = language
        self.min_confidence = min_confidence
        self.min_quality_score = min_quality_score
        self.use_faster_whisper = use_faster_whisper
        self.beam_size = beam_size
        self.temperature = temperature

        # Setup logging
        self.logger = setup_logger("voice_transcriber")

        # Initialize model
        self.model = None
        self.model_type = None
        self._initialize_model()

        # Initialize audio processor for quality assessment
        self.audio_processor = AudioProcessor()

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            if self.use_faster_whisper and FASTER_WHISPER_AVAILABLE:
                self.logger.info(
                    f"Initializing Faster-Whisper model: {self.model_name}"
                )
                self.model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type
                )
                self.model_type = "faster-whisper"

            elif WHISPER_AVAILABLE:
                self.logger.info(
                    f"Initializing OpenAI Whisper model: {self.model_name}"
                )
                self.model = whisper.load_model(self.model_name, device=self.device)
                self.model_type = "openai-whisper"

            else:
                raise ImportError(
                    "Neither Faster-Whisper nor OpenAI Whisper is available"
                )

            self.logger.info(
                f"Model initialized successfully: {self.model_type} on {self.device}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            raise

    def transcribe_audio(self, file_path: str) -> TranscriptionResult:
        """Transcribe a single audio file."""
        start_time = time.time()

        try:
            self.logger.info(f"Transcribing: {file_path}")

            # Quality assessment
            quality_metrics = self.audio_processor.assess_audio_quality(file_path)

            if quality_metrics.quality_score < self.min_quality_score:
                return TranscriptionResult(
                    file_path=file_path,
                    success=False,
                    quality_score=quality_metrics.quality_score,
                    error_message=f"Audio quality too low: {quality_metrics.quality_score:.2f} < {self.min_quality_score}",
                )

            # Transcribe based on model type
            if self.model_type == "faster-whisper":
                result = self._transcribe_faster_whisper(file_path, quality_metrics)
            else:
                result = self._transcribe_openai_whisper(file_path, quality_metrics)

            result.processing_time = time.time() - start_time

            # Calculate confidence score
            if result.segments:
                result.confidence_score = np.mean(
                    [seg.confidence for seg in result.segments]
                )

            # Filter by confidence
            if result.confidence_score < self.min_confidence:
                result.success = False
                result.error_message = f"Confidence too low: {result.confidence_score:.2f} < {self.min_confidence}"

            self.logger.info(
                f"Transcription complete: {len(result.segments)} segments, confidence: {result.confidence_score:.2f}"
            )
            return result

        except Exception as e:
            error_msg = f"Transcription failed for {file_path}: {e}"
            self.logger.error(error_msg)

            return TranscriptionResult(
                file_path=file_path,
                success=False,
                processing_time=time.time() - start_time,
                error_message=error_msg,
            )

    def _transcribe_faster_whisper(
        self, file_path: str, quality_metrics: AudioQualityMetrics
    ) -> TranscriptionResult:
        """Transcribe using Faster-Whisper."""
        try:
            segments, info = self.model.transcribe(
                file_path,
                language=self.language,
                beam_size=self.beam_size,
                temperature=self.temperature,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            transcription_segments = []
            full_text_parts = []

            for segment in segments:
                # Calculate confidence from avg_logprob
                confidence = self._logprob_to_confidence(segment.avg_logprob)

                transcription_segment = TranscriptionSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=confidence,
                    language=info.language,
                    no_speech_prob=segment.no_speech_prob,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    temperature=segment.temperature,
                    words=(
                        [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability,
                            }
                            for word in segment.words
                        ]
                        if hasattr(segment, "words") and segment.words
                        else None
                    ),
                )

                transcription_segments.append(transcription_segment)
                full_text_parts.append(segment.text.strip())

            return TranscriptionResult(
                file_path=file_path,
                success=True,
                language=info.language,
                language_confidence=info.language_probability,
                segments=transcription_segments,
                full_text=" ".join(full_text_parts),
                model_used=f"faster-whisper-{self.model_name}",
                quality_score=quality_metrics.quality_score,
                metadata={
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "vad_filter": True,
                    "beam_size": self.beam_size,
                    "temperature": self.temperature,
                },
            )

        except Exception as e:
            raise Exception(f"Faster-Whisper transcription failed: {e}")

    def _transcribe_openai_whisper(
        self, file_path: str, quality_metrics: AudioQualityMetrics
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        try:
            result = self.model.transcribe(
                file_path,
                language=self.language,
                temperature=self.temperature,
                word_timestamps=True,
                verbose=False,
            )

            transcription_segments = []

            for segment in result.get("segments", []):
                # OpenAI Whisper doesn't provide direct confidence scores
                # Estimate from no_speech_prob and avg_logprob
                confidence = self._estimate_openai_confidence(segment)

                transcription_segment = TranscriptionSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                    confidence=confidence,
                    language=result.get("language"),
                    avg_logprob=segment.get("avg_logprob"),
                    compression_ratio=segment.get("compression_ratio"),
                    temperature=segment.get("temperature"),
                    words=segment.get("words", []),
                )

                transcription_segments.append(transcription_segment)

            return TranscriptionResult(
                file_path=file_path,
                success=True,
                language=result.get("language"),
                segments=transcription_segments,
                full_text=result.get("text", "").strip(),
                model_used=f"openai-whisper-{self.model_name}",
                quality_score=quality_metrics.quality_score,
                metadata={"temperature": self.temperature, "word_timestamps": True},
            )

        except Exception as e:
            raise Exception(f"OpenAI Whisper transcription failed: {e}")

    def _logprob_to_confidence(self, avg_logprob: float) -> float:
        """Convert average log probability to confidence score (0-1)."""
        # Empirical mapping based on Whisper behavior
        # avg_logprob typically ranges from -1.0 (high confidence) to -3.0+ (low confidence)
        if avg_logprob >= -0.5:
            return 0.95
        if avg_logprob >= -1.0:
            return 0.85
        if avg_logprob >= -1.5:
            return 0.75
        if avg_logprob >= -2.0:
            return 0.65
        if avg_logprob >= -2.5:
            return 0.55
        if avg_logprob >= -3.0:
            return 0.45
        return 0.35

    def _estimate_openai_confidence(self, segment: dict) -> float:
        """Estimate confidence for OpenAI Whisper segments."""
        # Use avg_logprob if available
        if "avg_logprob" in segment:
            return self._logprob_to_confidence(segment["avg_logprob"])

        # Fallback: use compression ratio and other heuristics
        compression_ratio = segment.get("compression_ratio", 2.0)

        # Lower compression ratio generally indicates better transcription
        if compression_ratio < 1.5:
            return 0.9
        if compression_ratio < 2.0:
            return 0.8
        if compression_ratio < 2.5:
            return 0.7
        if compression_ratio < 3.0:
            return 0.6
        return 0.5

    def transcribe_batch(
        self, file_paths: list[str], output_dir: str | None = None
    ) -> BatchTranscriptionResult:
        """Transcribe multiple audio files in batch."""
        start_time = time.time()

        self.logger.info(f"Starting batch transcription of {len(file_paths)} files")

        results = []
        successful_count = 0
        total_segments = 0
        confidence_scores = []
        quality_scores = []
        errors = []

        for i, file_path in enumerate(file_paths):
            self.logger.info(
                f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}"
            )

            try:
                result = self.transcribe_audio(file_path)
                results.append(result)

                if result.success:
                    successful_count += 1
                    total_segments += len(result.segments)
                    confidence_scores.append(result.confidence_score)
                    quality_scores.append(result.quality_score)

                    # Save transcription if output directory specified
                    if output_dir:
                        self._save_transcription(result, output_dir)
                else:
                    errors.append(
                        result.error_message or f"Failed to transcribe {file_path}"
                    )

            except Exception as e:
                error_msg = f"Exception processing {file_path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

                results.append(
                    TranscriptionResult(
                        file_path=file_path, success=False, error_message=error_msg
                    )
                )

        total_processing_time = time.time() - start_time

        batch_result = BatchTranscriptionResult(
            total_files=len(file_paths),
            successful_files=successful_count,
            failed_files=len(file_paths) - successful_count,
            total_segments=total_segments,
            average_confidence=np.mean(confidence_scores) if confidence_scores else 0.0,
            average_quality=np.mean(quality_scores) if quality_scores else 0.0,
            total_processing_time=total_processing_time,
            results=results,
            errors=errors,
        )

        self.logger.info(
            f"Batch transcription complete: {successful_count}/{len(file_paths)} successful"
        )
        return batch_result

    def _save_transcription(self, result: TranscriptionResult, output_dir: str):
        """Save transcription result to files."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            file_stem = Path(result.file_path).stem

            # Save as JSON with full metadata
            json_path = Path(output_dir) / f"{file_stem}_transcription.json"
            transcription_data = {
                "file_path": result.file_path,
                "language": result.language,
                "language_confidence": result.language_confidence,
                "full_text": result.full_text,
                "confidence_score": result.confidence_score,
                "quality_score": result.quality_score,
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "metadata": result.metadata,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "confidence": seg.confidence,
                        "language": seg.language,
                        "no_speech_prob": seg.no_speech_prob,
                        "avg_logprob": seg.avg_logprob,
                        "compression_ratio": seg.compression_ratio,
                        "temperature": seg.temperature,
                        "words": seg.words,
                    }
                    for seg in result.segments
                ],
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            # Save as plain text
            txt_path = Path(output_dir) / f"{file_stem}_transcription.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result.full_text)

            # Save as SRT subtitle format
            srt_path = Path(output_dir) / f"{file_stem}_transcription.srt"
            self._save_as_srt(result.segments, srt_path)

            self.logger.debug(f"Transcription saved: {json_path}")

        except Exception as e:
            self.logger.error(
                f"Failed to save transcription for {result.file_path}: {e}"
            )

    def _save_as_srt(self, segments: list[TranscriptionSegment], output_path: Path):
        """Save transcription segments as SRT subtitle file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._seconds_to_srt_time(segment.start_time)
                    end_time = self._seconds_to_srt_time(segment.end_time)

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")

        except Exception as e:
            self.logger.warning(f"Failed to save SRT file: {e}")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def generate_transcription_report(self, result: BatchTranscriptionResult) -> str:
        """Generate a detailed transcription report."""
        report = []
        report.append("=" * 60)
        report.append("VOICE TRANSCRIPTION REPORT")
        report.append("=" * 60)
        report.append(f"Total Files: {result.total_files}")
        report.append(f"Successful: {result.successful_files}")
        report.append(f"Failed: {result.failed_files}")
        report.append(f"Total Segments: {result.total_segments}")
        report.append(f"Average Confidence: {result.average_confidence:.2f}")
        report.append(f"Average Quality: {result.average_quality:.2f}")
        report.append(f"Processing Time: {result.total_processing_time:.2f} seconds")
        report.append(
            f"Success Rate: {(result.successful_files/result.total_files*100):.1f}%"
        )
        report.append("")

        if result.results:
            report.append("DETAILED RESULTS:")
            report.append("-" * 40)
            for res in result.results:
                status = "✅ SUCCESS" if res.success else "❌ FAILED"
                filename = os.path.basename(res.file_path)
                report.append(
                    f"{status} | {filename} | {len(res.segments)} segments | conf: {res.confidence_score:.2f} | qual: {res.quality_score:.2f}"
                )
                if not res.success and res.error_message:
                    report.append(f"    Error: {res.error_message[:100]}...")

        if result.errors:
            report.append("")
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in result.errors[:10]:  # Limit to first 10 errors
                report.append(f"• {error[:150]}...")

        report.append("=" * 60)
        return "\n".join(report)

    def filter_by_quality(
        self,
        results: list[TranscriptionResult],
        min_confidence: float | None = None,
        min_quality: float | None = None,
    ) -> list[TranscriptionResult]:
        """Filter transcription results by quality thresholds."""
        min_conf = min_confidence or self.min_confidence
        min_qual = min_quality or self.min_quality_score

        filtered = []
        for result in results:
            if (
                result.success
                and result.confidence_score >= min_conf
                and result.quality_score >= min_qual
            ):
                filtered.append(result)

        self.logger.info(
            f"Quality filtering: {len(filtered)}/{len(results)} results passed"
        )
        return filtered


# Backward compatibility function
def transcribe_audio_files(
    file_paths: list[str],
    model_name: str = "base",
    output_dir: str | None = None,
    language: str | None = None,
    min_confidence: float = 0.6,
) -> BatchTranscriptionResult:
    """
    Transcribe audio files with enhanced capabilities.

    This function provides backward compatibility while offering
    the enhanced features of the new VoiceTranscriber.
    """
    transcriber = VoiceTranscriber(
        model_name=model_name, language=language, min_confidence=min_confidence
    )

    return transcriber.transcribe_batch(file_paths, output_dir)
