"""
Audio Extraction and Preprocessing Pipeline for Voice Training Data.

This module provides comprehensive audio processing capabilities including
quality control, segmentation, noise reduction, and preprocessing for
optimal transcription and voice training data preparation.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Optional imports with fallbacks
try:
    import librosa
    import soundfile as sf

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    from pydub.silence import split_on_silence

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import webrtcvad

    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

import builtins
import contextlib

from logger import setup_logger


@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics."""

    file_path: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int | None = None
    snr_db: float | None = None
    rms_level: float | None = None
    peak_level: float | None = None
    silence_ratio: float | None = None
    clipping_ratio: float | None = None
    spectral_centroid: float | None = None
    zero_crossing_rate: float | None = None
    quality_score: float | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class AudioSegment:
    """Represents a segmented audio chunk."""

    start_time: float
    end_time: float
    duration: float
    file_path: str
    confidence: float | None = None
    is_speech: bool = True
    quality_metrics: AudioQualityMetrics | None = None


@dataclass
class AudioProcessingResult:
    """Result of audio processing operation."""

    input_file: str
    success: bool
    output_files: list[str] = field(default_factory=list)
    segments: list[AudioSegment] = field(default_factory=list)
    quality_metrics: AudioQualityMetrics | None = None
    processing_time: float = 0.0
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)


class AudioProcessor:
    """
    Comprehensive audio processing pipeline for voice training data.

    Features:
    - Audio quality assessment and control
    - Voice activity detection and segmentation
    - Noise reduction and audio enhancement
    - Format conversion and standardization
    - Batch processing capabilities
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        min_segment_duration: float = 1.0,
        max_segment_duration: float = 30.0,
        silence_threshold: float = -40.0,
        min_silence_duration: float = 0.5,
        quality_threshold: float = 0.6,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.quality_threshold = quality_threshold

        # Setup logging
        self.logger = setup_logger("audio_processor")

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for optional dependencies and log availability."""
        deps = {
            "librosa": LIBROSA_AVAILABLE,
            "pydub": PYDUB_AVAILABLE,
            "webrtcvad": WEBRTCVAD_AVAILABLE,
        }

        for dep, available in deps.items():
            if available:
                self.logger.info(f"✅ {dep} available")
            else:
                self.logger.warning(
                    f"⚠️ {dep} not available - some features will be limited"
                )

    def assess_audio_quality(self, file_path: str) -> AudioQualityMetrics:
        """Comprehensive audio quality assessment."""
        start_time = time.time()

        try:
            # Basic file info using ffprobe
            basic_info = self._get_audio_info_ffprobe(file_path)

            metrics = AudioQualityMetrics(
                file_path=file_path,
                duration=basic_info.get("duration", 0.0),
                sample_rate=basic_info.get("sample_rate", 0),
                channels=basic_info.get("channels", 0),
                bit_depth=basic_info.get("bit_depth"),
            )

            # Advanced analysis with librosa if available
            if LIBROSA_AVAILABLE:
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    metrics = self._analyze_with_librosa(y, sr, metrics)
                except Exception as e:
                    self.logger.warning(f"Librosa analysis failed for {file_path}: {e}")

            # Pydub analysis if available
            if PYDUB_AVAILABLE:
                try:
                    metrics = self._analyze_with_pydub(file_path, metrics)
                except Exception as e:
                    self.logger.warning(f"Pydub analysis failed for {file_path}: {e}")

            # Calculate overall quality score
            metrics.quality_score = self._calculate_quality_score(metrics)

            # Identify issues
            metrics.issues = self._identify_quality_issues(metrics)

            processing_time = time.time() - start_time
            self.logger.debug(f"Quality assessment completed in {processing_time:.2f}s")

            return metrics

        except Exception as e:
            self.logger.error(f"Quality assessment failed for {file_path}: {e}")
            return AudioQualityMetrics(
                file_path=file_path,
                duration=0.0,
                sample_rate=0,
                channels=0,
                quality_score=0.0,
                issues=[f"Assessment failed: {e}"],
            )

    def _get_audio_info_ffprobe(self, file_path: str) -> dict:
        """Get basic audio information using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                file_path,
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Find audio stream
                audio_stream = None
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        audio_stream = stream
                        break

                if audio_stream:
                    return {
                        "duration": float(audio_stream.get("duration", 0)),
                        "sample_rate": int(audio_stream.get("sample_rate", 0)),
                        "channels": int(audio_stream.get("channels", 0)),
                        "bit_depth": audio_stream.get("bits_per_sample"),
                        "codec": audio_stream.get("codec_name"),
                        "bitrate": audio_stream.get("bit_rate"),
                    }

            return {}

        except Exception as e:
            self.logger.warning(f"ffprobe analysis failed: {e}")
            return {}

    def _analyze_with_librosa(
        self, y: np.ndarray, sr: int, metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Perform advanced audio analysis using librosa."""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        metrics.rms_level = float(np.mean(rms))

        # Peak level
        metrics.peak_level = float(np.max(np.abs(y)))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        metrics.zero_crossing_rate = float(np.mean(zcr))

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        metrics.spectral_centroid = float(np.mean(spectral_centroids))

        # Estimate SNR (simple approach)
        # Split into frames and estimate noise floor
        frame_length = sr // 10  # 100ms frames
        frames = librosa.util.frame(
            y, frame_length=frame_length, hop_length=frame_length // 2
        )
        frame_energy = np.mean(frames**2, axis=0)

        # Assume bottom 10% of frames are noise
        noise_threshold = np.percentile(frame_energy, 10)
        signal_energy = np.mean(frame_energy[frame_energy > noise_threshold])

        if noise_threshold > 0:
            snr_linear = signal_energy / noise_threshold
            metrics.snr_db = (
                float(10 * np.log10(snr_linear)) if snr_linear > 0 else -60.0
            )

        # Silence ratio (frames below threshold)
        silence_frames = np.sum(frame_energy < noise_threshold * 2)
        metrics.silence_ratio = float(silence_frames / len(frame_energy))

        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        metrics.clipping_ratio = float(clipped_samples / len(y))

        return metrics

    def _analyze_with_pydub(
        self, file_path: str, metrics: AudioQualityMetrics
    ) -> AudioQualityMetrics:
        """Perform audio analysis using pydub."""
        try:
            audio = AudioSegment.from_file(file_path)

            # Update basic metrics if not already set
            if metrics.duration == 0:
                metrics.duration = len(audio) / 1000.0
            if metrics.channels == 0:
                metrics.channels = audio.channels
            if metrics.sample_rate == 0:
                metrics.sample_rate = audio.frame_rate

            return metrics

        except Exception as e:
            self.logger.warning(f"Pydub analysis failed: {e}")
            return metrics

    def _calculate_quality_score(self, metrics: AudioQualityMetrics) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0

        # Duration check
        if metrics.duration < self.min_segment_duration:
            score *= 0.5
        elif metrics.duration > self.max_segment_duration * 2:
            score *= 0.8

        # Sample rate check
        if metrics.sample_rate < 8000:
            score *= 0.3
        elif metrics.sample_rate < 16000:
            score *= 0.7

        # SNR check
        if metrics.snr_db is not None:
            if metrics.snr_db < 10:
                score *= 0.4
            elif metrics.snr_db < 20:
                score *= 0.7

        # Silence ratio check
        if metrics.silence_ratio is not None:
            if metrics.silence_ratio > 0.8:
                score *= 0.3
            elif metrics.silence_ratio > 0.5:
                score *= 0.6

        # Clipping check
        if metrics.clipping_ratio is not None and metrics.clipping_ratio > 0.01:
            score *= 0.5

        return max(0.0, min(1.0, score))

    def _identify_quality_issues(self, metrics: AudioQualityMetrics) -> list[str]:
        """Identify specific quality issues."""
        issues = []

        if metrics.duration < self.min_segment_duration:
            issues.append(
                f"Too short: {metrics.duration:.1f}s < {self.min_segment_duration}s"
            )

        if metrics.sample_rate < 16000:
            issues.append(f"Low sample rate: {metrics.sample_rate}Hz")

        if metrics.snr_db is not None and metrics.snr_db < 15:
            issues.append(f"Low SNR: {metrics.snr_db:.1f}dB")

        if metrics.silence_ratio is not None and metrics.silence_ratio > 0.7:
            issues.append(f"High silence ratio: {metrics.silence_ratio:.1%}")

        if metrics.clipping_ratio is not None and metrics.clipping_ratio > 0.01:
            issues.append(f"Audio clipping detected: {metrics.clipping_ratio:.1%}")

        if metrics.channels > 2:
            issues.append(f"Too many channels: {metrics.channels}")

        return issues

    def segment_audio_by_silence(
        self, file_path: str, output_dir: str
    ) -> list[AudioSegment]:
        """Segment audio by detecting silence periods."""
        try:
            if not PYDUB_AVAILABLE:
                self.logger.warning("Pydub not available - using basic segmentation")
                return self._basic_segmentation(file_path, output_dir)

            audio = AudioSegment.from_file(file_path)

            # Split on silence
            segments = split_on_silence(
                audio,
                min_silence_len=int(self.min_silence_duration * 1000),  # Convert to ms
                silence_thresh=self.silence_threshold,
                keep_silence=250,  # Keep 250ms of silence at edges
            )

            audio_segments = []
            current_time = 0.0

            for i, segment in enumerate(segments):
                duration = len(segment) / 1000.0  # Convert to seconds

                if duration >= self.min_segment_duration:
                    # Export segment
                    segment_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
                    segment.export(segment_path, format="wav")

                    audio_segments.append(
                        AudioSegment(
                            start_time=current_time,
                            end_time=current_time + duration,
                            duration=duration,
                            file_path=segment_path,
                            is_speech=True,
                        )
                    )

                current_time += duration

            self.logger.info(
                f"Segmented {file_path} into {len(audio_segments)} segments"
            )
            return audio_segments

        except Exception as e:
            self.logger.error(f"Segmentation failed for {file_path}: {e}")
            return []

    def _basic_segmentation(
        self, file_path: str, output_dir: str
    ) -> list[AudioSegment]:
        """Basic segmentation using ffmpeg when pydub is not available."""
        try:
            # Get audio duration
            info = self._get_audio_info_ffprobe(file_path)
            duration = info.get("duration", 0)

            if duration == 0:
                return []

            segments = []
            segment_duration = min(self.max_segment_duration, duration)
            num_segments = int(np.ceil(duration / segment_duration))

            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                actual_duration = end_time - start_time

                if actual_duration >= self.min_segment_duration:
                    segment_path = os.path.join(output_dir, f"segment_{i:03d}.wav")

                    # Use ffmpeg to extract segment
                    cmd = [
                        "ffmpeg",
                        "-i",
                        file_path,
                        "-ss",
                        str(start_time),
                        "-t",
                        str(actual_duration),
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        str(self.target_sample_rate),
                        "-ac",
                        str(self.target_channels),
                        "-y",  # Overwrite output
                        segment_path,
                    ]

                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

                    if result.returncode == 0:
                        segments.append(
                            AudioSegment(
                                start_time=start_time,
                                end_time=end_time,
                                duration=actual_duration,
                                file_path=segment_path,
                                is_speech=True,
                            )
                        )

            return segments

        except Exception as e:
            self.logger.error(f"Basic segmentation failed: {e}")
            return []

    def enhance_audio(self, input_path: str, output_path: str) -> bool:
        """Enhance audio quality using available tools."""
        try:
            if PYDUB_AVAILABLE:
                return self._enhance_with_pydub(input_path, output_path)
            return self._enhance_with_ffmpeg(input_path, output_path)
        except Exception as e:
            self.logger.error(f"Audio enhancement failed: {e}")
            return False

    def _enhance_with_pydub(self, input_path: str, output_path: str) -> bool:
        """Enhance audio using pydub."""
        try:
            audio = AudioSegment.from_file(input_path)

            # Normalize audio
            audio = normalize(audio)

            # Convert to target format
            audio = audio.set_frame_rate(self.target_sample_rate)
            audio = audio.set_channels(self.target_channels)

            # Export enhanced audio
            audio.export(output_path, format="wav")

            return True

        except Exception as e:
            self.logger.error(f"Pydub enhancement failed: {e}")
            return False

    def _enhance_with_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """Enhance audio using ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.target_sample_rate),
                "-ac",
                str(self.target_channels),
                "-af",
                "volume=1.5,highpass=f=80,lowpass=f=8000",  # Basic filtering
                "-y",
                output_path,
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"FFmpeg enhancement failed: {e}")
            return False

    def process_audio_file(
        self, input_path: str, output_dir: str
    ) -> AudioProcessingResult:
        """Process a single audio file through the complete pipeline."""
        start_time = time.time()

        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Step 1: Quality assessment
            self.logger.info(f"Assessing quality for {input_path}")
            quality_metrics = self.assess_audio_quality(input_path)

            # Step 2: Check if enhancement is needed
            enhanced_path = input_path
            if quality_metrics.quality_score < self.quality_threshold:
                self.logger.info(
                    f"Enhancing audio (quality score: {quality_metrics.quality_score:.2f})"
                )
                enhanced_path = os.path.join(
                    output_dir, "enhanced_" + os.path.basename(input_path)
                )

                if not self.enhance_audio(input_path, enhanced_path):
                    self.logger.warning("Enhancement failed, using original file")
                    enhanced_path = input_path

            # Step 3: Segmentation
            self.logger.info(f"Segmenting audio: {enhanced_path}")
            segments = self.segment_audio_by_silence(enhanced_path, output_dir)

            # Step 4: Quality check segments
            valid_segments = []
            for segment in segments:
                segment_metrics = self.assess_audio_quality(segment.file_path)
                segment.quality_metrics = segment_metrics

                if segment_metrics.quality_score >= self.quality_threshold:
                    valid_segments.append(segment)
                else:
                    self.logger.debug(
                        f"Rejecting low-quality segment: {segment.file_path}"
                    )
                    # Remove low-quality segment file
                    with contextlib.suppress(builtins.BaseException):
                        os.remove(segment.file_path)

            processing_time = time.time() - start_time

            result = AudioProcessingResult(
                input_file=input_path,
                success=True,
                output_files=[seg.file_path for seg in valid_segments],
                segments=valid_segments,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                metadata={
                    "enhanced": enhanced_path != input_path,
                    "original_segments": len(segments),
                    "valid_segments": len(valid_segments),
                    "quality_threshold": self.quality_threshold,
                },
            )

            self.logger.info(
                f"Processed {input_path}: {len(valid_segments)} valid segments in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            error_msg = f"Processing failed for {input_path}: {e}"
            self.logger.error(error_msg)

            return AudioProcessingResult(
                input_file=input_path,
                success=False,
                processing_time=time.time() - start_time,
                error_message=error_msg,
            )

    def process_batch(
        self, input_files: list[str], output_base_dir: str
    ) -> list[AudioProcessingResult]:
        """Process multiple audio files in batch."""
        results = []

        for i, input_file in enumerate(input_files):
            self.logger.info(f"Processing file {i+1}/{len(input_files)}: {input_file}")

            # Create file-specific output directory
            file_name = Path(input_file).stem
            file_output_dir = os.path.join(output_base_dir, file_name)

            result = self.process_audio_file(input_file, file_output_dir)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        total_segments = sum(len(r.segments) for r in results)

        self.logger.info(
            f"Batch processing complete: {successful}/{len(input_files)} files, {total_segments} total segments"
        )

        return results


# Backward compatibility function
def process_audio_files(
    input_files: list[str],
    output_dir: str = "processed_audio",
    target_sample_rate: int = 16000,
    quality_threshold: float = 0.6,
) -> list[AudioProcessingResult]:
    """
    Process audio files with enhanced capabilities.

    This function provides backward compatibility while offering
    the enhanced features of the new AudioProcessor.
    """
    processor = AudioProcessor(
        target_sample_rate=target_sample_rate, quality_threshold=quality_threshold
    )

    return processor.process_batch(input_files, output_dir)
