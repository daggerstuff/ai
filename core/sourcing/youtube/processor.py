"""
YouTube channel processor - executes the discovery and evaluation pipeline.

Coordinates the full workflow:
1. Discover channels from YouTube API
2. Evaluate quality and metadata
3. Classify into therapeutic categories
4. Verify credentials and licensing
5. Score and rank channels
6. Generate report
"""

import logging
import os
import time
from typing import List, Optional

from ai.core.sourcing.youtube.api import (
    ChannelAnalyzer,
    ChannelHunterConfig,
    YouTubeAPI,
    YouTubeChannelHunter,
)
from ai.core.sourcing.youtube.models import (
    Channel,
    ChannelQualityThresholds,
    ChannelStatus,
    ContentCategory,
    QualityMetrics,
)

# Removed unused imports: ChannelRegistry, LicensingInfo, ChannelMonitor


logger = logging.getLogger(__name__)


class ChannelDiscoveryResults:
    """Results from a channel discovery run."""

    def __init__(self):
        self.total_searched: int = 0
        self.found_channels: List[Channel] = []
        self.qualified_channels: List[Channel] = []
        self.rejected_channels: List[Channel] = []
        self.time_elapsed_seconds: float = 0.0

        # Statistics
        self.languages: dict = {}
        self.categories: dict = {}
        self.professional_count: int = 0
        self.total_subscribers: int = 0
        self.total_videos: int = 0

    def add_channel(self, channel: Channel, qualified: bool):
        """Add a channel result."""
        self.found_channels.append(channel)
        self.total_searched += 1

        # Track by qualification
        if qualified:
            self.qualified_channels.append(channel)
            if channel.is_professional:
                self.professional_count += 1
        else:
            self.rejected_channels.append(channel)

        # Update statistics
        for lang in channel.languages:
            self.languages[lang] = self.languages.get(lang, 0) + 1

        for cat in channel.categories:
            key = cat.value
            self.categories[key] = self.categories.get(key, 0) + 1

        self.total_subscribers += channel.subscriber_count
        self.total_videos += channel.video_count


class ChannelProcessor:
    """
    Main processor for YouTube channel discovery and evaluation.

    Coordinates discovery, evaluation, and reporting phases.
    """

    def __init__(
        self,
        api_key: str,
        hunter_config: Optional[ChannelHunterConfig] = None,
        quality_thresholds: Optional[ChannelQualityThresholds] = None,
    ):
        self.api = YouTubeAPI(api_key)
        self.config = hunter_config or ChannelHunterConfig()
        self.thresholds = quality_thresholds or ChannelQualityThresholds()
        self.analyzer = ChannelAnalyzer()

    def run_discovery(
        self,
        progress_callback=None,
        max_searches: int = 20,
    ) -> ChannelDiscoveryResults:
        """
        Run full channel discovery pipeline.

        Args:
            progress_callback: Optional callback for progress updates
            max_searches: Maximum number of search terms to execute

        Returns:
            ChannelDiscoveryResults with all findings
        """
        logger.info("Starting channel discovery pipeline")
        start_time = time.time()

        results = ChannelDiscoveryResults()

        hunter = YouTubeChannelHunter(self.config)

        # Phase 1: Discover channels
        logger.info("Phase 1: Channel Discovery")
        discovered = hunter.discover_channels(progress_callback)

        for channel in discovered:
            if progress_callback:
                progress_callback(
                    0.5 + (0.3 * len(results.found_channels) / len(discovered)),
                    f"Evaluating {channel.channel_name}",
                )

            # Phase 2: Evaluate channel quality
            qualified = self._evaluate_full_channel(channel)
            results.add_channel(channel, qualified)

        results.time_elapsed_seconds = time.time() - start_time

        logger.info(
            f"Discovery complete: {len(results.qualified_channels)} qualified / "
            f"{len(results.found_channels)} found / {results.total_searched} searched"
        )

        return results

    def _evaluate_full_channel(self, channel: Channel) -> bool:
        """
        Perform full quality evaluation on a channel.

        Args:
            channel: Channel to evaluate

        Returns:
            True if channel passes quality thresholds
        """
        # Get channel details from API
        details = self.api.get_channel_details(channel.channel_id)
        if not details:
            logger.warning(f"Could not get details for channel {channel.channel_id}")
            return False

        # Update channel with API data
        channel.subscriber_count = details.get("subscriberCount", 0)
        channel.video_count = details.get("videoCount", 0)
        channel.created_date = details.get("publishedAt")
        channel.last_updated = details.get("publishedAt")

        # Get sample videos for analysis
        videos = self.api.get_channel_videos(channel.channel_id, max_results=10)
        if not videos:
            logger.warning(f"No videos found for channel {channel.channel_id}")
            return False

        # Extract features from videos
        video_titles = [v.get("title", "") for v in videos]
        video_descriptions = [v.get("description", "") for v in videos]
        video_tags = [v.get("tags", []) for v in videos]

        all_text = (
            f"{details.get('description', '')}\n\n"
            f"{' '.join(video_titles)}\n"
            f"{' '.join(video_descriptions)}"
        )

        # Analyze quality metrics
        metrics = QualityMetrics()
        for video in videos[:5]:  # Sample first 5 videos
            audio_id = video.get("audio_id")
            self._analyze_video(audio_id, metrics)

        channel.quality_metrics = metrics
        channel.quality_score = metrics.overall_score()

        # Classify into categories
        channel.categories = self.analyzer.classify_category(
            details.get("title", ""),
            details.get("description", ""),
            [tag for tags in video_tags for tag in tags],
        )

        # Detect language
        channel.primary_language = self.analyzer.detect_language(all_text)
        if len(channel.categories) > 0:
            for cat in channel.categories:
                if cat == ContentCategory.PROFESSIONAL_TRAINING:
                    channel.languages.add("en")
                    break

        # Verify professional credentials
        is_professional, credentials = self.analyzer.verify_professional(
            details.get("description", ""), channel.channel_id
        )
        channel.is_professional = is_professional
        channel.credentials = credentials

        # Extract licensing info
        channel.licensing = self.analyzer.extract_licensing_info(
            details.get("description", ""), video_descriptions
        )

        # Set status
        channel.status = ChannelStatus.ACTIVE
        channel.last_monitored = None  # No monitoring yet

        return self.thresholds.passes(metrics)

    def _analyze_video(self, video_id: str, metrics: QualityMetrics) -> None:
        """
        Analyze a single video for quality metrics.

        Downloads audio, analyzes quality, speech clarity, and production values.

        Args:
            video_id: YouTube video ID
            metrics: QualityMetrics object to update
        """
        import tempfile
        import os
        from pathlib import Path

        try:
            # Phase 1: Download audio for analysis
            audio_path = self._download_audio(video_id)
            if not audio_path:
                logger.warning(f"Could not download audio for video {video_id}")
                return

            try:
                # Phase 2: Analyze audio quality (sample rate, bitrate, noise)
                audio_quality = self._analyze_audio_quality(audio_path)
                metrics.production_quality = (
                    metrics.production_quality * 0.5 + audio_quality * 0.5
                )

                # Phase 3: Analyze speech clarity using Whisper
                speech_clarity = self._analyze_speech_clarity(audio_path)
                metrics.content_quality = max(metrics.content_quality, speech_clarity)

                # Phase 4: Evaluate production values
                production_score = self._evaluate_production_values(audio_path)
                metrics.production_quality = max(
                    metrics.production_quality, production_score
                )

                logger.info(
                    f"Video {video_id} analysis: audio={audio_quality:.2f}, "
                    f"speech={speech_clarity:.2f}, production={production_score:.2f}"
                )
            finally:
                # Cleanup downloaded file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        except Exception as e:
            logger.error(f"Error analyzing video {video_id}: {e}")

    def _download_audio(self, video_id: str) -> Optional[str]:
        """
        Download audio from YouTube video using yt-dlp.

        Args:
            video_id: YouTube video ID

        Returns:
            Path to downloaded audio file, or None on failure
        """
        import subprocess
        import tempfile

        try:
            # Create temp directory for audio
            temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
            output_path = os.path.join(temp_dir, f"{video_id}.wav")

            # Use yt-dlp to download audio
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",  # Best quality
                "-o",
                output_path,
                "--no-playlist",
                "--no-warnings",
                "--quiet",
                f"https://www.youtube.com/watch?v={video_id}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if result.returncode != 0:
                logger.warning(f"yt-dlp failed for {video_id}: {result.stderr}")
                return None

            # yt-dlp appends extension, find the actual file
            actual_path = output_path.replace(".wav", ".wav")
            if not os.path.exists(actual_path):
                # Try common extensions
                for ext in [".wav", ".m4a", ".webm"]:
                    candidate = output_path.replace(".wav", ext)
                    if os.path.exists(candidate):
                        actual_path = candidate
                        break

            if not os.path.exists(actual_path):
                logger.warning(f"Downloaded file not found for {video_id}")
                return None

            return actual_path

        except subprocess.TimeoutExpired:
            logger.warning(f"Download timeout for video {video_id}")
            return None
        except FileNotFoundError:
            logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            logger.error(f"Error downloading audio for {video_id}: {e}")
            return None

    def _analyze_audio_quality(self, audio_path: str) -> float:
        """
        Analyze technical audio quality.

        Measures sample rate, bit depth, noise floor, and dynamic range.

        Args:
            audio_path: Path to audio file

        Returns:
            Quality score 0.0-1.0
        """
        try:
            import librosa
            import numpy as np

            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)

            if len(y) == 0:
                return 0.0

            scores = []

            # 1. Sample rate quality (higher is better)
            sr_score = min(1.0, sr / 48000)  # 48kHz is excellent
            scores.append(sr_score)

            # 2. Signal-to-noise ratio estimation
            # Use quiet segments to estimate noise floor
            rms = librosa.feature.rms(y=y)[0]
            signal_rms = np.percentile(rms, 90)  # High energy segments
            noise_rms = np.percentile(rms, 10)  # Low energy segments

            if noise_rms > 0:
                snr = signal_rms / noise_rms
                snr_score = min(1.0, snr / 10)  # SNR of 10 is good
            else:
                snr_score = 1.0
            scores.append(snr_score)

            # 3. Dynamic range (avoid over-compression)
            dynamic_range = np.max(rms) - np.min(rms)
            range_score = min(1.0, dynamic_range * 5)  # Reasonable dynamic range
            scores.append(range_score)

            # 4. Clipping detection (penalize clipped audio)
            clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
            clipping_score = max(0, 1.0 - clipping_ratio * 10)
            scores.append(clipping_score)

            return float(np.mean(scores))

        except ImportError:
            logger.warning("librosa not installed, skipping audio quality analysis")
            return 0.5
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return 0.5

    def _analyze_speech_clarity(self, audio_path: str) -> float:
        """
        Analyze speech clarity using Whisper transcription.

        Transcribes audio and measures confidence/word clarity.

        Args:
            audio_path: Path to audio file

        Returns:
            Speech clarity score 0.0-1.0
        """
        try:
            from faster_whisper import WhisperModel

            # Use small model for faster processing
            model = WhisperModel("small", device="cpu", compute_type="int8")

            # Transcribe with word-level timestamps
            segments, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en",
            )

            if info.language_probability < 0.5:
                logger.warning(f"Low language confidence: {info.language_probability}")

            # Collect word-level confidences
            word_scores = []
            total_duration = 0.0
            speech_duration = 0.0

            for segment in segments:
                total_duration += segment.end - segment.start
                speech_duration += segment.end - segment.start

                if segment.words:
                    for word in segment.words:
                        # Whisper doesn't provide word confidence directly
                        # Use probability of the segment
                        word_scores.append(
                            segment.avg_logprob
                            if hasattr(segment, "avg_logprob")
                            else 0.5
                        )

            if not word_scores:
                return 0.5

            # Calculate scores
            avg_confidence = sum(word_scores) / len(word_scores)

            # Map log probabilities to 0-1 range (typical range -1 to 0)
            confidence_score = max(0, min(1, 1 + avg_confidence))

            # Speech ratio (how much of the audio is speech)
            if total_duration > 0:
                speech_ratio = speech_duration / total_duration
            else:
                speech_ratio = 0.5

            # Combine scores
            return 0.7 * confidence_score + 0.3 * speech_ratio

        except ImportError:
            logger.warning("faster-whisper not installed, using fallback")
            return self._fallback_speech_clarity(audio_path)
        except Exception as e:
            logger.error(f"Error analyzing speech clarity: {e}")
            return 0.5

    def _fallback_speech_clarity(self, audio_path: str) -> float:
        """
        Fallback speech clarity estimation without Whisper.

        Uses energy in speech frequency bands.

        Args:
            audio_path: Path to audio file

        Returns:
            Estimated speech clarity score 0.0-1.0
        """
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(audio_path, sr=None)

            # Speech is typically in 300-3400 Hz range
            # Use spectral centroid as a proxy for speech clarity
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            mean_cent = np.mean(cent)

            # Good speech typically has centroid around 1000-2000 Hz
            if mean_cent < 500 or mean_cent > 5000:
                cent_score = 0.3
            elif 1000 <= mean_cent <= 3000:
                cent_score = 1.0
            else:
                cent_score = 0.6

            # Spectral rolloff (measure of high frequency content)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mean_rolloff = np.mean(rolloff)

            # Good speech has rolloff around 3000-6000 Hz
            if mean_rolloff < 2000:
                rolloff_score = 0.4
            elif 3000 <= mean_rolloff <= 6000:
                rolloff_score = 1.0
            else:
                rolloff_score = 0.6

            return (cent_score + rolloff_score) / 2

        except Exception as e:
            logger.error(f"Error in fallback speech analysis: {e}")
            return 0.5

    def _evaluate_production_values(self, audio_path: str) -> float:
        """
        Evaluate production quality of audio.

        Considers consistency, editing quality, and professional markers.

        Args:
            audio_path: Path to audio file

        Returns:
            Production value score 0.0-1.0
        """
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(audio_path, sr=None)

            if len(y) == 0:
                return 0.0

            scores = []

            # 1. Volume consistency (professional audio is well-mastered)
            rms = librosa.feature.rms(y=y)[0]
            volume_variance = np.var(rms)
            consistency_score = max(0, 1.0 - volume_variance * 100)
            scores.append(consistency_score)

            # 2. Silence detection (too much silence = poor editing)
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)

            if silence_ratio < 0.1:
                silence_score = 1.0
            elif silence_ratio < 0.3:
                silence_score = 0.7
            elif silence_ratio < 0.5:
                silence_score = 0.4
            else:
                silence_score = 0.2
            scores.append(silence_score)

            # 3. Spectral flatness (measure of audio "richness")
            flatness = librosa.feature.spectral_flatness(y=y)
            mean_flatness = np.mean(flatness)

            # Lower flatness = more tonal/rich audio (good for speech)
            if mean_flatness < 0.1:
                richness_score = 1.0
            elif mean_flatness < 0.3:
                richness_score = 0.7
            else:
                richness_score = 0.4
            scores.append(richness_score)

            # 4. Duration check (very short or very long may indicate issues)
            duration = len(y) / sr
            if 60 <= duration <= 1800:  # 1-30 minutes is typical
                duration_score = 1.0
            elif duration < 30 or duration > 3600:
                duration_score = 0.5
            else:
                duration_score = 0.8
            scores.append(duration_score)

            return float(np.mean(scores))

        except ImportError:
            logger.warning("librosa not installed for production analysis")
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating production values: {e}")
            return 0.5

    def generate_report(self, results: ChannelDiscoveryResults) -> str:
        """
        Generate a comprehensive discovery report.

        Args:
            results: ChannelDiscoveryResults

        Returns:
            Formatted markdown report
        """
        report = f"""# YouTube Channel Discovery Report

## Summary

- **Total Searched:** {results.total_searched} search terms
- **Channels Found:** {len(results.found_channels)}
- **Qualified:** {len(results.qualified_channels)}
- **Rejected:** {len(results.rejected_channels)}
- **Professional Sources:** {results.professional_count}

## Content Statistics

**Languages:**
"""

        for lang, count in sorted(results.languages.items(), key=lambda x: -x[1]):
            report += f"- {lang}: {count} channels\n"

        report += "\n**Categories:**\n"
        for cat, count in sorted(results.categories.items(), key=lambda x: -x[1]):
            report += f"- {cat}: {count} channels\n"

        report += f"""
**Channel Metrics:**
- Total Subscribers: {results.total_subscribers:,}
- Total Videos: {results.total_videos:,}

## Qualified Channels

| # | Channel | Subscribers | Videos | Quality Score | Language | Categories |
|---|---------|-------------|--------|---------------|----------|------------|
"""

        for i, channel in enumerate(results.qualified_channels[:50]):
            cats = ", ".join([c.value for c in channel.categories])
            report += (
                f"| {i + 1} | {channel.channel_name} | {channel.subscriber_count:,} | "
                f"{channel.video_count:,} | {channel.quality_score:.2f} | "
                f"{channel.primary_language} | {cats} |\n"
            )

        return report

    def export_channels(self, results: ChannelDiscoveryResults, output_path: str):
        """
        Export qualified channels to JSON.

        Args:
            results: ChannelDiscoveryResults
            output_path: Path to output JSON file
        """
        import json

        # Export qualified channels to JSON
        # Using list comprehension for better performance
        channels_data = [
            {
                "channel_id": channel.channel_id,
                "channel_name": channel.channel_name,
                "channel_url": channel.channel_url,
                "subscriber_count": channel.subscriber_count,
                "video_count": channel.video_count,
                "total_views": channel.total_views,
                "languages": list(channel.languages),
                "primary_language": channel.primary_language,
                "categories": [c.value for c in channel.categories],
                "quality_score": channel.quality_score,
                "is_professional": channel.is_professional,
                "credentials": channel.credentials,
                "organization": channel.organization,
                "licensing": (
                    {
                        "cc_license": (
                            channel.licensing.cc_license if channel.licensing else False
                        ),
                        "cc_type": (
                            channel.licensing.cc_type if channel.licensing else None
                        ),
                        "commercial_use": (
                            channel.licensing.commercial_use
                            if channel.licensing
                            else False
                        ),
                    }
                    if channel.licensing
                    else None
                ),
                "status": channel.status.value,
            }
            for channel in results.qualified_channels
        ]

        with open(output_path, "w") as f:
            json.dump(channels_data, f, indent=2)

        logger.info(f"Exported {len(channels_data)} channels to {output_path}")


def run_pipeline(
    api_key: str,
    target_channels: int = 50,
    output_path: str = "qualified_channels.json",
    progress_callback=None,
) -> tuple[ChannelDiscoveryResults, str]:
    """
    Convenience function to run the full pipeline.

    Args:
        api_key: YouTube Data API key
        target_channels: Number of channels to find
        output_path: Path for JSON export
        progress_callback: Optional progress callback

    Returns:
        Tuple of (results, markdown_report)
    """
    config = ChannelHunterConfig(target_channels=target_channels)
    processor = ChannelProcessor(api_key, config)

    results = processor.run_discovery(progress_callback)
    report = processor.generate_report(results)

    if results.qualified_channels:
        processor.export_channels(results, output_path)

    return results, report
