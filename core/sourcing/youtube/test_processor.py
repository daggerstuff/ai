"""
Tests for YouTube processor video analysis methods.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import wave
import struct
import numpy as np

# Import directly to avoid circular import
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from core.sourcing.youtube.models import QualityMetrics
from core.sourcing.youtube.processor import ChannelProcessor


class TestVideoAnalysis:
    """Test video analysis methods."""

    @pytest.fixture
    def processor(self):
        """Create a ChannelProcessor instance."""
        return ChannelProcessor(api_key="test-api-key")

    @pytest.fixture
    def sample_audio(self, tmp_path):
        """Create a sample audio file for testing."""
        audio_path = tmp_path / "test_audio.wav"

        # Create a simple WAV file with sine wave
        sample_rate = 44100
        duration = 1.0
        frequency = 440

        n_samples = int(sample_rate * duration)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            sample = int(32767 * 0.5 * np.sin(2 * np.pi * frequency * t))
            samples.append(struct.pack("<h", sample))

        with wave.open(str(audio_path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(samples))

        return str(audio_path)

    def test_analyze_audio_quality(self, processor, sample_audio):
        """Test audio quality analysis."""
        score = processor._analyze_audio_quality(sample_audio)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Good audio should score reasonably
        assert score > 0.3

    def test_evaluate_production_values(self, processor, sample_audio):
        """Test production value scoring."""
        score = processor._evaluate_production_values(sample_audio)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fallback_speech_clarity(self, processor, sample_audio):
        """Test fallback speech clarity estimation."""
        score = processor._fallback_speech_clarity(sample_audio)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_analyze_video_updates_metrics(self, processor):
        """Test that _analyze_video updates QualityMetrics."""
        metrics = QualityMetrics()

        with patch.object(processor, "_download_audio", return_value=None):
            # Should handle None gracefully
            processor._analyze_video("test_video_id", metrics)
            # Metrics should remain unchanged
            assert metrics.production_quality == 0.0

    def test_download_audio_missing_yt_dlp(self, processor):
        """Test graceful handling when yt-dlp is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError("yt-dlp not found")):
            result = processor._download_audio("test_video_id")
            assert result is None

    def test_download_audio_timeout(self, processor):
        """Test handling of download timeout."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=120),
        ):
            result = processor._download_audio("test_video_id")
            assert result is None

    def test_analyze_video_full_flow(self, processor, sample_audio):
        """Test full video analysis flow with mocked audio."""
        metrics = QualityMetrics()

        with patch.object(processor, "_download_audio", return_value=sample_audio):
            processor._analyze_video("test_video_id", metrics)

            # Metrics should be updated
            assert metrics.production_quality > 0.0 or metrics.content_quality > 0.0


class TestQualityMetricsIntegration:
    """Test integration with QualityMetrics model."""

    def test_metrics_weights(self):
        """Test that QualityMetrics weights are correct."""
        metrics = QualityMetrics(
            content_quality=1.0,
            clinical_accuracy=1.0,
            production_quality=1.0,
            engagement_quality=1.0,
            credibility_score=1.0,
            consistency_score=1.0,
        )

        score = metrics.overall_score()
        assert score == 1.0

    def test_metrics_weighted_calculation(self):
        """Test weighted calculation matches expected weights."""
        # Clinical accuracy has highest weight (0.30)
        metrics = QualityMetrics(
            clinical_accuracy=1.0,
            content_quality=0.0,
            production_quality=0.0,
            engagement_quality=0.0,
            credibility_score=0.0,
            consistency_score=0.0,
        )

        score = metrics.overall_score()
        assert score == pytest.approx(0.30, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
