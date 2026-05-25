"""
Video quality analyzer for YouTube content.

Provides functions for:
- Audio quality assessment (sample rates, formats)
- Production value (清晰度)
- Content length analysis
- Metadata completeness
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class AudioFormat(StrEnum):
    """Audio format types."""

    MP3 = "mp3"
    WAV = "wav"
    AAC = "aac"
    OGG = "ogg"
    OPUS = "opus"
    WEBM = "webm"
    FLAC = "flac"
    M4A = "m4a"


@dataclass
class AudioQualityMetrics:
    """Quality metrics for audio analysis."""

    sample_rate: int | None = None  # e.g., 48000, 44100
    bitrate: int | None = None  # e.g., 128000 bps
    format: AudioFormat | None = None
    has_speech: bool = False
    has_music: bool = False
    noise_level: float = 0.0  # 0.0-1.0
    clarity_score: float = 0.0  # 0.0-1.0, how clear the speech is
    volume_consistency: float = 0.0  # 0.0-1.0
    has_timestamps: bool = False
    duration_range_hours: float = 0.0

    def overall_score(self) -> float:
        """Calculate overall audio quality score."""
        score = 0.0

        # Sample rate scoring
        if self.sample_rate:
            if self.sample_rate >= 48000:
                score += 0.3
            elif self.sample_rate >= 44100:
                score += 0.25
            elif self.sample_rate >= 16000:
                score += 0.2
            elif self.sample_rate >= 8000:
                score += 0.15

        # Quality scoring
        score += self.clarity_score * 0.3

        # Volume consistency
        score += self.volume_consistency * 0.2

        # Format preference (lossless formats get bonus)
        if self.format in [AudioFormat.FLAC, AudioFormat.OPUS, AudioFormat.MP3]:
            score += 0.2

        return min(score, 1.0)


@dataclass
class VideoContentMetrics:
    """Content analysis metrics."""

    title_length: int | None = None
    description_length: int | None = None
    tags_count: int = 0
    has_closed_captions: bool = False
    has_description: bool = False
    word_count_estimate: int | None = None  # Estimated word count
    speaking_proportion: float = 0.0  # Percentage that is speaking (vs. music, etc.)

    def content_quality_score(self) -> float:
        """Calculate content quality score (0.0-1.0)."""
        score = 0.0

        # Description presence indicates better documentation
        if self.has_description:
            score += 0.3

        # Tag count suggests well-organized content
        score += min(self.tags_count / 10, 0.3)

        # Reasonable length descriptions are valuable
        if self.description_length:
            if 50 <= self.description_length <= 500:
                score += 0.2
            elif 501 <= self.description_length <= 2000:
                score += 0.4

        return min(score, 1.0)


@dataclass
class ChannelEngagementMetrics:
    """Engagement metrics for a channel."""

    average_view_count: float = 0.0
    average_like_count: float = 0.0
    average_comment_count: float = 0.0
    comment_to_view_ratio: float = 0.0
    subscriber_growth_rate: float = 0.0  # Monthly growth rate
    video_upload_frequency_days: float = 0.0

    def engagement_score(self) -> float:
        """Calculate engagement quality score (0.0-1.0)."""
        score = 0.0

        # Video performance
        score += min(self.average_view_count / 1000, 0.3)

        # Comment engagement
        if self.comment_to_view_ratio > 0.005:  # 0.5% ratio
            score += 0.2
        if self.comment_to_view_ratio > 0.01:  # 1% ratio
            score += 0.3

        # Consistency (engagement that doesn't fluctuates)
        score += 0.2

        # Recent growth
        if self.subscriber_growth_rate > 0.05:  # +5% growth monthly
            score += 0.1
        elif self.subscriber_growth_rate > 0:  # Any positive growth
            score += 0.05

        # Regular uploads
        if 0 < self.video_upload_frequency_days <= 7:  # At least weekly
            score += 0.1

        return min(score, 1.0)


@dataclass
class ProfessionalIndicators:
    """Indicators of professional status."""

    verified_credentials: bool = False
    organization_affiliation: bool = False
    educational_institution: bool = False
    peer_review_publications: bool = False
    citations_in_literature: bool = False
    conference_presentations: bool = False
    social_media_presence: int = 0  # Number of platforms
    has_certifications: bool = False

    def professional_score(self) -> float:
        """Calculate professional credibility score (0.0-1.0)."""
        score = 0.0

        # Verified credentials (highest weight)
        if self.verified_credentials:
            score += 0.4

        # Educational institution
        if self.educational_institution:
            score += 0.3

        # Peer reviewed
        if self.peer_review_publications:
            score += 0.2

        # Academic citations
        if self.citations_in_literature:
            score += 0.15

        # Organizational affiliation
        if self.organization_affiliation:
            score += 0.15

        # Social media presence (multiple platforms suggests reach)
        if self.social_media_presence >= 3:
            score += 0.1

        # Certifications
        if self.has_certifications:
            score += 0.1

        return min(score, 1.0)


def analyze_audio_from_metadata(
    duration_seconds: int,
    bitrate: int,
    format_name: str,
    has_closed_captions: bool = False,
) -> AudioQualityMetrics:
    """Analyze audio quality from metadata alone."""
    metrics = AudioQualityMetrics()

    metrics.duration_range_hours = duration_seconds / 3600
    metrics.has_timestamps = True
    metrics.has_speech = True
    metrics.format = AudioFormat(format_name)

    # Sample rate estimation from bitrate
    if bitrate > 0 and duration_seconds > 0:
        estimated_sample_rate = int(bitrate * 8 / duration_seconds)
        quality = AudioQualityMetrics()

        if 44100 <= estimated_sample_rate <= 48000:
            quality.sample_rate = 48000
        elif 16000 <= estimated_sample_rate <= 44100:
            quality.sample_rate = 44100
        elif 8000 <= estimated_sample_rate <= 16000:
            quality.sample_rate = 8000
        else:
            quality.sample_rate = 8000

        metrics.sample_rate = quality.sample_rate

    # Format preferences
    format_preferences = [
        AudioFormat.FLAC,
        AudioFormat.OPUS,
        AudioFormat.MP3,
        AudioFormat.WEBM,
    ]

    if format_name in format_preferences:
        metrics.format = AudioFormat(format_name)
    else:
        metrics.format = AudioFormat.MP3

    metrics.has_closed_captions = has_closed_captions

    return metrics


def extract_channel_engagement(
    views: int,
    likes: int,
    comments: int,
    subscriber_count: int,
    age_months: float,
) -> ChannelEngagementMetrics:
    """Extract engagement metrics from raw numbers."""
    metrics = ChannelEngagementMetrics()

    # Video performance
    if subscriber_count > 0:
        metrics.average_view_count = views / subscriber_count
        metrics.average_like_count = likes / subscriber_count
        metrics.average_comment_count = comments / subscriber_count

    # Comment to view ratio
    if views > 0:
        metrics.comment_to_view_ratio = comments / views
        metrics.comment_to_view_ratio = min(metrics.comment_to_view_ratio, 0.5)  # Cap at 50%

    # Estimate growth rate (simplified)
    metrics.subscriber_growth_rate = min(subscriber_count / (1000 * (age_months / 12)), 1.0)

    # Upload frequency estimation
    if views > 0 and age_months > 0:
        estimated_videos_per_month = views / 7 / 30
        metrics.video_upload_frequency_days = estimated_videos_per_month / 10

    return metrics


def detect_closed_captions(description: str, tags: list[str]) -> bool:
    """Detect if content has optional closed captions."""
    text = f"{description} {' '.join(tags)}".lower()

    closed_captions_keywords = [
        "closed captions",
        "cc",
        "auto-generated",
        "auto generated",
        "machine translated",
        "ai generated",
    ]

    return any(keyword in text for keyword in closed_captions_keywords)


def detect_transcript_features(description: str) -> dict:
    """Detect transcript features from description."""
    features = {
        "has_timestamps": False,
        "has_speaker_diarization": False,
        "has_emotion_markers": False,
        "has_scene_descriptions": False,
        "has_music_licensing": False,
        "language_detected": None,
    }

    text = description.lower()

    # Timestamps
    time_keywords = ["timestamp", "time stamp", "timecode", "time:ms"]
    features["has_timestamps"] = any(kw in text for kw in time_keywords)

    # Speaker diarization
    diarization_keywords = [
        "speaker",
        "diarization",
        "speaker label",
        "speaker:",
        "person:",
        "person 1",
        "person 2",
        "interview",
    ]
    features["has_speaker_diarization"] = any(kw in text for kw in diarization_keywords)

    # Emotion markers
    emotion_keywords = [
        "emotion",
        "emotional tone",
        "emotional response",
        "tone analysis",
        "sentiment analysis",
    ]
    features["has_emotion_markers"] = any(kw in text for kw in emotion_keywords)

    # Scene descriptions
    features["has_scene_descriptions"] = len(text.split(":")) > 10

    # Language detection (simplified)
    language_indicators = {
        "español": "es",
        "español:": "es",
        "french": "fr",
        "français": "fr",
        "deutsch": "de",
        "alemán": "de",
        "português": "pt",
        "българскии": "bg",
        "русский": "ru",
        "中文": "zh",
    }

    detected_lang = None
    for term in language_indicators:
        if term in text:
            detected_lang = language_indicators[term]
            break

    features["language_detected"] = detected_lang

    return features


def analyze_video_content(
    title: str,
    description: str,
    tags: list[str],
    duration_seconds: int,
) -> VideoContentMetrics:
    """Analyze video content for quality."""
    metrics = VideoContentMetrics()

    metrics.title_length = len(title)
    metrics.description_length = len(description)
    metrics.tags_count = len(tags)

    # Detect closed captions
    metrics.has_closed_captions = detect_closed_captions(description, tags)

    # Estimate word count
    words = len(description.split()) + len(title.split())
    metrics.word_count_estimate = words + (words // 2)  # Assume avg word length 2

    # Estimate speaking proportion (simplified)
    if title and description:
        speaking_duration = min(duration_seconds / 3, duration_seconds * 2.0)
        words_per_minute = (words * 60) / speaking_duration
        metrics.speaking_proportion = words_per_minute / 1000  # ~100 wpm is normal

    metrics.has_description = len(description) > 0

    return metrics


def analyze_channel_professional(
    description: str, credentials: list[str], _channel_id: str, channel_url: str
) -> ProfessionalIndicators:
    """Analyze professional indicators."""
    indicators = ProfessionalIndicators()

    text = description.lower()

    # Check for credentials
    credentials_found = []
    for cred in credentials:
        cred_lower = cred.lower()
        if cred_lower in text or cred.replace(" ", " ") in text:
            credentials_found.append(cred)

    indicators.verified_credentials = len(credentials_found) > 0

    # Check for educational institution indicators
    edu_keywords = [
        "university",
        "university of",
        "college",
        "institute",
        "school of",
        "department of",
        "lab",
        "research",
    ]
    indicators.educational_institution = any(kw in text for kw in edu_keywords)

    # Check for peer-reviewed publications
    review_keywords = [
        "peer reviewed",
        "peer-reviewed",
        "journal publication",
        "conference presentation",
        "conference paper",
        "proceedings",
        "citation:",
        "cited by",
        "references",
    ]
    indicators.peer_review_publications = any(kw in text for kw in review_keywords)

    # Check for citations
    indicators.citations_in_literature = "cited by:" in text

    # Check organization affiliation
    org_keywords = [
        "university of",
        "university at",
        "hospital",
        "medical center",
        "clinic",
        "institute",
        "foundation",
        "association",
        "research lab",
        "department of psychology",
        "center for",
    ]
    indicators.organization_affiliation = any(kw in text for kw in org_keywords)

    # Check for certifications
    indicators.has_certifications = "certified" in text or "license" in text

    # Check social media presence (from URLs)
    social_domains = ["youtube", "instagram", "tiktok", "twitter"]
    # Very simplified check
    domains_in_url = [d for d in social_domains if d in channel_url]
    indicators.social_media_presence = len(domains_in_url) > 0

    # Additional professional indicators not yet implemented
    # - peer_review_publications
    # - citations_in_literature (requires additional context)
    # - conference_presentations

    return indicators
