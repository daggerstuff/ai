"""
Data models for YouTube channel curation.

Defines channel metadata, quality scoring, licensing info, and status tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ChannelStatus(StrEnum):
    """Health status of a channel."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    AT_RISK = "at_risk"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class ContentCategory(StrEnum):
    """Therapeutic content categories."""

    CPTSD_EDUCATION = "cptsd_education"
    TRAUMA_INFORMED = "trauma_informed"
    DBT_SKILLS = "dbt_skills"
    CBT_TECHNIQUES = "cbt_techniques"
    SOMATIC_THERAPY = "somatic_therapy"
    EMDR_THERAPY = "emdr_therapy"
    MINDFULNESS = "mindfulness"
    CR_SUPPORT = "cr_support"
    PROFESSIONAL_TRAINING = "professional_training"
    PATIENT_STORIES = "patient_stories"
    RECOVERY_JOURNEY = "recovery_journey"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a channel."""

    content_quality: float = 0.0  # 0.0-1.0, educational value
    clinical_accuracy: float = 0.0  # 0.0-1.0, evidence-based
    production_quality: float = 0.0  # 0.0-1.0, audio/video
    engagement_quality: float = 0.0  # 0.0-1.0, interaction
    credibility_score: float = 0.0  # 0.0-1.0, credentials
    consistency_score: float = 0.0  # 0.0-1.0, posting schedule

    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "content_quality": 0.25,
            "clinical_accuracy": 0.30,
            "production_quality": 0.15,
            "engagement_quality": 0.10,
            "credibility_score": 0.15,
            "consistency_score": 0.05,
        }
        return sum(getattr(self, metric) * weights[metric] for metric in weights)


@dataclass
class LicensingInfo:
    """Licensing and copyright information."""

    cc_license: bool = False
    cc_type: str | None = None  # BY, BY-SA, BY-NC, etc.
    commercial_use: bool = False
    attribution_required: bool = True
    modification_allowed: bool = False
    share_alike: bool = False
    notes: str | None = None
    verified_date: datetime | None = None


@dataclass
class Channel:
    """YouTube channel representation with all metadata."""

    channel_id: str
    channel_name: str
    channel_url: str
    subscriber_count: int = 0
    video_count: int = 0
    total_views: int = 0
    created_date: datetime | None = None
    last_updated: datetime | None = None

    # Content information
    categories: list[ContentCategory] = field(default_factory=list)
    primary_language: str = "en"
    languages: set[str] = field(default_factory=set)
    description: str | None = None

    # Quality assessment
    quality_score: float = 0.0  # 0.0-1.0 overall
    quality_metrics: QualityMetrics | None = None

    # Professional credentials
    is_professional: bool = False
    credentials: list[str] = field(default_factory=list)  # LCSW, PhD, LMFT, etc.
    organization: str | None = None  # Clinic, hospital, university
    verified_professional: bool = False

    # Licensing
    licensing: LicensingInfo | None = None

    # Monitoring
    status: ChannelStatus = ChannelStatus.UNKNOWN
    health_score: float = 0.0  # 0.0-1.0 channel health
    last_monitored: datetime | None = None
    alert_notes: list[str] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    source: str = "manual"  # manual, api_search, recommendation

    def min_quality_score(self, minimum: float = 0.8) -> bool:
        """Check if channel meets minimum quality threshold."""
        return self.quality_score >= minimum

    def is_licensed_for_use(self, commercial: bool = False) -> bool:
        """Check if channel content can be used."""
        if not self.licensing:
            return False
        return not (commercial and not self.licensing.commercial_use)


class ChannelQualityThresholds(BaseModel):
    """Quality thresholds for channel acceptance."""

    content_quality: float = Field(ge=0.0, le=1.0, default=0.75)
    clinical_accuracy: float = Field(ge=0.0, le=1.0, default=0.80)
    production_quality: float = Field(ge=0.0, le=1.0, default=0.70)
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.75)
    overall_minimum: float = Field(ge=0.0, le=1.0, default=0.80)

    def passes(self, metrics: QualityMetrics) -> bool:
        """Check if metrics meet all thresholds."""
        return (
            metrics.content_quality >= self.content_quality
            and metrics.clinical_accuracy >= self.clinical_accuracy
            and metrics.production_quality >= self.production_quality
            and metrics.credibility_score >= self.credibility_score
            and metrics.overall_score() >= self.overall_minimum
        )


class ChannelRegistry(BaseModel):
    """Registry of accepted channels."""

    total_channels: int = 0
    accepted_channels: int = 0
    pending_review: int = 0
    rejected_channels: int = 0

    languages: dict[str, int] = Field(default_factory=dict)
    categories: dict[str, int] = Field(default_factory=dict)

    total_subscribers: int = 0
    total_videos: int = 0

    def add_channel(self, channel: Channel):
        """Register a new channel."""
        self.total_channels += 1

        if channel.status == ChannelStatus.ACTIVE:
            self.accepted_channels += 1
            self.total_subscribers += channel.subscriber_count
            self.total_videos += channel.video_count

            # Track languages
            for lang in channel.languages:
                self.languages[lang] = self.languages.get(lang, 0) + 1

            # Track categories
            for cat in channel.categories:
                self.categories[cat.value] = self.categories.get(cat.value, 0) + 1

        elif channel.status == ChannelStatus.UNKNOWN:
            self.pending_review += 1
        else:
            self.rejected_channels += 1

    def summary(self) -> str:
        """Generate summary statistics."""
        return f"""
        Channel Registry Summary:
        - Total discovered: {self.total_channels}
        - Accepted: {self.accepted_channels}
        - Pending review: {self.pending_review}
        - Rejected: {self.rejected_channels}
        - Languages: {len(self.languages)}
        - Categories: {len(self.categories)}
        - Total subscribers: {self.total_subscribers:,}
        - Total videos: {self.total_videos:,}
        """
