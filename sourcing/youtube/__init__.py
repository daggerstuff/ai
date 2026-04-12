"""
YouTube Channel Curation System for CPTSD Dataset

Agent 1 (PIX-28): YouTube Source Expansion
Focus: Curate 50+ high-quality therapeutic YouTube channels

Components:
- Channel discovery and curation
- Quality scoring (0.0-1.0 scale)
- Licensing verification
- Automated monitoring
"""

__version__ = "1.0.0"

from ai.sourcing.youtube.api import (
    ChannelAnalyzer,
    YouTubeAPI,
    YouTubeChannelHunter,
)
from ai.sourcing.youtube.models import (
    Channel,
    ChannelQualityThresholds,
    ChannelStatus,
    ContentCategory,
    LicensingInfo,
    QualityMetrics,
)
from ai.sourcing.youtube.monitoring import (
    AlertCondition,
    ChannelMonitor,
    HealthCheck,
)

__all__ = [
    "AlertCondition",
    "Channel",
    "ChannelAnalyzer",
    "ChannelMonitor",
    "ChannelQuality",
    "ChannelStatus",
    "HealthCheck",
    "LicensingInfo",
    "QualityMetrics",
    "YouTubeAPI",
    "YouTubeChannelHunter",
]
