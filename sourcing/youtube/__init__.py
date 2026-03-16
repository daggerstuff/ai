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

from ai.sourcing.youtube.models import (
    Channel,
    ChannelQualityThresholds,
    QualityMetrics,
    LicensingInfo,
    ChannelStatus,
    ContentCategory,
)
from ai.sourcing.youtube.api import (
    YouTubeChannelHunter,
    YouTubeAPI,
    ChannelAnalyzer,
)
from ai.sourcing.youtube.monitoring import (
    ChannelMonitor,
    HealthCheck,
    AlertCondition,
)

__all__ = [
    "Channel",
    "ChannelQuality",
    "QualityMetrics",
    "LicensingInfo",
    "ChannelStatus",
    "YouTubeChannelHunter",
    "YouTubeAPI",
    "ChannelAnalyzer",
    "ChannelMonitor",
    "HealthCheck",
    "AlertCondition",
]
