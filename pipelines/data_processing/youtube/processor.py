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

import json
import logging
import time
from datetime import datetime

from ai.pipelines.data_processing.youtube.api import (
    ChannelAnalyzer,
    ChannelHunterConfig,
    YouTubeAPI,
    YouTubeChannelHunter,
)
from ai.pipelines.data_processing.youtube.models import (
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
        self.found_channels: list[Channel] = []
        self.qualified_channels: list[Channel] = []
        self.rejected_channels: list[Channel] = []
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
        hunter_config: ChannelHunterConfig | None = None,
        quality_thresholds: ChannelQualityThresholds | None = None,
    ):
        self.api = YouTubeAPI(api_key)
        self.config = hunter_config or ChannelHunterConfig()
        self.thresholds = quality_thresholds or ChannelQualityThresholds(
            content_quality=0.55,
            clinical_accuracy=0.50,
            production_quality=0.40,
            credibility_score=0.05,
            overall_minimum=0.40,
        )
        self.analyzer = ChannelAnalyzer()

    def run_discovery(
        self,
        progress_callback=None,
        _max_searches: int = 20,
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
        start_time = time.perf_counter()

        results = ChannelDiscoveryResults()

        hunter = YouTubeChannelHunter(self.config, api_key=self.api.api_key)

        # Phase 1: Discover channels
        logger.info("Phase 1: Channel Discovery")
        discovered = hunter.discover_channels(progress_callback)
        if _max_searches > 0:
            discovered = discovered[:_max_searches]

        for channel in discovered:
            if progress_callback:
                total = max(1, len(discovered))
                progress_callback(
                    0.5 + (0.3 * len(results.found_channels) / total),
                    f"Evaluating {channel.channel_name}",
                )

            # Phase 2: Evaluate channel quality
            qualified = self._evaluate_full_channel(channel)
            results.add_channel(channel, qualified)

        results.time_elapsed_seconds = max(0.0, time.perf_counter() - start_time)

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
        channel.subscriber_count = self._to_int(details.get("subscriberCount", 0))
        channel.video_count = self._to_int(details.get("videoCount", 0))
        channel.total_views = self._to_int(details.get("viewCount", 0))
        channel.created_date = self._parse_iso8601(details.get("publishedAt"))
        channel.last_updated = self._parse_iso8601(details.get("publishedAt"))

        # InnerTube returns 0 for unavailable stats; skip checks in that case
        if channel.subscriber_count > 0 and channel.subscriber_count < self.config.min_subscribers:
            return False
        if channel.video_count > 0 and channel.video_count < self.config.min_videos:
            return False

        # Get sample videos for analysis
        videos = self.api.get_channel_videos(channel.channel_id, max_results=10)
        if not videos:
            logger.warning(f"No videos found for channel {channel.channel_id}")
            return False

        # Extract features from videos
        video_titles = [v.get("snippet", {}).get("title", v.get("title", "")) for v in videos]
        video_descriptions = [v.get("snippet", {}).get("description", v.get("description", "")) for v in videos]
        video_tags = [v.get("snippet", {}).get("tags", v.get("tags", [])) for v in videos]

        all_text = f"{details.get('description', '')}\n\n{' '.join(video_titles)}\n{' '.join(video_descriptions)}"

        # Analyze quality metrics
        metrics = QualityMetrics()
        for video in videos[:5]:  # Sample first 5 videos
            self._analyze_video(video, metrics)

        channel.quality_metrics = metrics
        channel.quality_score = metrics.overall_score()

        # Classify into categories
        channel.categories = self.analyzer.classify_category(
            details.get("channelTitle", details.get("title", "")),
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
        channel.licensing = self.analyzer.extract_licensing_info(details.get("description", ""), video_descriptions)

        # Set status
        channel.status = ChannelStatus.ACTIVE
        channel.last_monitored = None  # No monitoring yet

        return self.thresholds.passes(metrics)

    def _analyze_video(self, video: dict, metrics: QualityMetrics):
        """
        Analyze a single video for quality metrics.

        Args:
            video: Raw video payload from YouTube API
            metrics: QualityMetrics object to update
        """
        if not video:
            return

        snippet = video.get("snippet", {})
        title = str(snippet.get("title", video.get("title", "")))
        description = str(snippet.get("description", video.get("description", "")))
        tags = list(snippet.get("tags", video.get("tags", [])) or [])
        stats = video.get("statistics", {})

        text = (title + " " + description).strip().lower()
        text_tags = " ".join(str(t).lower() for t in tags)

        # Content quality: documentation depth and clinical topic coverage.
        content_quality = 0.45
        if len(description) >= 80:
            content_quality += 0.25
        if any(term in text for term in ("therapy", "trauma", "cptsd", "dbt", "cbt")):
            content_quality += 0.2
        if "mindfulness" in text or "grounding" in text:
            content_quality += 0.1

        # Clinical relevance indicator for this clip.
        clinical_accuracy = 0.55
        if any(term in text_tags for term in ("therapeutic", "clinical", "trauma")):
            clinical_accuracy += 0.2
        if any(term in text for term in ("evidence", "treatment", "intervention", "session")):
            clinical_accuracy += 0.15

        # Production quality from title/description metadata confidence.
        production_quality = 0.5
        if snippet.get("contentDetails") or video.get("contentDetails"):
            production_quality += 0.1
        if any(term in text for term in ("guide", "explainer", "workshop", "session", "exercise")):
            production_quality += 0.1
        if any(kw in text for kw in ("hd", "full", "1080p")):
            production_quality += 0.1

        likes = self._to_int(stats.get("likeCount", 0))
        comments = self._to_int(stats.get("commentCount", 0))
        views = self._to_int(stats.get("viewCount", 0))
        ratio = (likes + comments) / max(1, views)
        engagement_quality = min(1.0, ratio + 0.2)

        sample_count = getattr(metrics, "_sample_count", 0)
        metrics.content_quality = (metrics.content_quality * sample_count + min(1.0, content_quality)) / (
            sample_count + 1
        )
        metrics.clinical_accuracy = (metrics.clinical_accuracy * sample_count + min(1.0, clinical_accuracy)) / (
            sample_count + 1
        )
        metrics.production_quality = (metrics.production_quality * sample_count + min(1.0, production_quality)) / (
            sample_count + 1
        )
        metrics.engagement_quality = (metrics.engagement_quality * sample_count + min(1.0, engagement_quality)) / (
            sample_count + 1
        )
        metrics.consistency_score = min(1.0, max(0.2, 0.35 + 0.65 * (sample_count + 1) / 5))
        metrics.credibility_score = min(
            1.0, metrics.credibility_score + 0.02 + (0.05 if any(len(tag) > 8 for tag in tags) else 0)
        )
        metrics._sample_count = sample_count + 1

    def _to_int(self, value) -> int:
        """Coerce unknown input types into a non-negative integer."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _parse_iso8601(self, value):
        """Parse YouTube datetime values into datetime objects when possible."""
        if value is None:
            return None
        if hasattr(value, "astimezone"):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

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
                "licensing": {
                    "cc_license": channel.licensing.cc_license if channel.licensing else False,
                    "cc_type": channel.licensing.cc_type if channel.licensing else None,
                    "commercial_use": channel.licensing.commercial_use if channel.licensing else False,
                }
                if channel.licensing
                else None,
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
