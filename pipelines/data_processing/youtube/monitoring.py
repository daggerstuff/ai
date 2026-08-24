"""
Automated monitoring system for YouTube channel health.

Provides:
- Channel health checks
- Alert conditions
- Status updates
- Historical tracking
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from ai.pipelines.data_processing.youtube.models import Channel, ChannelStatus

logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertCondition:
    """Condition that triggers an alert."""

    name: str
    severity: AlertSeverity
    description: str
    check: Callable[[Channel], bool]
    action: str

    def evaluate(self, channel: Channel) -> bool:
        """Evaluate if condition is met."""
        return self.check(channel)


@dataclass
class HealthCheck:
    """Result of a health check on a channel."""

    channel_id: str
    timestamp: datetime
    status: ChannelStatus
    health_score: float
    alerts: list[AlertCondition] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0


class ChannelMonitor:
    """
    Monitor YouTube channels for health and status changes.

    Features:
    - Periodic health checks
    - Alert conditions
    - Status tracking
    - Historical data management
    """

    # Default alert conditions
    DEFAULT_ALERT_CONDITIONS = [
        AlertCondition(
            name="channel_removed",
            severity=AlertSeverity.CRITICAL,
            description="Channel has been removed or made private",
            check=lambda c: c.status == ChannelStatus.REMOVED,
            action="Investigate removal cause, mark for replacement",
        ),
        AlertCondition(
            name="low_activity",
            severity=AlertSeverity.WARNING,
            description="No new content in 30+ days",
            check=lambda c: c.last_updated and datetime.now(UTC) - c.last_updated > timedelta(days=30),
            action="Monitor for 14 more days, then consider replacement",
        ),
        AlertCondition(
            name="subscriber_drop",
            severity=AlertSeverity.ERROR,
            description="Significant subscriber drop (>10%)",
            check=lambda c: False,  # Needs historical data
            action="Investigate cause, check content changes",
        ),
        AlertCondition(
            name="quality_decline",
            severity=AlertSeverity.WARNING,
            description="Quality score dropped below threshold",
            check=lambda c: c.quality_score < 0.7,
            action="Schedule content review, re-evaluate channel",
        ),
    ]

    def __init__(
        self,
        channels: list[Channel],
        alert_conditions: list[AlertCondition] | None = None,
    ):
        self.channels = channels
        self.alert_conditions = alert_conditions or self.DEFAULT_ALERT_CONDITIONS[:]
        self.health_history: dict[str, list[HealthCheck]] = {}

    def check_channel_health(self, channel: Channel) -> HealthCheck:
        """
        Perform comprehensive health check on a channel.

        Args:
            channel: Channel to check

        Returns:
            HealthCheck result
        """
        health_check = HealthCheck(
            channel_id=channel.channel_id,
            timestamp=datetime.now(UTC),
            status=channel.status,
            health_score=self._calculate_health_score(channel),
        )

        # Run all alert conditions
        for condition in self.alert_conditions:
            if condition.evaluate(channel):
                health_check.alerts.append(condition)

        # Categorize alert counts
        critical_alerts = sum(1 for a in health_check.alerts if a.severity == AlertSeverity.CRITICAL)
        error_alerts = sum(1 for a in health_check.alerts if a.severity == AlertSeverity.ERROR)
        warning_alerts = sum(1 for a in health_check.alerts if a.severity == AlertSeverity.WARNING)

        # Calculate passed/failed checks
        total_checks = len(self.alert_conditions) + 5  # + base checks
        health_check.checks_failed = critical_alerts + error_alerts + warning_alerts
        health_check.checks_passed = total_checks - health_check.checks_failed

        # Determine overall status
        if critical_alerts > 0:
            health_check.status = ChannelStatus.INACTIVE
        elif error_alerts > 0:
            health_check.status = ChannelStatus.AT_RISK
        elif warning_alerts > 0:
            health_check.status = ChannelStatus.ACTIVE
        else:
            health_check.status = ChannelStatus.ACTIVE

        # Add notes
        health_check.notes.extend(self._generate_notes(channel, health_check))

        return health_check

    def _calculate_health_score(self, channel: Channel) -> float:
        """
        Calculate overall health score (0.0-1.0).

        Args:
            channel: Channel to evaluate

        Returns:
            Health score
        """
        score = 0.0

        # Status weight (40%)
        if channel.status == ChannelStatus.ACTIVE:
            score += 0.4
        elif channel.status == ChannelStatus.AT_RISK:
            score += 0.2
        elif channel.status == ChannelStatus.UNKNOWN:
            score += 0.1

        # Quality weight (30%)
        score += channel.quality_score * 0.3

        # Activity weight (20%)
        if channel.last_updated:
            days_inactive = (datetime.now(UTC) - channel.last_updated).days
            if days_inactive < 7:
                score += 0.2
            elif days_inactive < 30:
                score += 0.15
            elif days_inactive < 90:
                score += 0.1

        # Content volume weight (10%)
        if channel.video_count >= 100:
            score += 0.1
        elif channel.video_count >= 50:
            score += 0.07
        elif channel.video_count >= 20:
            score += 0.05

        return min(score, 1.0)

    def _generate_notes(self, channel: Channel, health_check: HealthCheck) -> list[str]:
        """Generate descriptive notes for the health check."""
        notes = []

        notes.append(f"Subscribers: {channel.subscriber_count:,}")
        notes.append(f"Videos: {channel.video_count:,}")
        notes.append(f"Quality Score: {channel.quality_score:.2f}")

        if channel.last_updated:
            days_ago = (datetime.now(UTC) - channel.last_updated).days
            notes.append(f"Last update: {days_ago} days ago")

        if len(health_check.alerts) > 0:
            notes.append(f"Alerts: {len(health_check.alerts)}")
            for alert in health_check.alerts:
                notes.append(f"  - {alert.name}: {alert.description}")

        if channel.is_professional:
            notes.append(f"Professional: Yes ({', '.join(channel.credentials)})")

        return notes

    def run_all_checks(self) -> dict[str, HealthCheck]:
        """
        Run health checks on all monitored channels.

        Returns:
            Dictionary mapping channel_id to HealthCheck results
        """
        results = {}

        for channel in self.channels:
            health_check = self.check_channel_health(channel)
            results[channel.channel_id] = health_check

            # Store in history
            if channel.channel_id not in self.health_history:
                self.health_history[channel.channel_id] = []
            self.health_history[channel.channel_id].append(health_check)

            # Keep only last 30 days of history
            self._trim_history(channel.channel_id)

        return results

    def _trim_history(self, channel_id: str, days_to_keep: int = 30):
        """Trim health check history to specified days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)
        self.health_history[channel_id] = [
            check for check in self.health_history[channel_id] if check.timestamp >= cutoff_date
        ]

    def get_channels_at_risk(self) -> list[Channel]:
        """Get list of channels marked as at risk."""
        return [c for c in self.channels if c.status == ChannelStatus.AT_RISK]

    def get_inactive_channels(self) -> list[Channel]:
        """Get list of inactive channels."""
        return [c for c in self.channels if c.status == ChannelStatus.INACTIVE]

    def generate_health_report(self) -> str:
        """Generate comprehensive health report."""
        results = self.run_all_checks()

        active = sum(1 for r in results.values() if r.status == ChannelStatus.ACTIVE)
        at_risk = sum(1 for r in results.values() if r.status == ChannelStatus.AT_RISK)
        inactive = sum(1 for r in results.values() if r.status == ChannelStatus.INACTIVE)
        unknown = sum(1 for r in results.values() if r.status == ChannelStatus.UNKNOWN)

        report = f"""
Channel Health Report
Generated: {datetime.now(UTC).isoformat()}

Overall Status:
  Active: {active}
  At Risk: {at_risk}
  Inactive: {inactive}
  Unknown: {unknown}
  Total: {len(results)}

Critical Alerts (Immediate Action Required):
"""

        # Add critical alerts
        for channel_id, check in results.items():
            critical = [a for a in check.alerts if a.severity == AlertSeverity.CRITICAL]
            if critical:
                report += f"\n  {channel_id}: {len(critical)} critical alert(s)\n"
                for alert in critical:
                    report += f"    - {alert.name}: {alert.description}\n"

        # Add error alerts
        report += "\nError Alerts (Attention Required):\n"
        for channel_id, check in results.items():
            errors = [a for a in check.alerts if a.severity == AlertSeverity.ERROR]
            if errors:
                report += f"\n  {channel_id}: {len(errors)} error alert(s)\n"
                for alert in errors:
                    report += f"    - {alert.name}: {alert.description}\n"

        return report


# Standalone health check function for manual execution
def health_check_channel(channel: Channel) -> dict:
    """
    Quick health check of a single channel.

    Args:
        channel: Channel to check

    Returns:
        Dictionary with health status and metrics
    """
    monitor = ChannelMonitor([channel])
    health_check = monitor.check_channel_health(channel)

    today = datetime.now(UTC)

    # Activity status
    activity_status = "active"
    if channel.last_updated:
        days_inactive = (today - channel.last_updated).days
        if days_inactive > 90:
            activity_status = "inactive"
        elif days_inactive > 30:
            activity_status = "at_risk"

    return {
        "channel_id": channel.channel_id,
        "channel_name": channel.channel_name,
        "status": health_check.status.value,
        "activity_status": activity_status,
        "health_score": health_check.health_score,
        "quality_score": channel.quality_score,
        "subscribers": channel.subscriber_count,
        "videos": channel.video_count,
        "last_updated": channel.last_updated.isoformat() if channel.last_updated else None,
        "last_health_check": health_check.timestamp.isoformat(),
        "alerts": [
            {
                "name": a.name,
                "severity": a.severity.value,
                "description": a.description,
            }
            for a in health_check.alerts
        ],
        "notes": health_check.notes,
    }
