"""
Alerting system for monitoring training metrics and failures in the Pixelated Empathy AI project.
Detects unusual patterns in training metrics and system health.
"""

import json
import smtplib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
import psutil
import torch


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    METRIC_ANOMALY = "metric_anomaly"
    SYSTEM_RESOURCE = "system_resource"
    TRAINING_FAILURE = "training_failure"
    SAFETY_BREACH = "safety_breach"
    COST_THRESHOLD = "cost_threshold"
    PERFORMANCE_DROP = "performance_drop"


@dataclass
class Alert:
    """Definition of an alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: str
    run_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    name: str
    condition: Callable[[dict[str, Any]], bool]
    severity: AlertSeverity
    channels: list[AlertChannel]
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    escalation_minutes: int | None = None
    escalation_channels: list[AlertChannel] | None = None


@dataclass
class ChannelConfig:
    """Channel configuration."""

    channel_type: AlertChannel
    config: dict[str, Any]
    enabled: bool = True
    cooldown_minutes: int = 5  # Minimum time between alerts


class AlertManager:
    """Main alerting management system"""
    
    def __init__(self, 
                 smtp_server: Optional[str] = None,
                 smtp_port: Optional[int] = None,
                 smtp_username: Optional[str] = None,
                 smtp_password: Optional[str] = None,
                 email_recipients: Optional[List[str]] = None):
        self.alerts: List[Alert] = []
        self.rules: List[AlertRule] = self._create_default_rules()
        self.alert_history: Dict[str, List[datetime]] = {}
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.email_recipients = email_recipients or []
        self.logger = logging.getLogger(__name__)
        
        # For simulation purposes in this implementation
        self._alert_callback: Optional[Callable[[Alert], None]] = None
    
    def _create_default_rules(self) -> List[AlertRule]:
        """Create default alert rules"""
        rules = []
        
        # Loss explosion rule
        def loss_explosion_condition(metrics: Dict[str, Any]) -> bool:
            loss = metrics.get('train_loss', float('inf'))
            return loss > 1000.0
        
        rules.append(AlertRule(
            name="loss_explosion",
            description="Alert when training loss becomes extremely high",
            alert_type=AlertType.METRIC_ANOMALY,
            severity=AlertSeverity.CRITICAL,
            condition=loss_explosion_condition,
            message_template="Training loss has exploded to {train_loss:.4f}, indicating potential instability"
        ))
        
        # Loss plateau rule
        def loss_plateau_condition(metrics: Dict[str, Any]) -> bool:
            # Check if loss hasn't improved significantly in recent steps
            recent_losses = metrics.get('recent_losses', [])
            if len(recent_losses) >= 10:
                initial_loss = recent_losses[0]
                final_loss = recent_losses[-1]
                # If loss improvement is less than 1% over 10 measurements
                if initial_loss > 0 and ((initial_loss - final_loss) / initial_loss) < 0.01:
                    return True
            return False
        
        rules.append(AlertRule(
            name="loss_plateau",
            description="Alert when training loss stops improving",
            alert_type=AlertType.METRIC_ANOMALY,
            severity=AlertSeverity.MEDIUM,
            condition=loss_plateau_condition,
            message_template="Training loss has plateaued over recent measurements"
        ))
        
        # GPU memory exhaustion
        def gpu_memory_condition(metrics: Dict[str, Any]) -> bool:
            gpu_memory = metrics.get('gpu_memory_used_gb')
            gpu_memory_total = metrics.get('gpu_memory_total_gb')
            if gpu_memory and gpu_memory_total:
                return (gpu_memory / gpu_memory_total) > 0.95  # 95% threshold
            return False
        
        rules.append(AlertRule(
            name="gpu_memory_exhaustion",
            description="Alert when GPU memory is nearly exhausted",
            alert_type=AlertType.SYSTEM_RESOURCE,
            severity=AlertSeverity.HIGH,
            condition=gpu_memory_condition,
            message_template="GPU memory usage is critical at {gpu_memory_used_gb}GB out of {gpu_memory_total_gb}GB"
        ))
        
        # CPU usage high
        def cpu_usage_condition(metrics: Dict[str, Any]) -> bool:
            cpu_percent = metrics.get('cpu_percent', 0)
            return cpu_percent > 95.0
        
        rules.append(AlertRule(
            name="high_cpu_usage",
            description="Alert when CPU usage is extremely high",
            alert_type=AlertType.SYSTEM_RESOURCE,
            severity=AlertSeverity.MEDIUM,
            condition=cpu_usage_condition,
            message_template="CPU usage is critically high at {cpu_percent:.1f}%"
        ))
        
        # Safety metric degradation
        def safety_degradation_condition(metrics: Dict[str, Any]) -> bool:
            safety_score = metrics.get('overall_safety_score')
            if safety_score is not None:
                return safety_score < 0.5  # Safety score below threshold
            return False
        
        rules.append(AlertRule(
            name="safety_degradation",
            description="Alert when safety metrics degrade significantly",
            alert_type=AlertType.SAFETY_BREACH,
            severity=AlertSeverity.CRITICAL,
            condition=safety_degradation_condition,
            message_template="Overall safety score has dropped below threshold: {overall_safety_score:.3f}"
        ))
        
        # Cost threshold
        def cost_threshold_condition(metrics: Dict[str, Any]) -> bool:
            estimated_cost = metrics.get('estimated_cost_usd', 0)
            cost_threshold = metrics.get('cost_threshold_usd', float('inf'))
            return estimated_cost > cost_threshold
        
        rules.append(AlertRule(
            name="cost_threshold_breach",
            description="Alert when estimated cost exceeds threshold",
            alert_type=AlertType.COST_THRESHOLD,
            severity=AlertSeverity.MEDIUM,
            condition=cost_threshold_condition,
            message_template="Estimated cost of ${estimated_cost_usd:.2f} has exceeded threshold of ${cost_threshold_usd:.2f}"
        ))
        
        # Performance drop
        def performance_drop_condition(metrics: Dict[str, Any]) -> bool:
            current_perf = metrics.get('current_performance', float('inf'))
            baseline_perf = metrics.get('baseline_performance', float('inf'))
            if current_perf is not None and baseline_perf is not None:
                # Alert if performance has significantly degraded (e.g., loss increased by 50%)
                if baseline_perf > 0:
                    degradation = (current_perf - baseline_perf) / baseline_perf
                    return degradation > 0.5
            return False
        
        rules.append(AlertRule(
            name="performance_drop",
            description="Alert when model performance significantly degrades",
            alert_type=AlertType.PERFORMANCE_DROP,
            severity=AlertSeverity.HIGH,
            condition=performance_drop_condition,
            message_template="Model performance has degraded significantly: current {current_performance:.4f} vs baseline {baseline_performance:.4f}"
        ))
        
        return rules
    
    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.rules.append(rule)
    
    def evaluate_metrics(self, run_id: str, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate metrics against all rules and return triggered alerts"""
        triggered_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                # Check if we're still in cooldown period for this rule
                cooldown_key = f"{run_id}_{rule.name}"
                if cooldown_key in self.alert_history:
                    last_alert = self.alert_history[cooldown_key][-1]
                    cooldown_period = timedelta(minutes=rule.cooldown_minutes)
                    if datetime.now() - last_alert < cooldown_period:
                        continue
                
                if rule.condition(metrics):
                    # Generate alert
                    alert_id = f"alert_{rule.name}_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                    
                    # Format message with available metrics
                    message = rule.message_template
                    try:
                        message = message.format(**metrics)
                    except KeyError:
                        # If formatting fails, use the template as is
                        pass
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=rule.name.replace('_', ' ').title(),
                        description=message,
                        timestamp=datetime.utcnow().isoformat(),
                        run_id=run_id,
                        metadata=metrics
                    )
                    
                    triggered_alerts.append(alert)
                    
                    # Record alert time for cooldown
                    if cooldown_key not in self.alert_history:
                        self.alert_history[cooldown_key] = []
                    self.alert_history[cooldown_key].append(datetime.now())
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        # Add triggered alerts to our collection
        self.alerts.extend(triggered_alerts)
        
        # Send alerts if available
        for alert in triggered_alerts:
            self._send_alert(alert)
        
        return triggered_alerts
    
    def _send_alert(self, alert: Alert):
        """Send an alert through configured channels"""
        # Log the alert
        self.logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.description}")
        
        # Call the callback if available
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Send email if configured
        if self.email_recipients:
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        if not all([self.smtp_server, self.smtp_username, self.smtp_password, self.email_recipients]):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = ', '.join(self.email_recipients)
            msg['Subject'] = f"AI Training Alert: {alert.title} [{alert.severity.value.upper()}]"
            
            body = f"""
AI Training System Alert

Severity: {alert.severity.value.upper()}
Run ID: {alert.run_id or 'N/A'}
Title: {alert.title}
Description: {alert.description}
Timestamp: {alert.timestamp}

Metrics at time of alert:
{json.dumps(alert.metadata or {}, indent=2)}

Please investigate this issue as soon as possible.
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port or 587)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert email sent to {', '.join(self.email_recipients)}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookChannel:
    """Webhook alert channel."""

    def __init__(
        self,
        webhook_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.logger = get_logger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            payload = {
                "id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            self.logger.info(f"Webhook alert sent: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackChannel:
    """Slack alert channel."""

    def __init__(
        self,
        webhook_url: str,
        channel: str = "#alerts",
        username: str = "Dataset Monitor",
    ):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.logger = get_logger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Color coding by severity
            colors = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8B0000",
            }

            payload = {
                "channel": self.channel,
                "username": self.username,
                "attachments": [
                    {
                        "color": colors.get(alert.severity, "#36a64f"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                            {"title": "Source", "value": alert.source, "short": True},
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                            {"title": "Alert ID", "value": alert.id, "short": True},
                        ],
                        "footer": "Dataset Monitoring System",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()

            self.logger.info(f"Slack alert sent: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class ConsoleChannel:
    """Console alert channel."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to console."""
        try:
            severity_symbols = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
            }

            severity_symbols.get(alert.severity, "📢")

            if alert.metadata:
                pass

            return True

        except Exception as e:
            self.logger.error(f"Failed to send console alert: {e}")
            return False


class FileChannel:
    """File-based alert channel."""

    def __init__(self, file_path: str = "alerts.log"):
        self.file_path = file_path
        self.logger = get_logger(__name__)
        self.lock = threading.Lock()

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to file."""
        try:
            with self.lock, open(self.file_path, "a", encoding="utf-8") as f:
                alert_data = {
                    "id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata,
                }
                f.write(json.dumps(alert_data) + "\n")

            return True

        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")
            return False


class AlertingSystem:
    """Main alerting system."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Alert storage
        self.alerts: list[Alert] = []
        self.alert_history: dict[str, list[datetime]] = {}
        self.acknowledged_alerts: dict[str, datetime] = {}

        # Rules and channels
        self.rules: list[AlertRule] = []
        self.channels: dict[AlertChannel, Any] = {}

        # Threading
        self.alerts_lock = threading.Lock()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Default console channel
        self.channels[AlertChannel.CONSOLE] = ConsoleChannel()

        logger.info("AlertingSystem initialized")

    def add_channel(self, channel_config: ChannelConfig) -> None:
        """Add alert channel."""
        try:
            if channel_config.channel_type == AlertChannel.EMAIL:
                self.channels[AlertChannel.EMAIL] = EmailChannel(
                    **channel_config.config
                )
            elif channel_config.channel_type == AlertChannel.WEBHOOK:
                self.channels[AlertChannel.WEBHOOK] = WebhookChannel(
                    **channel_config.config
                )
            elif channel_config.channel_type == AlertChannel.SLACK:
                self.channels[AlertChannel.SLACK] = SlackChannel(
                    **channel_config.config
                )
            elif channel_config.channel_type == AlertChannel.FILE:
                self.channels[AlertChannel.FILE] = FileChannel(**channel_config.config)

            logger.info(f"Added alert channel: {channel_config.channel_type.value}")

        except Exception as e:
            logger.error(
                f"Failed to add channel {channel_config.channel_type.value}: {e}"
            )

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def check_conditions(self, metrics: dict[str, Any]) -> None:
        """Check all alert conditions against current metrics."""
        current_time = datetime.now()

        for rule in self.rules:
            try:
                if rule.condition(metrics):
                    # Check cooldown
                    if self._is_in_cooldown(rule.name, current_time):
                        continue

                    # Check rate limiting
                    if self._exceeds_rate_limit(rule.name, current_time):
                        continue

                    # Create and send alert
                    alert = Alert(
                        id=f"{rule.name}_{int(current_time.timestamp())}",
                        title=f"Alert: {rule.name}",
                        message=f"Condition triggered for rule: {rule.name}",
                        severity=rule.severity,
                        source="AlertingSystem",
                        metadata=metrics,
                    )

                    self.send_alert(alert, rule.channels)

                    # Record alert for cooldown and rate limiting
                    self._record_alert(rule.name, current_time)

            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {e}")

    def send_alert(self, alert: Alert, channels: list[AlertChannel]) -> None:
        """Send alert through specified channels."""
        with self.alerts_lock:
            self.alerts.append(alert)

        for channel_type in channels:
            if channel_type in self.channels:
                try:
                    success = self.channels[channel_type].send_alert(alert)
                    if success:
                        logger.info(f"Alert {alert.id} sent via {channel_type.value}")
                    else:
                        logger.error(
                            f"Failed to send alert {alert.id} via {channel_type.value}"
                        )
                except Exception as e:
                    logger.error(f"Error sending alert via {channel_type.value}: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    self.acknowledged_alerts[alert_id] = datetime.now()
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    return True

        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all active (unresolved) alerts."""
        with self.alerts_lock:
            return [alert for alert in self.alerts if not alert.resolved]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics."""
        with self.alerts_lock:
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts if not a.resolved])
            acknowledged_alerts = len([a for a in self.alerts if a.acknowledged])

            severity_counts = {}
            for severity in AlertSeverity:
                severity_counts[severity.value] = len(
                    [
                        a
                        for a in self.alerts
                        if a.severity == severity and not a.resolved
                    ]
                )

            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "acknowledged_alerts": acknowledged_alerts,
                "severity_breakdown": severity_counts,
                "last_alert": (
                    self.alerts[-1].timestamp.isoformat() if self.alerts else None
                ),
            }

    def _is_in_cooldown(self, rule_name: str, current_time: datetime) -> bool:
        """Check if rule is in cooldown period."""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if not rule:
            return False

        if rule_name not in self.alert_history:
            return False

        last_alerts = self.alert_history[rule_name]
        if not last_alerts:
            return False

        last_alert_time = max(last_alerts)
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)

        return current_time - last_alert_time < cooldown_period

    def _exceeds_rate_limit(self, rule_name: str, current_time: datetime) -> bool:
        """Check if rule exceeds rate limit."""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if not rule:
            return False

        if rule_name not in self.alert_history:
            return False

        # Count alerts in the last hour
        one_hour_ago = current_time - timedelta(hours=1)
        recent_alerts = [t for t in self.alert_history[rule_name] if t > one_hour_ago]

        return len(recent_alerts) >= rule.max_alerts_per_hour

    def _record_alert(self, rule_name: str, alert_time: datetime) -> None:
        """Record alert for cooldown and rate limiting."""
        if rule_name not in self.alert_history:
            self.alert_history[rule_name] = []

        self.alert_history[rule_name].append(alert_time)

        # Keep only last 24 hours of history
        cutoff_time = alert_time - timedelta(hours=24)
        self.alert_history[rule_name] = [
            t for t in self.alert_history[rule_name] if t > cutoff_time
        ]
        
        summary = self.get_alert_summary()
        for severity, count in summary.items():
            report.append(f"  {severity.upper()}: {count}")
        
        if alerts:
            report.extend([
                "",
                "Recent Alerts:"
            ])
            
            # Sort by timestamp, most recent first
            sorted_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
            for alert in sorted_alerts[:20]:  # Show last 20 alerts
                status = "RESOLVED" if alert.resolved else "ACTIVE"
                report.append(f"  [{alert.severity.value.upper()}] {alert.title} - {status}")
                report.append(f"      Run: {alert.run_id or 'N/A'}")
                report.append(f"      {alert.description}")
                report.append(f"      Time: {alert.timestamp}")
                report.append("")
        
        return "\n".join(report)


class TrainingMetricsMonitor:
    """Monitor for tracking training metrics and system health"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.run_metadata: Dict[str, Dict[str, Any]] = {}
    
    def track_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None):
        """Track metrics for a training run and check for anomalies"""
        # Add timestamp to metrics
        metrics['timestamp'] = datetime.utcnow().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # Add system resource metrics
        metrics.update(self._get_system_metrics())
        
        # Store metrics in history
        if run_id not in self.metrics_history:
            self.metrics_history[run_id] = []
        self.metrics_history[run_id].append(metrics)
        
        # Keep only recent history to avoid memory issues
        if len(self.metrics_history[run_id]) > 1000:
            self.metrics_history[run_id] = self.metrics_history[run_id][-500:]  # Keep last 500 entries
        
        # Update recent losses for plateau detection
        recent_losses = [m.get('train_loss') for m in self.metrics_history[run_id][-10:] if m.get('train_loss') is not None]
        if recent_losses:
            metrics['recent_losses'] = recent_losses
        
        # Evaluate metrics against alert rules
        triggered_alerts = self.alert_manager.evaluate_metrics(run_id, metrics)
        
        if triggered_alerts:
            self.logger.warning(f"Triggered {len(triggered_alerts)} alerts for run {run_id}")
        
        return triggered_alerts
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['memory_percent'] = psutil.virtual_memory().percent
        metrics['memory_used_gb'] = psutil.virtual_memory().used / (1024**3)
        metrics['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        
        # GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            metrics['gpu_utilization_percent'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            metrics['gpu_count'] = torch.cuda.device_count()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        metrics['disk_percent'] = (disk_usage.used / disk_usage.total) * 100
        metrics['disk_free_gb'] = disk_usage.free / (1024**3)
        
        return metrics
    
    def check_for_anomalies(self, run_id: str) -> List[str]:
        """Check for anomalies in the metric history"""
        if run_id not in self.metrics_history:
            return []
        
        anomalies = []
        metrics_list = self.metrics_history[run_id]
        
        if len(metrics_list) < 2:
            return []
        
        recent_metrics = metrics_list[-10:]  # Check last 10 metrics
        
        # Check for loss explosion
        losses = [m.get('train_loss') for m in recent_metrics if m.get('train_loss') is not None]
        if losses and max(losses) > 1000:
            anomalies.append("Loss explosion detected")
        
        # Check for abnormal gradients
        grad_norms = [m.get('grad_norm') for m in recent_metrics if m.get('grad_norm') is not None]
        if grad_norms and max(grad_norms) > 1000:
            anomalies.append("Gradient explosion detected")
        
        # Check for performance plateau
        if len(losses) >= 10:
            if all(l == losses[0] for l in losses):  # All losses are the same
                anomalies.append("Loss plateau detected")
        
        return anomalies
    
    def get_run_health_status(self, run_id: str) -> Dict[str, Any]:
        """Get health status for a specific run"""
        if run_id not in self.metrics_history:
            return {"status": "no_data", "message": "No metrics available for this run"}
        
        latest_metrics = self.metrics_history[run_id][-1] if self.metrics_history[run_id] else {}
        anomalies = self.check_for_anomalies(run_id)
        
        # Determine overall status
        active_alerts = self.alert_manager.get_alerts_by_run(run_id)
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved]
        
        if critical_alerts:
            status = "critical"
        elif anomalies:
            status = "warning"
        elif active_alerts:
            status = "attention_needed"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest_metrics.get('timestamp', 'N/A'),
            "latest_metrics": latest_metrics,
            "anomalies": anomalies,
            "active_alerts_count": len([a for a in active_alerts if not a.resolved]),
            "critical_alerts_count": len(critical_alerts),
            "message": f"Run has {len(anomalies)} anomalies and {len(active_alerts)} active alerts"
        }


class FailureDetector:
    """Component to detect and handle training failures"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        self.failure_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_failure(self, run_id: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record a training failure"""
        failure_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "metrics_snapshot": self._get_current_system_metrics()
        }
        
        if run_id not in self.failure_history:
            self.failure_history[run_id] = []
        self.failure_history[run_id].append(failure_record)
        
        # Create an alert for the failure
        alert_id = f"failure_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        alert = Alert(
            alert_id=alert_id,
            alert_type=AlertType.TRAINING_FAILURE,
            severity=AlertSeverity.CRITICAL,
            title="Training Run Failure",
            description=f"Training run {run_id} failed with {type(error).__name__}: {str(error)}",
            timestamp=datetime.utcnow().isoformat(),
            run_id=run_id,
            metadata={"error_type": type(error).__name__, "error_message": str(error), **(context or {})}
        )
        
        self.alert_manager.alerts.append(alert)
        self.alert_manager._send_alert(alert)
        
        self.logger.error(f"Recorded failure for run {run_id}: {type(error).__name__}: {str(error)}")
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics at time of failure"""
        metrics = {}
        try:
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['memory_percent'] = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass
        return metrics
    
    def get_failure_summary(self, run_id: str) -> Dict[str, Any]:
        """Get failure summary for a specific run"""
        failures = self.failure_history.get(run_id, [])
        return {
            "run_id": run_id,
            "total_failures": len(failures),
            "recent_failures": failures[-5:],  # Last 5 failures
            "failure_types": list(set(f["error_type"] for f in failures))
        }


def create_alert_system(smtp_config: Optional[Dict[str, Any]] = None) -> tuple[AlertManager, TrainingMetricsMonitor, FailureDetector]:
    """Create a complete alerting system"""
    # Create alert manager
    alert_manager = AlertManager(
        smtp_server=smtp_config.get("smtp_server") if smtp_config else None,
        smtp_port=smtp_config.get("smtp_port") if smtp_config else None,
        smtp_username=smtp_config.get("smtp_username") if smtp_config else None,
        smtp_password=smtp_config.get("smtp_password") if smtp_config else None,
        email_recipients=smtp_config.get("email_recipients") if smtp_config else []
    )
    
    # Create metrics monitor
    metrics_monitor = TrainingMetricsMonitor(alert_manager)
    
    # Create failure detector
    failure_detector = FailureDetector(alert_manager)
    
    return alert_manager, metrics_monitor, failure_detector


def test_alerting_system():
    """Test the alerting system"""
    logger.info("Testing Alerting System...")
    
    # Create the alerting system
    alert_manager, metrics_monitor, failure_detector = create_alert_system()
    
    # Set up a callback to capture alerts
    received_alerts = []
    def alert_callback(alert):
        received_alerts.append(alert)
        print(f"Received alert: {alert.title} - {alert.description}")
    
    alert_manager.set_alert_callback(alert_callback)
    
    # Test metric tracking and alerting
    run_id = "test_run_123"
    
    print("Testing normal metrics (should not trigger alerts)...")
    normal_metrics = {
        "train_loss": 1.5,
        "eval_loss": 1.4,
        "accuracy": 0.8,
        "learning_rate": 2e-5,
        "epoch": 1,
    }

    alerts = metrics_monitor.track_metrics(run_id, normal_metrics, step=100)
    print(f"Normal metrics triggered {len(alerts)} alerts")
    
    print("Testing anomalous metrics (should trigger alerts)...")
    anomalous_metrics = {
        "train_loss": 2000.0,  # Extremely high loss
        "eval_loss": 1500.0,
        "accuracy": 0.1,
        "learning_rate": 2e-5,
        "epoch": 5,
    }
    alerts = metrics_monitor.track_metrics(run_id, anomalous_metrics, step=500)
    print(f"Anomalous metrics triggered {len(alerts)} alerts")
    
    # Test GPU memory alert
    gpu_metrics = {
        "train_loss": 1.0,
        "gpu_memory_used_gb": 15.0,
        "gpu_memory_total_gb": 16.0,
    }
    alerts = metrics_monitor.track_metrics(run_id, gpu_metrics, step=600)
    print(f"GPU memory metrics triggered {len(alerts)} alerts")
    
    # Test cost threshold alert
    cost_metrics = {
        "train_loss": 0.8,
        "estimated_cost_usd": 500.0,
        "cost_threshold_usd": 100.0,  # Much lower than actual cost
    }
    alerts = metrics_monitor.track_metrics(run_id, cost_metrics, step=700)
    print(f"Cost metrics triggered {len(alerts)} alerts")
    
    # Test failure detection
    print("Testing failure detection...")
    try:
        raise RuntimeError("Test training failure")
    except RuntimeError as e:
        failure_detector.record_failure(run_id, e, {"step": 800, "model": "test_model"})
    
    # Get health status
    health_status = metrics_monitor.get_run_health_status(run_id)
    print(f"Run health status: {health_status['status']}")
    print(f"Anomalies found: {health_status['anomalies']}")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    print(f"Alert summary: {summary}")
    
    # Generate full report
    report = alert_manager.generate_alert_report(run_id)
    print(f"\nAlert Report:\n{report}")
    
    # Get failure summary
    failure_summary = failure_detector.get_failure_summary(run_id)
    print(f"\nFailure Summary: {failure_summary}")
    
    print(f"Total alerts received via callback: {len(received_alerts)}")
    print("Alerting system test completed!")


if __name__ == "__main__":
    test_alerting_system()
