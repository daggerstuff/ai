"""
Safety Monitoring Dashboard

Real-time safety metrics and alerts dashboard for the Pixelated Empathy
safety validation system. Provides comprehensive monitoring, alerting,
and reporting capabilities for safety-critical operations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import json
import threading
import time
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of safety metrics"""
    DETECTION_RATE = "detection_rate"
    RESPONSE_TIME = "response_time"
    VIOLATION_COUNT = "violation_count"
    COMPLIANCE_SCORE = "compliance_score"
    ERROR_RATE = "error_rate"
    SYSTEM_HEALTH = "system_health"
    CRISIS_DETECTION_RATE = "crisis_detection_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"


class DashboardStatus(Enum):
    """Dashboard operational status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class SafetyMetric:
    """Individual safety metric"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    threshold_breached: bool = False
    severity: AlertSeverity = AlertSeverity.LOW


@dataclass
class SafetyAlert:
    """Safety monitoring alert"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class SystemHealthStatus:
    """Overall system health status"""
    status: DashboardStatus
    overall_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    active_alerts: int = 0
    critical_alerts: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    uptime_hours: float = 0.0
    total_validations: int = 0
    successful_validations: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics summary"""
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    detection_accuracy: float = 0.0


class SafetyMonitoringDashboard:
    """
    Real-time safety monitoring dashboard
    
    Provides comprehensive monitoring of safety validation system including:
    - Real-time metrics collection and display
    - Automated alerting and notification
    - Performance monitoring and analysis
    - System health tracking
    - Historical data analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the safety monitoring dashboard"""
        
        self.config = self._load_config(config)
        
        # Metrics storage
        self.metrics_history: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=self.config["max_history_size"])
            for metric_type in MetricType
        }
        
        # Alerts storage
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.alert_history: deque = deque(maxlen=self.config["max_alert_history"])
        
        # System status
        self.system_health = SystemHealthStatus(
            status=DashboardStatus.HEALTHY,
            overall_score=1.0,
        )
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SafetyAlert], None]] = []
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "total_metrics_collected": 0,
            "total_alerts_generated": 0,
            "critical_alerts_generated": 0,
            "system_restarts": 0,
            "uptime_start": datetime.now(),
        }
        
        logger.info("Safety monitoring dashboard initialized")

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load dashboard configuration"""
        default_config = {
            # Thresholds
            "detection_rate_threshold": 0.9,  # 90% minimum detection rate
            "response_time_threshold": 1000,  # 1000ms maximum response time
            "error_rate_threshold": 0.05,     # 5% maximum error rate
            "compliance_score_threshold": 0.8, # 80% minimum compliance
            
            # Monitoring intervals
            "health_check_interval": 30,      # seconds
            "metrics_collection_interval": 10, # seconds
            "alert_check_interval": 5,        # seconds
            
            # Storage limits
            "max_history_size": 1000,         # metrics per type
            "max_alert_history": 500,         # total alerts
            
            # Alert settings
            "alert_cooldown_minutes": 5,      # minimum time between similar alerts
            "auto_resolve_minutes": 60,       # auto-resolve alerts after this time
            
            # Performance settings
            "performance_window_minutes": 15,  # window for performance calculations
            "throughput_calculation_window": 60, # seconds for throughput calculation
        }
        
        if config:
            default_config.update(config)
        
        return default_config

    def start_monitoring(self) -> None:
        """Start the monitoring dashboard"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.start_time = datetime.now()
        self.stats["uptime_start"] = self.start_time
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Safety monitoring dashboard started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring dashboard"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Safety monitoring dashboard stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform health check
                self._perform_health_check()
                
                # Check for alerts
                self._check_alerts()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep until next check
                time.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying

    def record_metric(self, metric_type: MetricType, value: float, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a safety metric"""
        if context is None:
            context = {}
        
        # Check if threshold is breached
        threshold_breached = self._check_threshold(metric_type, value)
        severity = self._determine_severity(metric_type, value, threshold_breached)
        
        metric = SafetyMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context,
            threshold_breached=threshold_breached,
            severity=severity,
        )
        
        # Store metric
        self.metrics_history[metric_type].append(metric)
        self.stats["total_metrics_collected"] += 1
        
        # Generate alert if threshold breached
        if threshold_breached:
            self._generate_threshold_alert(metric)
        
        logger.debug(f"Recorded metric: {metric_type.value} = {value}")

    def _check_threshold(self, metric_type: MetricType, value: float) -> bool:
        """Check if metric value breaches threshold"""
        thresholds = {
            MetricType.DETECTION_RATE: ("min", self.config["detection_rate_threshold"]),
            MetricType.RESPONSE_TIME: ("max", self.config["response_time_threshold"]),
            MetricType.ERROR_RATE: ("max", self.config["error_rate_threshold"]),
            MetricType.COMPLIANCE_SCORE: ("min", self.config["compliance_score_threshold"]),
        }
        
        if metric_type not in thresholds:
            return False
        
        threshold_type, threshold_value = thresholds[metric_type]
        
        if threshold_type == "min":
            return value < threshold_value
        elif threshold_type == "max":
            return value > threshold_value
        
        return False

    def _determine_severity(self, metric_type: MetricType, value: float, threshold_breached: bool) -> AlertSeverity:
        """Determine alert severity based on metric"""
        if not threshold_breached:
            return AlertSeverity.LOW
        
        # Critical thresholds
        critical_thresholds = {
            MetricType.DETECTION_RATE: 0.5,  # 50% detection rate is critical
            MetricType.RESPONSE_TIME: 5000,  # 5 second response time is critical
            MetricType.ERROR_RATE: 0.2,      # 20% error rate is critical
            MetricType.COMPLIANCE_SCORE: 0.3, # 30% compliance is critical
        }
        
        if metric_type in critical_thresholds:
            critical_threshold = critical_thresholds[metric_type]
            
            if metric_type in [MetricType.DETECTION_RATE, MetricType.COMPLIANCE_SCORE]:
                # Lower is worse
                if value <= critical_threshold:
                    return AlertSeverity.CRITICAL
                elif value <= critical_threshold * 1.5:
                    return AlertSeverity.HIGH
            else:
                # Higher is worse
                if value >= critical_threshold:
                    return AlertSeverity.CRITICAL
                elif value >= critical_threshold * 0.7:
                    return AlertSeverity.HIGH
        
        return AlertSeverity.MEDIUM

    def _generate_threshold_alert(self, metric: SafetyMetric) -> None:
        """Generate alert for threshold breach"""
        alert_id = f"threshold_{metric.metric_type.value}_{int(metric.timestamp.timestamp())}"
        
        # Check for recent similar alerts (cooldown)
        recent_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.alert_type == f"threshold_{metric.metric_type.value}"
            and (datetime.now() - alert.timestamp).total_seconds() < self.config["alert_cooldown_minutes"] * 60
        ]
        
        if recent_alerts:
            logger.debug(f"Skipping alert due to cooldown: {alert_id}")
            return
        
        # Create alert
        alert = SafetyAlert(
            alert_id=alert_id,
            alert_type=f"threshold_{metric.metric_type.value}",
            severity=metric.severity,
            title=f"{metric.metric_type.value.replace('_', ' ').title()} Threshold Breached",
            description=f"{metric.metric_type.value} value {metric.value} breached threshold",
            timestamp=metric.timestamp,
            metric_value=metric.value,
            context=metric.context,
        )
        
        self._add_alert(alert)

    def _add_alert(self, alert: SafetyAlert) -> None:
        """Add alert to active alerts and history"""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.stats["total_alerts_generated"] += 1
        
        if alert.severity == AlertSeverity.CRITICAL:
            self.stats["critical_alerts_generated"] += 1
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert generated: {alert.title} ({alert.severity.value})")

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.actions_taken.append(f"Acknowledged by {user} at {datetime.now()}")
        
        logger.info(f"Alert acknowledged: {alert_id} by {user}")
        return True

    def resolve_alert(self, alert_id: str, resolution: str = "Manual resolution", user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolution_time = datetime.now()
        alert.actions_taken.append(f"Resolved by {user}: {resolution}")
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id} by {user}")
        return True

    def _perform_health_check(self) -> None:
        """Perform system health check"""
        self.last_health_check = datetime.now()
        
        # Calculate component scores
        component_scores = {}
        
        # Detection rate score
        detection_metrics = list(self.metrics_history[MetricType.DETECTION_RATE])
        if detection_metrics:
            recent_detection = [m.value for m in detection_metrics[-10:]]  # Last 10 measurements
            avg_detection = statistics.mean(recent_detection)
            component_scores["detection_rate"] = min(1.0, avg_detection / self.config["detection_rate_threshold"])
        else:
            component_scores["detection_rate"] = 1.0
        
        # Response time score
        response_metrics = list(self.metrics_history[MetricType.RESPONSE_TIME])
        if response_metrics:
            recent_response = [m.value for m in response_metrics[-10:]]
            avg_response = statistics.mean(recent_response)
            component_scores["response_time"] = max(0.0, 1.0 - (avg_response / self.config["response_time_threshold"]))
        else:
            component_scores["response_time"] = 1.0
        
        # Error rate score
        error_metrics = list(self.metrics_history[MetricType.ERROR_RATE])
        if error_metrics:
            recent_errors = [m.value for m in error_metrics[-10:]]
            avg_error = statistics.mean(recent_errors)
            component_scores["error_rate"] = max(0.0, 1.0 - (avg_error / self.config["error_rate_threshold"]))
        else:
            component_scores["error_rate"] = 1.0
        
        # Alert score (based on active alerts)
        critical_alerts = sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.CRITICAL)
        high_alerts = sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.HIGH)
        
        alert_penalty = critical_alerts * 0.3 + high_alerts * 0.1
        component_scores["alerts"] = max(0.0, 1.0 - alert_penalty)
        
        # Calculate overall score
        overall_score = statistics.mean(component_scores.values()) if component_scores else 1.0
        
        # Determine status
        if overall_score >= 0.9:
            status = DashboardStatus.HEALTHY
        elif overall_score >= 0.7:
            status = DashboardStatus.WARNING
        elif overall_score >= 0.3:
            status = DashboardStatus.CRITICAL
        else:
            status = DashboardStatus.OFFLINE
        
        # Update system health
        self.system_health = SystemHealthStatus(
            status=status,
            overall_score=overall_score,
            component_scores=component_scores,
            active_alerts=len(self.active_alerts),
            critical_alerts=critical_alerts,
            last_update=datetime.now(),
            uptime_hours=(datetime.now() - self.start_time).total_seconds() / 3600,
        )

    def _check_alerts(self) -> None:
        """Check for alerts that need attention"""
        current_time = datetime.now()
        
        # Auto-resolve old alerts
        alerts_to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            if not alert.resolved:
                age_minutes = (current_time - alert.timestamp).total_seconds() / 60
                if age_minutes > self.config["auto_resolve_minutes"]:
                    alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id, "Auto-resolved due to age", "system")

    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=self.config["performance_window_minutes"])
        
        # Get recent response time metrics
        response_times = [
            m.value for m in self.metrics_history[MetricType.RESPONSE_TIME]
            if m.timestamp >= window_start
        ]
        
        if response_times:
            self.performance_metrics.avg_response_time = statistics.mean(response_times)
            self.performance_metrics.max_response_time = max(response_times)
            self.performance_metrics.min_response_time = min(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            if n >= 20:  # Only calculate percentiles with sufficient data
                self.performance_metrics.p95_response_time = sorted_times[int(n * 0.95)]
                self.performance_metrics.p99_response_time = sorted_times[int(n * 0.99)]
        
        # Calculate throughput
        throughput_window = current_time - timedelta(seconds=self.config["throughput_calculation_window"])
        recent_validations = sum(
            1 for m in self.metrics_history[MetricType.DETECTION_RATE]
            if m.timestamp >= throughput_window
        )
        self.performance_metrics.throughput_per_second = recent_validations / self.config["throughput_calculation_window"]
        
        # Calculate error rate
        error_metrics = [
            m.value for m in self.metrics_history[MetricType.ERROR_RATE]
            if m.timestamp >= window_start
        ]
        if error_metrics:
            self.performance_metrics.error_rate = statistics.mean(error_metrics)
        
        # Calculate detection accuracy
        detection_metrics = [
            m.value for m in self.metrics_history[MetricType.DETECTION_RATE]
            if m.timestamp >= window_start
        ]
        if detection_metrics:
            self.performance_metrics.detection_accuracy = statistics.mean(detection_metrics)

    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory issues"""
        # Metrics are automatically cleaned up by deque maxlen
        
        # Clean up old resolved alerts from history
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of history
        
        # Filter alert history
        self.alert_history = deque(
            [alert for alert in self.alert_history if alert.timestamp >= cutoff_time],
            maxlen=self.config["max_alert_history"]
        )

    def add_alert_callback(self, callback: Callable[[SafetyAlert], None]) -> None:
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[SafetyAlert], None]) -> None:
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data for display"""
        return {
            "system_health": {
                "status": self.system_health.status.value,
                "overall_score": self.system_health.overall_score,
                "component_scores": self.system_health.component_scores,
                "active_alerts": self.system_health.active_alerts,
                "critical_alerts": self.system_health.critical_alerts,
                "uptime_hours": self.system_health.uptime_hours,
                "last_update": self.system_health.last_update.isoformat(),
            },
            "performance_metrics": {
                "avg_response_time": self.performance_metrics.avg_response_time,
                "max_response_time": self.performance_metrics.max_response_time,
                "min_response_time": self.performance_metrics.min_response_time,
                "p95_response_time": self.performance_metrics.p95_response_time,
                "p99_response_time": self.performance_metrics.p99_response_time,
                "throughput_per_second": self.performance_metrics.throughput_per_second,
                "error_rate": self.performance_metrics.error_rate,
                "detection_accuracy": self.performance_metrics.detection_accuracy,
            },
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_value": alert.metric_value,
                    "acknowledged": alert.acknowledged,
                    "actions_taken": alert.actions_taken,
                }
                for alert in self.active_alerts.values()
            ],
            "recent_metrics": {
                metric_type.value: [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "threshold_breached": m.threshold_breached,
                        "severity": m.severity.value,
                    }
                    for m in list(self.metrics_history[metric_type])[-20:]  # Last 20 metrics
                ]
                for metric_type in MetricType
            },
            "statistics": {
                **self.stats,
                "uptime_start": self.stats["uptime_start"].isoformat(),
                "monitoring_active": self.monitoring_active,
            },
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get concise health summary"""
        return {
            "status": self.system_health.status.value,
            "score": round(self.system_health.overall_score, 3),
            "active_alerts": self.system_health.active_alerts,
            "critical_alerts": self.system_health.critical_alerts,
            "uptime_hours": round(self.system_health.uptime_hours, 1),
            "avg_response_time": round(self.performance_metrics.avg_response_time, 1),
            "detection_accuracy": round(self.performance_metrics.detection_accuracy, 3),
            "error_rate": round(self.performance_metrics.error_rate, 3),
        }

    def export_metrics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Export metrics for analysis"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
        
        exported_metrics = {}
        
        for metric_type in MetricType:
            metrics = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "context": m.context,
                    "threshold_breached": m.threshold_breached,
                    "severity": m.severity.value,
                }
                for m in self.metrics_history[metric_type]
                if start_time <= m.timestamp <= end_time
            ]
            exported_metrics[metric_type.value] = metrics
        
        return {
            "export_time": datetime.now().isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": exported_metrics,
            "alert_history": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None,
                    "actions_taken": alert.actions_taken,
                }
                for alert in self.alert_history
                if start_time <= alert.timestamp <= end_time
            ],
        }


# Example usage and testing
if __name__ == "__main__":
    import random
    
    async def demo_dashboard():
        """Demonstrate dashboard functionality"""
        print("=== SAFETY MONITORING DASHBOARD DEMO ===")
        
        # Create dashboard
        dashboard = SafetyMonitoringDashboard()
        
        # Add alert callback
        def alert_handler(alert: SafetyAlert):
            print(f"🚨 ALERT: {alert.title} ({alert.severity.value})")
        
        dashboard.add_alert_callback(alert_handler)
        
        # Start monitoring
        dashboard.start_monitoring()
        
        print("Dashboard started. Simulating metrics...")
        
        # Simulate metrics for 30 seconds
        for i in range(30):
            # Simulate various metrics
            detection_rate = random.uniform(0.7, 1.0)  # Sometimes below threshold
            response_time = random.uniform(200, 1500)  # Sometimes above threshold
            error_rate = random.uniform(0.0, 0.1)     # Sometimes above threshold
            
            dashboard.record_metric(MetricType.DETECTION_RATE, detection_rate)
            dashboard.record_metric(MetricType.RESPONSE_TIME, response_time)
            dashboard.record_metric(MetricType.ERROR_RATE, error_rate)
            
            # Print status every 10 iterations
            if i % 10 == 0:
                health = dashboard.get_health_summary()
                print(f"Health: {health['status']} (score: {health['score']}, alerts: {health['active_alerts']})")
            
            await asyncio.sleep(1)
        
        # Show final dashboard data
        print("\n=== FINAL DASHBOARD STATUS ===")
        health = dashboard.get_health_summary()
        for key, value in health.items():
            print(f"{key}: {value}")
        
        # Show active alerts
        dashboard_data = dashboard.get_dashboard_data()
        if dashboard_data["active_alerts"]:
            print(f"\n=== ACTIVE ALERTS ({len(dashboard_data['active_alerts'])}) ===")
            for alert in dashboard_data["active_alerts"]:
                print(f"- {alert['title']} ({alert['severity']})")
        
        # Stop monitoring
        dashboard.stop_monitoring()
        print("\nDashboard stopped.")
    
    # Run demo
    asyncio.run(demo_dashboard())
