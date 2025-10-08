#!/usr/bin/env python3
"""
Crisis Detection Monitoring System - Real-time alerts for detection failures.
Monitors the crisis detection system for failures, anomalies, and performance issues.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from crisis_intervention_detector import (
    CrisisDetection,
    CrisisInterventionDetector,
    CrisisLevel,
    CrisisType,
)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of monitoring alerts."""
    DETECTION_FAILURE = "detection_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ESCALATION_FAILURE = "escalation_failure"
    PATTERN_ANOMALY = "pattern_anomaly"
    SYSTEM_ERROR = "system_error"
    CONFIDENCE_ANOMALY = "confidence_anomaly"
    RESPONSE_TIME_VIOLATION = "response_time_violation"

@dataclass
class MonitoringAlert:
    """Monitoring alert data structure."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: datetime | None = None

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    avg_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    total_requests: int
    successful_detections: int
    failed_detections: int
    escalations_triggered: int
    confidence_scores: list[float]
    crisis_types_detected: dict[str, int]

class CrisisDetectionMonitor:
    """Real-time monitoring system for crisis detection."""

    def __init__(self, detector: CrisisInterventionDetector):
        self.detector = detector
        self.alerts = []
        self.performance_history = deque(maxlen=1000)  # Keep last 1000 requests
        self.alert_callbacks = []
        self.monitoring_active = False
        self.monitoring_thread = None

        # Monitoring thresholds
        self.thresholds = {
            "max_response_time_ms": 1000,
            "min_confidence_threshold": 0.1,
            "max_confidence_threshold": 1.0,
            "performance_window_minutes": 5,
            "min_detection_rate": 0.8,  # 80% minimum detection rate
            "max_error_rate": 0.05,     # 5% maximum error rate
            "escalation_timeout_minutes": 1
        }

        # Performance tracking
        self.request_times = deque(maxlen=100)
        self.detection_results = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0

        # Add monitoring callback to detector
        self.detector.add_escalation_callback(self._monitor_escalation)

        # Start monitoring thread
        self.start_monitoring()

    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("Crisis detection monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logging.info("Crisis detection monitoring stopped")

    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def monitor_detection(self, conversation: dict[str, Any], result: CrisisDetection, response_time_ms: float):
        """Monitor a crisis detection request."""
        self.total_requests += 1

        # Record performance metrics
        self.request_times.append(response_time_ms)
        self.detection_results.append({
            "timestamp": datetime.now(),
            "conversation_id": conversation.get("id", "unknown"),
            "crisis_level": result.crisis_level.value[0],
            "confidence": result.confidence_score,
            "crisis_types": [ct.value for ct in result.crisis_types],
            "response_time_ms": response_time_ms,
            "escalation_triggered": len(result.recommended_actions) > 0
        })

        # Check for immediate alerts
        self._check_response_time_alert(response_time_ms)
        self._check_confidence_anomaly(result.confidence_score, result.crisis_types)
        self._check_detection_patterns(conversation, result)

    def monitor_error(self, conversation: dict[str, Any], error: Exception):
        """Monitor detection errors."""
        self.error_count += 1
        self.total_requests += 1

        alert = MonitoringAlert(
            alert_id=f"error_{int(time.time())}_{self.error_count}",
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.HIGH,
            message=f"Crisis detection system error: {error!s}",
            details={
                "conversation_id": conversation.get("id", "unknown"),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "total_errors": self.error_count,
                "error_rate": self.error_count / self.total_requests
            },
            timestamp=datetime.now()
        )

        self._trigger_alert(alert)

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check performance metrics every 30 seconds
                self._check_performance_metrics()
                self._check_system_health()
                self._cleanup_old_alerts()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _monitor_escalation(self, detection: CrisisDetection, escalation):
        """Monitor escalation events."""
        # Check if escalation was triggered within acceptable time
        if escalation.response_time_minutes > self.thresholds["escalation_timeout_minutes"]:
            alert = MonitoringAlert(
                alert_id=f"escalation_timeout_{detection.detection_id}",
                alert_type=AlertType.ESCALATION_FAILURE,
                severity=AlertSeverity.HIGH,
                message=f"Escalation timeout: {escalation.response_time_minutes:.2f} minutes",
                details={
                    "detection_id": detection.detection_id,
                    "crisis_level": detection.crisis_level.value[0],
                    "response_time_minutes": escalation.response_time_minutes,
                    "threshold_minutes": self.thresholds["escalation_timeout_minutes"]
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _check_response_time_alert(self, response_time_ms: float):
        """Check for response time violations."""
        if response_time_ms > self.thresholds["max_response_time_ms"]:
            severity = AlertSeverity.CRITICAL if response_time_ms > 5000 else AlertSeverity.HIGH

            alert = MonitoringAlert(
                alert_id=f"response_time_{int(time.time())}",
                alert_type=AlertType.RESPONSE_TIME_VIOLATION,
                severity=severity,
                message=f"Response time violation: {response_time_ms:.1f}ms",
                details={
                    "response_time_ms": response_time_ms,
                    "threshold_ms": self.thresholds["max_response_time_ms"],
                    "avg_response_time": statistics.mean(self.request_times) if self.request_times else 0
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _check_confidence_anomaly(self, confidence: float, crisis_types: list[CrisisType]):
        """Check for confidence score anomalies."""
        # Check for impossible confidence scores
        if confidence < 0 or confidence > 1:
            alert = MonitoringAlert(
                alert_id=f"confidence_invalid_{int(time.time())}",
                alert_type=AlertType.CONFIDENCE_ANOMALY,
                severity=AlertSeverity.CRITICAL,
                message=f"Invalid confidence score: {confidence}",
                details={
                    "confidence_score": confidence,
                    "crisis_types": [ct.value for ct in crisis_types],
                    "valid_range": "0.0 - 1.0"
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

        # Check for crisis detected with zero confidence
        elif len(crisis_types) > 0 and confidence == 0.0:
            alert = MonitoringAlert(
                alert_id=f"confidence_zero_{int(time.time())}",
                alert_type=AlertType.CONFIDENCE_ANOMALY,
                severity=AlertSeverity.HIGH,
                message="Crisis detected with zero confidence",
                details={
                    "confidence_score": confidence,
                    "crisis_types": [ct.value for ct in crisis_types]
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _check_detection_patterns(self, conversation: dict[str, Any], result: CrisisDetection):
        """Check for detection pattern anomalies."""
        content = conversation.get("content", "")

        # Check for obvious crisis keywords with no detection
        crisis_keywords = [
            "kill myself", "suicide", "end my life", "want to die",
            "hurt myself", "cut myself", "overdose", "kill him",
            "hurt someone", "voices telling me", "going to hurt"
        ]

        has_crisis_keywords = any(keyword in content.lower() for keyword in crisis_keywords)
        has_crisis_detection = len(result.crisis_types) > 0

        if has_crisis_keywords and not has_crisis_detection:
            alert = MonitoringAlert(
                alert_id=f"detection_failure_{int(time.time())}",
                alert_type=AlertType.DETECTION_FAILURE,
                severity=AlertSeverity.CRITICAL,
                message="Potential crisis keywords detected but no crisis classification",
                details={
                    "conversation_id": conversation.get("id", "unknown"),
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "crisis_keywords_found": [kw for kw in crisis_keywords if kw in content.lower()],
                    "crisis_types_detected": [ct.value for ct in result.crisis_types],
                    "confidence_score": result.confidence_score
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _check_performance_metrics(self):
        """Check overall performance metrics."""
        if len(self.detection_results) < 10:  # Need minimum data
            return

        # Calculate recent performance metrics
        recent_results = list(self.detection_results)[-50:]  # Last 50 requests

        # Response time metrics
        response_times = [r["response_time_ms"] for r in recent_results]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)

        # Detection rate metrics
        crisis_detections = sum(1 for r in recent_results if r["crisis_types"])
        detection_rate = crisis_detections / len(recent_results)

        # Check for performance degradation
        if avg_response_time > self.thresholds["max_response_time_ms"] * 0.5:  # 50% of max threshold
            alert = MonitoringAlert(
                alert_id=f"performance_degradation_{int(time.time())}",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.MEDIUM,
                message=f"Performance degradation detected: {avg_response_time:.1f}ms average",
                details={
                    "avg_response_time_ms": avg_response_time,
                    "max_response_time_ms": max_response_time,
                    "sample_size": len(recent_results),
                    "threshold_ms": self.thresholds["max_response_time_ms"]
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

        # Check for low detection rate (potential system degradation)
        if detection_rate < self.thresholds["min_detection_rate"] and crisis_detections > 0:
            alert = MonitoringAlert(
                alert_id=f"low_detection_rate_{int(time.time())}",
                alert_type=AlertType.PATTERN_ANOMALY,
                severity=AlertSeverity.HIGH,
                message=f"Low crisis detection rate: {detection_rate:.1%}",
                details={
                    "detection_rate": detection_rate,
                    "crisis_detections": crisis_detections,
                    "total_requests": len(recent_results),
                    "min_threshold": self.thresholds["min_detection_rate"]
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _check_system_health(self):
        """Check overall system health."""
        if self.total_requests == 0:
            return

        # Check error rate
        error_rate = self.error_count / self.total_requests
        if error_rate > self.thresholds["max_error_rate"]:
            alert = MonitoringAlert(
                alert_id=f"high_error_rate_{int(time.time())}",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.HIGH,
                message=f"High error rate detected: {error_rate:.1%}",
                details={
                    "error_rate": error_rate,
                    "total_errors": self.error_count,
                    "total_requests": self.total_requests,
                    "max_threshold": self.thresholds["max_error_rate"]
                },
                timestamp=datetime.now()
            )
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: MonitoringAlert):
        """Trigger an alert and notify callbacks."""
        # Avoid duplicate alerts
        recent_alerts = [a for a in self.alerts if
                        a.alert_type == alert.alert_type and
                        (datetime.now() - a.timestamp).seconds < 300]  # 5 minutes

        if recent_alerts:
            return  # Skip duplicate alert

        self.alerts.append(alert)

        # Log the alert
        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[alert.severity]

        logging.log(log_level, f"CRISIS MONITOR ALERT [{alert.severity.value.upper()}]: {alert.message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")

    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts
                      if not alert.resolved or alert.timestamp > cutoff_time]

    def get_active_alerts(self) -> list[MonitoringAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.detection_results:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, [], {})

        recent_results = list(self.detection_results)
        response_times = [r["response_time_ms"] for r in recent_results]
        confidence_scores = [r["confidence"] for r in recent_results]

        # Count crisis types
        crisis_type_counts = defaultdict(int)
        successful_detections = 0
        escalations_triggered = 0

        for result in recent_results:
            if result["crisis_types"]:
                successful_detections += 1
                for crisis_type in result["crisis_types"]:
                    crisis_type_counts[crisis_type] += 1

            if result["escalation_triggered"]:
                escalations_triggered += 1

        return PerformanceMetrics(
            avg_response_time_ms=statistics.mean(response_times),
            max_response_time_ms=max(response_times),
            min_response_time_ms=min(response_times),
            total_requests=len(recent_results),
            successful_detections=successful_detections,
            failed_detections=len(recent_results) - successful_detections,
            escalations_triggered=escalations_triggered,
            confidence_scores=confidence_scores,
            crisis_types_detected=dict(crisis_type_counts)
        )

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logging.info(f"Alert resolved: {alert_id}")
                break

    def get_monitoring_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring report."""
        metrics = self.get_performance_metrics()
        active_alerts = self.get_active_alerts()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy" if not active_alerts else "alerts_active",
            "performance_metrics": asdict(metrics),
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "total_alerts_24h": len([a for a in self.alerts if
                                   (datetime.now() - a.timestamp).total_seconds() < 86400]),  # 24 hours in seconds
            "monitoring_thresholds": self.thresholds,
            "system_uptime": "monitoring_active" if self.monitoring_active else "monitoring_stopped"
        }

# Example usage and testing
def example_alert_callback(alert: MonitoringAlert):
    """Example alert callback for demonstration."""

def test_monitoring_system():
    """Test the monitoring system with various scenarios."""

    # Create detector and monitor
    detector = CrisisInterventionDetector()
    monitor = CrisisDetectionMonitor(detector)

    # Add test callback
    monitor.add_alert_callback(example_alert_callback)

    # Test normal operation
    start_time = time.time()
    result = detector.detect_crisis({"id": "test_1", "content": "I want to kill myself"})
    response_time = (time.time() - start_time) * 1000
    monitor.monitor_detection({"id": "test_1", "content": "I want to kill myself"}, result, response_time)

    # Test detection failure
    start_time = time.time()
    result = detector.detect_crisis({"id": "test_2", "content": "I want to kill myself tonight"})
    response_time = (time.time() - start_time) * 1000
    # Simulate a detection failure by creating a result with no crisis types
    from crisis_intervention_detector import CrisisDetection
    failed_result = CrisisDetection(
        detection_id="test_2_failed",
        conversation_id="test_2",
        crisis_level=CrisisLevel.ROUTINE,
        confidence_score=0.0,
        crisis_types=[],
        detected_indicators=[],
        risk_factors=[],
        protective_factors=[],
        recommended_actions=[],
        emergency_contacts=[],
        escalation_required=False
    )
    monitor.monitor_detection({"id": "test_2", "content": "I want to kill myself tonight"}, failed_result, response_time)

    # Test system error
    monitor.monitor_error({"id": "test_3", "content": "test error"}, Exception("Test exception"))

    # Test performance issue
    monitor.monitor_detection({"id": "test_4", "content": "test"}, result, 2000)  # 2 second response time

    # Get monitoring report
    monitor.get_monitoring_report()

    # Stop monitoring
    monitor.stop_monitoring()

    return len(monitor.get_active_alerts()) > 0

if __name__ == "__main__":
    success = test_monitoring_system()
