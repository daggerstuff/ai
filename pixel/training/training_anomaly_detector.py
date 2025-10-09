#!/usr/bin/env python3
"""
Training Anomaly Detection System for Pixel LLM Training

This module provides comprehensive anomaly detection during training with
statistical analysis, pattern recognition, alert generation, and real-time
monitoring capabilities for detecting training irregularities.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class AnomalyType(Enum):
    """Types of training anomalies"""

    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    MEMORY_LEAK = "memory_leak"
    CONVERGENCE_FAILURE = "convergence_failure"
    METRIC_DEGRADATION = "metric_degradation"
    TRAINING_STALL = "training_stall"
    VALIDATION_DIVERGENCE = "validation_divergence"
    EQ_REGRESSION = "eq_regression"
    CLINICAL_ACCURACY_DROP = "clinical_accuracy_drop"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection system"""

    # Statistical parameters
    rolling_window_size: int = 100
    anomaly_threshold_std: float = 2.0
    loss_spike_threshold: float = 0.5
    gradient_explosion_threshold: float = 100.0
    gradient_vanishing_threshold: float = 1e-7

    # Monitoring parameters
    check_interval_steps: int = 10
    memory_check_interval: int = 50
    convergence_patience: int = 200

    # Alert settings
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = True
    alert_cooldown_minutes: int = 15
    alert_log_path: str = "anomaly_alerts"

    # Metric thresholds
    eq_score_degradation_threshold: float = 0.1
    clinical_accuracy_drop_threshold: float = 0.05
    validation_divergence_threshold: float = 0.2


@dataclass
class TrainingMetrics:
    """Container for training metrics"""

    step: int
    epoch: int
    timestamp: datetime
    total_loss: float
    language_loss: float
    eq_scores: Dict[str, float]
    clinical_accuracy: Dict[str, float]
    persona_metrics: Dict[str, float]
    empathy_scores: Dict[str, float]
    gradient_norm: float
    memory_usage_mb: float
    learning_rate: float


@dataclass
class AnomalyAlert:
    """Anomaly alert data structure"""

    alert_id: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    timestamp: datetime
    step: int
    epoch: int
    description: str
    metrics: Dict[str, Any]
    suggested_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "alert_id": self.alert_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "epoch": self.epoch,
            "description": self.description,
            "metrics": self.metrics,
            "suggested_actions": self.suggested_actions,
        }


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using rolling statistics"""

    def __init__(self, window_size: int = 100, threshold_std: float = 2.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def add_metric(self, metric_name: str, value: float) -> None:
        """Add metric value to rolling window"""
        self.metric_windows[metric_name].append(value)

    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous based on rolling statistics"""
        window = self.metric_windows[metric_name]

        if len(window) < 10:  # Not enough data
            return False, 0.0

        window_mean = mean(window)
        window_std = stdev(window) if len(window) > 1 else 0.0

        if window_std == 0:
            return False, 0.0

        z_score = abs((value - window_mean) / window_std)
        is_anomaly = z_score > self.threshold_std

        return is_anomaly, z_score


class AlertManager:
    """Manages alert generation, deduplication, and distribution"""

    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history: List[AnomalyAlert] = []
        self.last_alert_times: Dict[AnomalyType, datetime] = {}
        self.alert_callbacks: List[Callable[[AnomalyAlert], None]] = []

        # Ensure alert directory exists
        Path(self.config.alert_log_path).mkdir(parents=True, exist_ok=True)

    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]) -> None:
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def generate_alert(self, alert: AnomalyAlert) -> bool:
        """Generate and process alert with cooldown and deduplication"""
        # Check cooldown period
        if alert.anomaly_type in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[alert.anomaly_type]
            if time_since_last < timedelta(minutes=self.config.alert_cooldown_minutes):
                return False

        # Update last alert time
        self.last_alert_times[alert.anomaly_type] = datetime.now()

        # Add to history
        self.alert_history.append(alert)

        # Log alert
        self.logger.warning(f"ANOMALY DETECTED: {alert.description}")

        # Save alert to file
        self._save_alert_to_file(alert)

        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        return True

    def _save_alert_to_file(self, alert: AnomalyAlert) -> None:
        """Save alert to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(self.config.alert_log_path) / f"alert_{timestamp}_{alert.alert_id}.json"

        with open(filename, "w") as f:
            json.dump(alert.to_dict(), f, indent=2, default=str)

    def get_recent_alerts(self, hours: int = 24) -> List[AnomalyAlert]:
        """Get alerts from recent hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class TrainingAnomalyDetector:
    """Main training anomaly detection system"""

    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=config.rolling_window_size,
            threshold_std=config.anomaly_threshold_std,
        )
        self.alert_manager = AlertManager(config)

        # Monitoring state
        self.metrics_history: List[TrainingMetrics] = []
        self.last_check_step = 0
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Pattern tracking
        self.convergence_counter = 0
        self.last_eq_scores: Dict[str, float] = {}
        self.last_clinical_accuracy: Dict[str, float] = {}

    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_manager.add_alert_callback(callback)

    def process_training_metrics(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Process training metrics and detect anomalies"""
        self.metrics_history.append(metrics)
        detected_alerts = []

        # Check if we should perform anomaly detection
        if metrics.step - self.last_check_step >= self.config.check_interval_steps:
            detected_alerts.extend(self._detect_loss_anomalies(metrics))
            detected_alerts.extend(self._detect_gradient_anomalies(metrics))
            detected_alerts.extend(self._detect_metric_degradation(metrics))
            detected_alerts.extend(self._detect_convergence_issues(metrics))

            if metrics.step % self.config.memory_check_interval == 0:
                detected_alerts.extend(self._detect_memory_issues(metrics))

            self.last_check_step = metrics.step

        # Process alerts
        for alert in detected_alerts:
            self.alert_manager.generate_alert(alert)

        return detected_alerts

    def _detect_loss_anomalies(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Detect loss-related anomalies"""
        alerts = []

        # Add metrics to statistical detector
        self.statistical_detector.add_metric("total_loss", metrics.total_loss)
        self.statistical_detector.add_metric("language_loss", metrics.language_loss)

        # Check for loss spikes
        is_anomaly, z_score = self.statistical_detector.detect_anomaly(
            "total_loss", metrics.total_loss
        )
        if is_anomaly and z_score > 3.0:  # High threshold for spikes
            alerts.append(
                AnomalyAlert(
                    alert_id=f"loss_spike_{metrics.step}",
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    severity=AlertSeverity.HIGH,
                    timestamp=metrics.timestamp,
                    step=metrics.step,
                    epoch=metrics.epoch,
                    description=f"Loss spike detected: {metrics.total_loss:.4f} (z-score: {z_score:.2f})",
                    metrics={"total_loss": metrics.total_loss, "z_score": z_score},
                    suggested_actions=[
                        "Check data batch for outliers",
                        "Consider reducing learning rate",
                        "Investigate gradient clipping",
                        "Examine input preprocessing",
                    ],
                )
            )

        # Check for loss plateau
        if len(self.metrics_history) >= 50:
            recent_losses = [m.total_loss for m in self.metrics_history[-50:]]
            loss_variance = np.var(recent_losses)
            if loss_variance < 1e-6:  # Very small variance indicates plateau
                alerts.append(
                    AnomalyAlert(
                        alert_id=f"loss_plateau_{metrics.step}",
                        anomaly_type=AnomalyType.LOSS_PLATEAU,
                        severity=AlertSeverity.MEDIUM,
                        timestamp=metrics.timestamp,
                        step=metrics.step,
                        epoch=metrics.epoch,
                        description=f"Loss plateau detected: variance={loss_variance:.2e}",
                        metrics={
                            "loss_variance": loss_variance,
                            "recent_losses": recent_losses[-10:],
                        },
                        suggested_actions=[
                            "Increase learning rate",
                            "Implement learning rate scheduling",
                            "Check for gradient vanishing",
                            "Consider model architecture changes",
                        ],
                    )
                )

        return alerts

    def _detect_gradient_anomalies(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Detect gradient-related anomalies"""
        alerts = []

        # Gradient explosion
        if metrics.gradient_norm > self.config.gradient_explosion_threshold:
            alerts.append(
                AnomalyAlert(
                    alert_id=f"gradient_explosion_{metrics.step}",
                    anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=metrics.timestamp,
                    step=metrics.step,
                    epoch=metrics.epoch,
                    description=f"Gradient explosion: norm={metrics.gradient_norm:.2e}",
                    metrics={"gradient_norm": metrics.gradient_norm},
                    suggested_actions=[
                        "Apply gradient clipping",
                        "Reduce learning rate",
                        "Check model initialization",
                        "Investigate batch composition",
                    ],
                )
            )

        # Gradient vanishing
        if metrics.gradient_norm < self.config.gradient_vanishing_threshold:
            alerts.append(
                AnomalyAlert(
                    alert_id=f"gradient_vanishing_{metrics.step}",
                    anomaly_type=AnomalyType.GRADIENT_VANISHING,
                    severity=AlertSeverity.HIGH,
                    timestamp=metrics.timestamp,
                    step=metrics.step,
                    epoch=metrics.epoch,
                    description=f"Gradient vanishing: norm={metrics.gradient_norm:.2e}",
                    metrics={"gradient_norm": metrics.gradient_norm},
                    suggested_actions=[
                        "Increase learning rate",
                        "Use skip connections",
                        "Apply batch normalization",
                        "Check activation functions",
                    ],
                )
            )

        return alerts

    def _detect_metric_degradation(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Detect degradation in EQ and clinical accuracy metrics"""
        alerts = []

        # EQ score degradation
        for domain, score in metrics.eq_scores.items():
            if domain in self.last_eq_scores:
                score_drop = self.last_eq_scores[domain] - score
                if score_drop > self.config.eq_score_degradation_threshold:
                    alerts.append(
                        AnomalyAlert(
                            alert_id=f"eq_regression_{domain}_{metrics.step}",
                            anomaly_type=AnomalyType.EQ_REGRESSION,
                            severity=AlertSeverity.HIGH,
                            timestamp=metrics.timestamp,
                            step=metrics.step,
                            epoch=metrics.epoch,
                            description=f"EQ regression in {domain}: dropped by {score_drop:.3f}",
                            metrics={
                                "domain": domain,
                                "current_score": score,
                                "previous_score": self.last_eq_scores[domain],
                            },
                            suggested_actions=[
                                "Review EQ training data quality",
                                "Check emotion encoding pipeline",
                                "Investigate empathy loss function",
                                "Validate emotion annotation accuracy",
                            ],
                        )
                    )

        self.last_eq_scores = metrics.eq_scores.copy()

        # Clinical accuracy degradation
        for category, accuracy in metrics.clinical_accuracy.items():
            if category in self.last_clinical_accuracy:
                accuracy_drop = self.last_clinical_accuracy[category] - accuracy
                if accuracy_drop > self.config.clinical_accuracy_drop_threshold:
                    alerts.append(
                        AnomalyAlert(
                            alert_id=f"clinical_drop_{category}_{metrics.step}",
                            anomaly_type=AnomalyType.CLINICAL_ACCURACY_DROP,
                            severity=AlertSeverity.HIGH,
                            timestamp=metrics.timestamp,
                            step=metrics.step,
                            epoch=metrics.epoch,
                            description=f"Clinical accuracy drop in {category}: dropped by {accuracy_drop:.3f}",
                            metrics={
                                "category": category,
                                "current_accuracy": accuracy,
                                "previous_accuracy": self.last_clinical_accuracy[category],
                            },
                            suggested_actions=[
                                "Review clinical training data",
                                "Check DSM-5/PDM-2 knowledge retrieval",
                                "Validate clinical annotation quality",
                                "Review therapeutic appropriateness training",
                            ],
                        )
                    )

        self.last_clinical_accuracy = metrics.clinical_accuracy.copy()

        return alerts

    def _detect_convergence_issues(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Detect convergence and training stall issues"""
        alerts = []

        # Check for training stall (minimal progress over time)
        if len(self.metrics_history) >= 100:
            recent_losses = [m.total_loss for m in self.metrics_history[-100:]]
            if len(set(f"{loss:.4f}" for loss in recent_losses)) < 5:  # Very little variation
                alerts.append(
                    AnomalyAlert(
                        alert_id=f"training_stall_{metrics.step}",
                        anomaly_type=AnomalyType.TRAINING_STALL,
                        severity=AlertSeverity.MEDIUM,
                        timestamp=metrics.timestamp,
                        step=metrics.step,
                        epoch=metrics.epoch,
                        description="Training appears to have stalled with minimal loss variation",
                        metrics={
                            "recent_loss_variation": len(
                                set(f"{loss:.4f}" for loss in recent_losses)
                            )
                        },
                        suggested_actions=[
                            "Consider learning rate adjustment",
                            "Check for gradient flow issues",
                            "Review data shuffling",
                            "Investigate curriculum learning",
                        ],
                    )
                )

        return alerts

    def _detect_memory_issues(self, metrics: TrainingMetrics) -> List[AnomalyAlert]:
        """Detect memory-related issues"""
        alerts = []

        # Add memory usage to statistical detector
        self.statistical_detector.add_metric("memory_usage", metrics.memory_usage_mb)

        # Check for memory leaks (consistently increasing memory usage)
        if len(self.metrics_history) >= 20:
            recent_memory = [m.memory_usage_mb for m in self.metrics_history[-20:]]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            if memory_trend > 50:  # Memory increasing by >50MB per step on average
                alerts.append(
                    AnomalyAlert(
                        alert_id=f"memory_leak_{metrics.step}",
                        anomaly_type=AnomalyType.MEMORY_LEAK,
                        severity=AlertSeverity.HIGH,
                        timestamp=metrics.timestamp,
                        step=metrics.step,
                        epoch=metrics.epoch,
                        description=f"Potential memory leak: trend={memory_trend:.2f} MB/step",
                        metrics={
                            "memory_trend": memory_trend,
                            "current_memory": metrics.memory_usage_mb,
                        },
                        suggested_actions=[
                            "Check for tensor accumulation",
                            "Review gradient computation graphs",
                            "Investigate data loader memory usage",
                            "Consider garbage collection tuning",
                        ],
                    )
                )

        return alerts

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        recent_alerts = self.alert_manager.get_recent_alerts(24)

        summary = {
            "total_alerts_24h": len(recent_alerts),
            "alerts_by_type": defaultdict(int),
            "alerts_by_severity": defaultdict(int),
            "recent_alerts": [alert.to_dict() for alert in recent_alerts[-10:]],
        }

        for alert in recent_alerts:
            summary["alerts_by_type"][alert.anomaly_type.value] += 1
            summary["alerts_by_severity"][alert.severity.value] += 1

        return dict(summary)


if __name__ == "__main__":
    config = AnomalyDetectionConfig()
    detector = TrainingAnomalyDetector(config)
    print("Training anomaly detection system initialized successfully!")
