"""
Dataset Acquisition Monitoring and Alerting System

This module provides real-time monitoring of dataset acquisition processes,
including progress tracking, error detection, and alerting capabilities.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class AcquisitionStatus(Enum):
    """Dataset acquisition status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AcquisitionTask:
    """Dataset acquisition task."""

    task_id: str
    dataset_name: str
    source_url: str
    status: AcquisitionStatus
    progress: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str | None = None
    file_count: int = 0
    total_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringMetrics:
    """Monitoring metrics."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    average_completion_time: float
    success_rate: float
    total_data_acquired: int
    last_update: datetime


class AcquisitionMonitor:
    """Real-time dataset acquisition monitoring system."""

    def __init__(self, monitoring_interval: int = 30):
        self.logger = get_logger(__name__)
        self.monitoring_interval = monitoring_interval
        self.tasks: dict[str, AcquisitionTask] = {}
        self.metrics_history: list[MonitoringMetrics] = []
        self.alert_callbacks: list[Callable] = []

        # Threading
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.tasks_lock = threading.Lock()

        # Configuration
        self.alert_thresholds = {
            "max_failure_rate": 0.3,  # 30% failure rate triggers alert
            "max_task_duration": 3600,  # 1 hour max task duration
            "min_success_rate": 0.7,  # 70% minimum success rate
            "max_concurrent_tasks": 10,
        }

        logger.info("AcquisitionMonitor initialized")

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Acquisition monitoring started")

    def stop_monitoring_system(self):
        """Stop the monitoring system."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Acquisition monitoring stopped")

    def register_task(
        self, task_id: str, dataset_name: str, source_url: str
    ) -> AcquisitionTask:
        """Register a new acquisition task."""
        with self.tasks_lock:
            task = AcquisitionTask(
                task_id=task_id,
                dataset_name=dataset_name,
                source_url=source_url,
                status=AcquisitionStatus.PENDING,
            )
            self.tasks[task_id] = task

        logger.info(f"Registered acquisition task: {task_id} for {dataset_name}")
        return task

    def start_task(self, task_id: str):
        """Mark task as started."""
        with self.tasks_lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = AcquisitionStatus.IN_PROGRESS
                self.tasks[task_id].start_time = datetime.now(tz=datetime.timezone.utc)
                logger.info(f"Started acquisition task: {task_id}")

    def update_progress(
        self,
        task_id: str,
        progress: float,
        file_count: int = 0,
        total_size: int = 0,
        metadata: dict | None = None,
    ):
        """Update task progress."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = min(100.0, max(0.0, progress))
                task.file_count = file_count
                task.total_size = total_size

                if metadata:
                    task.metadata.update(metadata)

                logger.debug(f"Updated progress for {task_id}: {progress:.1f}%")

    def complete_task(
        self, task_id: str, final_file_count: int = 0, final_size: int = 0
    ):
        """Mark task as completed."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = AcquisitionStatus.COMPLETED
                task.progress = 100.0
                task.end_time = datetime.now(tz=datetime.timezone.utc)
                task.file_count = final_file_count
                task.total_size = final_size

                logger.info(f"Completed acquisition task: {task_id}")
                self._check_completion_alerts(task)

    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = AcquisitionStatus.FAILED
                task.end_time = datetime.now(tz=datetime.timezone.utc)
                task.error_message = error_message

                logger.error(f"Failed acquisition task: {task_id} - {error_message}")
                self._trigger_failure_alert(task)

    def cancel_task(self, task_id: str):
        """Cancel a task."""
        with self.tasks_lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = AcquisitionStatus.CANCELLED
                self.tasks[task_id].end_time = datetime.now(tz=datetime.timezone.utc)

                logger.info(f"Cancelled acquisition task: {task_id}")

    def get_task_status(self, task_id: str) -> AcquisitionTask | None:
        """Get status of a specific task."""
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> list[AcquisitionTask]:
        """Get all tasks."""
        with self.tasks_lock:
            return list(self.tasks.values())

    def get_active_tasks(self) -> list[AcquisitionTask]:
        """Get currently active tasks."""
        with self.tasks_lock:
            return [
                task
                for task in self.tasks.values()
                if task.status == AcquisitionStatus.IN_PROGRESS
            ]

    def get_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics."""
        with self.tasks_lock:
            tasks = list(self.tasks.values())

            total_tasks = len(tasks)
            completed_tasks = sum(
                1 for t in tasks if t.status == AcquisitionStatus.COMPLETED
            )
            failed_tasks = sum(1 for t in tasks if t.status == AcquisitionStatus.FAILED)
            active_tasks = sum(
                1 for t in tasks if t.status == AcquisitionStatus.IN_PROGRESS
            )

            # Calculate average completion time
            completed_with_times = [
                t
                for t in tasks
                if t.status == AcquisitionStatus.COMPLETED
                and t.start_time
                and t.end_time
            ]

            if completed_with_times:
                completion_times = [
                    (t.end_time - t.start_time).total_seconds()
                    for t in completed_with_times
                ]
                average_completion_time = sum(completion_times) / len(completion_times)
            else:
                average_completion_time = 0.0

            # Calculate success rate
            finished_tasks = completed_tasks + failed_tasks
            success_rate = (
                completed_tasks / finished_tasks if finished_tasks > 0 else 1.0
            )

            # Calculate total data acquired
            total_data_acquired = sum(
                t.total_size for t in tasks if t.status == AcquisitionStatus.COMPLETED
            )

            return MonitoringMetrics(
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                active_tasks=active_tasks,
                average_completion_time=average_completion_time,
                success_rate=success_rate,
                total_data_acquired=total_data_acquired,
                last_update=datetime.now(tz=datetime.timezone.utc),
            )

    def add_alert_callback(self, callback: Callable[[str, dict], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")

    def generate_status_report(
        self, output_path: str = "acquisition_status_report.json"
    ) -> str:
        """Generate comprehensive status report."""
        metrics = self.get_metrics()
        tasks = self.get_all_tasks()

        report = {
            "report_timestamp": datetime.now(tz=datetime.timezone.utc).isoformat(),
            "metrics": {
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "active_tasks": metrics.active_tasks,
                "success_rate": metrics.success_rate,
                "average_completion_time": metrics.average_completion_time,
                "total_data_acquired": metrics.total_data_acquired,
            },
            "active_tasks": [
                {
                    "task_id": t.task_id,
                    "dataset_name": t.dataset_name,
                    "progress": t.progress,
                    "duration": (
                        (datetime.now(tz=datetime.timezone.utc) - t.start_time).total_seconds()
                        if t.start_time
                        else 0
                    ),
                }
                for t in tasks
                if t.status == AcquisitionStatus.IN_PROGRESS
            ],
            "recent_failures": [
                {
                    "task_id": t.task_id,
                    "dataset_name": t.dataset_name,
                    "error_message": t.error_message,
                    "failure_time": t.end_time.isoformat() if t.end_time else None,
                }
                for t in tasks
                if t.status == AcquisitionStatus.FAILED
                and t.end_time
                and t.end_time > datetime.now(tz=datetime.timezone.utc) - timedelta(hours=24)
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Status report generated: {output_path}")
        return output_path

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Update metrics
                current_metrics = self.get_metrics()
                self.metrics_history.append(current_metrics)

                # Keep only last 24 hours of metrics
                cutoff_time = datetime.now(tz=datetime.timezone.utc) - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.last_update > cutoff_time
                ]

                # Check for alerts
                self._check_system_alerts(current_metrics)

                # Check for stuck tasks
                self._check_stuck_tasks()

                # Sleep until next check
                self.stop_monitoring.wait(timeout=self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stop_monitoring.wait(timeout=60)  # Wait longer on error

    def _check_system_alerts(self, metrics: MonitoringMetrics):
        """Check for system-level alerts."""
        alerts = []

        # Check failure rate
        if metrics.success_rate < self.alert_thresholds["min_success_rate"]:
            alerts.append(
                {
                    "type": "low_success_rate",
                    "message": f"Success rate below threshold: {metrics.success_rate:.2%}",
                    "severity": "high",
                }
            )

        # Check concurrent tasks
        if metrics.active_tasks > self.alert_thresholds["max_concurrent_tasks"]:
            alerts.append(
                {
                    "type": "high_concurrent_tasks",
                    "message": f"Too many concurrent tasks: {metrics.active_tasks}",
                    "severity": "medium",
                }
            )

        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert["type"], alert)

    def _check_stuck_tasks(self):
        """Check for tasks that may be stuck."""
        current_time = datetime.now(tz=datetime.timezone.utc)
        max_duration = timedelta(seconds=self.alert_thresholds["max_task_duration"])

        with self.tasks_lock:
            for task in self.tasks.values():
                if (
                    task.status == AcquisitionStatus.IN_PROGRESS
                    and task.start_time
                    and current_time - task.start_time > max_duration
                ):

                    alert = {
                        "type": "stuck_task",
                        "message": f"Task {task.task_id} running for {current_time - task.start_time}",
                        "severity": "high",
                        "task_id": task.task_id,
                    }
                    self._trigger_alert("stuck_task", alert)

    def _check_completion_alerts(self, task: AcquisitionTask):
        """Check for completion-related alerts."""
        if task.start_time and task.end_time:
            duration = (task.end_time - task.start_time).total_seconds()

            # Alert for unusually fast completion (might indicate error)
            if duration < 10:  # Less than 10 seconds
                alert = {
                    "type": "fast_completion",
                    "message": f"Task {task.task_id} completed very quickly ({duration:.1f}s)",
                    "severity": "low",
                    "task_id": task.task_id,
                }
                self._trigger_alert("fast_completion", alert)

    def _trigger_failure_alert(self, task: AcquisitionTask):
        """Trigger alert for task failure."""
        alert = {
            "type": "task_failure",
            "message": f"Task {task.task_id} failed: {task.error_message}",
            "severity": "high",
            "task_id": task.task_id,
            "error_message": task.error_message,
        }
        self._trigger_alert("task_failure", alert)

    def _trigger_alert(self, alert_type: str, alert_data: dict):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


# Example alert callback
def default_alert_handler(alert_type: str, alert_data: dict):
    """Default alert handler that logs alerts."""
    severity = alert_data.get("severity", "unknown")
    message = alert_data.get("message", "No message")

    if severity == "high":
        logger.error(f"HIGH ALERT [{alert_type}]: {message}")
    elif severity == "medium":
        logger.warning(f"MEDIUM ALERT [{alert_type}]: {message}")
    else:
        logger.info(f"LOW ALERT [{alert_type}]: {message}")


# Example usage
if __name__ == "__main__":
    monitor = AcquisitionMonitor()
    monitor.add_alert_callback(default_alert_handler)
    monitor.start_monitoring()

    # Example task registration
    task = monitor.register_task(
        "test_task_1", "test_dataset", "http://example.com/data"
    )
    monitor.start_task("test_task_1")
    monitor.update_progress("test_task_1", 50.0, file_count=100)
    monitor.complete_task("test_task_1", final_file_count=200, final_size=1024000)

    # Generate report
    report_path = monitor.generate_status_report()

    monitor.stop_monitoring_system()
