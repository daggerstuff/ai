"""
Progress Tracking System for Dataset Downloads

Comprehensive progress tracking with real-time updates, ETA calculation,
speed monitoring, and resumable download support.
"""

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class ProgressStatus(Enum):
    """Progress tracking status."""

    PENDING = "pending"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""

    bytes_downloaded: int = 0
    total_bytes: int = 0
    files_downloaded: int = 0
    total_files: int = 0
    download_speed: float = 0.0  # bytes per second
    eta_seconds: float | None = None
    start_time: datetime | None = None
    last_update: datetime | None = None


@dataclass
class ProgressTask:
    """Individual progress tracking task."""

    task_id: str
    name: str
    status: ProgressStatus = ProgressStatus.PENDING
    metrics: ProgressMetrics = field(default_factory=ProgressMetrics)
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.metrics.total_bytes > 0:
            return (self.metrics.bytes_downloaded / self.metrics.total_bytes) * 100
        if self.metrics.total_files > 0:
            return (self.metrics.files_downloaded / self.metrics.total_files) * 100
        return 0.0


class ProgressTracker:
    """Comprehensive progress tracking system."""

    def __init__(self, update_interval: float = 1.0):
        self.logger = get_logger(__name__)
        self.update_interval = update_interval
        self.tasks: dict[str, ProgressTask] = {}
        self.callbacks: list[Callable[[str, ProgressTask], None]] = []

        # Threading
        self.tasks_lock = threading.Lock()
        self.update_thread = None
        self.stop_updates = threading.Event()

        # Persistence
        self.persistence_file = "progress_tracker_state.json"
        self.auto_save = True

        logger.info("ProgressTracker initialized")

    def create_task(
        self,
        task_id: str,
        name: str,
        total_bytes: int = 0,
        total_files: int = 0,
        metadata: dict | None = None,
    ) -> ProgressTask:
        """Create a new progress tracking task."""
        with self.tasks_lock:
            task = ProgressTask(
                task_id=task_id,
                name=name,
                metrics=ProgressMetrics(
                    total_bytes=total_bytes, total_files=total_files
                ),
                metadata=metadata or {},
            )
            self.tasks[task_id] = task

        logger.info(f"Created progress task: {task_id} - {name}")
        self._notify_callbacks(task_id, task)
        return task

    def start_task(self, task_id: str):
        """Start tracking progress for a task."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = ProgressStatus.IN_PROGRESS
                task.metrics.start_time = datetime.now()
                task.metrics.last_update = datetime.now()

        logger.info(f"Started progress tracking for task: {task_id}")
        self._notify_callbacks(task_id, task)

        # Start update thread if not running
        if not self.update_thread or not self.update_thread.is_alive():
            self.start_monitoring()

    def update_progress(
        self,
        task_id: str,
        bytes_downloaded: int | None = None,
        files_downloaded: int | None = None,
        additional_bytes: int | None = None,
        additional_files: int | None = None,
    ):
        """Update progress for a task."""
        with self.tasks_lock:
            if task_id not in self.tasks:
                logger.warning(f"Task not found: {task_id}")
                return

            task = self.tasks[task_id]
            current_time = datetime.now()

            # Update metrics
            if bytes_downloaded is not None:
                task.metrics.bytes_downloaded = bytes_downloaded
            elif additional_bytes is not None:
                task.metrics.bytes_downloaded += additional_bytes

            if files_downloaded is not None:
                task.metrics.files_downloaded = files_downloaded
            elif additional_files is not None:
                task.metrics.files_downloaded += additional_files

            # Calculate download speed
            if task.metrics.start_time and task.metrics.last_update:
                time_diff = (current_time - task.metrics.last_update).total_seconds()
                if time_diff > 0 and additional_bytes:
                    # Exponential moving average for smoother speed calculation
                    new_speed = additional_bytes / time_diff
                    if task.metrics.download_speed == 0:
                        task.metrics.download_speed = new_speed
                    else:
                        task.metrics.download_speed = (
                            0.7 * task.metrics.download_speed + 0.3 * new_speed
                        )

            # Calculate ETA
            if task.metrics.download_speed > 0 and task.metrics.total_bytes > 0:
                remaining_bytes = (
                    task.metrics.total_bytes - task.metrics.bytes_downloaded
                )
                task.metrics.eta_seconds = remaining_bytes / task.metrics.download_speed

            task.metrics.last_update = current_time

        self._notify_callbacks(task_id, task)

    def complete_task(self, task_id: str):
        """Mark task as completed."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = ProgressStatus.COMPLETED
                task.metrics.last_update = datetime.now()

                # Ensure progress is 100%
                if task.metrics.total_bytes > 0:
                    task.metrics.bytes_downloaded = task.metrics.total_bytes
                if task.metrics.total_files > 0:
                    task.metrics.files_downloaded = task.metrics.total_files

        logger.info(f"Completed progress task: {task_id}")
        self._notify_callbacks(task_id, task)

    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = ProgressStatus.FAILED
                task.error_message = error_message
                task.metrics.last_update = datetime.now()

        logger.error(f"Failed progress task: {task_id} - {error_message}")
        self._notify_callbacks(task_id, task)

    def pause_task(self, task_id: str):
        """Pause task progress tracking."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = ProgressStatus.PAUSED
                task.metrics.last_update = datetime.now()

        logger.info(f"Paused progress task: {task_id}")
        self._notify_callbacks(task_id, task)

    def resume_task(self, task_id: str):
        """Resume paused task."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == ProgressStatus.PAUSED:
                    task.status = ProgressStatus.IN_PROGRESS
                    task.metrics.last_update = datetime.now()

        logger.info(f"Resumed progress task: {task_id}")
        self._notify_callbacks(task_id, task)

    def cancel_task(self, task_id: str):
        """Cancel task progress tracking."""
        with self.tasks_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = ProgressStatus.CANCELLED
                task.metrics.last_update = datetime.now()

        logger.info(f"Cancelled progress task: {task_id}")
        self._notify_callbacks(task_id, task)

    def get_task(self, task_id: str) -> ProgressTask | None:
        """Get task by ID."""
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> list[ProgressTask]:
        """Get all tasks."""
        with self.tasks_lock:
            return list(self.tasks.values())

    def get_active_tasks(self) -> list[ProgressTask]:
        """Get currently active tasks."""
        with self.tasks_lock:
            return [
                task
                for task in self.tasks.values()
                if task.status == ProgressStatus.IN_PROGRESS
            ]

    def add_callback(self, callback: Callable[[str, ProgressTask], None]):
        """Add progress update callback."""
        self.callbacks.append(callback)
        logger.info("Added progress callback")

    def start_monitoring(self):
        """Start progress monitoring thread."""
        if self.update_thread and self.update_thread.is_alive():
            return

        self.stop_updates.clear()
        self.update_thread = threading.Thread(target=self._monitoring_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info("Progress monitoring started")

    def stop_monitoring(self):
        """Stop progress monitoring thread."""
        self.stop_updates.set()
        if self.update_thread:
            self.update_thread.join(timeout=5)

        logger.info("Progress monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_updates.is_set():
            try:
                # Auto-save state
                if self.auto_save:
                    self.save_state()

                # Check for stalled tasks
                self._check_stalled_tasks()

                # Sleep until next update
                self.stop_updates.wait(timeout=self.update_interval)

            except Exception as e:
                logger.error(f"Error in progress monitoring loop: {e}")
                self.stop_updates.wait(timeout=60)  # Wait longer on error

    def _check_stalled_tasks(self):
        """Check for tasks that may be stalled."""
        current_time = datetime.now()
        stall_threshold = timedelta(minutes=5)  # 5 minutes without update

        with self.tasks_lock:
            for task in self.tasks.values():
                if (
                    task.status == ProgressStatus.IN_PROGRESS
                    and task.metrics.last_update
                    and current_time - task.metrics.last_update > stall_threshold
                ):

                    logger.warning(f"Task may be stalled: {task.task_id}")
                    # Could trigger callback or automatic retry here

    def _notify_callbacks(self, task_id: str, task: ProgressTask):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(task_id, task)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def save_state(self, file_path: str | None = None):
        """Save current state to file."""
        if not file_path:
            file_path = self.persistence_file

        try:
            with self.tasks_lock:
                state = {"timestamp": datetime.now().isoformat(), "tasks": {}}

                for task_id, task in self.tasks.items():
                    state["tasks"][task_id] = {
                        "task_id": task.task_id,
                        "name": task.name,
                        "status": task.status.value,
                        "metrics": {
                            "bytes_downloaded": task.metrics.bytes_downloaded,
                            "total_bytes": task.metrics.total_bytes,
                            "files_downloaded": task.metrics.files_downloaded,
                            "total_files": task.metrics.total_files,
                            "download_speed": task.metrics.download_speed,
                            "eta_seconds": task.metrics.eta_seconds,
                            "start_time": (
                                task.metrics.start_time.isoformat()
                                if task.metrics.start_time
                                else None
                            ),
                            "last_update": (
                                task.metrics.last_update.isoformat()
                                if task.metrics.last_update
                                else None
                            ),
                        },
                        "error_message": task.error_message,
                        "metadata": task.metadata,
                    }

            with open(file_path, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Progress state saved to: {file_path}")

        except Exception as e:
            logger.error(f"Failed to save progress state: {e}")

    def load_state(self, file_path: str | None = None):
        """Load state from file."""
        if not file_path:
            file_path = self.persistence_file

        if not os.path.exists(file_path):
            logger.info("No previous progress state found")
            return

        try:
            with open(file_path) as f:
                state = json.load(f)

            with self.tasks_lock:
                for task_id, task_data in state.get("tasks", {}).items():
                    metrics_data = task_data.get("metrics", {})

                    metrics = ProgressMetrics(
                        bytes_downloaded=metrics_data.get("bytes_downloaded", 0),
                        total_bytes=metrics_data.get("total_bytes", 0),
                        files_downloaded=metrics_data.get("files_downloaded", 0),
                        total_files=metrics_data.get("total_files", 0),
                        download_speed=metrics_data.get("download_speed", 0.0),
                        eta_seconds=metrics_data.get("eta_seconds"),
                        start_time=(
                            datetime.fromisoformat(metrics_data["start_time"])
                            if metrics_data.get("start_time")
                            else None
                        ),
                        last_update=(
                            datetime.fromisoformat(metrics_data["last_update"])
                            if metrics_data.get("last_update")
                            else None
                        ),
                    )

                    task = ProgressTask(
                        task_id=task_data["task_id"],
                        name=task_data["name"],
                        status=ProgressStatus(task_data["status"]),
                        metrics=metrics,
                        error_message=task_data.get("error_message"),
                        metadata=task_data.get("metadata", {}),
                    )

                    self.tasks[task_id] = task

            logger.info(f"Progress state loaded from: {file_path}")

        except Exception as e:
            logger.error(f"Failed to load progress state: {e}")

    def get_summary(self) -> dict[str, Any]:
        """Get progress summary."""
        with self.tasks_lock:
            tasks = list(self.tasks.values())

            total_tasks = len(tasks)
            completed_tasks = sum(
                1 for t in tasks if t.status == ProgressStatus.COMPLETED
            )
            failed_tasks = sum(1 for t in tasks if t.status == ProgressStatus.FAILED)
            active_tasks = sum(
                1 for t in tasks if t.status == ProgressStatus.IN_PROGRESS
            )

            total_bytes = sum(t.metrics.total_bytes for t in tasks)
            downloaded_bytes = sum(t.metrics.bytes_downloaded for t in tasks)

            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "active_tasks": active_tasks,
                "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
                "total_bytes": total_bytes,
                "downloaded_bytes": downloaded_bytes,
                "overall_progress": (
                    downloaded_bytes / total_bytes if total_bytes > 0 else 0
                ),
                "timestamp": datetime.now().isoformat(),
            }


# Example progress callback
def console_progress_callback(task_id: str, task: ProgressTask):
    """Example console progress callback."""
    task.get_progress_percentage()
    (
        task.metrics.download_speed / (1024 * 1024)
        if task.metrics.download_speed
        else 0
    )

    if task.metrics.eta_seconds:
        f" ETA: {timedelta(seconds=int(task.metrics.eta_seconds))}"



# Example usage
if __name__ == "__main__":
    tracker = ProgressTracker()
    tracker.add_callback(console_progress_callback)
    tracker.start_monitoring()

    # Example task
    task = tracker.create_task(
        "download_1", "Test Dataset Download", total_bytes=1000000
    )
    tracker.start_task("download_1")

    # Simulate progress updates
    for i in range(0, 1000000, 100000):
        time.sleep(1)
        tracker.update_progress("download_1", bytes_downloaded=i)

    tracker.complete_task("download_1")
    tracker.stop_monitoring()
