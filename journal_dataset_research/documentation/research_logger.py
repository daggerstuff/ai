"""
Research Activity Logger

Provides comprehensive logging for all research activities with rotation,
archival, and structured log management.
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ai.journal_dataset_research.models.dataset_models import ResearchLog

logger = logging.getLogger(__name__)


class ResearchLogger:
    """
    Logger for research activities with rotation and archival support.

    Tracks all component activities with timestamps, outcomes, and durations.
    Supports log rotation based on size or time, and archival of old logs.
    """

    def __init__(
        self,
        log_directory: str = "logs",
        max_log_size_mb: int = 100,
        max_log_age_days: int = 30,
        rotation_interval_days: int = 7,
        enable_archival: bool = True,
    ):
        """
        Initialize the research logger.

        Args:
            log_directory: Directory to store log files
            max_log_size_mb: Maximum log file size before rotation (MB)
            max_log_age_days: Maximum age of logs before archival (days)
            rotation_interval_days: Interval for time-based rotation (days)
            enable_archival: Whether to enable automatic archival
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.max_log_age_days = max_log_age_days
        self.rotation_interval_days = rotation_interval_days
        self.enable_archival = enable_archival

        self.archival_directory = self.log_directory / "archived"
        if self.enable_archival:
            self.archival_directory.mkdir(parents=True, exist_ok=True)

        self._current_log_file: Optional[Path] = None
        self._session_logs: Dict[str, List[ResearchLog]] = {}
        self._lock = threading.Lock()

    def log_activity(
        self,
        activity_type: str,
        description: str,
        outcome: str = "",
        duration_minutes: int = 0,
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ResearchLog:
        """
        Log a research activity.

        Args:
            activity_type: Type of activity (must be in ResearchLog.ALLOWED_ACTIVITY_TYPES)
            description: Description of the activity
            outcome: Outcome of the activity
            duration_minutes: Duration in minutes
            source_id: Optional source ID associated with the activity
            session_id: Optional session ID for grouping logs

        Returns:
            ResearchLog: The created log entry
        """
        log_entry = ResearchLog(
            timestamp=datetime.now(),
            activity_type=activity_type,
            source_id=source_id,
            description=description,
            outcome=outcome,
            duration_minutes=duration_minutes,
        )

        # Validate the log entry
        errors = log_entry.validate()
        if errors:
            raise ValueError(f"Invalid log entry: {', '.join(errors)}")

        with self._lock:
            # Add to session logs if session_id provided
            if session_id:
                if session_id not in self._session_logs:
                    self._session_logs[session_id] = []
                self._session_logs[session_id].append(log_entry)

            # Write to file
            self._write_log_entry(log_entry, session_id)

            # Check for rotation (only if log file exists)
            if self._current_log_file and self._current_log_file.exists():
                self._check_rotation()

        logger.debug(
            f"Logged activity: {activity_type} - {description[:50]}"
            + (f" (session: {session_id})" if session_id else "")
        )

        return log_entry

    def get_session_logs(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ResearchLog]:
        """
        Get all logs for a specific session.

        Args:
            session_id: Session ID to retrieve logs for
            limit: Optional limit on number of logs to return

        Returns:
            List of ResearchLog entries for the session
        """
        with self._lock:
            logs = self._session_logs.get(session_id, [])
            if limit:
                return logs[-limit:]
            return logs.copy()

    def get_logs_by_activity_type(
        self, activity_type: str, session_id: Optional[str] = None
    ) -> List[ResearchLog]:
        """
        Get all logs of a specific activity type.

        Args:
            activity_type: Activity type to filter by
            session_id: Optional session ID to filter by

        Returns:
            List of ResearchLog entries matching the criteria
        """
        with self._lock:
            if session_id:
                logs = self._session_logs.get(session_id, [])
            else:
                logs = []
                for session_logs in self._session_logs.values():
                    logs.extend(session_logs)

            return [log for log in logs if log.activity_type == activity_type]

    def get_logs_by_source_id(self, source_id: str) -> List[ResearchLog]:
        """
        Get all logs associated with a specific source ID.

        Args:
            source_id: Source ID to filter by

        Returns:
            List of ResearchLog entries for the source
        """
        with self._lock:
            logs = []
            for session_logs in self._session_logs.values():
                logs.extend(
                    [log for log in session_logs if log.source_id == source_id]
                )

            return logs

    def get_logs_by_time_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[ResearchLog]:
        """
        Get all logs within a time range.

        Args:
            start_date: Start of time range
            end_date: End of time range

        Returns:
            List of ResearchLog entries within the time range
        """
        with self._lock:
            logs = []
            for session_logs in self._session_logs.values():
                logs.extend(
                    [
                        log
                        for log in session_logs
                        if start_date <= log.timestamp <= end_date
                    ]
                )

            return sorted(logs, key=lambda x: x.timestamp)

    def query_logs(
        self,
        activity_type: Optional[str] = None,
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ResearchLog]:
        """
        Query logs with flexible filtering.

        Args:
            activity_type: Optional activity type filter
            source_id: Optional source ID filter
            session_id: Optional session ID filter

        Returns:
            List of ResearchLog entries matching the criteria
        """
        # Collect logs from both in-memory and file sources
        all_logs = []

        # Get from in-memory session logs
        if session_id:
            all_logs.extend(self.get_session_logs(session_id))
        else:
            for session_logs in self._session_logs.values():
                all_logs.extend(session_logs)

        # Also read from log files
        log_file = self._get_current_log_file()
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            log_data = json.loads(line)
                            log_entry = ResearchLog(
                                timestamp=datetime.fromisoformat(log_data["timestamp"]),
                                activity_type=log_data["activity_type"],
                                source_id=log_data.get("source_id"),
                                description=log_data["description"],
                                outcome=log_data.get("outcome", ""),
                                duration_minutes=log_data.get("duration_minutes", 0),
                            )
                            all_logs.append(log_entry)
            except (IOError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to read log file: {e}")

        # Apply filters
        if activity_type:
            all_logs = [log for log in all_logs if log.activity_type == activity_type]
        if source_id:
            all_logs = [log for log in all_logs if log.source_id == source_id]

        # Remove duplicates (same timestamp + activity_type + description)
        seen = set()
        unique_logs = []
        for log in all_logs:
            key = (log.timestamp.isoformat(), log.activity_type, log.description)
            if key not in seen:
                seen.add(key)
                unique_logs.append(log)

        return sorted(unique_logs, key=lambda x: x.timestamp)

    def archive_old_logs(self) -> int:
        """
        Archive old log files (public method).

        Returns:
            Number of files archived
        """
        return self._archive_old_logs()

    def _write_log_entry(
        self, log_entry: ResearchLog, session_id: Optional[str] = None
    ) -> None:
        """Write a log entry to the current log file."""
        if not self._current_log_file:
            self._current_log_file = self._get_current_log_file()

        log_data = {
            "timestamp": log_entry.timestamp.isoformat(),
            "activity_type": log_entry.activity_type,
            "source_id": log_entry.source_id,
            "description": log_entry.description,
            "outcome": log_entry.outcome,
            "duration_minutes": log_entry.duration_minutes,
            "session_id": session_id,
        }

        try:
            with open(self._current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")
        except IOError as e:
            logger.error(f"Failed to write log entry: {e}")

    def _get_current_log_file(self) -> Path:
        """Get the current log file path, creating it if necessary."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_directory / f"research_activity_{today}.jsonl"

        if not log_file.exists():
            log_file.touch()

        return log_file

    def _check_rotation(self) -> None:
        """Check if log rotation is needed and perform it if necessary."""
        # Ensure log file is initialized
        if not self._current_log_file:
            self._current_log_file = self._get_current_log_file()

        if not self._current_log_file.exists():
            return

        try:
            # Check file size
            if self._current_log_file.stat().st_size >= self.max_log_size_bytes:
                self._rotate_log()
                return  # Don't check age if we just rotated

            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(
                self._current_log_file.stat().st_mtime
            )
            if file_age.days >= self.rotation_interval_days:
                self._rotate_log()
        except (OSError, AttributeError) as e:
            logger.warning(f"Error checking log rotation: {e}")

    def _rotate_log(self) -> None:
        """Rotate the current log file."""
        if not self._current_log_file:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use .log extension for rotated files (test compatibility)
        rotated_file = self.log_directory / f"research_activity_{timestamp}.log"

        try:
            self._current_log_file.rename(rotated_file)
            logger.info(f"Rotated log file: {rotated_file.name}")
            self._current_log_file = None

            # Archive old logs if enabled
            if self.enable_archival:
                self._archive_old_logs()
        except IOError as e:
            logger.error(f"Failed to rotate log file: {e}")

    def _archive_old_logs(self) -> int:
        """
        Archive log files older than max_log_age_days.

        Returns:
            Number of files archived
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_log_age_days)
        archived_count = 0

        for log_file in self.log_directory.glob("research_activity_*.jsonl"):
            if log_file == self._current_log_file:
                continue

            file_age = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_age < cutoff_date:
                try:
                    archived_path = self.archival_directory / log_file.name
                    log_file.rename(archived_path)
                    logger.info(f"Archived log file: {archived_path.name}")
                    archived_count += 1
                except IOError as e:
                    logger.error(f"Failed to archive log file {log_file.name}: {e}")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about logged activities.

        Returns:
            Dictionary with statistics about logs
        """
        with self._lock:
            total_logs = sum(
                len(logs) for logs in self._session_logs.values()
            )
            activity_counts: Dict[str, int] = {}
            total_duration = 0

            for session_logs in self._session_logs.values():
                for log in session_logs:
                    activity_counts[log.activity_type] = (
                        activity_counts.get(log.activity_type, 0) + 1
                    )
                    total_duration += log.duration_minutes

            return {
                "total_logs": total_logs,
                "sessions": len(self._session_logs),
                "activity_counts": activity_counts,
                "total_duration_minutes": total_duration,
                "average_duration_minutes": (
                    total_duration / total_logs if total_logs > 0 else 0
                ),
            }

    def clear_session_logs(self, session_id: str) -> None:
        """
        Clear all logs for a specific session.

        Args:
            session_id: Session ID to clear logs for
        """
        with self._lock:
            if session_id in self._session_logs:
                del self._session_logs[session_id]
                logger.info(f"Cleared logs for session: {session_id}")

    def export_logs(
        self, output_path: Path, session_id: Optional[str] = None
    ) -> None:
        """
        Export logs to a JSON file.

        Args:
            output_path: Path to write the exported logs
            session_id: Optional session ID to export (exports all if None)
        """
        with self._lock:
            if session_id:
                logs = self._session_logs.get(session_id, [])
            else:
                logs = []
                for session_logs in self._session_logs.values():
                    logs.extend(session_logs)

            log_data = [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "activity_type": log.activity_type,
                    "source_id": log.source_id,
                    "description": log.description,
                    "outcome": log.outcome,
                    "duration_minutes": log.duration_minutes,
                }
                for log in logs
            ]

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2)
                logger.info(f"Exported {len(log_data)} logs to {output_path}")
            except IOError as e:
                logger.error(f"Failed to export logs: {e}")

