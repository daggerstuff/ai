#!/usr/bin/env python3
"""
Quality Validation Progress Tracking System for Pixelated Empathy AI
Tracks progress of distributed quality validation across multiple workers
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading
import sqlite3
from enum import Enum
import asyncio
from collections import defaultdict

# Redis for real-time progress updates
try:
    import redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using local tracking only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class BatchStatus(Enum):
    """Batch status enumeration"""
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskProgress:
    """Individual task progress information"""
    task_id: str
    batch_id: str
    file_path: str
    status: TaskStatus
    worker_id: Optional[str] = None
    progress_percentage: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    estimated_completion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskProgress':
        """Create from dictionary"""
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


@dataclass
class BatchProgress:
    """Batch progress information"""
    batch_id: str
    batch_name: str
    status: BatchStatus
    total_tasks: int
    queued_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    progress_percentage: float = 0.0
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchProgress':
        """Create from dictionary"""
        data['status'] = BatchStatus(data['status'])
        return cls(**data)


class ProgressTracker:
    """Tracks progress of quality validation tasks"""
    
    def __init__(self, db_path: str = None, redis_url: str = None):
        self.db_path = db_path or "progress_tracking.db"
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Initialize database
        self._init_database()
        
        # Initialize Redis connection
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for real-time progress tracking")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                self.redis_client = None
        
        # In-memory cache for fast access
        self.task_cache: Dict[str, TaskProgress] = {}
        self.batch_cache: Dict[str, BatchProgress] = {}
        self.lock = threading.Lock()
        
        # Progress update callbacks
        self.progress_callbacks: List[Callable] = []
        
        # Load existing data from database
        self._load_from_database()
    
    def _init_database(self):
        """Initialize SQLite database for progress tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_progress (
                    batch_id TEXT PRIMARY KEY,
                    batch_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_tasks INTEGER NOT NULL,
                    queued_tasks INTEGER DEFAULT 0,
                    processing_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    failed_tasks INTEGER DEFAULT 0,
                    cancelled_tasks INTEGER DEFAULT 0,
                    progress_percentage REAL DEFAULT 0.0,
                    started_at TEXT,
                    estimated_completion TEXT,
                    completed_at TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_progress (
                    task_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    progress_percentage REAL DEFAULT 0.0,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    estimated_completion TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (batch_id) REFERENCES batch_progress (batch_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_progress(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_batch ON task_progress(batch_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON task_progress(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_worker ON task_progress(worker_id)")
    
    def _load_from_database(self):
        """Load existing progress data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load batch progress
                cursor = conn.execute("SELECT * FROM batch_progress")
                for row in cursor.fetchall():
                    batch_progress = BatchProgress(
                        batch_id=row[0],
                        batch_name=row[1],
                        status=BatchStatus(row[2]),
                        total_tasks=row[3],
                        queued_tasks=row[4],
                        processing_tasks=row[5],
                        completed_tasks=row[6],
                        failed_tasks=row[7],
                        cancelled_tasks=row[8],
                        progress_percentage=row[9],
                        started_at=row[10],
                        estimated_completion=row[11],
                        completed_at=row[12]
                    )
                    self.batch_cache[batch_progress.batch_id] = batch_progress
                
                # Load task progress
                cursor = conn.execute("SELECT * FROM task_progress")
                for row in cursor.fetchall():
                    task_progress = TaskProgress(
                        task_id=row[0],
                        batch_id=row[1],
                        file_path=row[2],
                        status=TaskStatus(row[3]),
                        worker_id=row[4],
                        progress_percentage=row[5],
                        started_at=row[6],
                        completed_at=row[7],
                        error_message=row[8],
                        retry_count=row[9],
                        estimated_completion=row[10]
                    )
                    self.task_cache[task_progress.task_id] = task_progress
                
                logger.info(f"Loaded {len(self.batch_cache)} batches and {len(self.task_cache)} tasks from database")
                
        except Exception as e:
            logger.error(f"Failed to load progress data from database: {e}")
    
    def create_batch(self, batch_id: str, batch_name: str, task_ids: List[str]) -> BatchProgress:
        """Create a new batch for tracking"""
        with self.lock:
            batch_progress = BatchProgress(
                batch_id=batch_id,
                batch_name=batch_name,
                status=BatchStatus.CREATED,
                total_tasks=len(task_ids),
                queued_tasks=len(task_ids)
            )
            
            # Store in cache and database
            self.batch_cache[batch_id] = batch_progress
            self._store_batch_in_db(batch_progress)
            
            # Create task progress entries
            for task_id in task_ids:
                task_progress = TaskProgress(
                    task_id=task_id,
                    batch_id=batch_id,
                    file_path="",  # Will be updated when task starts
                    status=TaskStatus.PENDING
                )
                self.task_cache[task_id] = task_progress
                self._store_task_in_db(task_progress)
            
            # Publish to Redis
            self._publish_batch_update(batch_progress)
            
            logger.info(f"Created batch {batch_id} with {len(task_ids)} tasks")
            return batch_progress
    
    def update_task_progress(self, task_id: str, **kwargs):
        """Update progress for a specific task"""
        with self.lock:
            if task_id not in self.task_cache:
                logger.warning(f"Task {task_id} not found in cache")
                return
            
            task_progress = self.task_cache[task_id]
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(task_progress, field):
                    if field == 'status' and isinstance(value, str):
                        value = TaskStatus(value)
                    setattr(task_progress, field, value)
            
            # Update timestamps
            if task_progress.status == TaskStatus.PROCESSING and not task_progress.started_at:
                task_progress.started_at = datetime.now(timezone.utc).isoformat()
            elif task_progress.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if not task_progress.completed_at:
                    task_progress.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Store in database
            self._store_task_in_db(task_progress)
            
            # Update batch progress
            self._update_batch_progress(task_progress.batch_id)
            
            # Publish to Redis
            self._publish_task_update(task_progress)
            
            # Trigger callbacks
            self._trigger_callbacks('task_updated', task_progress)
    
    def _update_batch_progress(self, batch_id: str):
        """Update batch progress based on task statuses"""
        if batch_id not in self.batch_cache:
            return
        
        batch_progress = self.batch_cache[batch_id]
        
        # Count tasks by status
        task_counts = defaultdict(int)
        for task in self.task_cache.values():
            if task.batch_id == batch_id:
                task_counts[task.status] += 1
        
        # Update batch counters
        batch_progress.queued_tasks = task_counts[TaskStatus.PENDING] + task_counts[TaskStatus.QUEUED]
        batch_progress.processing_tasks = task_counts[TaskStatus.PROCESSING] + task_counts[TaskStatus.RETRYING]
        batch_progress.completed_tasks = task_counts[TaskStatus.COMPLETED]
        batch_progress.failed_tasks = task_counts[TaskStatus.FAILED]
        batch_progress.cancelled_tasks = task_counts[TaskStatus.CANCELLED]
        
        # Calculate progress percentage
        finished_tasks = batch_progress.completed_tasks + batch_progress.failed_tasks + batch_progress.cancelled_tasks
        batch_progress.progress_percentage = (finished_tasks / batch_progress.total_tasks) * 100 if batch_progress.total_tasks > 0 else 0
        
        # Update batch status
        if batch_progress.processing_tasks > 0:
            batch_progress.status = BatchStatus.PROCESSING
            if not batch_progress.started_at:
                batch_progress.started_at = datetime.now(timezone.utc).isoformat()
        elif finished_tasks == batch_progress.total_tasks:
            if batch_progress.failed_tasks == 0:
                batch_progress.status = BatchStatus.COMPLETED
            else:
                batch_progress.status = BatchStatus.FAILED
            if not batch_progress.completed_at:
                batch_progress.completed_at = datetime.now(timezone.utc).isoformat()
        elif batch_progress.queued_tasks > 0:
            batch_progress.status = BatchStatus.QUEUED
        
        # Estimate completion time
        if batch_progress.status == BatchStatus.PROCESSING:
            batch_progress.estimated_completion = self._estimate_completion_time(batch_id)
        
        # Store in database
        self._store_batch_in_db(batch_progress)
        
        # Publish to Redis
        self._publish_batch_update(batch_progress)
        
        # Trigger callbacks
        self._trigger_callbacks('batch_updated', batch_progress)
    
    def _estimate_completion_time(self, batch_id: str) -> Optional[str]:
        """Estimate completion time for a batch"""
        try:
            # Get completed tasks for this batch
            completed_tasks = [
                task for task in self.task_cache.values()
                if task.batch_id == batch_id and task.status == TaskStatus.COMPLETED
                and task.started_at and task.completed_at
            ]
            
            if len(completed_tasks) < 3:  # Need at least 3 samples
                return None
            
            # Calculate average processing time
            total_time = 0
            for task in completed_tasks:
                start_time = datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(task.completed_at.replace('Z', '+00:00'))
                total_time += (end_time - start_time).total_seconds()
            
            avg_time = total_time / len(completed_tasks)
            
            # Get remaining tasks
            batch_progress = self.batch_cache[batch_id]
            remaining_tasks = batch_progress.total_tasks - batch_progress.completed_tasks - batch_progress.failed_tasks - batch_progress.cancelled_tasks
            
            # Estimate completion time
            estimated_seconds = remaining_tasks * avg_time
            estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=estimated_seconds)
            
            return estimated_completion.isoformat()
            
        except Exception as e:
            logger.warning(f"Failed to estimate completion time: {e}")
            return None
    
    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get progress for a specific batch"""
        return self.batch_cache.get(batch_id)
    
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get progress for a specific task"""
        return self.task_cache.get(task_id)
    
    def get_batch_tasks(self, batch_id: str) -> List[TaskProgress]:
        """Get all tasks for a batch"""
        return [task for task in self.task_cache.values() if task.batch_id == batch_id]
    
    def get_worker_tasks(self, worker_id: str) -> List[TaskProgress]:
        """Get all tasks assigned to a worker"""
        return [task for task in self.task_cache.values() if task.worker_id == worker_id]
    
    def get_active_batches(self) -> List[BatchProgress]:
        """Get all active (non-completed) batches"""
        return [
            batch for batch in self.batch_cache.values()
            if batch.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
        ]
    
    def get_batch_statistics(self, batch_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a batch"""
        batch_progress = self.get_batch_progress(batch_id)
        if not batch_progress:
            return {}
        
        tasks = self.get_batch_tasks(batch_id)
        
        # Worker statistics
        worker_stats = defaultdict(lambda: {'tasks': 0, 'completed': 0, 'failed': 0})
        for task in tasks:
            if task.worker_id:
                worker_stats[task.worker_id]['tasks'] += 1
                if task.status == TaskStatus.COMPLETED:
                    worker_stats[task.worker_id]['completed'] += 1
                elif task.status == TaskStatus.FAILED:
                    worker_stats[task.worker_id]['failed'] += 1
        
        # Processing time statistics
        processing_times = []
        for task in tasks:
            if task.started_at and task.completed_at:
                try:
                    start_time = datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(task.completed_at.replace('Z', '+00:00'))
                    processing_times.append((end_time - start_time).total_seconds())
                except Exception:
                    pass
        
        time_stats = {}
        if processing_times:
            import statistics
            time_stats = {
                'mean': statistics.mean(processing_times),
                'median': statistics.median(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'std_dev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            }
        
        return {
            'batch_progress': batch_progress.to_dict(),
            'worker_statistics': dict(worker_stats),
            'processing_time_statistics': time_stats,
            'task_count_by_status': {
                status.value: len([t for t in tasks if t.status == status])
                for status in TaskStatus
            }
        }
    
    def cancel_batch(self, batch_id: str):
        """Cancel a batch and all its pending/processing tasks"""
        with self.lock:
            if batch_id not in self.batch_cache:
                logger.warning(f"Batch {batch_id} not found")
                return
            
            batch_progress = self.batch_cache[batch_id]
            batch_progress.status = BatchStatus.CANCELLED
            batch_progress.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Cancel all non-completed tasks
            for task in self.task_cache.values():
                if task.batch_id == batch_id and task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    self._store_task_in_db(task)
                    self._publish_task_update(task)
            
            # Update batch progress
            self._update_batch_progress(batch_id)
            
            logger.info(f"Cancelled batch {batch_id}")
    
    def pause_batch(self, batch_id: str):
        """Pause a batch (mark as paused, but don't cancel tasks)"""
        with self.lock:
            if batch_id not in self.batch_cache:
                logger.warning(f"Batch {batch_id} not found")
                return
            
            batch_progress = self.batch_cache[batch_id]
            batch_progress.status = BatchStatus.PAUSED
            
            self._store_batch_in_db(batch_progress)
            self._publish_batch_update(batch_progress)
            
            logger.info(f"Paused batch {batch_id}")
    
    def resume_batch(self, batch_id: str):
        """Resume a paused batch"""
        with self.lock:
            if batch_id not in self.batch_cache:
                logger.warning(f"Batch {batch_id} not found")
                return
            
            batch_progress = self.batch_cache[batch_id]
            if batch_progress.status == BatchStatus.PAUSED:
                # Determine appropriate status based on task states
                self._update_batch_progress(batch_id)
                
                logger.info(f"Resumed batch {batch_id}")
    
    def add_progress_callback(self, callback: Callable):
        """Add a callback function for progress updates"""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """Remove a progress callback"""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def _store_batch_in_db(self, batch_progress: BatchProgress):
        """Store batch progress in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO batch_progress (
                        batch_id, batch_name, status, total_tasks, queued_tasks,
                        processing_tasks, completed_tasks, failed_tasks, cancelled_tasks,
                        progress_percentage, started_at, estimated_completion, completed_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    batch_progress.batch_id,
                    batch_progress.batch_name,
                    batch_progress.status.value,
                    batch_progress.total_tasks,
                    batch_progress.queued_tasks,
                    batch_progress.processing_tasks,
                    batch_progress.completed_tasks,
                    batch_progress.failed_tasks,
                    batch_progress.cancelled_tasks,
                    batch_progress.progress_percentage,
                    batch_progress.started_at,
                    batch_progress.estimated_completion,
                    batch_progress.completed_at
                ))
        except Exception as e:
            logger.error(f"Failed to store batch progress in database: {e}")
    
    def _store_task_in_db(self, task_progress: TaskProgress):
        """Store task progress in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO task_progress (
                        task_id, batch_id, file_path, status, worker_id,
                        progress_percentage, started_at, completed_at, error_message,
                        retry_count, estimated_completion, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    task_progress.task_id,
                    task_progress.batch_id,
                    task_progress.file_path,
                    task_progress.status.value,
                    task_progress.worker_id,
                    task_progress.progress_percentage,
                    task_progress.started_at,
                    task_progress.completed_at,
                    task_progress.error_message,
                    task_progress.retry_count,
                    task_progress.estimated_completion
                ))
        except Exception as e:
            logger.error(f"Failed to store task progress in database: {e}")
    
    def _publish_batch_update(self, batch_progress: BatchProgress):
        """Publish batch update to Redis"""
        if self.redis_client:
            try:
                channel = f"batch_progress:{batch_progress.batch_id}"
                message = json.dumps(batch_progress.to_dict())
                self.redis_client.publish(channel, message)
            except Exception as e:
                logger.warning(f"Failed to publish batch update to Redis: {e}")
    
    def _publish_task_update(self, task_progress: TaskProgress):
        """Publish task update to Redis"""
        if self.redis_client:
            try:
                channel = f"task_progress:{task_progress.task_id}"
                message = json.dumps(task_progress.to_dict())
                self.redis_client.publish(channel, message)
            except Exception as e:
                logger.warning(f"Failed to publish task update to Redis: {e}")
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old completed batches and tasks"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get old batch IDs
                cursor = conn.execute("""
                    SELECT batch_id FROM batch_progress
                    WHERE status IN ('completed', 'failed', 'cancelled')
                    AND completed_at < ?
                """, (cutoff_str,))
                
                old_batch_ids = [row[0] for row in cursor.fetchall()]
                
                if old_batch_ids:
                    # Delete old tasks
                    placeholders = ','.join('?' * len(old_batch_ids))
                    conn.execute(f"""
                        DELETE FROM task_progress
                        WHERE batch_id IN ({placeholders})
                    """, old_batch_ids)
                    
                    # Delete old batches
                    conn.execute(f"""
                        DELETE FROM batch_progress
                        WHERE batch_id IN ({placeholders})
                    """, old_batch_ids)
                    
                    # Remove from cache
                    for batch_id in old_batch_ids:
                        self.batch_cache.pop(batch_id, None)
                        # Remove associated tasks from cache
                        task_ids_to_remove = [
                            task_id for task_id, task in self.task_cache.items()
                            if task.batch_id == batch_id
                        ]
                        for task_id in task_ids_to_remove:
                            self.task_cache.pop(task_id, None)
                    
                    logger.info(f"Cleaned up {len(old_batch_ids)} old batches")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


def main():
    """Main CLI interface"""
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description="Quality Validation Progress Tracker")
    parser.add_argument('--db-path', help="Database path for storing progress")
    parser.add_argument('--redis-url', help="Redis URL for real-time updates")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show batch status')
    status_parser.add_argument('batch_id', help='Batch ID to check')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all batches')
    list_parser.add_argument('--active-only', action='store_true', help='Show only active batches')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show batch statistics')
    stats_parser.add_argument('batch_id', help='Batch ID for statistics')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a batch')
    cancel_parser.add_argument('batch_id', help='Batch ID to cancel')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create progress tracker
    tracker = ProgressTracker(args.db_path, args.redis_url)
    
    if args.command == 'status':
        batch_progress = tracker.get_batch_progress(args.batch_id)
        if batch_progress:
            print(json.dumps(batch_progress.to_dict(), indent=2))
        else:
            print(f"Batch not found: {args.batch_id}")
    
    elif args.command == 'list':
        if args.active_only:
            batches = tracker.get_active_batches()
        else:
            batches = list(tracker.batch_cache.values())
        
        batch_data = [batch.to_dict() for batch in batches]
        print(json.dumps(batch_data, indent=2))
    
    elif args.command == 'stats':
        stats = tracker.get_batch_statistics(args.batch_id)
        if stats:
            print(json.dumps(stats, indent=2))
        else:
            print(f"No statistics found for batch: {args.batch_id}")
    
    elif args.command == 'cancel':
        tracker.cancel_batch(args.batch_id)
        print(f"Cancelled batch: {args.batch_id}")
    
    elif args.command == 'cleanup':
        tracker.cleanup_old_data(args.days)
        print(f"Cleaned up data older than {args.days} days")


if __name__ == '__main__':
    main()
