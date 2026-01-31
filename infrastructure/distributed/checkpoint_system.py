#!/usr/bin/env python3
"""
Processing Checkpoint System for Pixelated Empathy AI
Implements checkpoint creation, storage, and recovery for fault tolerance
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import threading
from pathlib import Path
import gzip
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointType(Enum):
    """Types of checkpoints"""
    PROCESSING_STATE = "processing_state"
    BATCH_PROGRESS = "batch_progress"
    MODEL_STATE = "model_state"
    QUEUE_STATE = "queue_state"
    SYSTEM_STATE = "system_state"
    CUSTOM = "custom"

class CheckpointStatus(Enum):
    """Checkpoint status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    ARCHIVED = "archived"

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    process_id: str
    task_id: str
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    size_bytes: int = 0
    compression: bool = True
    encryption: bool = False
    ttl_hours: int = 24
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.checkpoint_type, str):
            self.checkpoint_type = CheckpointType(self.checkpoint_type)
        if isinstance(self.status, str):
            self.status = CheckpointStatus(self.status)

@dataclass
class ProcessingState:
    """State of a processing operation"""
    process_id: str
    task_id: str
    current_step: str
    total_steps: int
    completed_steps: int
    progress_percentage: float
    start_time: datetime
    last_update: datetime
    estimated_completion: Optional[datetime] = None
    current_batch: Optional[Dict[str, Any]] = None
    processed_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self, completed_steps: int = None, current_step: str = None):
        """Update processing progress"""
        if completed_steps is not None:
            self.completed_steps = completed_steps
            self.progress_percentage = (completed_steps / self.total_steps) * 100
        
        if current_step is not None:
            self.current_step = current_step
        
        self.last_update = datetime.utcnow()
        
        # Estimate completion time
        if self.progress_percentage > 0:
            elapsed = (self.last_update - self.start_time).total_seconds()
            total_estimated = elapsed / (self.progress_percentage / 100)
            remaining = total_estimated - elapsed
            self.estimated_completion = self.last_update + timedelta(seconds=remaining)

class CheckpointStorage:
    """Storage backend for checkpoints"""
    
    def __init__(self, storage_path: str = "/home/vivi/pixelated/ai/infrastructure/distributed/checkpoints"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "checkpoints.db"
        self.data_path = self.storage_path / "data"
        self.data_path.mkdir(exist_ok=True)
        self.setup_database()
    
    def setup_database(self):
        """Initialize checkpoint database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    checkpoint_type TEXT NOT NULL,
                    process_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    version TEXT DEFAULT '1.0',
                    description TEXT,
                    tags TEXT,
                    dependencies TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    compression BOOLEAN DEFAULT 1,
                    encryption BOOLEAN DEFAULT 0,
                    ttl_hours INTEGER DEFAULT 24,
                    status TEXT DEFAULT 'active',
                    file_path TEXT,
                    checksum TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_process_task 
                ON checkpoints (process_id, task_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON checkpoints (created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON checkpoints (status)
            """)
    
    def save_checkpoint(self, metadata: CheckpointMetadata, data: Any) -> str:
        """Save checkpoint data and metadata"""
        
        # Generate file path
        file_name = f"{metadata.checkpoint_id}.pkl"
        if metadata.compression:
            file_name += ".gz"
        
        file_path = self.data_path / file_name
        
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Compress if enabled
            if metadata.compression:
                with gzip.open(file_path, 'wb') as f:
                    f.write(serialized_data)
            else:
                with open(file_path, 'wb') as f:
                    f.write(serialized_data)
            
            # Calculate size and checksum
            metadata.size_bytes = file_path.stat().st_size
            checksum = self._calculate_checksum(file_path)
            
            # Save metadata to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints 
                    (checkpoint_id, checkpoint_type, process_id, task_id, created_at,
                     version, description, tags, dependencies, size_bytes, compression,
                     encryption, ttl_hours, status, file_path, checksum, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.checkpoint_id,
                    metadata.checkpoint_type.value,
                    metadata.process_id,
                    metadata.task_id,
                    metadata.created_at.isoformat(),
                    metadata.version,
                    metadata.description,
                    json.dumps(metadata.tags),
                    json.dumps(metadata.dependencies),
                    metadata.size_bytes,
                    metadata.compression,
                    metadata.encryption,
                    metadata.ttl_hours,
                    metadata.status.value,
                    str(file_path),
                    checksum,
                    json.dumps(asdict(metadata))
                ))
            
            logger.info(f"Saved checkpoint {metadata.checkpoint_id} ({metadata.size_bytes} bytes)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {metadata.checkpoint_id}: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> tuple[CheckpointMetadata, Any]:
        """Load checkpoint data and metadata"""
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT checkpoint_type, process_id, task_id, created_at, version,
                       description, tags, dependencies, size_bytes, compression,
                       encryption, ttl_hours, status, file_path, checksum, metadata
                FROM checkpoints WHERE checkpoint_id = ?
            """, (checkpoint_id,)).fetchone()
            
            if not row:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            # Reconstruct metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=CheckpointType(row[0]),
                process_id=row[1],
                task_id=row[2],
                created_at=datetime.fromisoformat(row[3]),
                version=row[4],
                description=row[5] or "",
                tags=json.loads(row[6]) if row[6] else [],
                dependencies=json.loads(row[7]) if row[7] else [],
                size_bytes=row[8],
                compression=bool(row[9]),
                encryption=bool(row[10]),
                ttl_hours=row[11],
                status=CheckpointStatus(row[12])
            )
            
            file_path = Path(row[13])
            stored_checksum = row[14]
            
            # Verify file exists and checksum
            if not file_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
            
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != stored_checksum:
                raise ValueError(f"Checkpoint {checkpoint_id} corrupted (checksum mismatch)")
            
            # Load data
            try:
                if metadata.compression:
                    with gzip.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                
                logger.info(f"Loaded checkpoint {checkpoint_id}")
                return metadata, data
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
                raise
    
    def list_checkpoints(self, process_id: str = None, task_id: str = None, 
                        checkpoint_type: CheckpointType = None,
                        status: CheckpointStatus = None) -> List[CheckpointMetadata]:
        """List checkpoints with optional filters"""
        
        query = "SELECT * FROM checkpoints WHERE 1=1"
        params = []
        
        if process_id:
            query += " AND process_id = ?"
            params.append(process_id)
        
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        
        if checkpoint_type:
            query += " AND checkpoint_type = ?"
            params.append(checkpoint_type.value)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            
            checkpoints = []
            for row in rows:
                metadata = CheckpointMetadata(
                    checkpoint_id=row[0],
                    checkpoint_type=CheckpointType(row[1]),
                    process_id=row[2],
                    task_id=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    version=row[5],
                    description=row[6] or "",
                    tags=json.loads(row[7]) if row[7] else [],
                    dependencies=json.loads(row[8]) if row[8] else [],
                    size_bytes=row[9],
                    compression=bool(row[10]),
                    encryption=bool(row[11]),
                    ttl_hours=row[12],
                    status=CheckpointStatus(row[13])
                )
                checkpoints.append(metadata)
            
            return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path
                row = conn.execute(
                    "SELECT file_path FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,)
                ).fetchone()
                
                if not row:
                    return False
                
                file_path = Path(row[0])
                
                # Delete file
                if file_path.exists():
                    file_path.unlink()
                
                # Delete database record
                conn.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,)
                )
            
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_expired_checkpoints(self) -> int:
        """Clean up expired checkpoints"""
        
        deleted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Find expired checkpoints
            cutoff_time = datetime.utcnow()
            
            expired_checkpoints = conn.execute("""
                SELECT checkpoint_id, file_path, created_at, ttl_hours
                FROM checkpoints 
                WHERE status = 'active'
            """).fetchall()
            
            for checkpoint_id, file_path, created_at_str, ttl_hours in expired_checkpoints:
                created_at = datetime.fromisoformat(created_at_str)
                expiry_time = created_at + timedelta(hours=ttl_hours)
                
                if cutoff_time > expiry_time:
                    # Delete expired checkpoint
                    try:
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()
                        
                        conn.execute(
                            "UPDATE checkpoints SET status = 'expired' WHERE checkpoint_id = ?",
                            (checkpoint_id,)
                        )
                        
                        deleted_count += 1
                        logger.info(f"Expired checkpoint {checkpoint_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to expire checkpoint {checkpoint_id}: {e}")
        
        return deleted_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Count by status
            status_counts = {}
            for status in CheckpointStatus:
                count = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE status = ?",
                    (status.value,)
                ).fetchone()[0]
                status_counts[status.value] = count
            
            # Total size
            total_size = conn.execute(
                "SELECT SUM(size_bytes) FROM checkpoints WHERE status = 'active'"
            ).fetchone()[0] or 0
            
            # Count by type
            type_counts = {}
            for checkpoint_type in CheckpointType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE checkpoint_type = ? AND status = 'active'",
                    (checkpoint_type.value,)
                ).fetchone()[0]
                type_counts[checkpoint_type.value] = count
        
        return {
            "status_counts": status_counts,
            "type_counts": type_counts,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_path": str(self.storage_path)
        }
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

class CheckpointManager:
    """High-level checkpoint management system"""
    
    def __init__(self, storage_path: str = None):
        self.storage = CheckpointStorage(storage_path)
        self.active_processes: Dict[str, ProcessingState] = {}
        self.checkpoint_callbacks: Dict[str, List[Callable]] = {}
        self.auto_checkpoint_interval = 300  # 5 minutes
        self.cleanup_interval = 3600  # 1 hour
        self._cleanup_thread = None
        self._running = False
    
    def start_background_tasks(self):
        """Start background cleanup and auto-checkpoint tasks"""
        
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("Started checkpoint background tasks")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        logger.info("Stopped checkpoint background tasks")
    
    def register_process(self, process_id: str, task_id: str, total_steps: int,
                        description: str = "") -> ProcessingState:
        """Register a new processing operation"""
        
        state = ProcessingState(
            process_id=process_id,
            task_id=task_id,
            current_step="initialized",
            total_steps=total_steps,
            completed_steps=0,
            progress_percentage=0.0,
            start_time=datetime.utcnow(),
            last_update=datetime.utcnow(),
            metadata={"description": description}
        )
        
        self.active_processes[process_id] = state
        
        # Create initial checkpoint
        self.create_checkpoint(
            process_id=process_id,
            task_id=task_id,
            checkpoint_type=CheckpointType.PROCESSING_STATE,
            data=state,
            description=f"Initial state for {description}"
        )
        
        logger.info(f"Registered process {process_id} for task {task_id}")
        return state
    
    def create_checkpoint(self, process_id: str, task_id: str, 
                         checkpoint_type: CheckpointType, data: Any,
                         description: str = "", tags: List[str] = None,
                         ttl_hours: int = 24) -> str:
        """Create a new checkpoint"""
        
        checkpoint_id = f"{process_id}_{checkpoint_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            created_at=datetime.utcnow(),
            process_id=process_id,
            task_id=task_id,
            description=description,
            tags=tags or [],
            ttl_hours=ttl_hours
        )
        
        try:
            file_path = self.storage.save_checkpoint(metadata, data)
            
            # Trigger callbacks
            self._trigger_callbacks(checkpoint_id, "created", metadata)
            
            logger.info(f"Created checkpoint {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def update_process_progress(self, process_id: str, completed_steps: int = None,
                              current_step: str = None, metadata: Dict[str, Any] = None,
                              auto_checkpoint: bool = True) -> ProcessingState:
        """Update process progress and optionally create checkpoint"""
        
        if process_id not in self.active_processes:
            raise ValueError(f"Process {process_id} not registered")
        
        state = self.active_processes[process_id]
        state.update_progress(completed_steps, current_step)
        
        if metadata:
            state.metadata.update(metadata)
        
        # Auto-checkpoint if enabled and significant progress made
        if auto_checkpoint and completed_steps is not None:
            progress_delta = completed_steps - (state.metadata.get("last_checkpoint_step", 0))
            if progress_delta >= max(1, state.total_steps // 20):  # Checkpoint every 5% progress
                self.create_checkpoint(
                    process_id=process_id,
                    task_id=state.task_id,
                    checkpoint_type=CheckpointType.PROCESSING_STATE,
                    data=state,
                    description=f"Progress checkpoint at {state.progress_percentage:.1f}%"
                )
                state.metadata["last_checkpoint_step"] = completed_steps
        
        return state
    
    def complete_process(self, process_id: str, final_data: Any = None) -> str:
        """Mark process as completed and create final checkpoint"""
        
        if process_id not in self.active_processes:
            raise ValueError(f"Process {process_id} not registered")
        
        state = self.active_processes[process_id]
        state.completed_steps = state.total_steps
        state.progress_percentage = 100.0
        state.current_step = "completed"
        state.last_update = datetime.utcnow()
        
        # Create final checkpoint
        checkpoint_data = {
            "state": state,
            "final_data": final_data
        }
        
        checkpoint_id = self.create_checkpoint(
            process_id=process_id,
            task_id=state.task_id,
            checkpoint_type=CheckpointType.PROCESSING_STATE,
            data=checkpoint_data,
            description="Final completion checkpoint",
            ttl_hours=168  # Keep completion checkpoints for 1 week
        )
        
        # Mark older checkpoints as completed
        self._mark_process_checkpoints_completed(process_id)
        
        # Remove from active processes
        del self.active_processes[process_id]
        
        logger.info(f"Completed process {process_id}")
        return checkpoint_id
    
    def recover_process(self, process_id: str) -> Optional[ProcessingState]:
        """Recover process state from latest checkpoint"""
        
        checkpoints = self.storage.list_checkpoints(
            process_id=process_id,
            checkpoint_type=CheckpointType.PROCESSING_STATE,
            status=CheckpointStatus.ACTIVE
        )
        
        if not checkpoints:
            logger.warning(f"No checkpoints found for process {process_id}")
            return None
        
        # Get latest checkpoint
        latest_checkpoint = checkpoints[0]
        
        try:
            metadata, data = self.storage.load_checkpoint(latest_checkpoint.checkpoint_id)
            
            # Extract state from checkpoint data
            if isinstance(data, ProcessingState):
                state = data
            elif isinstance(data, dict) and "state" in data:
                state = data["state"]
            else:
                logger.error(f"Invalid checkpoint data format for {process_id}")
                return None
            
            # Restore to active processes
            self.active_processes[process_id] = state
            
            logger.info(f"Recovered process {process_id} from checkpoint {latest_checkpoint.checkpoint_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to recover process {process_id}: {e}")
            return None
    
    def add_checkpoint_callback(self, event_type: str, callback: Callable):
        """Add callback for checkpoint events"""
        
        if event_type not in self.checkpoint_callbacks:
            self.checkpoint_callbacks[event_type] = []
        
        self.checkpoint_callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, checkpoint_id: str, event_type: str, metadata: CheckpointMetadata):
        """Trigger registered callbacks"""
        
        if event_type in self.checkpoint_callbacks:
            for callback in self.checkpoint_callbacks[event_type]:
                try:
                    callback(checkpoint_id, event_type, metadata)
                except Exception as e:
                    logger.error(f"Callback error for {event_type}: {e}")
    
    def _mark_process_checkpoints_completed(self, process_id: str):
        """Mark all checkpoints for a process as completed"""
        
        with sqlite3.connect(self.storage.db_path) as conn:
            conn.execute("""
                UPDATE checkpoints 
                SET status = 'completed' 
                WHERE process_id = ? AND status = 'active'
            """, (process_id,))
    
    def _background_cleanup(self):
        """Background cleanup task"""
        
        while self._running:
            try:
                # Clean up expired checkpoints
                deleted_count = self.storage.cleanup_expired_checkpoints()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired checkpoints")
                
                # Sleep until next cleanup
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "active_processes": len(self.active_processes),
            "process_details": {
                pid: {
                    "task_id": state.task_id,
                    "progress": state.progress_percentage,
                    "current_step": state.current_step,
                    "start_time": state.start_time.isoformat(),
                    "estimated_completion": state.estimated_completion.isoformat() if state.estimated_completion else None
                }
                for pid, state in self.active_processes.items()
            }
        }

# Example usage and testing
async def example_usage():
    """Example of how to use the checkpoint system"""
    
    # Initialize checkpoint manager
    manager = CheckpointManager()
    manager.start_background_tasks()
    
    try:
        # Register a processing operation
        process_id = "example_process_001"
        task_id = "data_processing_task"
        
        state = manager.register_process(
            process_id=process_id,
            task_id=task_id,
            total_steps=100,
            description="Example data processing operation"
        )
        
        print(f"Registered process: {process_id}")
        
        # Simulate processing with checkpoints
        for step in range(0, 101, 10):
            # Update progress
            manager.update_process_progress(
                process_id=process_id,
                completed_steps=step,
                current_step=f"Processing batch {step//10 + 1}",
                metadata={"current_batch": step//10 + 1}
            )
            
            print(f"Progress: {step}% - {state.current_step}")
            
            # Simulate some work
            await asyncio.sleep(0.1)
        
        # Complete the process
        final_checkpoint = manager.complete_process(
            process_id=process_id,
            final_data={"result": "success", "items_processed": 1000}
        )
        
        print(f"Process completed with final checkpoint: {final_checkpoint}")
        
        # Test recovery
        print("\nTesting recovery...")
        recovered_state = manager.recover_process(process_id)
        if recovered_state:
            print(f"Recovered state: {recovered_state.progress_percentage}% complete")
        
        # Get system stats
        stats = manager.get_system_stats()
        print(f"\nSystem stats: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        manager.stop_background_tasks()

if __name__ == "__main__":
    asyncio.run(example_usage())
