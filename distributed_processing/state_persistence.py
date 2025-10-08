#!/usr/bin/env python3
"""
Progress State Persistence System for Pixelated Empathy AI
Maintains processing state across system restarts and failures
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import fcntl
from pathlib import Path

from checkpoint_system import CheckpointManager, CheckpointType, ProcessingState
from auto_resume_engine import AutoResumeEngine, ResumeConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistenceLevel(Enum):
    """Levels of state persistence"""
    MINIMAL = "minimal"          # Basic progress only
    STANDARD = "standard"        # Progress + metadata
    COMPREHENSIVE = "comprehensive"  # Full state including intermediate results
    PARANOID = "paranoid"        # Everything + redundant backups

class StateScope(Enum):
    """Scope of state persistence"""
    PROCESS = "process"          # Individual process state
    TASK = "task"               # Task-level aggregated state
    SYSTEM = "system"           # System-wide state
    GLOBAL = "global"           # Cross-system state

@dataclass
class PersistentState:
    """Persistent state container"""
    state_id: str
    scope: StateScope
    process_id: Optional[str]
    task_id: Optional[str]
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if isinstance(self.scope, str):
            self.scope = StateScope(self.scope)

@dataclass
class PersistenceConfig:
    """Configuration for state persistence"""
    persistence_level: PersistenceLevel = PersistenceLevel.STANDARD
    persistence_interval_seconds: int = 30
    state_retention_hours: int = 168  # 1 week
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    compression_enabled: bool = True
    encryption_enabled: bool = False
    redundancy_copies: int = 2
    storage_path: str = "/home/vivi/pixelated/ai/distributed_processing/persistent_state"
    lock_timeout_seconds: int = 30

class StatePersistenceManager:
    """Manages persistence of processing state across system restarts"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, 
                 config: PersistenceConfig = None):
        self.checkpoint_manager = checkpoint_manager
        self.config = config or PersistenceConfig()
        
        # Storage setup
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "persistent_state.db"
        self.backup_path = self.storage_path / "backups"
        self.backup_path.mkdir(exist_ok=True)
        
        # State management
        self.active_states: Dict[str, PersistentState] = {}
        self.state_locks: Dict[str, threading.Lock] = {}
        self.persistence_active = False
        self.persistence_thread = None
        
        # Performance tracking
        self.persistence_metrics = {
            "states_persisted": 0,
            "states_restored": 0,
            "persistence_errors": 0,
            "avg_persistence_time": 0.0,
            "last_persistence": None
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize persistence database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persistent_states (
                    state_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    process_id TEXT,
                    task_id TEXT,
                    state_data TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    version INTEGER DEFAULT 1,
                    checksum TEXT,
                    file_path TEXT,
                    backup_paths TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_process_task 
                ON persistent_states (process_id, task_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scope_updated 
                ON persistent_states (scope, updated_at)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (state_id) REFERENCES persistent_states (state_id)
                )
            """)
    
    def start_persistence(self):
        """Start automatic state persistence"""
        
        if self.persistence_active:
            return
        
        self.persistence_active = True
        self.persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
        self.persistence_thread.start()
        
        logger.info("Started state persistence system")
    
    def stop_persistence(self):
        """Stop automatic state persistence"""
        
        self.persistence_active = False
        if self.persistence_thread:
            self.persistence_thread.join(timeout=10)
        
        # Final persistence of all active states
        self._persist_all_states()
        
        logger.info("Stopped state persistence system")
    
    def register_persistent_state(self, process_id: str, task_id: str = None,
                                scope: StateScope = StateScope.PROCESS,
                                initial_data: Dict[str, Any] = None) -> str:
        """Register a new persistent state"""
        
        state_id = f"{scope.value}_{process_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        persistent_state = PersistentState(
            state_id=state_id,
            scope=scope,
            process_id=process_id,
            task_id=task_id,
            state_data=initial_data or {},
            metadata={
                "registered_at": datetime.utcnow().isoformat(),
                "persistence_level": self.config.persistence_level.value
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.active_states[state_id] = persistent_state
        self.state_locks[state_id] = threading.Lock()
        
        # Immediate persistence for registration
        self._persist_state(state_id)
        
        logger.info(f"Registered persistent state {state_id} for {process_id}")
        return state_id
    
    def update_persistent_state(self, state_id: str, updates: Dict[str, Any],
                              metadata_updates: Dict[str, Any] = None):
        """Update persistent state data"""
        
        if state_id not in self.active_states:
            logger.warning(f"State {state_id} not found for update")
            return
        
        with self.state_locks[state_id]:
            state = self.active_states[state_id]
            
            # Update state data
            state.state_data.update(updates)
            
            # Update metadata
            if metadata_updates:
                state.metadata.update(metadata_updates)
            
            # Update version and timestamp
            state.version += 1
            state.updated_at = datetime.utcnow()
            
            # Mark for persistence
            state.metadata["needs_persistence"] = True
    
    def get_persistent_state(self, state_id: str) -> Optional[PersistentState]:
        """Get current persistent state"""
        
        return self.active_states.get(state_id)
    
    def restore_system_state(self) -> Dict[str, Any]:
        """Restore system state from persistence on startup"""
        
        restoration_results = {
            "restored_states": 0,
            "restored_processes": [],
            "failed_restorations": 0,
            "restoration_errors": []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all active persistent states
                states = conn.execute("""
                    SELECT state_id, scope, process_id, task_id, state_data, 
                           metadata, created_at, updated_at, version, checksum, file_path
                    FROM persistent_states
                    ORDER BY updated_at DESC
                """).fetchall()
                
                for state_row in states:
                    try:
                        state_id = state_row[0]
                        
                        # Reconstruct persistent state
                        persistent_state = PersistentState(
                            state_id=state_id,
                            scope=StateScope(state_row[1]),
                            process_id=state_row[2],
                            task_id=state_row[3],
                            state_data=json.loads(state_row[4]),
                            metadata=json.loads(state_row[5]) if state_row[5] else {},
                            created_at=datetime.fromisoformat(state_row[6]),
                            updated_at=datetime.fromisoformat(state_row[7]),
                            version=state_row[8],
                            checksum=state_row[9]
                        )
                        
                        # Load additional data from file if exists
                        if state_row[10]:  # file_path
                            file_path = Path(state_row[10])
                            if file_path.exists():
                                additional_data = self._load_state_file(file_path)
                                if additional_data:
                                    persistent_state.state_data.update(additional_data)
                        
                        # Restore to active states
                        self.active_states[state_id] = persistent_state
                        self.state_locks[state_id] = threading.Lock()
                        
                        restoration_results["restored_states"] += 1
                        if persistent_state.process_id:
                            restoration_results["restored_processes"].append(persistent_state.process_id)
                        
                        logger.info(f"Restored persistent state {state_id}")
                        
                    except Exception as e:
                        restoration_results["failed_restorations"] += 1
                        restoration_results["restoration_errors"].append(str(e))
                        logger.error(f"Failed to restore state {state_id}: {e}")
                
                # Update metrics
                self.persistence_metrics["states_restored"] = restoration_results["restored_states"]
                
        except Exception as e:
            logger.error(f"Failed to restore system state: {e}")
            restoration_results["restoration_errors"].append(str(e))
        
        logger.info(f"System state restoration completed: {restoration_results}")
        return restoration_results
    
    def create_system_snapshot(self) -> str:
        """Create a complete system state snapshot"""
        
        snapshot_id = f"snapshot_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        snapshot_path = self.storage_path / f"{snapshot_id}.json"
        
        try:
            # Collect all system state
            system_state = {
                "snapshot_id": snapshot_id,
                "created_at": datetime.utcnow().isoformat(),
                "active_states": {},
                "checkpoint_stats": self.checkpoint_manager.get_system_stats(),
                "persistence_metrics": self.persistence_metrics.copy()
            }
            
            # Include all active persistent states
            for state_id, state in self.active_states.items():
                system_state["active_states"][state_id] = {
                    "state_id": state.state_id,
                    "scope": state.scope.value,
                    "process_id": state.process_id,
                    "task_id": state.task_id,
                    "state_data": state.state_data,
                    "metadata": state.metadata,
                    "created_at": state.created_at.isoformat(),
                    "updated_at": state.updated_at.isoformat(),
                    "version": state.version
                }
            
            # Save snapshot
            with open(snapshot_path, 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
            # Compress if enabled
            if self.config.compression_enabled:
                import gzip
                with open(snapshot_path, 'rb') as f_in:
                    with gzip.open(f"{snapshot_path}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                snapshot_path.unlink()  # Remove uncompressed version
                snapshot_path = Path(f"{snapshot_path}.gz")
            
            logger.info(f"Created system snapshot {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create system snapshot: {e}")
            raise
    
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """Restore system state from a snapshot"""
        
        try:
            # Find snapshot file
            snapshot_files = [
                self.storage_path / f"{snapshot_id}.json",
                self.storage_path / f"{snapshot_id}.json.gz"
            ]
            
            snapshot_path = None
            for path in snapshot_files:
                if path.exists():
                    snapshot_path = path
                    break
            
            if not snapshot_path:
                logger.error(f"Snapshot {snapshot_id} not found")
                return False
            
            # Load snapshot data
            if snapshot_path.suffix == '.gz':
                import gzip
                with gzip.open(snapshot_path, 'rt') as f:
                    system_state = json.load(f)
            else:
                with open(snapshot_path, 'r') as f:
                    system_state = json.load(f)
            
            # Clear current active states
            self.active_states.clear()
            self.state_locks.clear()
            
            # Restore states from snapshot
            for state_id, state_data in system_state["active_states"].items():
                persistent_state = PersistentState(
                    state_id=state_data["state_id"],
                    scope=StateScope(state_data["scope"]),
                    process_id=state_data["process_id"],
                    task_id=state_data["task_id"],
                    state_data=state_data["state_data"],
                    metadata=state_data["metadata"],
                    created_at=datetime.fromisoformat(state_data["created_at"]),
                    updated_at=datetime.fromisoformat(state_data["updated_at"]),
                    version=state_data["version"]
                )
                
                self.active_states[state_id] = persistent_state
                self.state_locks[state_id] = threading.Lock()
            
            logger.info(f"Restored system state from snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from snapshot {snapshot_id}: {e}")
            return False
    
    def _persistence_loop(self):
        """Main persistence loop"""
        
        while self.persistence_active:
            try:
                start_time = time.time()
                
                # Persist all states that need it
                persisted_count = self._persist_all_states()
                
                # Update metrics
                if persisted_count > 0:
                    persistence_time = time.time() - start_time
                    self.persistence_metrics["states_persisted"] += persisted_count
                    self.persistence_metrics["last_persistence"] = datetime.utcnow()
                    
                    # Update average persistence time
                    if self.persistence_metrics["avg_persistence_time"] == 0:
                        self.persistence_metrics["avg_persistence_time"] = persistence_time
                    else:
                        self.persistence_metrics["avg_persistence_time"] = (
                            self.persistence_metrics["avg_persistence_time"] * 0.9 + 
                            persistence_time * 0.1
                        )
                
                # Cleanup old states
                self._cleanup_old_states()
                
                # Create backup if needed
                self._maybe_create_backup()
                
                # Sleep until next persistence cycle
                time.sleep(self.config.persistence_interval_seconds)
                
            except Exception as e:
                self.persistence_metrics["persistence_errors"] += 1
                logger.error(f"Persistence loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _persist_all_states(self) -> int:
        """Persist all states that need persistence"""
        
        persisted_count = 0
        
        for state_id in list(self.active_states.keys()):
            try:
                state = self.active_states[state_id]
                
                # Check if state needs persistence
                if (state.metadata.get("needs_persistence", False) or
                    self.config.persistence_level == PersistenceLevel.PARANOID):
                    
                    if self._persist_state(state_id):
                        persisted_count += 1
                        
                        # Clear persistence flag
                        state.metadata["needs_persistence"] = False
                        
            except Exception as e:
                logger.error(f"Failed to persist state {state_id}: {e}")
        
        return persisted_count
    
    def _persist_state(self, state_id: str) -> bool:
        """Persist a single state"""
        
        if state_id not in self.active_states:
            return False
        
        try:
            with self.state_locks[state_id]:
                state = self.active_states[state_id]
                
                # Prepare data for persistence
                state_data_json = json.dumps(state.state_data)
                metadata_json = json.dumps(state.metadata)
                
                # Calculate checksum
                import hashlib
                checksum = hashlib.sha256(state_data_json.encode()).hexdigest()
                state.checksum = checksum
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO persistent_states
                        (state_id, scope, process_id, task_id, state_data, metadata,
                         created_at, updated_at, version, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        state.state_id, state.scope.value, state.process_id, state.task_id,
                        state_data_json, metadata_json, state.created_at.isoformat(),
                        state.updated_at.isoformat(), state.version, checksum
                    ))
                    
                    # Record history
                    conn.execute("""
                        INSERT INTO state_history (state_id, action, details)
                        VALUES (?, ?, ?)
                    """, (state_id, "persisted", f"Version {state.version}"))
                
                # Save additional data to file if comprehensive persistence
                if self.config.persistence_level in [PersistenceLevel.COMPREHENSIVE, PersistenceLevel.PARANOID]:
                    self._save_state_file(state)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to persist state {state_id}: {e}")
            return False
    
    def _save_state_file(self, state: PersistentState):
        """Save state data to separate file for comprehensive persistence"""
        
        file_path = self.storage_path / f"{state.state_id}.pkl"
        
        try:
            # Prepare comprehensive data
            comprehensive_data = {
                "state_data": state.state_data,
                "metadata": state.metadata,
                "checkpoint_data": None
            }
            
            # Include checkpoint data if available
            if state.process_id:
                try:
                    checkpoints = self.checkpoint_manager.storage.list_checkpoints(
                        process_id=state.process_id
                    )
                    if checkpoints:
                        latest_checkpoint = checkpoints[0]
                        _, checkpoint_data = self.checkpoint_manager.storage.load_checkpoint(
                            latest_checkpoint.checkpoint_id
                        )
                        comprehensive_data["checkpoint_data"] = checkpoint_data
                except Exception as e:
                    logger.debug(f"Could not include checkpoint data: {e}")
            
            # Save with pickle for complex data structures
            with open(file_path, 'wb') as f:
                pickle.dump(comprehensive_data, f)
            
            # Update database with file path
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE persistent_states SET file_path = ? WHERE state_id = ?
                """, (str(file_path), state.state_id))
            
        except Exception as e:
            logger.error(f"Failed to save state file for {state.state_id}: {e}")
    
    def _load_state_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load state data from file"""
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data.get("state_data", {})
        except Exception as e:
            logger.error(f"Failed to load state file {file_path}: {e}")
            return None
    
    def _cleanup_old_states(self):
        """Clean up old persistent states"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.state_retention_hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find old states
                old_states = conn.execute("""
                    SELECT state_id, file_path FROM persistent_states
                    WHERE updated_at < ?
                """, (cutoff_time.isoformat(),)).fetchall()
                
                for state_id, file_path in old_states:
                    # Remove from active states
                    if state_id in self.active_states:
                        del self.active_states[state_id]
                    if state_id in self.state_locks:
                        del self.state_locks[state_id]
                    
                    # Remove file if exists
                    if file_path:
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM persistent_states WHERE state_id = ?", (state_id,))
                    conn.execute("DELETE FROM state_history WHERE state_id = ?", (state_id,))
                
                if old_states:
                    logger.info(f"Cleaned up {len(old_states)} old persistent states")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
    
    def _maybe_create_backup(self):
        """Create backup if needed"""
        
        if not self.config.backup_enabled:
            return
        
        # Check if backup is needed
        last_backup_file = self.backup_path / "last_backup_time"
        
        should_backup = True
        if last_backup_file.exists():
            try:
                with open(last_backup_file, 'r') as f:
                    last_backup_str = f.read().strip()
                last_backup = datetime.fromisoformat(last_backup_str)
                
                time_since_backup = datetime.utcnow() - last_backup
                should_backup = time_since_backup.total_seconds() > (self.config.backup_interval_hours * 3600)
            except Exception:
                should_backup = True
        
        if should_backup:
            try:
                snapshot_id = self.create_system_snapshot()
                
                # Move snapshot to backup directory
                snapshot_files = list(self.storage_path.glob(f"{snapshot_id}*"))
                for snapshot_file in snapshot_files:
                    backup_file = self.backup_path / snapshot_file.name
                    snapshot_file.rename(backup_file)
                
                # Update last backup time
                with open(last_backup_file, 'w') as f:
                    f.write(datetime.utcnow().isoformat())
                
                logger.info(f"Created backup snapshot {snapshot_id}")
                
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
    
    def get_persistence_status(self) -> Dict[str, Any]:
        """Get comprehensive persistence status"""
        
        return {
            "persistence_active": self.persistence_active,
            "active_states_count": len(self.active_states),
            "persistence_config": asdict(self.config),
            "metrics": self.persistence_metrics.copy(),
            "storage_info": {
                "storage_path": str(self.storage_path),
                "db_size_mb": self.db_path.stat().st_size / (1024*1024) if self.db_path.exists() else 0,
                "backup_count": len(list(self.backup_path.glob("*"))) if self.backup_path.exists() else 0
            }
        }

# Example usage
async def example_state_persistence():
    """Example of using state persistence system"""
    
    from checkpoint_system import CheckpointManager
    
    # Initialize systems
    checkpoint_manager = CheckpointManager()
    
    config = PersistenceConfig(
        persistence_level=PersistenceLevel.COMPREHENSIVE,
        persistence_interval_seconds=5,  # Fast for demo
        backup_enabled=True
    )
    
    persistence_manager = StatePersistenceManager(checkpoint_manager, config)
    
    try:
        # Start persistence
        persistence_manager.start_persistence()
        
        # Register some persistent states
        process_state_id = persistence_manager.register_persistent_state(
            process_id="demo_process_001",
            task_id="demo_task",
            scope=StateScope.PROCESS,
            initial_data={
                "progress": 0,
                "items_processed": [],
                "current_batch": None
            }
        )
        
        # Simulate processing with state updates
        for i in range(10):
            persistence_manager.update_persistent_state(
                state_id=process_state_id,
                updates={
                    "progress": i * 10,
                    "items_processed": list(range(i)),
                    "current_batch": f"batch_{i}"
                },
                metadata_updates={
                    "last_update": datetime.utcnow().isoformat(),
                    "update_count": i + 1
                }
            )
            
            await asyncio.sleep(1)
        
        # Create system snapshot
        snapshot_id = persistence_manager.create_system_snapshot()
        print(f"Created snapshot: {snapshot_id}")
        
        # Get status
        status = persistence_manager.get_persistence_status()
        print(f"Persistence status: {json.dumps(status, indent=2, default=str)}")
        
        # Test restoration
        restoration_results = persistence_manager.restore_system_state()
        print(f"Restoration results: {restoration_results}")
        
    finally:
        persistence_manager.stop_persistence()

if __name__ == "__main__":
    asyncio.run(example_state_persistence())
