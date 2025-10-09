#!/usr/bin/env python3
"""
Partial Result Recovery System for Pixelated Empathy AI
Recovers and continues processing from partial results after interruptions
"""

import asyncio
import json
import logging
import pickle
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultType(Enum):
    """Types of partial results"""
    INTERMEDIATE = "intermediate"
    BATCH = "batch"
    CHECKPOINT = "checkpoint"
    FINAL = "final"

class RecoveryStrategy(Enum):
    """Strategies for result recovery"""
    MERGE_RESULTS = "merge_results"
    REPLACE_RESULTS = "replace_results"
    APPEND_RESULTS = "append_results"
    VALIDATE_AND_MERGE = "validate_and_merge"

@dataclass
class PartialResult:
    """Container for partial processing results"""
    result_id: str
    process_id: str
    task_id: str
    result_type: ResultType
    data: Any
    metadata: Dict[str, Any]
    created_at: datetime
    sequence_number: int = 0
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.result_type, str):
            self.result_type = ResultType(self.result_type)

class PartialResultManager:
    """Manages partial results for recovery and continuation"""
    
    def __init__(self, storage_path: str = "/home/vivi/pixelated/ai/distributed_processing/partial_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "partial_results.db"
        self.data_path = self.storage_path / "data"
        self.data_path.mkdir(exist_ok=True)
        
        self.active_results: Dict[str, PartialResult] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize partial results database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS partial_results (
                    result_id TEXT PRIMARY KEY,
                    process_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    sequence_number INTEGER DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    checksum TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    dependencies TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_process_sequence 
                ON partial_results (process_id, sequence_number)
            """)
    
    def store_partial_result(self, process_id: str, task_id: str, 
                           result_type: ResultType, data: Any,
                           metadata: Dict[str, Any] = None,
                           sequence_number: int = 0) -> str:
        """Store a partial result"""
        
        result_id = f"{process_id}_{result_type.value}_{sequence_number}_{uuid.uuid4().hex[:8]}"
        
        # Create partial result
        partial_result = PartialResult(
            result_id=result_id,
            process_id=process_id,
            task_id=task_id,
            result_type=result_type,
            data=data,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            sequence_number=sequence_number
        )
        
        # Calculate checksum
        data_str = json.dumps(data, sort_keys=True, default=str)
        partial_result.checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Save data to file
        file_path = self.data_path / f"{result_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO partial_results 
                (result_id, process_id, task_id, result_type, sequence_number,
                 created_at, checksum, file_path, metadata, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id, process_id, task_id, result_type.value, sequence_number,
                partial_result.created_at.isoformat(), partial_result.checksum,
                str(file_path), json.dumps(metadata or {}), json.dumps([])
            ))
        
        self.active_results[result_id] = partial_result
        logger.info(f"Stored partial result {result_id}")
        return result_id
    
    def recover_partial_results(self, process_id: str, 
                              result_types: List[ResultType] = None) -> List[PartialResult]:
        """Recover partial results for a process"""
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT result_id, process_id, task_id, result_type, sequence_number,
                       created_at, checksum, file_path, metadata, dependencies
                FROM partial_results 
                WHERE process_id = ?
            """
            params = [process_id]
            
            if result_types:
                placeholders = ','.join('?' * len(result_types))
                query += f" AND result_type IN ({placeholders})"
                params.extend([rt.value for rt in result_types])
            
            query += " ORDER BY sequence_number ASC"
            
            results = conn.execute(query, params).fetchall()
            
            recovered_results = []
            for row in results:
                try:
                    # Load data from file
                    file_path = Path(row[7])
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    else:
                        logger.warning(f"Data file not found for result {row[0]}")
                        continue
                    
                    # Create partial result object
                    partial_result = PartialResult(
                        result_id=row[0],
                        process_id=row[1],
                        task_id=row[2],
                        result_type=ResultType(row[3]),
                        data=data,
                        metadata=json.loads(row[8]) if row[8] else {},
                        created_at=datetime.fromisoformat(row[5]),
                        sequence_number=row[4],
                        checksum=row[6],
                        dependencies=json.loads(row[9]) if row[9] else []
                    )
                    
                    recovered_results.append(partial_result)
                    
                except Exception as e:
                    logger.error(f"Failed to recover result {row[0]}: {e}")
            
            logger.info(f"Recovered {len(recovered_results)} partial results for {process_id}")
            return recovered_results
    
    def merge_partial_results(self, results: List[PartialResult], 
                            strategy: RecoveryStrategy = RecoveryStrategy.MERGE_RESULTS) -> Any:
        """Merge multiple partial results into a single result"""
        
        if not results:
            return None
        
        if len(results) == 1:
            return results[0].data
        
        # Sort by sequence number
        sorted_results = sorted(results, key=lambda r: r.sequence_number)
        
        if strategy == RecoveryStrategy.MERGE_RESULTS:
            return self._merge_results_data(sorted_results)
        elif strategy == RecoveryStrategy.REPLACE_RESULTS:
            return sorted_results[-1].data  # Return latest
        elif strategy == RecoveryStrategy.APPEND_RESULTS:
            return self._append_results_data(sorted_results)
        elif strategy == RecoveryStrategy.VALIDATE_AND_MERGE:
            return self._validate_and_merge_results(sorted_results)
        else:
            return sorted_results[-1].data
    
    def _merge_results_data(self, results: List[PartialResult]) -> Any:
        """Merge result data intelligently based on data type"""
        
        if not results:
            return None
        
        first_data = results[0].data
        
        # Handle different data types
        if isinstance(first_data, dict):
            merged = {}
            for result in results:
                if isinstance(result.data, dict):
                    merged.update(result.data)
            return merged
        
        elif isinstance(first_data, list):
            merged = []
            for result in results:
                if isinstance(result.data, list):
                    merged.extend(result.data)
            return merged
        
        elif isinstance(first_data, (int, float)):
            return sum(result.data for result in results if isinstance(result.data, (int, float)))
        
        else:
            # For other types, return the latest
            return results[-1].data
    
    def _append_results_data(self, results: List[PartialResult]) -> List[Any]:
        """Append all results into a list"""
        
        return [result.data for result in results]
    
    def _validate_and_merge_results(self, results: List[PartialResult]) -> Any:
        """Validate checksums and merge results"""
        
        # Validate checksums first
        valid_results = []
        for result in results:
            if result.checksum:
                data_str = json.dumps(result.data, sort_keys=True, default=str)
                calculated_checksum = hashlib.sha256(data_str.encode()).hexdigest()
                
                if calculated_checksum == result.checksum:
                    valid_results.append(result)
                else:
                    logger.warning(f"Checksum mismatch for result {result.result_id}")
            else:
                valid_results.append(result)  # No checksum to validate
        
        return self._merge_results_data(valid_results)
    
    def continue_processing_from_results(self, process_id: str, 
                                       continuation_handler: Callable,
                                       result_types: List[ResultType] = None) -> Any:
        """Continue processing from recovered partial results"""
        
        # Recover partial results
        recovered_results = self.recover_partial_results(process_id, result_types)
        
        if not recovered_results:
            logger.info(f"No partial results found for {process_id}")
            return None
        
        # Group results by type
        results_by_type = {}
        for result in recovered_results:
            if result.result_type not in results_by_type:
                results_by_type[result.result_type] = []
            results_by_type[result.result_type].append(result)
        
        # Merge results for each type
        merged_results = {}
        for result_type, type_results in results_by_type.items():
            merged_results[result_type.value] = self.merge_partial_results(
                type_results, RecoveryStrategy.VALIDATE_AND_MERGE
            )
        
        # Call continuation handler with merged results
        try:
            return continuation_handler(merged_results, recovered_results)
        except Exception as e:
            logger.error(f"Continuation handler failed for {process_id}: {e}")
            raise
    
    def cleanup_partial_results(self, process_id: str = None, 
                              older_than_hours: int = 24) -> int:
        """Clean up old partial results"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            if process_id:
                query = "SELECT result_id, file_path FROM partial_results WHERE process_id = ? AND created_at < ?"
                params = [process_id, cutoff_time.isoformat()]
            else:
                query = "SELECT result_id, file_path FROM partial_results WHERE created_at < ?"
                params = [cutoff_time.isoformat()]
            
            old_results = conn.execute(query, params).fetchall()
            
            cleaned_count = 0
            for result_id, file_path in old_results:
                try:
                    # Remove file
                    if file_path:
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM partial_results WHERE result_id = ?", (result_id,))
                    
                    # Remove from active results
                    if result_id in self.active_results:
                        del self.active_results[result_id]
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup result {result_id}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old partial results")
            return cleaned_count
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about partial result recovery"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Count by result type
            type_counts = {}
            for result_type in ResultType:
                count = conn.execute(
                    "SELECT COUNT(*) FROM partial_results WHERE result_type = ?",
                    (result_type.value,)
                ).fetchone()[0]
                type_counts[result_type.value] = count
            
            # Count by process
            process_counts = conn.execute("""
                SELECT process_id, COUNT(*) as count
                FROM partial_results
                GROUP BY process_id
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            
            # Total size
            total_results = conn.execute("SELECT COUNT(*) FROM partial_results").fetchone()[0]
            
            # Storage size
            storage_size = sum(f.stat().st_size for f in self.data_path.glob("*.pkl"))
        
        return {
            "total_results": total_results,
            "results_by_type": type_counts,
            "top_processes": [{"process_id": p[0], "count": p[1]} for p in process_counts],
            "storage_size_mb": round(storage_size / (1024*1024), 2),
            "active_results": len(self.active_results)
        }

# Example usage
async def example_partial_result_recovery():
    """Example of using partial result recovery"""
    
    manager = PartialResultManager()
    
    # Simulate storing partial results during processing
    process_id = "example_process_001"
    task_id = "data_processing"
    
    # Store some intermediate results
    for i in range(5):
        batch_data = {
            "batch_id": i,
            "processed_items": list(range(i*10, (i+1)*10)),
            "batch_stats": {"count": 10, "errors": 0}
        }
        
        manager.store_partial_result(
            process_id=process_id,
            task_id=task_id,
            result_type=ResultType.BATCH,
            data=batch_data,
            metadata={"batch_number": i, "timestamp": datetime.utcnow().isoformat()},
            sequence_number=i
        )
    
    # Store an intermediate checkpoint
    checkpoint_data = {
        "total_processed": 50,
        "current_position": "batch_4",
        "accumulated_stats": {"total_items": 50, "total_errors": 0}
    }
    
    manager.store_partial_result(
        process_id=process_id,
        task_id=task_id,
        result_type=ResultType.CHECKPOINT,
        data=checkpoint_data,
        metadata={"checkpoint_type": "progress"},
        sequence_number=100
    )
    
    # Simulate recovery after interruption
    def continuation_handler(merged_results: Dict[str, Any], 
                           all_results: List[PartialResult]) -> Dict[str, Any]:
        """Handle continuation from partial results"""
        
        print(f"Continuing from {len(all_results)} partial results")
        
        # Get batch results
        batch_results = merged_results.get("batch", [])
        checkpoint_data = merged_results.get("checkpoint", {})
        
        print(f"Recovered batch data: {len(batch_results) if isinstance(batch_results, list) else 'merged'}")
        print(f"Checkpoint data: {checkpoint_data}")
        
        # Continue processing from where we left off
        continuation_result = {
            "recovered_batches": len(all_results) - 1,  # Exclude checkpoint
            "last_checkpoint": checkpoint_data,
            "continuation_successful": True
        }
        
        return continuation_result
    
    # Test recovery
    result = manager.continue_processing_from_results(
        process_id=process_id,
        continuation_handler=continuation_handler
    )
    
    print(f"Continuation result: {result}")
    
    # Get statistics
    stats = manager.get_recovery_statistics()
    print(f"Recovery statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(example_partial_result_recovery())
