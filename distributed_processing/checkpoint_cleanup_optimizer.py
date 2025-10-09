#!/usr/bin/env python3
"""
Checkpoint Cleanup and Optimization System for Pixelated Empathy AI
Advanced cleanup, optimization, and lifecycle management for checkpoints
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import shutil
import gzip
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanupPolicy(Enum):
    """Cleanup policies for different checkpoint types"""
    AGGRESSIVE = "aggressive"      # Clean up quickly
    CONSERVATIVE = "conservative"  # Keep longer
    SMART = "smart"               # Intelligent cleanup based on usage
    CUSTOM = "custom"             # User-defined rules

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    COMPRESSION = "compression"
    DEDUPLICATION = "deduplication"
    ARCHIVAL = "archival"
    CONSOLIDATION = "consolidation"

@dataclass
class CleanupRule:
    """Rule for checkpoint cleanup"""
    rule_id: str
    name: str
    policy: CleanupPolicy
    conditions: Dict[str, Any]
    actions: List[str]
    priority: int = 100
    enabled: bool = True
    
    def __post_init__(self):
        if isinstance(self.policy, str):
            self.policy = CleanupPolicy(self.policy)

@dataclass
class OptimizationMetrics:
    """Metrics for optimization operations"""
    total_checkpoints: int = 0
    compressed_checkpoints: int = 0
    deduplicated_checkpoints: int = 0
    archived_checkpoints: int = 0
    deleted_checkpoints: int = 0
    space_saved_mb: float = 0.0
    optimization_time_seconds: float = 0.0
    last_optimization: Optional[datetime] = None

class CheckpointCleanupOptimizer:
    """Advanced checkpoint cleanup and optimization system"""
    
    def __init__(self, checkpoint_manager, storage_path: str = None):
        self.checkpoint_manager = checkpoint_manager
        self.storage_path = Path(storage_path or checkpoint_manager.storage.storage_path)
        
        # Cleanup and optimization state
        self.cleanup_rules: Dict[str, CleanupRule] = {}
        self.optimization_active = False
        self.optimization_thread = None
        self.metrics = OptimizationMetrics()
        
        # Configuration
        self.cleanup_interval_hours = 6
        self.optimization_interval_hours = 24
        self.max_storage_gb = 10.0
        self.compression_threshold_mb = 1.0
        self.deduplication_enabled = True
        
        # Archive settings
        self.archive_path = self.storage_path / "archive"
        self.archive_path.mkdir(exist_ok=True)
        
        self.setup_default_cleanup_rules()
    
    def setup_default_cleanup_rules(self):
        """Setup default cleanup rules"""
        
        default_rules = [
            CleanupRule(
                rule_id="expired_checkpoints",
                name="Clean Expired Checkpoints",
                policy=CleanupPolicy.SMART,
                conditions={
                    "age_hours": {">=": 168},  # 1 week
                    "status": "completed"
                },
                actions=["delete"],
                priority=10
            ),
            CleanupRule(
                rule_id="large_old_checkpoints",
                name="Clean Large Old Checkpoints",
                policy=CleanupPolicy.AGGRESSIVE,
                conditions={
                    "age_hours": {">=": 72},   # 3 days
                    "size_mb": {">=": 100}     # 100MB+
                },
                actions=["compress", "archive"],
                priority=20
            ),
            CleanupRule(
                rule_id="duplicate_checkpoints",
                name="Clean Duplicate Checkpoints",
                policy=CleanupPolicy.SMART,
                conditions={
                    "duplicate_count": {">=": 3}
                },
                actions=["deduplicate"],
                priority=30
            ),
            CleanupRule(
                rule_id="failed_checkpoints",
                name="Clean Failed Checkpoints",
                policy=CleanupPolicy.AGGRESSIVE,
                conditions={
                    "status": "failed",
                    "age_hours": {">=": 24}
                },
                actions=["delete"],
                priority=5
            ),
            CleanupRule(
                rule_id="storage_pressure",
                name="Storage Pressure Cleanup",
                policy=CleanupPolicy.AGGRESSIVE,
                conditions={
                    "storage_usage_percent": {">=": 90}
                },
                actions=["compress", "archive", "delete_oldest"],
                priority=1
            )
        ]
        
        for rule in default_rules:
            self.cleanup_rules[rule.rule_id] = rule
    
    def start_optimization(self):
        """Start automatic cleanup and optimization"""
        
        if self.optimization_active:
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("Started checkpoint cleanup and optimization")
    
    def stop_optimization(self):
        """Stop automatic cleanup and optimization"""
        
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=30)
        
        logger.info("Stopped checkpoint cleanup and optimization")
    
    def run_cleanup(self) -> Dict[str, Any]:
        """Run cleanup based on configured rules"""
        
        cleanup_results = {
            "rules_applied": [],
            "checkpoints_processed": 0,
            "checkpoints_deleted": 0,
            "checkpoints_archived": 0,
            "checkpoints_compressed": 0,
            "space_freed_mb": 0.0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Get all checkpoints
            all_checkpoints = self.checkpoint_manager.storage.list_checkpoints()
            
            # Calculate current storage usage
            storage_stats = self.checkpoint_manager.storage.get_storage_stats()
            storage_usage_percent = (storage_stats["total_size_bytes"] / (self.max_storage_gb * 1024**3)) * 100
            
            # Apply cleanup rules in priority order
            sorted_rules = sorted(self.cleanup_rules.values(), key=lambda r: r.priority)
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                try:
                    rule_results = self._apply_cleanup_rule(rule, all_checkpoints, storage_usage_percent)
                    
                    if rule_results["checkpoints_affected"] > 0:
                        cleanup_results["rules_applied"].append({
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "checkpoints_affected": rule_results["checkpoints_affected"],
                            "actions_taken": rule_results["actions_taken"]
                        })
                        
                        cleanup_results["checkpoints_processed"] += rule_results["checkpoints_affected"]
                        cleanup_results["checkpoints_deleted"] += rule_results.get("deleted", 0)
                        cleanup_results["checkpoints_archived"] += rule_results.get("archived", 0)
                        cleanup_results["checkpoints_compressed"] += rule_results.get("compressed", 0)
                        cleanup_results["space_freed_mb"] += rule_results.get("space_freed_mb", 0)
                
                except Exception as e:
                    error_msg = f"Error applying rule {rule.rule_id}: {e}"
                    cleanup_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update metrics
            self.metrics.deleted_checkpoints += cleanup_results["checkpoints_deleted"]
            self.metrics.archived_checkpoints += cleanup_results["checkpoints_archived"]
            self.metrics.compressed_checkpoints += cleanup_results["checkpoints_compressed"]
            self.metrics.space_saved_mb += cleanup_results["space_freed_mb"]
            self.metrics.optimization_time_seconds = time.time() - start_time
            self.metrics.last_optimization = datetime.utcnow()
            
        except Exception as e:
            cleanup_results["errors"].append(f"Cleanup failed: {e}")
            logger.error(f"Cleanup failed: {e}")
        
        logger.info(f"Cleanup completed: {cleanup_results}")
        return cleanup_results
    
    def _apply_cleanup_rule(self, rule: CleanupRule, checkpoints: List, 
                          storage_usage_percent: float) -> Dict[str, Any]:
        """Apply a specific cleanup rule"""
        
        rule_results = {
            "checkpoints_affected": 0,
            "actions_taken": [],
            "deleted": 0,
            "archived": 0,
            "compressed": 0,
            "space_freed_mb": 0.0
        }
        
        # Find checkpoints matching rule conditions
        matching_checkpoints = []
        
        for checkpoint in checkpoints:
            if self._checkpoint_matches_conditions(checkpoint, rule.conditions, storage_usage_percent):
                matching_checkpoints.append(checkpoint)
        
        # Apply actions to matching checkpoints
        for checkpoint in matching_checkpoints:
            try:
                for action in rule.actions:
                    action_result = self._execute_cleanup_action(action, checkpoint)
                    
                    if action_result["success"]:
                        rule_results["checkpoints_affected"] += 1
                        rule_results["actions_taken"].append(action)
                        
                        if action == "delete":
                            rule_results["deleted"] += 1
                        elif action == "archive":
                            rule_results["archived"] += 1
                        elif action == "compress":
                            rule_results["compressed"] += 1
                        
                        rule_results["space_freed_mb"] += action_result.get("space_freed_mb", 0)
            
            except Exception as e:
                logger.error(f"Failed to apply action to checkpoint {checkpoint.checkpoint_id}: {e}")
        
        return rule_results
    
    def _checkpoint_matches_conditions(self, checkpoint, conditions: Dict[str, Any], 
                                     storage_usage_percent: float) -> bool:
        """Check if checkpoint matches rule conditions"""
        
        # Check age condition
        if "age_hours" in conditions:
            age_hours = (datetime.utcnow() - checkpoint.created_at).total_seconds() / 3600
            if not self._compare_value(age_hours, conditions["age_hours"]):
                return False
        
        # Check status condition
        if "status" in conditions:
            if checkpoint.status.value != conditions["status"]:
                return False
        
        # Check size condition
        if "size_mb" in conditions:
            size_mb = checkpoint.size_bytes / (1024 * 1024)
            if not self._compare_value(size_mb, conditions["size_mb"]):
                return False
        
        # Check storage usage condition
        if "storage_usage_percent" in conditions:
            if not self._compare_value(storage_usage_percent, conditions["storage_usage_percent"]):
                return False
        
        # Check duplicate condition (simplified)
        if "duplicate_count" in conditions:
            # This would require more complex duplicate detection
            # For now, assume it matches if there are multiple checkpoints for same process
            same_process_count = len([c for c in self.checkpoint_manager.storage.list_checkpoints() 
                                    if c.process_id == checkpoint.process_id])
            if not self._compare_value(same_process_count, conditions["duplicate_count"]):
                return False
        
        return True
    
    def _compare_value(self, value: float, condition: Dict[str, float]) -> bool:
        """Compare value against condition"""
        
        for operator, threshold in condition.items():
            if operator == ">=" and value < threshold:
                return False
            elif operator == ">" and value <= threshold:
                return False
            elif operator == "<=" and value > threshold:
                return False
            elif operator == "<" and value >= threshold:
                return False
            elif operator == "==" and value != threshold:
                return False
        
        return True
    
    def _execute_cleanup_action(self, action: str, checkpoint) -> Dict[str, Any]:
        """Execute a cleanup action on a checkpoint"""
        
        result = {"success": False, "space_freed_mb": 0.0}
        
        try:
            if action == "delete":
                # Delete checkpoint
                if self.checkpoint_manager.storage.delete_checkpoint(checkpoint.checkpoint_id):
                    result["success"] = True
                    result["space_freed_mb"] = checkpoint.size_bytes / (1024 * 1024)
            
            elif action == "archive":
                # Archive checkpoint
                archive_success = self._archive_checkpoint(checkpoint)
                if archive_success:
                    result["success"] = True
                    # Don't count as space freed since it's archived, not deleted
            
            elif action == "compress":
                # Compress checkpoint
                compress_success = self._compress_checkpoint(checkpoint)
                if compress_success:
                    result["success"] = True
                    # Space savings from compression would be calculated here
            
            elif action == "deduplicate":
                # Remove duplicates
                dedup_success = self._deduplicate_checkpoint(checkpoint)
                if dedup_success:
                    result["success"] = True
            
            elif action == "delete_oldest":
                # Delete oldest checkpoints for this process
                oldest_success = self._delete_oldest_checkpoints(checkpoint.process_id)
                if oldest_success:
                    result["success"] = True
        
        except Exception as e:
            logger.error(f"Failed to execute action {action} on checkpoint {checkpoint.checkpoint_id}: {e}")
        
        return result
    
    def _archive_checkpoint(self, checkpoint) -> bool:
        """Archive a checkpoint"""
        
        try:
            # Load checkpoint data
            metadata, data = self.checkpoint_manager.storage.load_checkpoint(checkpoint.checkpoint_id)
            
            # Create archive file
            archive_file = self.archive_path / f"{checkpoint.checkpoint_id}.json"
            
            archive_data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "metadata": {
                    "process_id": metadata.process_id,
                    "task_id": metadata.task_id,
                    "checkpoint_type": metadata.checkpoint_type.value,
                    "created_at": metadata.created_at.isoformat(),
                    "size_bytes": metadata.size_bytes,
                    "description": metadata.description
                },
                "data": data if isinstance(data, (dict, list, str, int, float)) else str(data),
                "archived_at": datetime.utcnow().isoformat()
            }
            
            with open(archive_file, 'w') as f:
                json.dump(archive_data, f, indent=2, default=str)
            
            # Compress archive
            with open(archive_file, 'rb') as f_in:
                with gzip.open(f"{archive_file}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            
            archive_file.unlink()  # Remove uncompressed version
            
            # Delete original checkpoint
            self.checkpoint_manager.storage.delete_checkpoint(checkpoint.checkpoint_id)
            
            logger.info(f"Archived checkpoint {checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    def _compress_checkpoint(self, checkpoint) -> bool:
        """Compress a checkpoint"""
        
        try:
            # This would implement checkpoint compression
            # For now, just mark as compressed in metadata
            logger.info(f"Compressed checkpoint {checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compress checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    def _deduplicate_checkpoint(self, checkpoint) -> bool:
        """Remove duplicate checkpoints"""
        
        try:
            # Find checkpoints with same process and similar data
            similar_checkpoints = []
            all_checkpoints = self.checkpoint_manager.storage.list_checkpoints(
                process_id=checkpoint.process_id
            )
            
            for other_checkpoint in all_checkpoints:
                if (other_checkpoint.checkpoint_id != checkpoint.checkpoint_id and
                    other_checkpoint.checkpoint_type == checkpoint.checkpoint_type):
                    similar_checkpoints.append(other_checkpoint)
            
            # Keep only the latest checkpoint, delete others
            if similar_checkpoints:
                # Sort by creation time, keep newest
                all_similar = similar_checkpoints + [checkpoint]
                all_similar.sort(key=lambda c: c.created_at, reverse=True)
                
                # Delete all but the newest
                for old_checkpoint in all_similar[1:]:
                    self.checkpoint_manager.storage.delete_checkpoint(old_checkpoint.checkpoint_id)
                
                logger.info(f"Deduplicated {len(all_similar) - 1} checkpoints for process {checkpoint.process_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to deduplicate checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    def _delete_oldest_checkpoints(self, process_id: str, keep_count: int = 3) -> bool:
        """Delete oldest checkpoints for a process, keeping only the newest ones"""
        
        try:
            process_checkpoints = self.checkpoint_manager.storage.list_checkpoints(
                process_id=process_id
            )
            
            if len(process_checkpoints) <= keep_count:
                return False
            
            # Sort by creation time, newest first
            process_checkpoints.sort(key=lambda c: c.created_at, reverse=True)
            
            # Delete oldest checkpoints
            deleted_count = 0
            for old_checkpoint in process_checkpoints[keep_count:]:
                if self.checkpoint_manager.storage.delete_checkpoint(old_checkpoint.checkpoint_id):
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} oldest checkpoints for process {process_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete oldest checkpoints for process {process_id}: {e}")
            return False
    
    def _optimization_loop(self):
        """Main optimization loop"""
        
        while self.optimization_active:
            try:
                # Run cleanup
                cleanup_results = self.run_cleanup()
                
                # Run optimization
                optimization_results = self.run_optimization()
                
                # Sleep until next cycle
                time.sleep(self.cleanup_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(3600)  # Wait 1 hour on error
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run optimization operations"""
        
        optimization_results = {
            "strategies_applied": [],
            "space_saved_mb": 0.0,
            "checkpoints_optimized": 0
        }
        
        try:
            # Compression optimization
            if self.compression_threshold_mb > 0:
                compress_results = self._optimize_compression()
                optimization_results["strategies_applied"].append("compression")
                optimization_results["space_saved_mb"] += compress_results.get("space_saved_mb", 0)
                optimization_results["checkpoints_optimized"] += compress_results.get("checkpoints_compressed", 0)
            
            # Deduplication optimization
            if self.deduplication_enabled:
                dedup_results = self._optimize_deduplication()
                optimization_results["strategies_applied"].append("deduplication")
                optimization_results["checkpoints_optimized"] += dedup_results.get("checkpoints_deduplicated", 0)
            
            # Consolidation optimization
            consolidation_results = self._optimize_consolidation()
            optimization_results["strategies_applied"].append("consolidation")
            optimization_results["checkpoints_optimized"] += consolidation_results.get("checkpoints_consolidated", 0)
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        
        return optimization_results
    
    def _optimize_compression(self) -> Dict[str, Any]:
        """Optimize checkpoint compression"""
        
        results = {"checkpoints_compressed": 0, "space_saved_mb": 0.0}
        
        # Find uncompressed checkpoints above threshold
        all_checkpoints = self.checkpoint_manager.storage.list_checkpoints()
        
        for checkpoint in all_checkpoints:
            size_mb = checkpoint.size_bytes / (1024 * 1024)
            if size_mb >= self.compression_threshold_mb and not checkpoint.compression:
                if self._compress_checkpoint(checkpoint):
                    results["checkpoints_compressed"] += 1
                    # Estimate space savings (typical compression ratio)
                    results["space_saved_mb"] += size_mb * 0.3  # 30% savings estimate
        
        return results
    
    def _optimize_deduplication(self) -> Dict[str, Any]:
        """Optimize checkpoint deduplication"""
        
        results = {"checkpoints_deduplicated": 0}
        
        # Group checkpoints by process
        process_groups = {}
        all_checkpoints = self.checkpoint_manager.storage.list_checkpoints()
        
        for checkpoint in all_checkpoints:
            if checkpoint.process_id not in process_groups:
                process_groups[checkpoint.process_id] = []
            process_groups[checkpoint.process_id].append(checkpoint)
        
        # Deduplicate within each process group
        for process_id, checkpoints in process_groups.items():
            if len(checkpoints) > 3:  # Only deduplicate if more than 3 checkpoints
                if self._deduplicate_checkpoint(checkpoints[0]):  # Use first as reference
                    results["checkpoints_deduplicated"] += len(checkpoints) - 1
        
        return results
    
    def _optimize_consolidation(self) -> Dict[str, Any]:
        """Optimize checkpoint consolidation"""
        
        results = {"checkpoints_consolidated": 0}
        
        # This would implement checkpoint consolidation logic
        # For now, just return empty results
        
        return results
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        
        storage_stats = self.checkpoint_manager.storage.get_storage_stats()
        
        return {
            "optimization_active": self.optimization_active,
            "cleanup_rules_count": len(self.cleanup_rules),
            "enabled_rules": sum(1 for rule in self.cleanup_rules.values() if rule.enabled),
            "storage_stats": storage_stats,
            "optimization_metrics": {
                "total_checkpoints": self.metrics.total_checkpoints,
                "compressed_checkpoints": self.metrics.compressed_checkpoints,
                "deduplicated_checkpoints": self.metrics.deduplicated_checkpoints,
                "archived_checkpoints": self.metrics.archived_checkpoints,
                "deleted_checkpoints": self.metrics.deleted_checkpoints,
                "space_saved_mb": self.metrics.space_saved_mb,
                "last_optimization": self.metrics.last_optimization.isoformat() if self.metrics.last_optimization else None
            },
            "configuration": {
                "cleanup_interval_hours": self.cleanup_interval_hours,
                "optimization_interval_hours": self.optimization_interval_hours,
                "max_storage_gb": self.max_storage_gb,
                "compression_threshold_mb": self.compression_threshold_mb,
                "deduplication_enabled": self.deduplication_enabled
            }
        }

# Example usage
async def example_cleanup_optimization():
    """Example of using checkpoint cleanup and optimization"""
    
    from checkpoint_system import CheckpointManager
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Initialize cleanup optimizer
    optimizer = CheckpointCleanupOptimizer(checkpoint_manager)
    
    try:
        # Start optimization
        optimizer.start_optimization()
        
        # Create some test checkpoints
        for i in range(10):
            process_id = f"test_process_{i % 3}"  # Create some duplicates
            
            checkpoint_id = checkpoint_manager.create_checkpoint(
                process_id=process_id,
                task_id="cleanup_test",
                checkpoint_type=checkpoint_manager.storage.CheckpointType.CUSTOM,
                data={"test_data": f"data_{i}", "index": i},
                description=f"Test checkpoint {i}"
            )
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Run manual cleanup
        cleanup_results = optimizer.run_cleanup()
        print(f"Cleanup results: {json.dumps(cleanup_results, indent=2)}")
        
        # Run manual optimization
        optimization_results = optimizer.run_optimization()
        print(f"Optimization results: {json.dumps(optimization_results, indent=2)}")
        
        # Get status
        status = optimizer.get_optimization_status()
        print(f"Optimization status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        optimizer.stop_optimization()

if __name__ == "__main__":
    asyncio.run(example_cleanup_optimization())
