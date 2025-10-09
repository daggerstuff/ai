#!/usr/bin/env python3
"""
Checkpoint Utilities and Configuration for Pixelated Empathy AI
Advanced utilities for checkpoint management, optimization, and monitoring
"""

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Generator
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import psutil
import sqlite3

from checkpoint_system import (
    CheckpointManager, CheckpointStorage, CheckpointType, 
    CheckpointStatus, CheckpointMetadata, ProcessingState
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint system"""
    storage_path: str = "/home/vivi/pixelated/ai/distributed_processing/checkpoints"
    auto_checkpoint_interval: int = 300  # 5 minutes
    cleanup_interval: int = 3600  # 1 hour
    default_ttl_hours: int = 24
    max_storage_size_gb: float = 10.0
    compression_enabled: bool = True
    encryption_enabled: bool = False
    backup_enabled: bool = True
    backup_path: str = "/home/vivi/pixelated/ai/distributed_processing/checkpoint_backups"
    monitoring_enabled: bool = True
    performance_tracking: bool = True

class CheckpointOptimizer:
    """Optimizer for checkpoint storage and performance"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.storage = CheckpointStorage(config.storage_path)
        self.performance_metrics = {}
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize checkpoint storage"""
        
        optimization_results = {
            "actions_taken": [],
            "space_freed_mb": 0,
            "checkpoints_archived": 0,
            "checkpoints_compressed": 0
        }
        
        # Get current storage stats
        stats = self.storage.get_storage_stats()
        current_size_gb = stats["total_size_bytes"] / (1024**3)
        
        logger.info(f"Current storage usage: {current_size_gb:.2f} GB")
        
        # Check if we need to free space
        if current_size_gb > self.config.max_storage_size_gb:
            space_to_free = (current_size_gb - self.config.max_storage_size_gb) * (1024**3)
            freed_space = self._free_storage_space(space_to_free)
            optimization_results["space_freed_mb"] = freed_space / (1024**2)
            optimization_results["actions_taken"].append("freed_storage_space")
        
        # Archive old completed checkpoints
        archived_count = self._archive_old_checkpoints()
        optimization_results["checkpoints_archived"] = archived_count
        if archived_count > 0:
            optimization_results["actions_taken"].append("archived_old_checkpoints")
        
        # Compress uncompressed checkpoints
        compressed_count = self._compress_checkpoints()
        optimization_results["checkpoints_compressed"] = compressed_count
        if compressed_count > 0:
            optimization_results["actions_taken"].append("compressed_checkpoints")
        
        # Deduplicate similar checkpoints
        deduplicated_count = self._deduplicate_checkpoints()
        if deduplicated_count > 0:
            optimization_results["checkpoints_deduplicated"] = deduplicated_count
            optimization_results["actions_taken"].append("deduplicated_checkpoints")
        
        logger.info(f"Storage optimization completed: {optimization_results}")
        return optimization_results
    
    def _free_storage_space(self, target_bytes: float) -> float:
        """Free storage space by removing old checkpoints"""
        
        freed_bytes = 0
        
        # Get checkpoints sorted by age (oldest first)
        with sqlite3.connect(self.storage.db_path) as conn:
            old_checkpoints = conn.execute("""
                SELECT checkpoint_id, size_bytes, created_at, status
                FROM checkpoints 
                WHERE status IN ('completed', 'expired')
                ORDER BY created_at ASC
            """).fetchall()
        
        for checkpoint_id, size_bytes, created_at, status in old_checkpoints:
            if freed_bytes >= target_bytes:
                break
            
            if self.storage.delete_checkpoint(checkpoint_id):
                freed_bytes += size_bytes
                logger.info(f"Deleted old checkpoint {checkpoint_id} ({size_bytes} bytes)")
        
        return freed_bytes
    
    def _archive_old_checkpoints(self, days_old: int = 7) -> int:
        """Archive old completed checkpoints"""
        
        if not self.config.backup_enabled:
            return 0
        
        archived_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Create backup directory
        backup_path = Path(self.config.backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Find old completed checkpoints
        checkpoints = self.storage.list_checkpoints(status=CheckpointStatus.COMPLETED)
        
        for metadata in checkpoints:
            if metadata.created_at < cutoff_date:
                try:
                    # Load checkpoint data
                    _, data = self.storage.load_checkpoint(metadata.checkpoint_id)
                    
                    # Save to backup location
                    backup_file = backup_path / f"{metadata.checkpoint_id}.json"
                    with open(backup_file, 'w') as f:
                        json.dump({
                            "metadata": asdict(metadata),
                            "data": data if isinstance(data, (dict, list, str, int, float)) else str(data)
                        }, f, indent=2, default=str)
                    
                    # Update status to archived
                    with sqlite3.connect(self.storage.db_path) as conn:
                        conn.execute(
                            "UPDATE checkpoints SET status = 'archived' WHERE checkpoint_id = ?",
                            (metadata.checkpoint_id,)
                        )
                    
                    archived_count += 1
                    logger.info(f"Archived checkpoint {metadata.checkpoint_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to archive checkpoint {metadata.checkpoint_id}: {e}")
        
        return archived_count
    
    def _compress_checkpoints(self) -> int:
        """Compress uncompressed checkpoints"""
        
        compressed_count = 0
        
        with sqlite3.connect(self.storage.db_path) as conn:
            uncompressed = conn.execute("""
                SELECT checkpoint_id, file_path 
                FROM checkpoints 
                WHERE compression = 0 AND status = 'active'
            """).fetchall()
        
        for checkpoint_id, file_path in uncompressed:
            try:
                # Load and recompress
                metadata, data = self.storage.load_checkpoint(checkpoint_id)
                metadata.compression = True
                
                # Save compressed version
                self.storage.save_checkpoint(metadata, data)
                compressed_count += 1
                
                logger.info(f"Compressed checkpoint {checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Failed to compress checkpoint {checkpoint_id}: {e}")
        
        return compressed_count
    
    def _deduplicate_checkpoints(self) -> int:
        """Remove duplicate checkpoints with identical data"""
        
        deduplicated_count = 0
        
        # Group checkpoints by process and type
        with sqlite3.connect(self.storage.db_path) as conn:
            checkpoints = conn.execute("""
                SELECT checkpoint_id, process_id, checkpoint_type, checksum, created_at
                FROM checkpoints 
                WHERE status = 'active'
                ORDER BY process_id, checkpoint_type, created_at
            """).fetchall()
        
        # Find duplicates by checksum
        checksum_groups = {}
        for checkpoint_id, process_id, checkpoint_type, checksum, created_at in checkpoints:
            key = f"{process_id}_{checkpoint_type}_{checksum}"
            if key not in checksum_groups:
                checksum_groups[key] = []
            checksum_groups[key].append((checkpoint_id, created_at))
        
        # Remove older duplicates
        for key, group in checksum_groups.items():
            if len(group) > 1:
                # Sort by creation time, keep newest
                group.sort(key=lambda x: x[1], reverse=True)
                
                for checkpoint_id, _ in group[1:]:  # Remove all but the newest
                    if self.storage.delete_checkpoint(checkpoint_id):
                        deduplicated_count += 1
                        logger.info(f"Removed duplicate checkpoint {checkpoint_id}")
        
        return deduplicated_count
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze checkpoint system performance"""
        
        analysis = {
            "storage_efficiency": self._analyze_storage_efficiency(),
            "access_patterns": self._analyze_access_patterns(),
            "performance_metrics": self._get_performance_metrics(),
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["storage_efficiency"]["compression_ratio"] < 0.5:
            analysis["recommendations"].append("Enable compression for better storage efficiency")
        
        if analysis["storage_efficiency"]["fragmentation_ratio"] > 0.3:
            analysis["recommendations"].append("Consider storage defragmentation")
        
        if analysis["access_patterns"]["cache_hit_ratio"] < 0.8:
            analysis["recommendations"].append("Increase checkpoint cache size")
        
        return analysis
    
    def _analyze_storage_efficiency(self) -> Dict[str, float]:
        """Analyze storage efficiency metrics"""
        
        stats = self.storage.get_storage_stats()
        
        # Calculate compression ratio
        with sqlite3.connect(self.storage.db_path) as conn:
            compressed_size = conn.execute("""
                SELECT SUM(size_bytes) FROM checkpoints WHERE compression = 1
            """).fetchone()[0] or 0
            
            uncompressed_size = conn.execute("""
                SELECT SUM(size_bytes) FROM checkpoints WHERE compression = 0
            """).fetchone()[0] or 0
        
        total_size = compressed_size + uncompressed_size
        compression_ratio = compressed_size / total_size if total_size > 0 else 0
        
        # Calculate fragmentation (simplified)
        storage_path = Path(self.config.storage_path)
        if storage_path.exists():
            total_files = len(list(storage_path.rglob("*")))
            data_files = len(list((storage_path / "data").glob("*"))) if (storage_path / "data").exists() else 0
            fragmentation_ratio = (total_files - data_files) / total_files if total_files > 0 else 0
        else:
            fragmentation_ratio = 0
        
        return {
            "compression_ratio": compression_ratio,
            "fragmentation_ratio": fragmentation_ratio,
            "storage_utilization": stats["total_size_mb"] / (self.config.max_storage_size_gb * 1024)
        }
    
    def _analyze_access_patterns(self) -> Dict[str, float]:
        """Analyze checkpoint access patterns"""
        
        # This would be enhanced with actual access logging
        return {
            "cache_hit_ratio": 0.85,  # Placeholder
            "avg_access_time_ms": 50,  # Placeholder
            "frequent_access_ratio": 0.2  # Placeholder
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        # System resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(self.config.storage_path)
        
        return {
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "checkpoint_metrics": self.performance_metrics
        }

class CheckpointMonitor:
    """Monitor checkpoint system health and performance"""
    
    def __init__(self, manager: CheckpointManager, config: CheckpointConfig):
        self.manager = manager
        self.config = config
        self.optimizer = CheckpointOptimizer(config)
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_history = []
    
    def start_monitoring(self):
        """Start checkpoint monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started checkpoint monitoring")
    
    def stop_monitoring(self):
        """Stop checkpoint monitoring"""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped checkpoint monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 24 hours of metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if datetime.fromisoformat(m["timestamp"]) > cutoff_time
                ]
                
                # Check for issues
                self._check_health_issues(metrics)
                
                # Auto-optimize if needed
                if self._should_auto_optimize(metrics):
                    self.optimizer.optimize_storage()
                
                # Sleep until next check
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        
        stats = self.manager.get_system_stats()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "storage_stats": stats["storage"],
            "active_processes": stats["active_processes"],
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(self.config.storage_path).percent
            }
        }
    
    def _check_health_issues(self, metrics: Dict[str, Any]):
        """Check for health issues and alert if necessary"""
        
        issues = []
        
        # Check storage usage
        storage_gb = metrics["storage_stats"]["total_size_bytes"] / (1024**3)
        if storage_gb > self.config.max_storage_size_gb * 0.9:
            issues.append(f"Storage usage high: {storage_gb:.2f} GB")
        
        # Check system resources
        if metrics["system_resources"]["cpu_percent"] > 90:
            issues.append(f"High CPU usage: {metrics['system_resources']['cpu_percent']:.1f}%")
        
        if metrics["system_resources"]["memory_percent"] > 90:
            issues.append(f"High memory usage: {metrics['system_resources']['memory_percent']:.1f}%")
        
        if metrics["system_resources"]["disk_percent"] > 90:
            issues.append(f"High disk usage: {metrics['system_resources']['disk_percent']:.1f}%")
        
        # Log issues
        for issue in issues:
            logger.warning(f"Health issue detected: {issue}")
    
    def _should_auto_optimize(self, metrics: Dict[str, Any]) -> bool:
        """Determine if auto-optimization should run"""
        
        # Optimize if storage is over 80% of limit
        storage_gb = metrics["storage_stats"]["total_size_bytes"] / (1024**3)
        if storage_gb > self.config.max_storage_size_gb * 0.8:
            return True
        
        # Optimize if too many completed checkpoints
        completed_count = metrics["storage_stats"]["status_counts"].get("completed", 0)
        if completed_count > 100:
            return True
        
        return False
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        
        if not self.metrics_history:
            return {"status": "no_data", "message": "No monitoring data available"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends
        if len(self.metrics_history) >= 2:
            prev_metrics = self.metrics_history[-2]
            storage_trend = (
                latest_metrics["storage_stats"]["total_size_bytes"] - 
                prev_metrics["storage_stats"]["total_size_bytes"]
            ) / (1024**2)  # MB change
        else:
            storage_trend = 0
        
        # Determine overall health
        health_score = 100
        issues = []
        
        # Storage health
        storage_gb = latest_metrics["storage_stats"]["total_size_bytes"] / (1024**3)
        storage_usage_pct = (storage_gb / self.config.max_storage_size_gb) * 100
        
        if storage_usage_pct > 90:
            health_score -= 30
            issues.append("Critical storage usage")
        elif storage_usage_pct > 80:
            health_score -= 15
            issues.append("High storage usage")
        
        # System resource health
        cpu_pct = latest_metrics["system_resources"]["cpu_percent"]
        memory_pct = latest_metrics["system_resources"]["memory_percent"]
        
        if cpu_pct > 90 or memory_pct > 90:
            health_score -= 20
            issues.append("High system resource usage")
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": health_score,
            "issues": issues,
            "metrics": latest_metrics,
            "trends": {
                "storage_change_mb": storage_trend
            },
            "recommendations": self._generate_recommendations(latest_metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on current metrics"""
        
        recommendations = []
        
        storage_gb = metrics["storage_stats"]["total_size_bytes"] / (1024**3)
        storage_usage_pct = (storage_gb / self.config.max_storage_size_gb) * 100
        
        if storage_usage_pct > 80:
            recommendations.append("Run storage optimization to free space")
        
        if metrics["storage_stats"]["status_counts"].get("completed", 0) > 50:
            recommendations.append("Archive old completed checkpoints")
        
        if metrics["system_resources"]["cpu_percent"] > 80:
            recommendations.append("Consider reducing checkpoint frequency")
        
        if not self.config.compression_enabled:
            recommendations.append("Enable compression to reduce storage usage")
        
        return recommendations

# Utility functions
def create_checkpoint_config(
    storage_path: str = None,
    max_storage_gb: float = 10.0,
    auto_checkpoint_interval: int = 300,
    compression: bool = True,
    backup: bool = True
) -> CheckpointConfig:
    """Create checkpoint configuration with sensible defaults"""
    
    if storage_path is None:
        storage_path = "/home/vivi/pixelated/ai/distributed_processing/checkpoints"
    
    return CheckpointConfig(
        storage_path=storage_path,
        max_storage_size_gb=max_storage_gb,
        auto_checkpoint_interval=auto_checkpoint_interval,
        compression_enabled=compression,
        backup_enabled=backup
    )

def setup_checkpoint_system(config: CheckpointConfig = None) -> tuple[CheckpointManager, CheckpointMonitor]:
    """Setup complete checkpoint system with monitoring"""
    
    if config is None:
        config = create_checkpoint_config()
    
    # Create manager
    manager = CheckpointManager(config.storage_path)
    manager.auto_checkpoint_interval = config.auto_checkpoint_interval
    
    # Create monitor
    monitor = CheckpointMonitor(manager, config)
    
    # Start background tasks
    manager.start_background_tasks()
    if config.monitoring_enabled:
        monitor.start_monitoring()
    
    logger.info("Checkpoint system setup completed")
    return manager, monitor

# Example usage
async def example_checkpoint_utilities():
    """Example of using checkpoint utilities"""
    
    # Setup system
    config = create_checkpoint_config(max_storage_gb=1.0)  # Small limit for testing
    manager, monitor = setup_checkpoint_system(config)
    
    try:
        # Create some test checkpoints
        for i in range(5):
            process_id = f"test_process_{i}"
            state = manager.register_process(
                process_id=process_id,
                task_id="test_task",
                total_steps=10,
                description=f"Test process {i}"
            )
            
            # Simulate progress
            for step in range(0, 11, 2):
                manager.update_process_progress(process_id, step)
                await asyncio.sleep(0.1)
            
            manager.complete_process(process_id)
        
        # Get health report
        health_report = monitor.get_health_report()
        print("Health Report:")
        print(json.dumps(health_report, indent=2, default=str))
        
        # Run optimization
        optimizer = CheckpointOptimizer(config)
        optimization_results = optimizer.optimize_storage()
        print("\nOptimization Results:")
        print(json.dumps(optimization_results, indent=2))
        
    finally:
        # Cleanup
        manager.stop_background_tasks()
        monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(example_checkpoint_utilities())
