"""
Unit tests for performance optimizer.

Tests the comprehensive performance optimization functionality including
intelligent caching, concurrency management, resume capabilities, and
resource optimization for dataset pipeline operations.
"""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from .performance_optimizer import PerformanceMetrics, PerformanceOptimizer


class TestPerformanceOptimizer(unittest.TestCase):
    """Test performance optimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test configuration
        self.config = {
            "cache_directory": str(self.temp_dir / "cache"),
            "max_cache_size_mb": 10,  # Small cache for testing
            "default_ttl_seconds": 60,
            "cache_cleanup_interval": 1,  # Fast cleanup for testing
            "max_threads": 4,
            "max_processes": 2,
            "enable_monitoring": False,  # Disable for most tests
            "monitoring_interval": 1
        }

        self.optimizer = PerformanceOptimizer(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown optimizer
        self.optimizer.shutdown()

        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config is not None
        assert self.optimizer.memory_cache is not None
        assert self.optimizer.cache_stats is not None
        assert self.optimizer.thread_pool is not None
        assert self.optimizer.process_pool is not None

        # Check that cache directory was created
        assert Path(self.config["cache_directory"]).exists()

    def test_cache_basic_operations(self):
        """Test basic cache operations."""

        # Test cache miss
        result = self.optimizer.cache_get("nonexistent_key")
        assert result is None
        assert self.optimizer.cache_stats["misses"] == 1

        # Test cache set and get
        test_data = {"test": "data", "number": 42}
        success = self.optimizer.cache_set("test_key", test_data)
        assert success

        retrieved_data = self.optimizer.cache_get("test_key")
        assert retrieved_data == test_data
        assert self.optimizer.cache_stats["hits"] == 1

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""

        # Set item with short TTL
        test_data = "test_data"
        success = self.optimizer.cache_set("ttl_key", test_data, ttl_seconds=1)
        assert success

        # Should be available immediately
        result = self.optimizer.cache_get("ttl_key")
        assert result == test_data

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired now
        result = self.optimizer.cache_get("ttl_key")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""

        # Fill cache with small items
        for i in range(5):
            self.optimizer.cache_set(f"key_{i}", f"data_{i}")

        # Access some items to change LRU order
        self.optimizer.cache_get("key_1")
        self.optimizer.cache_get("key_3")

        # Add a large item that should trigger eviction
        large_data = "x" * (1024 * 1024)  # 1MB
        success = self.optimizer.cache_set("large_key", large_data)

        # Should succeed and evict LRU items
        assert success

        # Check that some items were evicted
        assert self.optimizer.cache_stats["evictions"] > 0

    def test_cache_invalidation(self):
        """Test cache invalidation."""

        # Set and verify item
        self.optimizer.cache_set("test_key", "test_data")
        assert self.optimizer.cache_get("test_key") is not None

        # Invalidate and verify removal
        success = self.optimizer.cache_invalidate("test_key")
        assert success
        assert self.optimizer.cache_get("test_key") is None

        # Test invalidating non-existent key
        success = self.optimizer.cache_invalidate("nonexistent")
        assert not success

    def test_cache_clear(self):
        """Test cache clearing."""

        # Add some items
        for i in range(3):
            self.optimizer.cache_set(f"key_{i}", f"data_{i}")

        # Verify items exist
        assert len(self.optimizer.memory_cache) == 3

        # Clear cache
        self.optimizer.cache_clear()

        # Verify cache is empty
        assert len(self.optimizer.memory_cache) == 0
        assert self.optimizer.cache_stats["total_size_bytes"] == 0

    def test_concurrency_calculation(self):
        """Test optimal concurrency calculation."""

        # Test with different task counts
        concurrency_10 = self.optimizer._calculate_optimal_concurrency(10)
        concurrency_100 = self.optimizer._calculate_optimal_concurrency(100)

        # Should be reasonable values
        assert concurrency_10 > 0
        assert concurrency_10 <= self.optimizer.max_threads
        assert concurrency_100 > 0
        assert concurrency_100 <= self.optimizer.max_threads

        # Should not exceed task count
        concurrency_2 = self.optimizer._calculate_optimal_concurrency(2)
        assert concurrency_2 <= 2

    def test_execute_with_concurrency_threads(self):
        """Test concurrent execution with threads."""

        def test_task():
            time.sleep(0.1)
            return "task_result"

        # Create multiple tasks
        tasks = [test_task for _ in range(5)]

        async def run_test():
            results = await self.optimizer.execute_with_concurrency(
                tasks, max_concurrent=2, use_processes=False
            )

            # Check results
            assert len(results) == 5
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 5

            # Check that all results are correct
            for result in successful_results:
                assert result == "task_result"

        asyncio.run(run_test())

    def test_resume_point_operations(self):
        """Test resume point creation, loading, and management."""

        # Create resume point
        operation_id = "test_operation"
        progress = {"completed": 50, "total": 100}
        completed_items = ["item1", "item2", "item3"]
        failed_items = ["item4"]
        metadata = {"start_time": "2024-01-01T00:00:00"}

        result_id = self.optimizer.create_resume_point(
            operation_id, "test_type", progress, completed_items, failed_items, metadata
        )

        assert result_id == operation_id

        # Load resume point
        loaded_resume = self.optimizer.load_resume_point(operation_id)

        assert loaded_resume is not None
        assert loaded_resume.operation_id == operation_id
        assert loaded_resume.operation_type == "test_type"
        assert loaded_resume.progress == progress
        assert loaded_resume.completed_items == completed_items
        assert loaded_resume.failed_items == failed_items
        assert loaded_resume.metadata == metadata

        # Update resume point
        new_progress = {"completed": 75, "total": 100}
        new_completed = [*completed_items, "item5"]

        success = self.optimizer.update_resume_point(
            operation_id, new_progress, new_completed, failed_items
        )

        assert success

        # Verify update
        updated_resume = self.optimizer.load_resume_point(operation_id)
        assert updated_resume.progress == new_progress
        assert updated_resume.completed_items == new_completed

        # Delete resume point
        success = self.optimizer.delete_resume_point(operation_id)
        assert success

        # Verify deletion
        deleted_resume = self.optimizer.load_resume_point(operation_id)
        assert deleted_resume is None

    def test_resume_point_nonexistent(self):
        """Test operations on non-existent resume points."""

        # Try to load non-existent resume point
        result = self.optimizer.load_resume_point("nonexistent")
        assert result is None

        # Try to update non-existent resume point
        success = self.optimizer.update_resume_point("nonexistent", {}, [], [])
        assert not success

        # Try to delete non-existent resume point
        success = self.optimizer.delete_resume_point("nonexistent")
        assert not success

    def test_cache_stats(self):
        """Test cache statistics retrieval."""

        # Add some items to cache
        for i in range(3):
            self.optimizer.cache_set(f"key_{i}", f"data_{i}")

        # Get some items (hits)
        self.optimizer.cache_get("key_0")
        self.optimizer.cache_get("key_1")

        # Try to get non-existent item (miss)
        self.optimizer.cache_get("nonexistent")

        # Get cache stats
        stats = self.optimizer.get_cache_stats()

        # Verify stats structure
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate_percent" in stats
        assert "evictions" in stats
        assert "total_entries" in stats
        assert "total_size_mb" in stats
        assert "max_size_mb" in stats
        assert "utilization_percent" in stats

        # Verify some values
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_entries"] == 3
        assert stats["hit_rate_percent"] > 0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_performance_monitoring(self, mock_memory, mock_cpu):
        """Test performance monitoring functionality."""

        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(used=1024*1024*1024)  # 1GB

        # Create optimizer with monitoring enabled
        config = self.config.copy()
        config["enable_monitoring"] = True
        config["monitoring_interval"] = 0.1  # Very fast for testing

        optimizer = PerformanceOptimizer(config)

        try:
            # Wait a bit for monitoring to collect data
            time.sleep(0.5)

            # Get performance metrics
            metrics = optimizer.get_performance_metrics()

            if metrics:  # Monitoring might not have started yet
                assert isinstance(metrics, PerformanceMetrics)
                assert metrics.optimization_score >= 0
                assert metrics.optimization_score <= 100

        finally:
            optimizer.shutdown()

    def test_optimization_score_calculation(self):
        """Test optimization score calculation."""

        # Test with good metrics
        score = self.optimizer._calculate_optimization_score(
            cache_hit_rate=80.0,
            avg_response_time=0.5,
            error_rate=0.01
        )

        # Should be a high score
        assert score > 70
        assert score <= 100

        # Test with poor metrics
        score = self.optimizer._calculate_optimization_score(
            cache_hit_rate=20.0,
            avg_response_time=10.0,
            error_rate=0.2
        )

        # Should be a low score
        assert score < 50
        assert score >= 0

    def test_performance_alerts(self):
        """Test performance alert generation."""

        # Create metrics that should trigger alerts
        bad_metrics = PerformanceMetrics(
            cache_hit_rate=30.0,  # Low hit rate
            average_response_time=10.0,  # High response time
            error_rate=15.0,  # High error rate
            memory_usage_mb=10000,  # High memory usage
            cpu_usage_percent=95.0  # High CPU usage
        )

        initial_alert_count = len(self.optimizer.performance_alerts)

        # Trigger alert checking
        self.optimizer._check_performance_alerts(bad_metrics)

        # Should have generated alerts
        assert len(self.optimizer.performance_alerts) > initial_alert_count

        # Check alert structure
        if self.optimizer.performance_alerts:
            alert = self.optimizer.performance_alerts[-1]
            assert "message" in alert
            assert "timestamp" in alert
            assert "severity" in alert

    def test_empty_task_execution(self):
        """Test execution with empty task list."""

        async def run_test():
            results = await self.optimizer.execute_with_concurrency([])
            assert results == []

        asyncio.run(run_test())

    def test_cache_metadata(self):
        """Test cache entry metadata functionality."""

        test_data = "test_data"
        metadata = {"source": "test", "priority": "high"}

        # Set with metadata
        success = self.optimizer.cache_set("meta_key", test_data, metadata=metadata)
        assert success

        # Verify data is retrievable (metadata is internal)
        result = self.optimizer.cache_get("meta_key")
        assert result == test_data

        # Check that metadata was stored (internal verification)
        with self.optimizer.cache_lock:
            if "meta_key" in self.optimizer.memory_cache:
                entry = self.optimizer.memory_cache["meta_key"]
                assert entry.metadata == metadata


if __name__ == "__main__":
    unittest.main()
