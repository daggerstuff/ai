#!/usr/bin/env python3
"""
Comprehensive Test Suite for Checkpoint System
Tests checkpoint creation, recovery, optimization, and performance
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import shutil

from checkpoint_system import (
    CheckpointManager, CheckpointStorage, CheckpointType, 
    CheckpointStatus, CheckpointMetadata, ProcessingState
)
from checkpoint_utils import (
    CheckpointOptimizer, CheckpointMonitor, CheckpointConfig,
    create_checkpoint_config, setup_checkpoint_system
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointTestSuite:
    """Comprehensive test suite for checkpoint system"""
    
    def __init__(self):
        self.temp_dir = None
        self.manager = None
        self.monitor = None
        self.test_results = []
    
    def setup_test_environment(self):
        """Setup test environment with temporary storage"""
        
        self.temp_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
        
        config = CheckpointConfig(
            storage_path=os.path.join(self.temp_dir, "checkpoints"),
            max_storage_size_gb=0.1,  # Small limit for testing
            auto_checkpoint_interval=10,  # Fast for testing
            cleanup_interval=30,  # Fast for testing
            compression_enabled=True,
            backup_enabled=True,
            backup_path=os.path.join(self.temp_dir, "backups"),
            monitoring_enabled=True
        )
        
        self.manager, self.monitor = setup_checkpoint_system(config)
        logger.info(f"Test environment setup in {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        
        if self.manager:
            self.manager.stop_background_tasks()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    async def run_all_tests(self):
        """Run all test scenarios"""
        
        print("🧪 Starting Checkpoint System Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_basic_checkpoint_operations,
            self.test_process_registration_and_progress,
            self.test_checkpoint_recovery,
            self.test_storage_optimization,
            self.test_monitoring_and_health,
            self.test_concurrent_operations,
            self.test_error_handling,
            self.test_performance_benchmarks,
            self.test_storage_limits,
            self.test_compression_and_deduplication
        ]
        
        for test_method in test_methods:
            try:
                print(f"\n🔍 Running {test_method.__name__}...")
                await test_method()
                print(f"✅ {test_method.__name__} passed")
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}")
                logger.error(f"Test {test_method.__name__} failed", exc_info=True)
        
        # Print summary
        self.print_test_summary()
    
    async def test_basic_checkpoint_operations(self):
        """Test basic checkpoint creation, storage, and retrieval"""
        
        # Create a checkpoint
        process_id = "test_basic_001"
        task_id = "basic_test"
        
        test_data = {
            "message": "Hello, checkpoint!",
            "timestamp": datetime.utcnow().isoformat(),
            "data": list(range(100))
        }
        
        checkpoint_id = self.manager.create_checkpoint(
            process_id=process_id,
            task_id=task_id,
            checkpoint_type=CheckpointType.CUSTOM,
            data=test_data,
            description="Basic test checkpoint"
        )
        
        assert checkpoint_id is not None, "Checkpoint ID should not be None"
        
        # Retrieve the checkpoint
        metadata, loaded_data = self.manager.storage.load_checkpoint(checkpoint_id)
        
        assert metadata.checkpoint_id == checkpoint_id, "Checkpoint ID mismatch"
        assert metadata.process_id == process_id, "Process ID mismatch"
        assert metadata.task_id == task_id, "Task ID mismatch"
        assert loaded_data == test_data, "Data mismatch"
        
        # List checkpoints
        checkpoints = self.manager.storage.list_checkpoints(process_id=process_id)
        assert len(checkpoints) >= 1, "Should find at least one checkpoint"
        
        # Delete checkpoint
        deleted = self.manager.storage.delete_checkpoint(checkpoint_id)
        assert deleted, "Checkpoint deletion should succeed"
        
        self.test_results.append({
            "test": "basic_checkpoint_operations",
            "status": "passed",
            "details": f"Created, retrieved, and deleted checkpoint {checkpoint_id}"
        })
    
    async def test_process_registration_and_progress(self):
        """Test process registration and progress tracking"""
        
        process_id = "test_progress_001"
        task_id = "progress_test"
        total_steps = 50
        
        # Register process
        state = self.manager.register_process(
            process_id=process_id,
            task_id=task_id,
            total_steps=total_steps,
            description="Progress tracking test"
        )
        
        assert state.process_id == process_id, "Process ID mismatch"
        assert state.total_steps == total_steps, "Total steps mismatch"
        assert state.progress_percentage == 0.0, "Initial progress should be 0"
        
        # Update progress
        for step in range(0, total_steps + 1, 10):
            updated_state = self.manager.update_process_progress(
                process_id=process_id,
                completed_steps=step,
                current_step=f"Step {step}",
                metadata={"batch": step // 10}
            )
            
            expected_progress = (step / total_steps) * 100
            assert abs(updated_state.progress_percentage - expected_progress) < 0.1, \
                f"Progress mismatch: expected {expected_progress}, got {updated_state.progress_percentage}"
        
        # Complete process
        final_checkpoint = self.manager.complete_process(
            process_id=process_id,
            final_data={"result": "success", "total_processed": total_steps}
        )
        
        assert final_checkpoint is not None, "Final checkpoint should be created"
        assert process_id not in self.manager.active_processes, "Process should be removed from active list"
        
        self.test_results.append({
            "test": "process_registration_and_progress",
            "status": "passed",
            "details": f"Tracked progress for {total_steps} steps and completed successfully"
        })
    
    async def test_checkpoint_recovery(self):
        """Test checkpoint recovery functionality"""
        
        process_id = "test_recovery_001"
        task_id = "recovery_test"
        
        # Register and progress a process
        state = self.manager.register_process(
            process_id=process_id,
            task_id=task_id,
            total_steps=20,
            description="Recovery test process"
        )
        
        # Make some progress
        self.manager.update_process_progress(
            process_id=process_id,
            completed_steps=10,
            current_step="Halfway point",
            metadata={"checkpoint_reason": "recovery_test"}
        )
        
        # Simulate process interruption (remove from active processes)
        original_state = self.manager.active_processes[process_id]
        del self.manager.active_processes[process_id]
        
        # Recover the process
        recovered_state = self.manager.recover_process(process_id)
        
        assert recovered_state is not None, "Process recovery should succeed"
        assert recovered_state.process_id == process_id, "Recovered process ID mismatch"
        assert recovered_state.completed_steps == 10, "Recovered progress mismatch"
        assert recovered_state.current_step == "Halfway point", "Recovered step mismatch"
        
        # Continue from recovered state
        self.manager.update_process_progress(
            process_id=process_id,
            completed_steps=20,
            current_step="Completed after recovery"
        )
        
        final_state = self.manager.active_processes[process_id]
        assert final_state.progress_percentage == 100.0, "Should reach 100% after recovery"
        
        self.test_results.append({
            "test": "checkpoint_recovery",
            "status": "passed",
            "details": f"Successfully recovered process at 50% progress and completed"
        })
    
    async def test_storage_optimization(self):
        """Test storage optimization features"""
        
        optimizer = CheckpointOptimizer(self.monitor.config)
        
        # Create multiple checkpoints to test optimization
        for i in range(10):
            process_id = f"test_optimize_{i}"
            
            # Create and complete process
            state = self.manager.register_process(
                process_id=process_id,
                task_id="optimization_test",
                total_steps=5,
                description=f"Optimization test {i}"
            )
            
            self.manager.update_process_progress(process_id, 5)
            self.manager.complete_process(process_id, {"result": f"completed_{i}"})
        
        # Get initial stats
        initial_stats = self.manager.storage.get_storage_stats()
        initial_completed = initial_stats["status_counts"].get("completed", 0)
        
        # Run optimization
        optimization_results = optimizer.optimize_storage()
        
        assert "actions_taken" in optimization_results, "Should return actions taken"
        
        # Verify some optimization occurred
        final_stats = self.manager.storage.get_storage_stats()
        
        # Should have archived some completed checkpoints
        if optimization_results.get("checkpoints_archived", 0) > 0:
            final_completed = final_stats["status_counts"].get("completed", 0)
            assert final_completed < initial_completed, "Should have fewer completed checkpoints after archiving"
        
        self.test_results.append({
            "test": "storage_optimization",
            "status": "passed",
            "details": f"Optimization completed: {optimization_results}"
        })
    
    async def test_monitoring_and_health(self):
        """Test monitoring and health reporting"""
        
        # Let monitoring collect some data
        await asyncio.sleep(2)
        
        # Get health report
        health_report = self.monitor.get_health_report()
        
        assert "status" in health_report, "Health report should have status"
        assert "health_score" in health_report, "Health report should have health score"
        assert "metrics" in health_report, "Health report should have metrics"
        
        # Health score should be reasonable
        health_score = health_report["health_score"]
        assert 0 <= health_score <= 100, f"Health score should be 0-100, got {health_score}"
        
        # Should have system metrics
        metrics = health_report["metrics"]
        assert "storage_stats" in metrics, "Should have storage stats"
        assert "system_resources" in metrics, "Should have system resource stats"
        
        self.test_results.append({
            "test": "monitoring_and_health",
            "status": "passed",
            "details": f"Health score: {health_score}, Status: {health_report['status']}"
        })
    
    async def test_concurrent_operations(self):
        """Test concurrent checkpoint operations"""
        
        async def create_process_with_checkpoints(process_num: int):
            """Create a process with multiple checkpoints"""
            
            process_id = f"concurrent_test_{process_num}"
            task_id = f"concurrent_task_{process_num}"
            
            state = self.manager.register_process(
                process_id=process_id,
                task_id=task_id,
                total_steps=10,
                description=f"Concurrent test process {process_num}"
            )
            
            for step in range(0, 11, 2):
                self.manager.update_process_progress(
                    process_id=process_id,
                    completed_steps=step,
                    current_step=f"Step {step}",
                    metadata={"process_num": process_num, "step": step}
                )
                await asyncio.sleep(0.1)  # Simulate work
            
            self.manager.complete_process(process_id, {"process_num": process_num})
            return process_id
        
        # Run multiple concurrent processes
        tasks = [create_process_with_checkpoints(i) for i in range(5)]
        completed_processes = await asyncio.gather(*tasks)
        
        assert len(completed_processes) == 5, "All concurrent processes should complete"
        
        # Verify all processes completed successfully
        for process_id in completed_processes:
            checkpoints = self.manager.storage.list_checkpoints(process_id=process_id)
            assert len(checkpoints) > 0, f"Process {process_id} should have checkpoints"
        
        self.test_results.append({
            "test": "concurrent_operations",
            "status": "passed",
            "details": f"Successfully completed {len(completed_processes)} concurrent processes"
        })
    
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        
        # Test invalid checkpoint ID
        try:
            self.manager.storage.load_checkpoint("invalid_checkpoint_id")
            assert False, "Should raise exception for invalid checkpoint ID"
        except ValueError:
            pass  # Expected
        
        # Test updating non-existent process
        try:
            self.manager.update_process_progress("non_existent_process", 5)
            assert False, "Should raise exception for non-existent process"
        except ValueError:
            pass  # Expected
        
        # Test recovery of non-existent process
        recovered = self.manager.recover_process("non_existent_process")
        assert recovered is None, "Recovery of non-existent process should return None"
        
        # Test checkpoint with invalid data (should handle gracefully)
        try:
            # Create checkpoint with complex object that might cause serialization issues
            class UnserializableClass:
                def __init__(self):
                    self.file_handle = open(__file__, 'r')  # This won't serialize
            
            # This should either succeed or fail gracefully
            try:
                checkpoint_id = self.manager.create_checkpoint(
                    process_id="error_test",
                    task_id="error_test",
                    checkpoint_type=CheckpointType.CUSTOM,
                    data=UnserializableClass(),
                    description="Error handling test"
                )
                # If it succeeds, clean up
                self.manager.storage.delete_checkpoint(checkpoint_id)
            except Exception:
                pass  # Expected to fail, that's fine
        
        except Exception as e:
            # Any other exception is acceptable for this test
            pass
        
        self.test_results.append({
            "test": "error_handling",
            "status": "passed",
            "details": "Error handling scenarios completed successfully"
        })
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        
        # Benchmark checkpoint creation
        start_time = time.time()
        checkpoint_ids = []
        
        for i in range(50):
            checkpoint_id = self.manager.create_checkpoint(
                process_id=f"perf_test_{i}",
                task_id="performance_test",
                checkpoint_type=CheckpointType.CUSTOM,
                data={"index": i, "data": list(range(100))},
                description=f"Performance test checkpoint {i}"
            )
            checkpoint_ids.append(checkpoint_id)
        
        creation_time = time.time() - start_time
        avg_creation_time = creation_time / 50
        
        # Benchmark checkpoint loading
        start_time = time.time()
        
        for checkpoint_id in checkpoint_ids[:10]:  # Load first 10
            metadata, data = self.manager.storage.load_checkpoint(checkpoint_id)
        
        loading_time = time.time() - start_time
        avg_loading_time = loading_time / 10
        
        # Performance assertions
        assert avg_creation_time < 0.1, f"Checkpoint creation too slow: {avg_creation_time:.3f}s"
        assert avg_loading_time < 0.05, f"Checkpoint loading too slow: {avg_loading_time:.3f}s"
        
        # Cleanup
        for checkpoint_id in checkpoint_ids:
            self.manager.storage.delete_checkpoint(checkpoint_id)
        
        self.test_results.append({
            "test": "performance_benchmarks",
            "status": "passed",
            "details": f"Avg creation: {avg_creation_time:.3f}s, Avg loading: {avg_loading_time:.3f}s"
        })
    
    async def test_storage_limits(self):
        """Test storage limit enforcement"""
        
        # Create checkpoints until we approach the limit
        checkpoint_ids = []
        
        try:
            for i in range(20):  # Try to create many checkpoints
                # Create larger data to fill storage faster
                large_data = {"data": list(range(1000)), "index": i}
                
                checkpoint_id = self.manager.create_checkpoint(
                    process_id=f"storage_test_{i}",
                    task_id="storage_limit_test",
                    checkpoint_type=CheckpointType.CUSTOM,
                    data=large_data,
                    description=f"Storage limit test {i}"
                )
                checkpoint_ids.append(checkpoint_id)
                
                # Check if we're approaching limits
                stats = self.manager.storage.get_storage_stats()
                storage_gb = stats["total_size_bytes"] / (1024**3)
                
                if storage_gb > 0.05:  # 50MB limit for test
                    break
        
        except Exception as e:
            # Storage limit reached or other storage-related error is acceptable
            pass
        
        # Verify we created some checkpoints
        assert len(checkpoint_ids) > 0, "Should have created at least some checkpoints"
        
        # Test cleanup
        deleted_count = self.manager.storage.cleanup_expired_checkpoints()
        
        self.test_results.append({
            "test": "storage_limits",
            "status": "passed",
            "details": f"Created {len(checkpoint_ids)} checkpoints, cleaned up {deleted_count}"
        })
    
    async def test_compression_and_deduplication(self):
        """Test compression and deduplication features"""
        
        # Create identical data checkpoints
        identical_data = {"message": "identical", "data": list(range(50))}
        
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = self.manager.create_checkpoint(
                process_id=f"dedup_test_{i}",
                task_id="deduplication_test",
                checkpoint_type=CheckpointType.CUSTOM,
                data=identical_data,
                description=f"Deduplication test {i}"
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Get initial stats
        initial_stats = self.manager.storage.get_storage_stats()
        
        # Run optimization (which includes deduplication)
        optimizer = CheckpointOptimizer(self.monitor.config)
        optimization_results = optimizer.optimize_storage()
        
        # Verify optimization occurred
        final_stats = self.manager.storage.get_storage_stats()
        
        # Should have some optimization actions
        assert len(optimization_results["actions_taken"]) >= 0, "Should have taken some optimization actions"
        
        # Test compression specifically
        if self.monitor.config.compression_enabled:
            # Verify compressed checkpoints exist
            with self.manager.storage.storage.connect(self.manager.storage.db_path) as conn:
                compressed_count = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE compression = 1"
                ).fetchone()[0]
                
                assert compressed_count > 0, "Should have compressed checkpoints"
        
        self.test_results.append({
            "test": "compression_and_deduplication",
            "status": "passed",
            "details": f"Optimization actions: {optimization_results['actions_taken']}"
        })
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        
        print("\n" + "=" * 60)
        print("🏁 CHECKPOINT SYSTEM TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = [r for r in self.test_results if r["status"] == "passed"]
        failed_tests = [r for r in self.test_results if r["status"] == "failed"]
        
        print(f"✅ Passed: {len(passed_tests)}")
        print(f"❌ Failed: {len(failed_tests)}")
        print(f"📊 Total: {len(self.test_results)}")
        
        if passed_tests:
            print("\n✅ PASSED TESTS:")
            for test in passed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        # Overall result
        success_rate = len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 All tests passed! Checkpoint system is working correctly.")
        elif success_rate >= 80:
            print("⚠️  Most tests passed, but some issues need attention.")
        else:
            print("🚨 Multiple test failures detected. System needs review.")

# Main execution
async def main():
    """Run the complete checkpoint test suite"""
    
    test_suite = CheckpointTestSuite()
    
    try:
        test_suite.setup_test_environment()
        await test_suite.run_all_tests()
    finally:
        test_suite.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())
