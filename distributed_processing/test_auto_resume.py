#!/usr/bin/env python3
"""
Comprehensive Test Suite for Automatic Resume System
Tests interruption detection, resume strategies, orchestration, and recovery scenarios
"""

import asyncio
import json
import logging
import os
import signal
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import threading

from auto_resume_engine import (
    AutoResumeEngine, InterruptionContext, InterruptionType, 
    ResumeStrategy, ResumeConfiguration, ProcessingInterruptionDetector
)
from resume_orchestrator import (
    ResumeOrchestrator, ProcessPriority, DependencyType, ProcessDependency
)
from checkpoint_system import CheckpointManager, CheckpointType, ProcessingState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoResumeTestSuite:
    """Comprehensive test suite for automatic resume system"""
    
    def __init__(self):
        self.temp_dir = None
        self.checkpoint_manager = None
        self.resume_engine = None
        self.orchestrator = None
        self.test_results = []
    
    def setup_test_environment(self):
        """Setup test environment with temporary storage"""
        
        self.temp_dir = tempfile.mkdtemp(prefix="auto_resume_test_")
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            storage_path=os.path.join(self.temp_dir, "checkpoints")
        )
        
        # Initialize resume engine with test configuration
        config = ResumeConfiguration(
            auto_resume_enabled=True,
            max_resume_attempts=2,
            resume_delay_seconds=1,  # Fast for testing
            timeout_threshold_minutes=1,  # Short for testing
            memory_threshold_percent=95.0,
            cpu_threshold_percent=98.0
        )
        
        self.resume_engine = AutoResumeEngine(self.checkpoint_manager, config)
        self.orchestrator = ResumeOrchestrator(self.checkpoint_manager, config)
        
        logger.info(f"Test environment setup in {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        
        if self.resume_engine:
            self.resume_engine.stop()
        
        if self.orchestrator:
            self.orchestrator.stop()
        
        if self.checkpoint_manager:
            self.checkpoint_manager.stop_background_tasks()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    async def run_all_tests(self):
        """Run all test scenarios"""
        
        print("🧪 Starting Automatic Resume System Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_basic_resume_functionality,
            self.test_interruption_detection,
            self.test_resume_strategies,
            self.test_process_orchestration,
            self.test_dependency_management,
            self.test_priority_handling,
            self.test_resource_management,
            self.test_concurrent_resumes,
            self.test_error_recovery,
            self.test_performance_benchmarks
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
    
    async def test_basic_resume_functionality(self):
        """Test basic resume functionality"""
        
        # Create a test resume handler
        resume_called = False
        resume_data = {}
        
        async def test_resume_handler(resume_point: Dict[str, Any], 
                                    interruption_context: InterruptionContext = None):
            nonlocal resume_called, resume_data
            resume_called = True
            resume_data = resume_point
            
            # Simulate successful resume
            original_state = resume_point["original_state"]
            
            # Continue from resume point
            for step in range(resume_point["resume_step"], original_state.total_steps + 1):
                self.checkpoint_manager.update_process_progress(
                    process_id=original_state.process_id,
                    completed_steps=step,
                    current_step=f"Resumed step {step}"
                )
                await asyncio.sleep(0.01)
            
            self.checkpoint_manager.complete_process(original_state.process_id)
            return True
        
        # Start resume engine
        self.resume_engine.start()
        
        # Register a process
        process_id = "test_basic_resume"
        task_id = "basic_resume_test"
        
        self.resume_engine.register_process(
            process_id=process_id,
            task_id=task_id,
            resume_handler=test_resume_handler
        )
        
        # Create a process with some progress
        state = self.checkpoint_manager.register_process(
            process_id=process_id,
            task_id=task_id,
            total_steps=10,
            description="Basic resume test"
        )
        
        # Make some progress
        for step in range(5):
            self.checkpoint_manager.update_process_progress(
                process_id=process_id,
                completed_steps=step,
                current_step=f"Step {step}"
            )
        
        # Test resume
        interruption = InterruptionContext(
            interruption_id=str(uuid.uuid4()),
            process_id=process_id,
            task_id=task_id,
            interruption_type=InterruptionType.PROCESS_CRASH,
            timestamp=datetime.utcnow()
        )
        
        success = await self.resume_engine.resume_process(process_id, interruption)
        
        assert success, "Resume should succeed"
        assert resume_called, "Resume handler should be called"
        assert resume_data["process_id"] == process_id, "Resume data should contain correct process ID"
        
        self.test_results.append({
            "test": "basic_resume_functionality",
            "status": "passed",
            "details": f"Successfully resumed process from step {resume_data['resume_step']}"
        })
    
    async def test_interruption_detection(self):
        """Test interruption detection mechanisms"""
        
        detector = ProcessingInterruptionDetector(self.resume_engine.config)
        
        # Test callback registration
        interruptions_detected = []
        
        def interruption_callback(interruption: InterruptionContext):
            interruptions_detected.append(interruption)
        
        detector.add_interruption_callback(interruption_callback)
        
        # Start monitoring
        detector.start_monitoring()
        
        try:
            # Test heartbeat timeout detection
            process_id = "timeout_test_process"
            detector.register_process_heartbeat(process_id)
            
            # Wait for timeout (should be detected quickly in test config)
            await asyncio.sleep(2)
            
            # Should have detected timeout
            timeout_interruptions = [
                i for i in interruptions_detected 
                if i.interruption_type == InterruptionType.TIMEOUT
            ]
            
            assert len(timeout_interruptions) > 0, "Should detect timeout interruption"
            
            # Test manual interruption trigger
            manual_interruption = InterruptionContext(
                interruption_id=str(uuid.uuid4()),
                process_id="manual_test",
                task_id="manual_test",
                interruption_type=InterruptionType.USER_TERMINATION,
                timestamp=datetime.utcnow()
            )
            
            detector._trigger_interruption_callbacks(manual_interruption)
            
            # Should have received manual interruption
            manual_interruptions = [
                i for i in interruptions_detected
                if i.interruption_type == InterruptionType.USER_TERMINATION
            ]
            
            assert len(manual_interruptions) > 0, "Should detect manual interruption"
            
        finally:
            detector.stop_monitoring()
        
        self.test_results.append({
            "test": "interruption_detection",
            "status": "passed",
            "details": f"Detected {len(interruptions_detected)} interruptions"
        })
    
    async def test_resume_strategies(self):
        """Test different resume strategies"""
        
        strategies_tested = []
        
        async def strategy_test_handler(resume_point: Dict[str, Any], 
                                      interruption_context: InterruptionContext = None):
            strategy = resume_point["strategy"]
            strategies_tested.append(strategy)
            
            # Verify strategy was applied correctly
            original_state = resume_point["original_state"]
            resume_step = resume_point["resume_step"]
            
            if strategy == ResumeStrategy.EXACT_CONTINUATION:
                assert resume_step == original_state.completed_steps, "Exact continuation should resume from current step"
            elif strategy == ResumeStrategy.ROLLBACK_AND_RETRY:
                assert resume_step < original_state.completed_steps, "Rollback should resume from earlier step"
            elif strategy == ResumeStrategy.PARTIAL_RESTART:
                assert resume_step <= original_state.completed_steps * 0.9, "Partial restart should go back significantly"
            elif strategy == ResumeStrategy.FULL_RESTART:
                assert resume_step == 0, "Full restart should start from beginning"
            
            return True
        
        self.resume_engine.start()
        
        # Test different interruption types that trigger different strategies
        test_cases = [
            (InterruptionType.SYSTEM_SHUTDOWN, ResumeStrategy.EXACT_CONTINUATION),
            (InterruptionType.PROCESS_CRASH, ResumeStrategy.ROLLBACK_AND_RETRY),
            (InterruptionType.MEMORY_EXHAUSTION, ResumeStrategy.PARTIAL_RESTART),
        ]
        
        for i, (interruption_type, expected_strategy) in enumerate(test_cases):
            process_id = f"strategy_test_{i}"
            task_id = f"strategy_test_task_{i}"
            
            # Register process
            self.resume_engine.register_process(
                process_id=process_id,
                task_id=task_id,
                resume_handler=strategy_test_handler
            )
            
            # Create process with progress
            state = self.checkpoint_manager.register_process(
                process_id=process_id,
                task_id=task_id,
                total_steps=20,
                description=f"Strategy test {i}"
            )
            
            # Make progress
            self.checkpoint_manager.update_process_progress(
                process_id=process_id,
                completed_steps=10,
                current_step="Mid-progress"
            )
            
            # Create interruption
            interruption = InterruptionContext(
                interruption_id=str(uuid.uuid4()),
                process_id=process_id,
                task_id=task_id,
                interruption_type=interruption_type,
                timestamp=datetime.utcnow()
            )
            
            # Test resume
            success = await self.resume_engine.resume_process(process_id, interruption)
            assert success, f"Resume should succeed for {interruption_type.value}"
        
        # Verify all strategies were tested
        assert len(strategies_tested) == len(test_cases), "All strategies should be tested"
        
        self.test_results.append({
            "test": "resume_strategies",
            "status": "passed",
            "details": f"Tested {len(strategies_tested)} resume strategies"
        })
    
    async def test_process_orchestration(self):
        """Test process orchestration functionality"""
        
        self.orchestrator.start()
        
        # Track resume calls
        resume_calls = []
        
        async def orchestrated_handler(resume_point: Dict[str, Any], 
                                     interruption_context: InterruptionContext = None):
            resume_calls.append({
                "process_id": resume_point["process_id"],
                "timestamp": datetime.utcnow(),
                "strategy": resume_point["strategy"]
            })
            
            # Simulate resume work
            await asyncio.sleep(0.5)
            return True
        
        # Register multiple processes with different priorities
        processes = [
            ("critical_process", ProcessPriority.CRITICAL),
            ("high_process", ProcessPriority.HIGH),
            ("medium_process", ProcessPriority.MEDIUM),
            ("low_process", ProcessPriority.LOW)
        ]
        
        for process_id, priority in processes:
            self.orchestrator.register_resumable_process(
                process_id=process_id,
                task_id=f"{process_id}_task",
                resume_handler=orchestrated_handler,
                priority=priority
            )
        
        # Request resumes for all processes
        for process_id, _ in processes:
            await self.orchestrator.request_resume(process_id)
        
        # Wait for orchestration
        await asyncio.sleep(3)
        
        # Verify resumes occurred
        assert len(resume_calls) > 0, "Should have resume calls"
        
        # Verify priority ordering (critical should be first)
        if len(resume_calls) > 1:
            first_call = resume_calls[0]
            assert "critical" in first_call["process_id"], "Critical process should resume first"
        
        # Get orchestrator status
        status = self.orchestrator.get_orchestrator_status()
        assert status["orchestrator_active"], "Orchestrator should be active"
        
        self.test_results.append({
            "test": "process_orchestration",
            "status": "passed",
            "details": f"Orchestrated {len(resume_calls)} process resumes"
        })
    
    async def test_dependency_management(self):
        """Test process dependency management"""
        
        self.orchestrator.start()
        
        # Track completion order
        completion_order = []
        
        async def dependency_handler(resume_point: Dict[str, Any], 
                                   interruption_context: InterruptionContext = None):
            process_id = resume_point["process_id"]
            completion_order.append(process_id)
            
            # Simulate different completion times
            if "fast" in process_id:
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(0.5)
            
            return True
        
        # Register processes with dependencies
        self.orchestrator.register_resumable_process(
            process_id="dependency_fast",
            task_id="fast_task",
            resume_handler=dependency_handler,
            priority=ProcessPriority.MEDIUM
        )
        
        self.orchestrator.register_resumable_process(
            process_id="dependent_slow",
            task_id="slow_task", 
            resume_handler=dependency_handler,
            priority=ProcessPriority.HIGH  # Higher priority but should wait for dependency
        )
        
        # Add dependency
        self.orchestrator.add_process_dependency(
            dependent_process="dependent_slow",
            dependency_process="dependency_fast",
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        # Create processes with some progress to make them resumable
        for process_id in ["dependency_fast", "dependent_slow"]:
            state = self.checkpoint_manager.register_process(
                process_id=process_id,
                task_id=f"{process_id}_task",
                total_steps=10,
                description=f"Dependency test {process_id}"
            )
            
            # Make partial progress
            self.checkpoint_manager.update_process_progress(
                process_id=process_id,
                completed_steps=5,
                current_step="Partial progress"
            )
        
        # Request resumes
        await self.orchestrator.request_resume("dependent_slow")
        await self.orchestrator.request_resume("dependency_fast")
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Verify dependency order was respected
        if len(completion_order) >= 2:
            fast_index = completion_order.index("dependency_fast")
            slow_index = completion_order.index("dependent_slow")
            assert fast_index < slow_index, "Dependency should complete before dependent"
        
        self.test_results.append({
            "test": "dependency_management",
            "status": "passed",
            "details": f"Managed dependencies with completion order: {completion_order}"
        })
    
    async def test_priority_handling(self):
        """Test priority-based resume handling"""
        
        self.orchestrator.start()
        
        # Track resume order
        resume_order = []
        
        async def priority_handler(resume_point: Dict[str, Any], 
                                 interruption_context: InterruptionContext = None):
            process_id = resume_point["process_id"]
            resume_order.append(process_id)
            await asyncio.sleep(0.1)
            return True
        
        # Register processes with different priorities
        priorities = [
            ("low_priority", ProcessPriority.LOW),
            ("critical_priority", ProcessPriority.CRITICAL),
            ("medium_priority", ProcessPriority.MEDIUM),
            ("high_priority", ProcessPriority.HIGH)
        ]
        
        for process_id, priority in priorities:
            self.orchestrator.register_resumable_process(
                process_id=process_id,
                task_id=f"{process_id}_task",
                resume_handler=priority_handler,
                priority=priority
            )
        
        # Request all resumes at once
        for process_id, _ in priorities:
            await self.orchestrator.request_resume(process_id)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify priority order
        assert len(resume_order) > 0, "Should have processed some resumes"
        
        # Critical should be first if multiple processed
        if len(resume_order) > 1:
            assert "critical" in resume_order[0], "Critical priority should be processed first"
        
        self.test_results.append({
            "test": "priority_handling",
            "status": "passed",
            "details": f"Processed resumes in order: {resume_order}"
        })
    
    async def test_resource_management(self):
        """Test resource management during resumes"""
        
        self.orchestrator.start()
        
        # Track resource usage
        resource_usage_log = []
        
        async def resource_handler(resume_point: Dict[str, Any], 
                                 interruption_context: InterruptionContext = None):
            process_id = resume_point["process_id"]
            
            # Log current resource usage
            status = self.orchestrator.get_orchestrator_status()
            resource_usage_log.append({
                "process_id": process_id,
                "resource_utilization": status["resource_utilization"]
            })
            
            await asyncio.sleep(0.3)
            return True
        
        # Register processes with different resource requirements
        self.orchestrator.register_resumable_process(
            process_id="cpu_intensive_1",
            task_id="cpu_task_1",
            resume_handler=resource_handler,
            resource_requirements={"cpu_intensive": 1}
        )
        
        self.orchestrator.register_resumable_process(
            process_id="cpu_intensive_2", 
            task_id="cpu_task_2",
            resume_handler=resource_handler,
            resource_requirements={"cpu_intensive": 2}  # Requires more CPU
        )
        
        # Request resumes
        await self.orchestrator.request_resume("cpu_intensive_1")
        await self.orchestrator.request_resume("cpu_intensive_2")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify resource management occurred
        assert len(resource_usage_log) > 0, "Should have resource usage logs"
        
        # Check that resource limits were respected
        for log_entry in resource_usage_log:
            cpu_utilization = log_entry["resource_utilization"]["cpu_intensive"]
            assert cpu_utilization["used"] <= cpu_utilization["total"], "Should not exceed resource limits"
        
        self.test_results.append({
            "test": "resource_management",
            "status": "passed",
            "details": f"Managed resources for {len(resource_usage_log)} processes"
        })
    
    async def test_concurrent_resumes(self):
        """Test concurrent resume operations"""
        
        self.orchestrator.start()
        
        # Track concurrent operations
        active_resumes = []
        max_concurrent = 0
        
        async def concurrent_handler(resume_point: Dict[str, Any], 
                                   interruption_context: InterruptionContext = None):
            process_id = resume_point["process_id"]
            
            # Track active resumes
            active_resumes.append(process_id)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_resumes))
            
            # Simulate work
            await asyncio.sleep(0.5)
            
            # Remove from active
            active_resumes.remove(process_id)
            return True
        
        # Register multiple processes
        process_count = 8
        for i in range(process_count):
            process_id = f"concurrent_process_{i}"
            self.orchestrator.register_resumable_process(
                process_id=process_id,
                task_id=f"concurrent_task_{i}",
                resume_handler=concurrent_handler,
                priority=ProcessPriority.MEDIUM
            )
        
        # Request all resumes simultaneously
        for i in range(process_count):
            await self.orchestrator.request_resume(f"concurrent_process_{i}")
        
        # Wait for all to complete
        await asyncio.sleep(3)
        
        # Verify concurrency was managed
        assert max_concurrent > 1, "Should have concurrent resumes"
        assert max_concurrent <= self.orchestrator.max_concurrent_resumes, "Should respect concurrency limits"
        
        self.test_results.append({
            "test": "concurrent_resumes",
            "status": "passed",
            "details": f"Max concurrent resumes: {max_concurrent}/{process_count}"
        })
    
    async def test_error_recovery(self):
        """Test error recovery in resume operations"""
        
        self.resume_engine.start()
        
        # Track retry attempts
        retry_attempts = []
        
        async def failing_handler(resume_point: Dict[str, Any], 
                                interruption_context: InterruptionContext = None):
            process_id = resume_point["process_id"]
            retry_attempts.append(process_id)
            
            # Fail first few attempts, succeed on last
            if len(retry_attempts) < 3:
                raise Exception(f"Simulated failure for {process_id}")
            
            return True
        
        # Register process with failing handler
        process_id = "error_recovery_test"
        self.resume_engine.register_process(
            process_id=process_id,
            task_id="error_recovery_task",
            resume_handler=failing_handler
        )
        
        # Create process state
        state = self.checkpoint_manager.register_process(
            process_id=process_id,
            task_id="error_recovery_task",
            total_steps=10,
            description="Error recovery test"
        )
        
        self.checkpoint_manager.update_process_progress(
            process_id=process_id,
            completed_steps=5,
            current_step="Mid-progress"
        )
        
        # Create interruption that will trigger retries
        interruption = InterruptionContext(
            interruption_id=str(uuid.uuid4()),
            process_id=process_id,
            task_id="error_recovery_task",
            interruption_type=InterruptionType.PROCESS_CRASH,
            timestamp=datetime.utcnow()
        )
        
        # Trigger automatic retry through interruption handling
        self.resume_engine._handle_interruption(interruption)
        
        # Wait for retries to complete
        await asyncio.sleep(5)
        
        # Verify retries occurred
        assert len(retry_attempts) >= 2, "Should have retry attempts"
        
        self.test_results.append({
            "test": "error_recovery",
            "status": "passed",
            "details": f"Completed {len(retry_attempts)} retry attempts"
        })
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks for resume operations"""
        
        self.orchestrator.start()
        
        # Performance tracking
        resume_times = []
        
        async def benchmark_handler(resume_point: Dict[str, Any], 
                                  interruption_context: InterruptionContext = None):
            start_time = time.time()
            
            # Simulate minimal resume work
            await asyncio.sleep(0.01)
            
            end_time = time.time()
            resume_times.append(end_time - start_time)
            return True
        
        # Register multiple processes for benchmarking
        process_count = 20
        for i in range(process_count):
            process_id = f"benchmark_process_{i}"
            self.orchestrator.register_resumable_process(
                process_id=process_id,
                task_id=f"benchmark_task_{i}",
                resume_handler=benchmark_handler,
                priority=ProcessPriority.MEDIUM
            )
        
        # Measure total orchestration time
        start_time = time.time()
        
        # Request all resumes
        for i in range(process_count):
            await self.orchestrator.request_resume(f"benchmark_process_{i}")
        
        # Wait for completion
        await asyncio.sleep(5)
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        if resume_times:
            avg_resume_time = sum(resume_times) / len(resume_times)
            max_resume_time = max(resume_times)
            min_resume_time = min(resume_times)
        else:
            avg_resume_time = max_resume_time = min_resume_time = 0
        
        # Performance assertions
        assert avg_resume_time < 1.0, f"Average resume time too slow: {avg_resume_time:.3f}s"
        assert total_time < 10.0, f"Total orchestration time too slow: {total_time:.3f}s"
        
        self.test_results.append({
            "test": "performance_benchmarks",
            "status": "passed",
            "details": f"Avg resume: {avg_resume_time:.3f}s, Total: {total_time:.3f}s, Processes: {len(resume_times)}"
        })
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        
        print("\n" + "=" * 60)
        print("🏁 AUTOMATIC RESUME SYSTEM TEST SUMMARY")
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
            print("🎉 All tests passed! Automatic resume system is working correctly.")
        elif success_rate >= 80:
            print("⚠️  Most tests passed, but some issues need attention.")
        else:
            print("🚨 Multiple test failures detected. System needs review.")

# Main execution
async def main():
    """Run the complete auto-resume test suite"""
    
    test_suite = AutoResumeTestSuite()
    
    try:
        test_suite.setup_test_environment()
        await test_suite.run_all_tests()
    finally:
        test_suite.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())
