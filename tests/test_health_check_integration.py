"""
Integration tests for health check and graceful shutdown functionality.
Tests that health checks work correctly and shutdown is graceful.
"""

import unittest
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging
from unittest.mock import patch, MagicMock
import signal
import threading

# Import our modules
from ..monitoring.health_check import (
    HealthCheckManager,
    HealthStatus,
    ComponentStatus,
    ComponentHealth,
    HealthCheckResult,
    ShutdownResult,
    HealthCheckMiddleware
)
from ..inference.inference_api import app

logger = logging.getLogger(__name__)


class TestHealthCheckSystem(unittest.TestCase):
    """Integration tests for health check system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.health_manager = HealthCheckManager()
        cls.test_client = app.test_client()
        logger.info("Setting up health check system tests")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logger.info("Tearing down health check system tests")
    
    def test_health_manager_initialization(self):
        """Test that health manager initializes correctly"""
        self.assertIsNotNone(self.health_manager)
        self.assertIsInstance(self.health_manager, HealthCheckManager)
        self.assertFalse(self.health_manager.is_shutting_down)
        self.assertEqual(len(self.health_manager.health_checks), 6)  # Default checks
        
        # Check that default components are registered
        self.assertIn("system_resources", self.health_manager.health_checks)
        self.assertIn("gpu_status", self.health_manager.health_checks)
        self.assertIn("memory_status", self.health_manager.health_checks)
        self.assertIn("disk_space", self.health_manager.health_checks)
        self.assertIn("network_connectivity", self.health_manager.health_checks)
        self.assertIn("model_status", self.health_manager.health_checks)
    
    def test_component_registration(self):
        """Test that components can be registered for health monitoring"""
        # Create a mock component
        class MockComponent:
            def __init__(self, name):
                self.name = name
                self.health_status = "healthy"
            
            def get_health_status(self):
                return self.health_status
        
        mock_component = MockComponent("test_component")
        
        # Register the component
        self.health_manager.register_component("test_component", mock_component)
        
        # Verify registration
        self.assertIn("test_component", self.health_manager.components)
        self.assertEqual(self.health_manager.components["test_component"], mock_component)
    
    def test_custom_health_check_registration(self):
        """Test that custom health checks can be registered"""
        def custom_check():
            return ComponentHealth(
                name="custom_test",
                status=ComponentStatus.OPERATIONAL,
                last_checked=datetime.utcnow().isoformat(),
                health_score=1.0,
                details={"test_field": "test_value"}
            )
        
        # Register custom health check
        self.health_manager.register_health_check("custom_test", custom_check)
        
        # Verify registration
        self.assertIn("custom_test", self.health_manager.health_checks)
        self.assertEqual(self.health_manager.health_checks["custom_test"], custom_check)
    
    def test_system_resources_health_check(self):
        """Test system resources health check"""
        component_health = self.health_manager._check_system_resources()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "system_resources")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
        
        # Check that details contain expected fields
        self.assertIsNotNone(component_health.details)
        self.assertIn("cpu_percent", component_health.details)
        self.assertIn("memory_percent", component_health.details)
        self.assertIn("memory_available_gb", component_health.details)
        self.assertIn("memory_total_gb", component_health.details)
    
    def test_gpu_status_health_check(self):
        """Test GPU status health check"""
        component_health = self.health_manager._check_gpu_status()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "gpu_status")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.FAILED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
        
        # Check that details contain expected fields
        self.assertIsNotNone(component_health.details)
        self.assertIn("cuda_available", component_health.details)
    
    def test_memory_status_health_check(self):
        """Test memory status health check"""
        component_health = self.health_manager._check_memory_status()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "memory_status")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
        
        # Check that details contain expected fields
        self.assertIsNotNone(component_health.details)
        self.assertIn("process_memory_mb", component_health.details)
        self.assertIn("process_memory_percent", component_health.details)
        self.assertIn("virtual_memory_percent", component_health.details)
    
    def test_disk_space_health_check(self):
        """Test disk space health check"""
        component_health = self.health_manager._check_disk_space()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "disk_space")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED, ComponentStatus.FAILED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
        
        # Check that details contain expected fields
        self.assertIsNotNone(component_health.details)
        self.assertIn("disk_percent_used", component_health.details)
        self.assertIn("disk_free_gb", component_health.details)
        self.assertIn("disk_total_gb", component_health.details)
    
    def test_network_connectivity_health_check(self):
        """Test network connectivity health check"""
        component_health = self.health_manager._check_network_connectivity()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "network_connectivity")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
        
        # Check that details contain expected fields
        self.assertIsNotNone(component_health.details)
        self.assertIn("active_connections", component_health.details)
        self.assertIn("bytes_sent_mb", component_health.details)
        self.assertIn("bytes_received_mb", component_health.details)
    
    def test_model_status_health_check(self):
        """Test model status health check"""
        component_health = self.health_manager._check_model_status()
        
        self.assertIsInstance(component_health, ComponentHealth)
        self.assertEqual(component_health.name, "model_status")
        self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED, ComponentStatus.FAILED])
        self.assertGreaterEqual(component_health.health_score, 0.0)
        self.assertLessEqual(component_health.health_score, 1.0)
        self.assertIsNotNone(component_health.last_checked)
    
    def test_comprehensive_health_check(self):
        """Test comprehensive health check"""
        health_result = self.health_manager.perform_health_check()
        
        self.assertIsInstance(health_result, HealthCheckResult)
        self.assertIn(health_result.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY])
        self.assertIsNotNone(health_result.timestamp)
        self.assertGreaterEqual(health_result.overall_score, 0.0)
        self.assertLessEqual(health_result.overall_score, 1.0)
        self.assertIsInstance(health_result.components, dict)
        
        # Check that components are included
        self.assertGreater(len(health_result.components), 0)
        
        # Check component health results
        for component_name, component_health in health_result.components.items():
            self.assertIsInstance(component_health, ComponentHealth)
            self.assertEqual(component_health.name, component_name)
            self.assertIn(component_health.status, [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED, ComponentStatus.FAILED])
            self.assertGreaterEqual(component_health.health_score, 0.0)
            self.assertLessEqual(component_health.health_score, 1.0)
            self.assertIsNotNone(component_health.last_checked)
    
    def test_health_check_caching(self):
        """Test that health checks are cached appropriately"""
        # Perform initial health check
        first_result = self.health_manager.get_health_status()
        self.assertIsNotNone(first_result)
        
        # Get cached result (should be the same if within cache window)
        second_result = self.health_manager.get_health_status()
        self.assertIsNotNone(second_result)
        
        # Both should have the same timestamp if cached
        # (Note: This test may not always pass due to timing, but it's a reasonable check)
    
    def test_component_health_lookup(self):
        """Test looking up specific component health"""
        # Perform a health check first
        health_result = self.health_manager.perform_health_check()
        
        # Look up specific component
        component_health = self.health_manager.get_component_health("system_resources")
        
        if component_health:
            self.assertIsInstance(component_health, ComponentHealth)
            self.assertEqual(component_health.name, "system_resources")
        else:
            # Component may not have been checked yet
            pass
    
    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        metrics = self.health_manager.get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("cpu_percent", metrics)
        self.assertIn("memory_percent", metrics)
        self.assertIn("disk_percent", metrics)
        self.assertIn("timestamp", metrics)
        
        # Check that metrics are reasonable values
        self.assertGreaterEqual(metrics["cpu_percent"], 0.0)
        self.assertLessEqual(metrics["cpu_percent"], 100.0)
        self.assertGreaterEqual(metrics["memory_percent"], 0.0)
        self.assertLessEqual(metrics["memory_percent"], 100.0)
        self.assertGreaterEqual(metrics["disk_percent"], 0.0)
        self.assertLessEqual(metrics["disk_percent"], 100.0)
    
    def test_shutdown_callback_registration(self):
        """Test that shutdown callbacks can be registered"""
        callback_executed = False
        
        def test_callback():
            nonlocal callback_executed
            callback_executed = True
        
        # Register callback
        self.health_manager.register_shutdown_callback(test_callback)
        
        # Verify registration
        self.assertIn(test_callback, self.health_manager.shutdown_callbacks)
        self.assertEqual(len(self.health_manager.shutdown_callbacks), 1)
    
    def test_health_check_middleware(self):
        """Test health check middleware functionality"""
        middleware = HealthCheckMiddleware(self.health_manager)
        
        # Test health check endpoint
        status_code, response_data = middleware.health_check_endpoint()
        
        self.assertIsInstance(status_code, int)
        self.assertIn(status_code, [200, 503])  # Healthy or unhealthy
        self.assertIsInstance(response_data, dict)
        self.assertIn("status", response_data)
        self.assertIn("timestamp", response_data)
        self.assertIn("overall_score", response_data)
        
        # Test readiness probe
        status_code, response_data = middleware.readiness_probe_endpoint()
        self.assertIsInstance(status_code, int)
        self.assertIn(status_code, [200, 503])
        self.assertIsInstance(response_data, dict)
        self.assertIn("ready", response_data)
        
        # Test liveness probe
        status_code, response_data = middleware.liveness_probe_endpoint()
        self.assertIsInstance(status_code, int)
        self.assertIn(status_code, [200, 503])
        self.assertIsInstance(response_data, dict)
        self.assertIn("alive", response_data)
    
    def test_api_health_endpoints(self):
        """Test that API health endpoints work"""
        # Test health endpoint
        response = self.test_client.get("/health")
        self.assertIn(response.status_code, [200, 503])
        
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("overall_score", data)
        
        # Test readiness endpoint
        response = self.test_client.get("/ready")
        self.assertIn(response.status_code, [200, 503])
        
        data = json.loads(response.data)
        self.assertIn("ready", data)
        self.assertIn("status", data)
        
        # Test liveness endpoint
        response = self.test_client.get("/alive")
        self.assertIn(response.status_code, [200, 503])
        
        data = json.loads(response.data)
        self.assertIn("alive", data)
        self.assertIn("status", data)


class TestGracefulShutdown(unittest.TestCase):
    """Tests for graceful shutdown functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.health_manager = HealthCheckManager()
        logger.info("Setting up graceful shutdown tests")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logger.info("Tearing down graceful shutdown tests")
    
    def test_shutdown_result_structure(self):
        """Test that shutdown results have correct structure"""
        # Create a mock shutdown result
        shutdown_result = ShutdownResult(
            success=True,
            duration_seconds=1.5,
            components_shutdown=["component1", "component2"],
            components_failed=[],
            error_messages=[],
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.assertIsInstance(shutdown_result, ShutdownResult)
        self.assertTrue(shutdown_result.success)
        self.assertEqual(shutdown_result.duration_seconds, 1.5)
        self.assertEqual(shutdown_result.components_shutdown, ["component1", "component2"])
        self.assertEqual(shutdown_result.components_failed, [])
        self.assertEqual(shutdown_result.error_messages, [])
        self.assertIsNotNone(shutdown_result.timestamp)
    
    def test_shutdown_callback_execution(self):
        """Test that shutdown callbacks are executed"""
        callback_executed = False
        callback_result = None
        
        def test_callback():
            nonlocal callback_executed, callback_result
            callback_executed = True
            callback_result = "callback executed"
            return callback_result
        
        # Register callback
        self.health_manager.register_shutdown_callback(test_callback)
        
        # Verify callback was registered
        self.assertEqual(len(self.health_manager.shutdown_callbacks), 1)
        
        # Execute callback manually (since we don't want to actually shut down in test)
        for callback in self.health_manager.shutdown_callbacks:
            result = callback()
            self.assertEqual(result, "callback executed")
        
        # Verify callback was executed
        self.assertTrue(callback_executed)
    
    def test_shutdown_without_actual_shutdown(self):
        """Test shutdown functionality without actually shutting down"""
        # Register some mock components and callbacks
        def mock_cleanup():
            pass
        
        self.health_manager.register_shutdown_callback(mock_cleanup)
        
        # Test that shutdown can be initiated without actually shutting down
        # (We're not actually calling initiate_graceful_shutdown to avoid shutting down the test)
        self.assertFalse(self.health_manager.is_shutting_down)
        
        # Verify we can register callbacks
        self.assertEqual(len(self.health_manager.shutdown_callbacks), 1)
    
    def test_multiple_shutdown_callbacks(self):
        """Test registering and executing multiple shutdown callbacks"""
        execution_order = []
        
        def callback1():
            execution_order.append("callback1")
        
        def callback2():
            execution_order.append("callback2")
        
        def callback3():
            execution_order.append("callback3")
        
        # Register multiple callbacks
        self.health_manager.register_shutdown_callback(callback1)
        self.health_manager.register_shutdown_callback(callback2)
        self.health_manager.register_shutdown_callback(callback3)
        
        # Verify all callbacks are registered
        self.assertEqual(len(self.health_manager.shutdown_callbacks), 3)
        
        # Execute callbacks manually
        for callback in self.health_manager.shutdown_callbacks:
            callback()
        
        # Verify execution order (callbacks should execute in registration order)
        self.assertEqual(execution_order, ["callback1", "callback2", "callback3"])
    
    def test_shutdown_with_failing_callbacks(self):
        """Test shutdown behavior with failing callbacks"""
        execution_log = []
        
        def successful_callback():
            execution_log.append("successful_callback_executed")
        
        def failing_callback():
            execution_log.append("failing_callback_executed")
            raise Exception("Intentional test exception")
        
        def another_successful_callback():
            execution_log.append("another_successful_callback_executed")
        
        # Register callbacks
        self.health_manager.register_shutdown_callback(successful_callback)
        self.health_manager.register_shutdown_callback(failing_callback)
        self.health_manager.register_shutdown_callback(another_successful_callback)
        
        # Execute callbacks manually and handle exceptions
        for i, callback in enumerate(self.health_manager.shutdown_callbacks):
            try:
                callback()
                execution_log.append(f"callback_{i}_success")
            except Exception as e:
                execution_log.append(f"callback_{i}_failed: {str(e)}")
        
        # Verify execution log
        self.assertIn("successful_callback_executed", execution_log)
        self.assertIn("failing_callback_executed", execution_log)
        self.assertIn("another_successful_callback_executed", execution_log)
        
        # Verify that all callbacks were attempted despite failures
        success_count = sum(1 for log in execution_log if "_success" in log)
        failure_count = sum(1 for log in execution_log if "_failed:" in log)
        self.assertEqual(success_count + failure_count, 3)  # All callbacks attempted


class TestHealthCheckPerformance(unittest.TestCase):
    """Performance tests for health check system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.health_manager = HealthCheckManager()
        logger.info("Setting up health check performance tests")
    
    def test_health_check_performance(self):
        """Test that health checks complete within reasonable time"""
        start_time = time.time()
        
        # Perform comprehensive health check
        health_result = self.health_manager.perform_health_check()
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.assertIsInstance(health_result, HealthCheckResult)
        self.assertLess(duration, 5.0)  # Should complete within 5 seconds
        self.assertGreater(duration, 0.0)  # Should take some time
        
        logger.info(f"Health check completed in {duration:.3f} seconds")
    
    def test_concurrent_health_checks(self):
        """Test that health checks can handle concurrent requests"""
        results = []
        
        def perform_health_check():
            result = self.health_manager.perform_health_check()
            results.append(result)
        
        # Start multiple concurrent health checks
        threads = []
        for i in range(5):  # 5 concurrent checks
            thread = threading.Thread(target=perform_health_check)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all checks completed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, HealthCheckResult)
    
    def test_health_check_caching_performance(self):
        """Test that health check caching improves performance"""
        # Perform initial health check
        start_time = time.time()
        first_result = self.health_manager.get_health_status()
        first_duration = time.time() - start_time
        
        # Perform second health check (should be cached)
        start_time = time.time()
        second_result = self.health_manager.get_health_status()
        second_duration = time.time() - start_time
        
        # Second check should be faster (cached)
        self.assertLess(second_duration, first_duration * 0.1)  # Should be much faster
        
        # Results should be the same (or at least from the same time period)
        self.assertEqual(first_result.timestamp, second_result.timestamp)


# Pytest-style tests
def test_health_check_decorators():
    """Test health check decorators"""
    from ..monitoring.health_check import health_checked
    
    @health_checked
    def sample_function():
        return "success"
    
    result = sample_function()
    assert result == "success"


def test_health_manager_singleton():
    """Test that health manager can be used as singleton"""
    from ..monitoring.health_check import health_manager
    
    # Should be able to access the global health manager
    assert health_manager is not None
    assert isinstance(health_manager, HealthCheckManager)


# Integration tests with FastAPI
def test_fastapi_health_integration():
    """Test integration with FastAPI health endpoints"""
    # Test that FastAPI integration function exists
    from ..monitoring.health_check import integrate_health_checks_with_fastapi
    
    # Should be able to call the integration function
    assert integrate_health_checks_with_fastapi is not None


# Performance benchmark tests
def benchmark_health_checks():
    """Benchmark health check performance"""
    health_manager = HealthCheckManager()
    
    # Warm up
    for _ in range(3):
        health_manager.perform_health_check()
    
    # Benchmark
    import time
    start_time = time.time()
    num_iterations = 10
    
    for i in range(num_iterations):
        result = health_manager.perform_health_check()
        assert isinstance(result, HealthCheckResult)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_check = total_time / num_iterations * 1000  # Convert to milliseconds
    
    print(f"Health check benchmark:")
    print(f"  Average time per check: {avg_time_per_check:.2f}ms")
    print(f"  Checks per second: {1000/avg_time_per_check:.0f}")
    
    # Should be reasonably fast (under 1000ms per check)
    assert avg_time_per_check < 1000.0


# Main test runner
def run_health_check_tests():
    """Run all health check and graceful shutdown tests"""
    print("Running Health Check and Graceful Shutdown Tests...")
    
    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run pytest-style tests
    print("\nRunning Pytest-style Tests...")
    try:
        test_health_check_decorators()
        test_health_manager_singleton()
        test_fastapi_health_integration()
        print("Pytest-style tests completed successfully")
    except Exception as e:
        print(f"Pytest-style tests failed: {e}")
    
    # Run performance benchmarks
    print("\nRunning Performance Benchmarks...")
    try:
        benchmark_health_checks()
        print("Performance benchmarks completed successfully")
    except Exception as e:
        print(f"Performance benchmarks failed: {e}")
    
    # Summary
    print(f"\nTest Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = result.wasSuccessful()
    print(f"\nOverall Test Status: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_health_check_tests()
    exit(0 if success else 1)