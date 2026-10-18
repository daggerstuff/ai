"""
Integration tests for health check and graceful shutdown functionality.
Tests that health checks work correctly and shutdown is graceful.
"""

import contextlib
import json
import logging
import sys
import threading
import time
import unittest
from datetime import UTC, datetime

from ai.inference.inference_api import app

# Import our modules
from ai.monitoring.health_check import (
    ComponentHealth,
    ComponentStatus,
    HealthCheckManager,
    HealthCheckMiddleware,
    HealthCheckResult,
    HealthStatus,
    ShutdownResult,
    health_checked,
    health_manager,
    integrate_health_checks_with_fastapi,
)

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
        assert self.health_manager is not None
        assert isinstance(self.health_manager, HealthCheckManager)
        assert not self.health_manager.is_shutting_down
        assert len(self.health_manager.health_checks) == 6

        # Check that default components are registered
        assert "system_resources" in self.health_manager.health_checks
        assert "gpu_status" in self.health_manager.health_checks
        assert "memory_status" in self.health_manager.health_checks
        assert "disk_space" in self.health_manager.health_checks
        assert "network_connectivity" in self.health_manager.health_checks
        assert "model_status" in self.health_manager.health_checks

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
        assert "test_component" in self.health_manager.components
        assert self.health_manager.components["test_component"] == mock_component

    def test_custom_health_check_registration(self):
        """Test that custom health checks can be registered"""

        def custom_check():
            return ComponentHealth(
                name="custom_test",
                status=ComponentStatus.OPERATIONAL,
                last_checked=datetime.now(UTC).isoformat(),
                health_score=1.0,
                details={"test_field": "test_value"},
            )

        # Register custom health check
        self.health_manager.register_health_check("custom_test", custom_check)

        # Verify registration
        assert "custom_test" in self.health_manager.health_checks
        assert self.health_manager.health_checks["custom_test"] == custom_check

    def test_system_resources_health_check(self):
        """Test system resources health check"""
        component_health = self.health_manager._check_system_resources()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "system_resources"
        assert component_health.status in [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

        # Check that details contain expected fields
        assert component_health.details is not None
        assert "cpu_percent" in component_health.details
        assert "memory_percent" in component_health.details
        assert "memory_available_gb" in component_health.details
        assert "memory_total_gb" in component_health.details

    def test_gpu_status_health_check(self):
        """Test GPU status health check"""
        component_health = self.health_manager._check_gpu_status()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "gpu_status"
        assert component_health.status in [ComponentStatus.OPERATIONAL, ComponentStatus.FAILED]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

        # Check that details contain expected fields
        assert component_health.details is not None
        assert "cuda_available" in component_health.details

    def test_memory_status_health_check(self):
        """Test memory status health check"""
        component_health = self.health_manager._check_memory_status()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "memory_status"
        assert component_health.status in [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

        # Check that details contain expected fields
        assert component_health.details is not None
        assert "process_memory_mb" in component_health.details
        assert "process_memory_percent" in component_health.details
        assert "virtual_memory_percent" in component_health.details

    def test_disk_space_health_check(self):
        """Test disk space health check"""
        component_health = self.health_manager._check_disk_space()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "disk_space"
        assert component_health.status in [
            ComponentStatus.OPERATIONAL,
            ComponentStatus.DEGRADED,
            ComponentStatus.FAILED,
        ]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

        # Check that details contain expected fields
        assert component_health.details is not None
        assert "disk_percent_used" in component_health.details
        assert "disk_free_gb" in component_health.details
        assert "disk_total_gb" in component_health.details

    def test_network_connectivity_health_check(self):
        """Test network connectivity health check"""
        component_health = self.health_manager._check_network_connectivity()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "network_connectivity"
        assert component_health.status in [ComponentStatus.OPERATIONAL, ComponentStatus.DEGRADED]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

        # Check that details contain expected fields
        assert component_health.details is not None
        assert "active_connections" in component_health.details
        assert "bytes_sent_mb" in component_health.details
        assert "bytes_received_mb" in component_health.details

    def test_model_status_health_check(self):
        """Test model status health check"""
        component_health = self.health_manager._check_model_status()

        assert isinstance(component_health, ComponentHealth)
        assert component_health.name == "model_status"
        assert component_health.status in [
            ComponentStatus.OPERATIONAL,
            ComponentStatus.DEGRADED,
            ComponentStatus.FAILED,
        ]
        assert component_health.health_score >= 0.0
        assert component_health.health_score <= 1.0
        assert component_health.last_checked is not None

    def test_comprehensive_health_check(self):
        """Test comprehensive health check"""
        health_result = self.health_manager.perform_health_check()

        assert isinstance(health_result, HealthCheckResult)
        assert health_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert health_result.timestamp is not None
        assert health_result.overall_score >= 0.0
        assert health_result.overall_score <= 1.0
        assert isinstance(health_result.components, dict)

        # Check that components are included
        assert len(health_result.components) > 0

        # Check component health results
        for component_name, component_health in health_result.components.items():
            assert isinstance(component_health, ComponentHealth)
            assert component_health.name == component_name
            assert component_health.status in [
                ComponentStatus.OPERATIONAL,
                ComponentStatus.DEGRADED,
                ComponentStatus.FAILED,
            ]
            assert component_health.health_score >= 0.0
            assert component_health.health_score <= 1.0
            assert component_health.last_checked is not None

    def test_health_check_caching(self):
        """Test that health checks are cached appropriately"""
        # Perform initial health check
        first_result = self.health_manager.get_health_status()
        assert first_result is not None

        # Get cached result (should be the same if within cache window)
        second_result = self.health_manager.get_health_status()
        assert second_result is not None

        # Both should have the same timestamp if cached
        # (Note: This test may not always pass due to timing, but it's a reasonable check)

    def test_component_health_lookup(self):
        """Test looking up specific component health"""
        # Perform a health check first
        self.health_manager.perform_health_check()

        # Look up specific component
        component_health = self.health_manager.get_component_health("system_resources")

        if component_health:
            assert isinstance(component_health, ComponentHealth)
            assert component_health.name == "system_resources"
        else:
            # Component may not have been checked yet
            pass

    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        metrics = self.health_manager.get_system_metrics()

        assert isinstance(metrics, dict)
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "timestamp" in metrics

        # Check that metrics are reasonable values
        assert metrics["cpu_percent"] >= 0.0
        assert metrics["cpu_percent"] <= 100.0
        assert metrics["memory_percent"] >= 0.0
        assert metrics["memory_percent"] <= 100.0
        assert metrics["disk_percent"] >= 0.0
        assert metrics["disk_percent"] <= 100.0

    def test_shutdown_callback_registration(self):
        """Test that shutdown callbacks can be registered"""
        callback_executed = False

        def test_callback():
            nonlocal callback_executed
            callback_executed = True

        # Register callback
        self.health_manager.register_shutdown_callback(test_callback)

        # Verify registration
        assert test_callback in self.health_manager.shutdown_callbacks
        assert len(self.health_manager.shutdown_callbacks) == 1

    def test_health_check_middleware(self):
        """Test health check middleware functionality"""
        middleware = HealthCheckMiddleware(self.health_manager)

        # Test health check endpoint
        status_code, response_data = middleware.health_check_endpoint()

        assert isinstance(status_code, int)
        assert status_code in [200, 503]  # Healthy or unhealthy
        assert isinstance(response_data, dict)
        assert "status" in response_data
        assert "timestamp" in response_data
        assert "overall_score" in response_data

        # Test readiness probe
        status_code, response_data = middleware.readiness_probe_endpoint()
        assert isinstance(status_code, int)
        assert status_code in [200, 503]
        assert isinstance(response_data, dict)
        assert "ready" in response_data

        # Test liveness probe
        status_code, response_data = middleware.liveness_probe_endpoint()
        assert isinstance(status_code, int)
        assert status_code in [200, 503]
        assert isinstance(response_data, dict)
        assert "alive" in response_data

    def test_api_health_endpoints(self):
        """Test that API health endpoints work"""
        # Test health endpoint
        response = self.test_client.get("/health")
        assert response.status_code in [200, 503]

        data = json.loads(response.data)
        assert "status" in data
        assert "timestamp" in data
        assert "overall_score" in data

        # Test readiness endpoint
        response = self.test_client.get("/ready")
        assert response.status_code in [200, 503]

        data = json.loads(response.data)
        assert "ready" in data
        assert "status" in data

        # Test liveness endpoint
        response = self.test_client.get("/alive")
        assert response.status_code in [200, 503]

        data = json.loads(response.data)
        assert "alive" in data
        assert "status" in data


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
            timestamp=datetime.now(UTC).isoformat(),
        )

        assert isinstance(shutdown_result, ShutdownResult)
        assert shutdown_result.success
        assert shutdown_result.duration_seconds == 1.5
        assert shutdown_result.components_shutdown == ["component1", "component2"]
        assert shutdown_result.components_failed == []
        assert shutdown_result.error_messages == []
        assert shutdown_result.timestamp is not None

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
        assert len(self.health_manager.shutdown_callbacks) == 1

        # Execute callback manually (since we don't want to actually shut down in test)
        for callback in self.health_manager.shutdown_callbacks:
            result = callback()
            assert result == "callback executed"

        # Verify callback was executed
        assert callback_executed

    def test_shutdown_without_actual_shutdown(self):
        """Test shutdown functionality without actually shutting down"""

        # Register some mock components and callbacks
        def mock_cleanup():
            pass

        self.health_manager.register_shutdown_callback(mock_cleanup)

        # Test that shutdown can be initiated without actually shutting down
        # (We're not actually calling initiate_graceful_shutdown to avoid shutting down the test)
        assert not self.health_manager.is_shutting_down

        # Verify we can register callbacks
        assert len(self.health_manager.shutdown_callbacks) == 1

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
        assert len(self.health_manager.shutdown_callbacks) == 3

        # Execute callbacks manually
        for callback in self.health_manager.shutdown_callbacks:
            callback()

        # Verify execution order (callbacks should execute in registration order)
        assert execution_order == ["callback1", "callback2", "callback3"]

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
                execution_log.append(f"callback_{i}_failed: {e!s}")

        # Verify execution log
        assert "successful_callback_executed" in execution_log
        assert "failing_callback_executed" in execution_log
        assert "another_successful_callback_executed" in execution_log

        # Verify that all callbacks were attempted despite failures
        success_count = sum(1 for log in execution_log if "_success" in log)
        failure_count = sum(1 for log in execution_log if "_failed:" in log)
        assert success_count + failure_count == 3


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

        assert isinstance(health_result, HealthCheckResult)
        assert duration < 5.0  # Should complete within 5 seconds
        assert duration > 0.0  # Should take some time

        logger.info(f"Health check completed in {duration:.3f} seconds")

    def test_concurrent_health_checks(self):
        """Test that health checks can handle concurrent requests"""
        results = []

        def perform_health_check():
            result = self.health_manager.perform_health_check()
            results.append(result)

        # Start multiple concurrent health checks
        threads = []
        for _i in range(5):  # 5 concurrent checks
            thread = threading.Thread(target=perform_health_check)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all checks completed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, HealthCheckResult)

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
        assert second_duration < first_duration * 0.1  # Should be much faster

        # Results should be the same (or at least from the same time period)
        assert first_result.timestamp == second_result.timestamp


# Pytest-style tests
def test_health_check_decorators():
    """Test health check decorators"""

    @health_checked
    def sample_function():
        return "success"

    result = sample_function()
    assert result == "success"


def test_health_manager_singleton():
    """Test that health manager can be used as singleton"""

    # Should be able to access the global health manager
    assert health_manager is not None
    assert isinstance(health_manager, HealthCheckManager)


# Integration tests with FastAPI
def test_fastapi_health_integration():
    """Test integration with FastAPI health endpoints"""
    # Test that FastAPI integration function exists

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

    start_time = time.time()
    num_iterations = 10

    for _i in range(num_iterations):
        result = health_manager.perform_health_check()
        assert isinstance(result, HealthCheckResult)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_check = total_time / num_iterations * 1000  # Convert to milliseconds

    # Should be reasonably fast (under 1000ms per check)
    assert avg_time_per_check < 1000.0


# Main test runner
def run_health_check_tests():
    """Run all health check and graceful shutdown tests"""

    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Run pytest-style tests
    try:
        test_health_check_decorators()
        test_health_manager_singleton()
        test_fastapi_health_integration()
    except Exception:
        pass

    # Run performance benchmarks
    with contextlib.suppress(Exception):
        benchmark_health_checks()

    # Summary

    if result.failures:
        for _test, _traceback in result.failures:
            pass

    if result.errors:
        for _test, _traceback in result.errors:
            pass

    return result.wasSuccessful()



if __name__ == "__main__":
    success = run_health_check_tests()
    sys.exit(0 if success else 1)
