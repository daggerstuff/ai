#!/usr/bin/env python3
"""
Unit tests for Health Check and Disaster Recovery Systems
"""

import asyncio

# Add parent directory to path for imports
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


from distributed_processing.disaster_recovery import (
    DisasterRecoveryManager,
    DisasterType,
    RecoveryStatus,
)
from distributed_processing.health_check import (
    ComponentType,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
)


class TestHealthCheckSystem(unittest.TestCase):
    """Test cases for Health Check System"""

    def setUp(self):
        """Set up test fixtures"""
        self.health_manager = HealthCheckManager()

    def test_register_health_check(self):
        """Test registering health checks"""

        async def mock_check():
            return {"status": "ok"}

        self.health_manager.register_health_check("test_service", ComponentType.API, mock_check)

        assert "test_service" in self.health_manager.health_checks
        assert "test_service" in self.health_manager.component_configs

    def test_run_health_check_success(self):
        """Test running a successful health check"""

        async def mock_check():
            return HealthCheckResult(
                component_name="test_service",
                component_type=ComponentType.API,
                status=HealthStatus.HEALTHY,
                message="Service is healthy",
                timestamp=datetime.now(UTC),
                response_time_ms=15.5,
            )

        self.health_manager.register_health_check("test_service", ComponentType.API, mock_check)

        result = asyncio.run(self.health_manager.run_health_check("test_service"))

        assert result.component_name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.response_time_ms > 0

    def test_run_health_check_failure(self):
        """Test running a failing health check"""

        async def mock_check():
            raise Exception("Service unavailable")

        self.health_manager.register_health_check("failing_service", ComponentType.API, mock_check)

        result = asyncio.run(self.health_manager.run_health_check("failing_service"))

        assert result.component_name == "failing_service"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Service unavailable" in result.message

    def test_run_all_health_checks(self):
        """Test running all health checks"""

        # Register some mock health checks
        async def healthy_check():
            return HealthCheckResult(
                component_name="healthy_service",
                component_type=ComponentType.API,
                status=HealthStatus.HEALTHY,
                message="Healthy",
                timestamp=datetime.now(UTC),
            )

        async def degraded_check():
            return HealthCheckResult(
                component_name="degraded_service",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.DEGRADED,
                message="Degraded performance",
                timestamp=datetime.now(UTC),
            )

        self.health_manager.register_health_check("healthy_service", ComponentType.API, healthy_check)
        self.health_manager.register_health_check("degraded_service", ComponentType.DATABASE, degraded_check)

        report = asyncio.run(self.health_manager.run_all_health_checks())

        assert len(report.component_checks) == 2
        assert report.overall_status == HealthStatus.DEGRADED
        assert report.system_metrics is not None
        assert len(report.recommendations) > 0

    def test_calculate_overall_health(self):
        """Test overall health calculation"""
        # All healthy
        results_healthy = [
            HealthCheckResult(
                "service1",
                ComponentType.API,
                HealthStatus.HEALTHY,
                "OK",
                datetime.now(UTC),
            ),
            HealthCheckResult(
                "service2",
                ComponentType.DATABASE,
                HealthStatus.HEALTHY,
                "OK",
                datetime.now(UTC),
            ),
        ]
        status = self.health_manager._calculate_overall_health(results_healthy)
        assert status == HealthStatus.HEALTHY

        # One degraded
        results_degraded = [
            HealthCheckResult(
                "service1",
                ComponentType.API,
                HealthStatus.HEALTHY,
                "OK",
                datetime.now(UTC),
            ),
            HealthCheckResult(
                "service2",
                ComponentType.DATABASE,
                HealthStatus.DEGRADED,
                "Slow",
                datetime.now(UTC),
            ),
        ]
        status = self.health_manager._calculate_overall_health(results_degraded)
        assert status == HealthStatus.DEGRADED

        # Critical component unhealthy
        results_critical_unhealthy = [
            HealthCheckResult(
                "database",
                ComponentType.DATABASE,
                HealthStatus.UNHEALTHY,
                "Down",
                datetime.now(UTC),
            ),
            HealthCheckResult("api", ComponentType.API, HealthStatus.HEALTHY, "OK", datetime.now(UTC)),
        ]
        status = self.health_manager._calculate_overall_health(results_critical_unhealthy)
        assert status == HealthStatus.UNHEALTHY


class TestDisasterRecoverySystem(unittest.TestCase):
    """Test cases for Disaster Recovery System"""

    def setUp(self):
        """Set up test fixtures"""
        self.dr_manager = DisasterRecoveryManager()

    def test_initialize_default_plans(self):
        """Test initialization of default recovery plans"""
        assert len(self.dr_manager.recovery_plans) > 0
        assert "db_failure_recovery" in self.dr_manager.recovery_plans
        assert "data_corruption_recovery" in self.dr_manager.recovery_plans
        assert "security_breach_recovery" in self.dr_manager.recovery_plans

    def test_get_recovery_plan(self):
        """Test getting recovery plan by disaster type"""
        plan = self.dr_manager.get_recovery_plan(DisasterType.HARDWARE_FAILURE)
        assert plan is not None
        assert plan.plan_id == "db_failure_recovery"

        # Test for non-existent plan
        plan = self.dr_manager.get_recovery_plan(DisasterType.NATURAL_DISASTER)
        assert plan is None

    def test_start_recovery_session(self):
        """Test starting a recovery session"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)
        assert session_id is not None
        assert session_id in self.dr_manager.active_sessions

        session = self.dr_manager.active_sessions[session_id]
        assert session.disaster_type == DisasterType.HARDWARE_FAILURE
        assert session.status == RecoveryStatus.NOT_STARTED

    def test_execute_recovery_step(self):
        """Test executing a recovery step"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)

        # Execute first step (which has no dependencies)
        success = asyncio.run(self.dr_manager.execute_recovery_step(session_id, "db_001"))
        assert success

        # Check session status
        session = self.dr_manager.active_sessions[session_id]
        assert session.status == RecoveryStatus.IN_PROGRESS
        assert "db_001" in session.completed_steps

    def test_execute_recovery_plan(self):
        """Test executing complete recovery plan"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)

        # Execute the plan
        asyncio.run(self.dr_manager.execute_recovery_plan(session_id))

        # Session should be moved to history
        assert session_id not in self.dr_manager.active_sessions

        # Check that a session with this ID exists in history
        session_found = False
        for session in self.dr_manager.recovery_history:
            if session.session_id == session_id:
                session_found = True
                break

        assert session_found


if __name__ == "__main__":
    # Run tests
    unittest.main()
