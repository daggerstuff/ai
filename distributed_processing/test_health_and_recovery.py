#!/usr/bin/env python3
"""
Unit tests for Health Check and Disaster Recovery Systems
"""

import unittest
import asyncio
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from distributed_processing.health_check import HealthCheckManager, HealthStatus, ComponentType, HealthCheckResult
from distributed_processing.disaster_recovery import DisasterRecoveryManager, DisasterType, RecoveryStatus


class TestHealthCheckSystem(unittest.TestCase):
    """Test cases for Health Check System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.health_manager = HealthCheckManager()
    
    def test_register_health_check(self):
        """Test registering health checks"""
        async def mock_check():
            return {"status": "ok"}
        
        self.health_manager.register_health_check(
            "test_service", 
            ComponentType.API, 
            mock_check
        )
        
        self.assertIn("test_service", self.health_manager.health_checks)
        self.assertIn("test_service", self.health_manager.component_configs)
    
    def test_run_health_check_success(self):
        """Test running a successful health check"""
        async def mock_check():
            return HealthCheckResult(
                component_name="test_service",
                component_type=ComponentType.API,
                status=HealthStatus.HEALTHY,
                message="Service is healthy",
                timestamp=datetime.utcnow(),
                response_time_ms=15.5
            )
        
        self.health_manager.register_health_check(
            "test_service", 
            ComponentType.API, 
            mock_check
        )
        
        result = asyncio.run(self.health_manager.run_health_check("test_service"))
        
        self.assertEqual(result.component_name, "test_service")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.message, "Service is healthy")
        self.assertGreater(result.response_time_ms, 0)
    
    def test_run_health_check_failure(self):
        """Test running a failing health check"""
        async def mock_check():
            raise Exception("Service unavailable")
        
        self.health_manager.register_health_check(
            "failing_service", 
            ComponentType.API, 
            mock_check
        )
        
        result = asyncio.run(self.health_manager.run_health_check("failing_service"))
        
        self.assertEqual(result.component_name, "failing_service")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("Service unavailable", result.message)
    
    def test_run_all_health_checks(self):
        """Test running all health checks"""
        # Register some mock health checks
        async def healthy_check():
            return HealthCheckResult(
                component_name="healthy_service",
                component_type=ComponentType.API,
                status=HealthStatus.HEALTHY,
                message="Healthy",
                timestamp=datetime.utcnow()
            )
        
        async def degraded_check():
            return HealthCheckResult(
                component_name="degraded_service",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.DEGRADED,
                message="Degraded performance",
                timestamp=datetime.utcnow()
            )
        
        self.health_manager.register_health_check("healthy_service", ComponentType.API, healthy_check)
        self.health_manager.register_health_check("degraded_service", ComponentType.DATABASE, degraded_check)
        
        report = asyncio.run(self.health_manager.run_all_health_checks())
        
        self.assertEqual(len(report.component_checks), 2)
        self.assertEqual(report.overall_status, HealthStatus.DEGRADED)
        self.assertIsNotNone(report.system_metrics)
        self.assertGreater(len(report.recommendations), 0)
    
    def test_calculate_overall_health(self):
        """Test overall health calculation"""
        # All healthy
        results_healthy = [
            HealthCheckResult("service1", ComponentType.API, HealthStatus.HEALTHY, "OK", datetime.utcnow()),
            HealthCheckResult("service2", ComponentType.DATABASE, HealthStatus.HEALTHY, "OK", datetime.utcnow())
        ]
        status = self.health_manager._calculate_overall_health(results_healthy)
        self.assertEqual(status, HealthStatus.HEALTHY)
        
        # One degraded
        results_degraded = [
            HealthCheckResult("service1", ComponentType.API, HealthStatus.HEALTHY, "OK", datetime.utcnow()),
            HealthCheckResult("service2", ComponentType.DATABASE, HealthStatus.DEGRADED, "Slow", datetime.utcnow())
        ]
        status = self.health_manager._calculate_overall_health(results_degraded)
        self.assertEqual(status, HealthStatus.DEGRADED)
        
        # Critical component unhealthy
        results_critical_unhealthy = [
            HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.UNHEALTHY, "Down", datetime.utcnow()),
            HealthCheckResult("api", ComponentType.API, HealthStatus.HEALTHY, "OK", datetime.utcnow())
        ]
        status = self.health_manager._calculate_overall_health(results_critical_unhealthy)
        self.assertEqual(status, HealthStatus.UNHEALTHY)


class TestDisasterRecoverySystem(unittest.TestCase):
    """Test cases for Disaster Recovery System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dr_manager = DisasterRecoveryManager()
    
    def test_initialize_default_plans(self):
        """Test initialization of default recovery plans"""
        self.assertGreater(len(self.dr_manager.recovery_plans), 0)
        self.assertIn("db_failure_recovery", self.dr_manager.recovery_plans)
        self.assertIn("data_corruption_recovery", self.dr_manager.recovery_plans)
        self.assertIn("security_breach_recovery", self.dr_manager.recovery_plans)
    
    def test_get_recovery_plan(self):
        """Test getting recovery plan by disaster type"""
        plan = self.dr_manager.get_recovery_plan(DisasterType.HARDWARE_FAILURE)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "db_failure_recovery")
        
        # Test for non-existent plan
        plan = self.dr_manager.get_recovery_plan(DisasterType.NATURAL_DISASTER)
        self.assertIsNone(plan)
    
    def test_start_recovery_session(self):
        """Test starting a recovery session"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.dr_manager.active_sessions)
        
        session = self.dr_manager.active_sessions[session_id]
        self.assertEqual(session.disaster_type, DisasterType.HARDWARE_FAILURE)
        self.assertEqual(session.status, RecoveryStatus.NOT_STARTED)
    
    def test_execute_recovery_step(self):
        """Test executing a recovery step"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)
        
        # Execute first step (which has no dependencies)
        success = asyncio.run(self.dr_manager.execute_recovery_step(session_id, "db_001"))
        self.assertTrue(success)
        
        # Check session status
        session = self.dr_manager.active_sessions[session_id]
        self.assertEqual(session.status, RecoveryStatus.IN_PROGRESS)
        self.assertIn("db_001", session.completed_steps)
    
    def test_execute_recovery_plan(self):
        """Test executing complete recovery plan"""
        session_id = self.dr_manager.start_recovery_session(DisasterType.HARDWARE_FAILURE)
        
        # Execute the plan
        final_status = asyncio.run(self.dr_manager.execute_recovery_plan(session_id))
        
        # Session should be moved to history
        self.assertNotIn(session_id, self.dr_manager.active_sessions)
        
        # Check that a session with this ID exists in history
        session_found = False
        for session in self.dr_manager.recovery_history:
            if session.session_id == session_id:
                session_found = True
                break
        
        self.assertTrue(session_found)


if __name__ == '__main__':
    # Run tests
    unittest.main()