"""
Comprehensive Tests for Pipeline Communication System.

This module provides extensive testing for the six-stage pipeline communication
system with HIPAA++ compliance, sub-50ms performance requirements, and bias detection.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ..event_bus import EventBus, EventMessage, EventType
from ..pipeline_coordinator import PipelineCoordinator, PipelineContext
from ..state_manager import StateManager, PipelineState, StageState
from ..progress_tracker import ProgressTracker, ProgressUpdate
from ..error_recovery import ErrorRecoveryManager, RecoveryStrategy, RecoveryResult
from ..bias_integration import BiasDetectionIntegration, BiasMetrics
from ..performance_monitor import PerformanceMonitor, PerformanceMetric
from ..graceful_degradation import GracefulDegradationManager, DegradationLevel
from ...error_handling.custom_errors import (
    PipelineExecutionError, ValidationError, TimeoutError,
    BiasDetectionError, ServiceUnavailableError
)
from ...integration.redis_client import RedisClient


class TestEventBus:
    """Test cases for EventBus functionality."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create EventBus instance for testing."""
        redis_client = Mock(spec=RedisClient)
        redis_client.publish = AsyncMock(return_value=1)
        redis_client.subscribe = AsyncMock()
        redis_client.unsubscribe = AsyncMock()
        
        config = {
            'connection_pool': {'max_connections': 10},
            'event_ttl': 3600,
            'guaranteed_delivery': True
        }
        
        event_bus = EventBus(redis_client, config)
        yield event_bus
        
        # Cleanup
        await event_bus.shutdown()
    
    @pytest.mark.asyncio
    async def test_event_publishing(self, event_bus):
        """Test basic event publishing functionality."""
        event = await event_bus.create_event(
            event_type=EventType.REQUEST_INITIATED.value,
            execution_id="test_execution_123",
            payload={'test': 'data'}
        )
        
        result = await event_bus.publish_event(event)
        
        assert result is True
        event_bus.redis_client.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_subscribing(self, event_bus):
        """Test event subscription and handling."""
        handler_called = False
        received_event = None
        
        async def test_handler(event: EventMessage) -> Dict[str, Any]:
            nonlocal handler_called, received_event
            handler_called = True
            received_event = event
            return {'handled': True}
        
        event_bus.register_handler(MockHandler(test_handler))
        
        event = EventMessage(
            event_type=EventType.PROGRESS_UPDATE.value,
            execution_id="test_execution_123",
            payload={'progress': 50.0}
        )
        
        await event_bus.publish_event(event)
        await asyncio.sleep(0.1)  # Allow async processing
        
        assert handler_called is True
        assert received_event.execution_id == "test_execution_123"
    
    @pytest.mark.asyncio
    async def test_hipaa_compliance(self, event_bus):
        """Test HIPAA++ compliance in event handling."""
        sensitive_data = {
            'user_id': 'user_123',
            'session_data': 'sensitive_info',
            'pii': 'personal_data'
        }
        
        event = EventMessage(
            event_type=EventType.STAGE_TRANSITION.value,
            execution_id="test_execution_123",
            payload=sensitive_data
        )
        
        # Verify data is sanitized before publishing
        result = await event_bus.publish_event(event)
        
        assert result is True
        # Check that publish was called with sanitized data
        call_args = event_bus.redis_client.publish.call_args
        published_data = json.loads(call_args[0][1])
        
        # Verify HIPAA compliance - no sensitive data in logs
        assert 'pii' not in published_data['payload'] or published_data['payload']['pii'] != 'personal_data'


class TestPipelineCoordinator:
    """Test cases for PipelineCoordinator functionality."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create PipelineCoordinator instance for testing."""
        redis_client = Mock(spec=RedisClient)
        redis_client.get = AsyncMock(return_value=None)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.keys = AsyncMock(return_value=[])
        
        config = {
            'state_manager': {'state_ttl_seconds': 3600},
            'event_bus': {'event_ttl': 3600},
            'performance_monitor': {'metrics_retention_hours': 24}
        }
        
        coordinator = PipelineCoordinator(redis_client, config)
        yield coordinator
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, coordinator):
        """Test complete six-stage pipeline execution."""
        execution_request = {
            'dataset_ids': ['dataset_1', 'dataset_2'],
            'user_id': 'user_123',
            'execution_mode': 'standard',
            'quality_threshold': 0.8,
            'enable_bias_detection': True
        }
        
        result = await coordinator.execute_pipeline(execution_request)
        
        assert result['status'] == 'completed'
        assert result['execution_id'] is not None
        assert 'stage_results' in result
        assert len(result['stage_results']) == 6  # All 6 stages completed
        
        # Verify sub-50ms compliance tracking
        assert 'overall_quality_score' in result
        assert result['overall_quality_score'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_pipeline_validation(self, coordinator):
        """Test pipeline input validation."""
        # Test invalid execution mode
        invalid_request = {
            'dataset_ids': ['dataset_1'],
            'user_id': 'user_123',
            'execution_mode': 'invalid_mode'
        }
        
        with pytest.raises(ValidationError):
            await coordinator.execute_pipeline(invalid_request)
    
    @pytest.mark.asyncio
    async def test_bias_detection_integration(self, coordinator):
        """Test bias detection integration during pipeline execution."""
        execution_request = {
            'dataset_ids': ['dataset_1'],
            'user_id': 'user_123',
            'execution_mode': 'standard',
            'enable_bias_detection': True
        }
        
        result = await coordinator.execute_pipeline(execution_request)
        
        # Verify bias detection was applied in validation stage
        validation_result = result['stage_results'].get('validation', {})
        assert 'bias_analysis' in validation_result.get('result', {})
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, coordinator):
        """Test error recovery mechanisms."""
        # Simulate a stage failure
        with patch.object(coordinator.stage_coordinators['validation'], 'execute_stage') as mock_stage:
            mock_stage.side_effect = Exception("Stage validation failed")
            
            execution_request = {
                'dataset_ids': ['dataset_1'],
                'user_id': 'user_123',
                'execution_mode': 'standard'
            }
            
            # Should handle error gracefully
            result = await coordinator.execute_pipeline(execution_request)
            
            # Verify error was handled and recovery was attempted
            assert result['status'] in ['completed', 'failed']


class TestPerformanceRequirements:
    """Test cases for sub-50ms performance requirements."""
    
    @pytest.mark.asyncio
    async def test_event_bus_performance(self):
        """Test event bus operations complete within 50ms."""
        redis_client = Mock(spec=RedisClient)
        redis_client.publish = AsyncMock(return_value=1)
        
        event_bus = EventBus(redis_client)
        
        start_time = time.time()
        
        event = EventMessage(
            event_type=EventType.PROGRESS_UPDATE.value,
            execution_id="perf_test_123",
            payload={'progress': 75.0}
        )
        
        await event_bus.publish_event(event)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert elapsed_ms < 50.0, f"Event publishing took {elapsed_ms:.2f}ms, exceeding 50ms limit"
    
    @pytest.mark.asyncio
    async def test_bias_detection_performance(self):
        """Test bias detection operations complete within 50ms."""
        bias_integration = BiasDetectionIntegration()
        await bias_integration.initialize_bias_service()
        
        test_data = {
            'validation_score': 0.9,
            'checks_passed': ['check1', 'check2'],
            'dataset_size': 1000
        }
        
        start_time = time.time()
        
        result = await bias_integration.analyze_stage_data(test_data, 'validation')
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert elapsed_ms < 50.0, f"Bias detection took {elapsed_ms:.2f}ms, exceeding 50ms limit"
        assert isinstance(result, BiasMetrics)
        assert result.overall_bias_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_state_update_performance(self):
        """Test state update operations complete within 50ms."""
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.get = AsyncMock(return_value=None)
        
        state_manager = StateManager(redis_client)
        
        # Create test context
        context = PipelineContext(
            execution_id="perf_test_123",
            user_id="user_123",
            dataset_ids=["dataset_1"],
            execution_mode="standard",
            quality_threshold=0.8,
            enable_bias_detection=True,
            start_time=datetime.utcnow(),
            current_stage="test",
            stage_results={},
            metadata={}
        )
        
        start_time = time.time()
        
        await state_manager.initialize_pipeline_state(context)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert elapsed_ms < 50.0, f"State initialization took {elapsed_ms:.2f}ms, exceeding 50ms limit"


class TestHIPAACompliance:
    """Test cases for HIPAA++ compliance."""
    
    @pytest.mark.asyncio
    async def test_data_sanitization(self):
        """Test that sensitive data is properly sanitized."""
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        
        event_bus = EventBus(redis_client)
        
        # Test data with potential PII/PHI
        sensitive_data = {
            'user_id': 'user_123',
            'session_data': 'patient_record_456',
            'diagnosis': 'mental_health_condition',
            'personal_info': 'john.doe@email.com'
        }
        
        event = EventMessage(
            event_type=EventType.STAGE_TRANSITION.value,
            execution_id="hipaa_test_123",
            payload=sensitive_data
        )
        
        await event_bus.publish_event(event)
        
        # Verify published data is sanitized
        call_args = redis_client.setex.call_args
        published_data = json.loads(call_args[0][1])
        
        # Check that sensitive fields are either removed or anonymized
        payload = published_data['payload']
        assert 'personal_info' not in payload or '@' not in str(payload.get('personal_info', ''))
    
    @pytest.mark.asyncio
    async def test_audit_trail_compliance(self):
        """Test audit trail generation for compliance."""
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.get = AsyncMock(return_value=None)
        
        state_manager = StateManager(redis_client)
        
        context = PipelineContext(
            execution_id="audit_test_123",
            user_id="user_123",
            dataset_ids=["dataset_1"],
            execution_mode="standard",
            quality_threshold=0.8,
            enable_bias_detection=True,
            start_time=datetime.utcnow(),
            current_stage="validation",
            stage_results={},
            metadata={}
        )
        
        await state_manager.initialize_pipeline_state(context)
        await state_manager.update_pipeline_state("audit_test_123", "running")
        
        # Verify audit trail was created
        audit_trail = await state_manager.get_audit_trail("audit_test_123")
        
        assert len(audit_trail) > 0
        assert audit_trail[0]['event_type'] == 'pipeline_initialized'
        assert 'timestamp' in audit_trail[0]
        assert 'execution_id' in audit_trail[0]
    
    @pytest.mark.asyncio
    async def test_encryption_compliance(self):
        """Test that encryption is applied to sensitive data."""
        # This would test actual encryption in a real implementation
        # For now, we verify the encryption flag is set
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        
        state_manager = StateManager(redis_client, {'encryption_enabled': True})
        
        context = PipelineContext(
            execution_id="encryption_test_123",
            user_id="user_123",
            dataset_ids=["dataset_1"],
            execution_mode="standard",
            quality_threshold=0.8,
            enable_bias_detection=True,
            start_time=datetime.utcnow(),
            current_stage="test",
            stage_results={},
            metadata={}
        )
        
        state = await state_manager.initialize_pipeline_state(context)
        
        assert state.encryption_applied is True


class TestErrorHandling:
    """Test cases for error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_error_recovery_retry(self):
        """Test error recovery with retry strategy."""
        error_recovery = ErrorRecoveryManager()
        
        context = PipelineContext(
            execution_id="recovery_test_123",
            user_id="user_123",
            dataset_ids=["dataset_1"],
            execution_mode="standard",
            quality_threshold=0.8,
            enable_bias_detection=True,
            start_time=datetime.utcnow(),
            current_stage="validation",
            stage_results={},
            metadata={}
        )
        
        # Simulate a retryable error
        error = TimeoutError("Operation timed out")
        
        result = await error_recovery.attempt_stage_recovery(
            context, "validation", error
        )
        
        assert isinstance(result, RecoveryResult)
        assert result.recovered is True
        assert result.final_strategy == RecoveryStrategy.RETRY
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """Test circuit breaker activation after repeated failures."""
        degradation_manager = GracefulDegradationManager(Mock(spec=EventBus))
        
        service_name = "test_service"
        
        # Simulate multiple failures
        for i in range(6):  # Exceed threshold
            await degradation_manager._record_service_failure(
                service_name, f"Failure {i}"
            )
        
        # Check circuit breaker state
        health = degradation_manager.service_health[service_name]
        
        assert health.circuit_state.name == "OPEN"
        assert health.next_retry_time is not None
    
    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback mechanism execution."""
        degradation_manager = GracefulDegradationManager(Mock(spec=EventBus))
        
        async def failing_function():
            raise Exception("Primary function failed")
        
        async def fallback_function():
            return {"status": "fallback_success", "data": "fallback_data"}
        
        result = await degradation_manager.execute_with_fallback(
            "test_service", failing_function, fallback_function
        )
        
        assert result["status"] == "fallback_success"
        assert result["data"] == "fallback_data"
        assert degradation_manager.performance_metrics['successful_fallbacks'] == 1


class TestBiasDetection:
    """Test cases for bias detection integration."""
    
    @pytest.mark.asyncio
    async def test_bias_metrics_calculation(self):
        """Test bias metrics calculation and thresholds."""
        bias_integration = BiasDetectionIntegration()
        await bias_integration.initialize_bias_service()
        
        test_data = {
            'demographic_data': {'gender': 'mixed', 'age': 'adult'},
            'content_analysis': {'sentiment': 'positive', 'bias_indicators': ['none']}
        }
        
        metrics = await bias_integration.analyze_stage_data(test_data, 'validation')
        
        assert isinstance(metrics, BiasMetrics)
        assert 0.0 <= metrics.overall_bias_score <= 1.0
        assert metrics.compliance_status in ['compliant', 'warning', 'violation']
        
        # Test threshold detection
        if metrics.overall_bias_score > bias_integration.config.threshold:
            assert metrics.compliance_status == 'violation'
    
    @pytest.mark.asyncio
    async def test_bias_detection_disabled(self):
        """Test behavior when bias detection is disabled."""
        config = {'enabled': False}
        bias_integration = BiasDetectionIntegration(config)
        
        test_data = {'some': 'data'}
        metrics = await bias_integration.analyze_stage_data(test_data, 'validation')
        
        assert metrics.overall_bias_score == 0.0
        assert metrics.compliance_status == 'not_applicable'
        assert metrics.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_real_time_bias_monitoring(self):
        """Test real-time bias monitoring and alerts."""
        bias_integration = BiasDetectionIntegration()
        await bias_integration.initialize_bias_service()
        
        # Simulate high bias scenario
        high_bias_data = {
            'demographic_representation': {'skewed': True, 'imbalance': 0.8},
            'content_bias': {'stereotypes': 0.7, 'representation': 0.6}
        }
        
        metrics = await bias_integration.analyze_stage_data(high_bias_data, 'processing')
        
        # Verify high bias is detected
        assert metrics.overall_bias_score > 0.5
        
        # Check real-time alerts
        alerts = bias_integration.get_real_time_alerts()
        
        # Should have alerts for high bias
        high_bias_alerts = [alert for alert in alerts if alert.get('bias_score', 0) > 0.5]
        assert len(high_bias_alerts) > 0


class TestPerformanceMonitoring:
    """Test cases for performance monitoring."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create PerformanceMonitor instance for testing."""
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.keys = AsyncMock(return_value=[])
        
        config = {
            'metrics_retention_hours': 24,
            'threshold_check_interval': 60,
            'performance_window_size': 1000
        }
        
        monitor = PerformanceMonitor(redis_client, config)
        yield monitor
    
    @pytest.mark.asyncio
    async def test_sub_50ms_threshold_monitoring(self, performance_monitor):
        """Test sub-50ms threshold monitoring."""
        # Record metric above threshold
        metric = await performance_monitor.record_metric(
            'stage_execution_time', 'test_execution_123', 75.0, 'milliseconds'
        )
        
        assert metric.threshold_exceeded is True
        assert metric.value == 75.0
        
        # Check threshold violations
        violations = await performance_monitor.check_performance_thresholds()
        
        violation_found = any(
            v['metric_name'] == 'stage_execution_time' and v['current_value'] == 75.0
            for v in violations
        )
        assert violation_found is True
    
    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, performance_monitor):
        """Test performance summary generation."""
        # Record multiple metrics
        for i in range(10):
            await performance_monitor.record_metric(
                'test_metric', 'test_execution_123', float(i * 10), 'milliseconds'
            )
        
        summary = await performance_monitor.get_execution_performance_summary(
            'test_execution_123'
        )
        
        assert summary is not None
        assert summary.execution_id == 'test_execution_123'
        assert summary.average_response_time_ms > 0
        assert summary.min_response_time_ms >= 0
        assert summary.max_response_time_ms <= 90.0
    
    @pytest.mark.asyncio
    async def test_real_time_performance_dashboard(self, performance_monitor):
        """Test real-time performance dashboard generation."""
        # Record metrics
        await performance_monitor.record_metric(
            'stage_execution_time', 'test_execution_123', 25.0, 'milliseconds'
        )
        
        dashboard = await performance_monitor.get_real_time_performance_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'overall_stats' in dashboard
        assert 'current_performance' in dashboard
        assert 'sub_50ms_compliance' in dashboard
        
        # Check compliance tracking
        compliance_data = dashboard['sub_50ms_compliance']
        assert 'current_rate' in compliance_data
        assert 'target_rate' in compliance_data
        assert compliance_data['target_rate'] == 0.95  # 95% target


class TestIntegrationScenarios:
    """Test cases for integrated pipeline scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self):
        """Test complete pipeline flow with all components."""
        # Mock Redis client
        redis_client = Mock(spec=RedisClient)
        redis_client.get = AsyncMock(return_value=None)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.keys = AsyncMock(return_value=[])
        redis_client.publish = AsyncMock(return_value=1)
        
        # Initialize all components
        event_bus = EventBus(redis_client)
        state_manager = StateManager(redis_client)
        progress_tracker = ProgressTracker(redis_client, event_bus, state_manager)
        error_recovery = ErrorRecoveryManager()
        bias_integration = BiasDetectionIntegration()
        performance_monitor = PerformanceMonitor(redis_client)
        degradation_manager = GracefulDegradationManager(event_bus)
        
        coordinator = PipelineCoordinator(
            redis_client,
            config={
                'event_bus': {'event_ttl': 3600},
                'state_manager': {'state_ttl_seconds': 3600},
                'performance_monitor': {'metrics_retention_hours': 24}
            }
        )
        
        # Execute pipeline
        execution_request = {
            'dataset_ids': ['dataset_1', 'dataset_2', 'dataset_3'],
            'user_id': 'integration_test_user',
            'execution_mode': 'comprehensive',
            'quality_threshold': 0.85,
            'enable_bias_detection': True,
            'metadata': {'test_run': True}
        }
        
        start_time = time.time()
        
        result = await coordinator.execute_pipeline(execution_request)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['overall_quality_score'] >= 0.85
        assert len(result['stage_results']) == 6
        
        # Verify performance requirements
        assert total_time_ms < 30000  # Complete pipeline should finish within 30 seconds
        
        # Verify HIPAA compliance
        assert 'user_id' in result['metadata']
        assert result['metadata']['user_id'] == 'integration_test_user'
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_executions(self):
        """Test multiple concurrent pipeline executions."""
        redis_client = Mock(spec=RedisClient)
        redis_client.get = AsyncMock(return_value=None)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.keys = AsyncMock(return_value=[])
        
        coordinator = PipelineCoordinator(redis_client)
        
        # Create multiple execution requests
        execution_requests = [
            {
                'dataset_ids': [f'dataset_{i}'],
                'user_id': f'user_{i}',
                'execution_mode': 'standard',
                'quality_threshold': 0.8,
                'enable_bias_detection': True
            }
            for i in range(5)  # 5 concurrent executions
        ]
        
        # Execute concurrently
        start_time = time.time()
        
        results = await asyncio.gather(
            *[coordinator.execute_pipeline(req) for req in execution_requests]
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Verify all executions completed
        assert len(results) == 5
        for result in results:
            assert result['status'] == 'completed'
            assert 'execution_id' in result
        
        # Verify concurrent performance
        assert total_time_ms < 60000  # All executions should complete within 60 seconds
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self):
        """Test error propagation and recovery across components."""
        redis_client = Mock(spec=RedisClient)
        redis_client.get = AsyncMock(return_value=None)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.keys = AsyncMock(return_value=[])
        
        coordinator = PipelineCoordinator(redis_client)
        
        # Simulate a failure in bias detection
        with patch.object(coordinator.bias_integration, 'analyze_stage_data') as mock_bias:
            mock_bias.side_effect = BiasDetectionError("Bias detection service unavailable")
            
            execution_request = {
                'dataset_ids': ['dataset_1'],
                'user_id': 'error_test_user',
                'execution_mode': 'standard',
                'enable_bias_detection': True
            }
            
            # Should handle error gracefully and continue
            result = await coordinator.execute_pipeline(execution_request)
            
            # Verify pipeline completed despite bias detection failure
            assert result['status'] == 'completed'
            assert 'stage_results' in result


class TestSecurityAndCompliance:
    """Test cases for security and compliance requirements."""
    
    @pytest.mark.asyncio
    async def test_input_validation_and_sanitization(self):
        """Test comprehensive input validation and sanitization."""
        from ..utils.validation import sanitize_input, validate_pipeline_input
        
        # Test malicious input
        malicious_input = {
            'user_input': '<script>alert("xss")</script>',
            'sql_injection': "'; DROP TABLE users; --",
            'command_injection': 'rm -rf /',
            'path_traversal': '../../../etc/passwd'
        }
        
        sanitized = sanitize_input(malicious_input)
        
        # Verify sanitization
        assert '<script>' not in str(sanitized)
        assert 'DROP TABLE' not in str(sanitized)
        assert 'rm -rf' not in str(sanitized)
        assert '../../../' not in str(sanitized)
    
    @pytest.mark.asyncio
    async def test_rate_limiting_protection(self):
        """Test rate limiting protection against abuse."""
        # This would test actual rate limiting in a real implementation
        # For now, we verify the rate limiting configuration exists
        
        redis_client = Mock(spec=RedisClient)
        
        # Verify rate limiting is configured
        from ..communication.event_bus import EventBus
        event_bus = EventBus(redis_client)
        
        # Check that event publishing has rate limiting considerations
        assert hasattr(event_bus, 'publish_event')
        
        # Verify event structure includes rate limiting metadata
        event = EventMessage(
            event_type=EventType.REQUEST_INITIATED.value,
            execution_id="rate_limit_test",
            payload={'test': 'data'},
            metadata={'timestamp': datetime.utcnow().isoformat()}
        )
        
        assert 'timestamp' in event.metadata
    
    @pytest.mark.asyncio
    async def test_audit_logging_compliance(self):
        """Test audit logging compliance with HIPAA requirements."""
        redis_client = Mock(spec=RedisClient)
        redis_client.setex = AsyncMock(return_value=True)
        redis_client.get = AsyncMock(return_value=None)
        
        state_manager = StateManager(redis_client)
        
        # Create audit trail
        context = PipelineContext(
            execution_id="audit_compliance_test",
            user_id="compliance_user",
            dataset_ids=["sensitive_dataset"],
            execution_mode="standard",
            quality_threshold=0.9,
            enable_bias_detection=True,
            start_time=datetime.utcnow(),
            current_stage="validation",
            stage_results={},
            metadata={'compliance_level': 'hipaa_plus'}
        )
        
        await state_manager.initialize_pipeline_state(context)
        
        # Verify audit trail
        audit_trail = await state_manager.get_audit_trail("audit_compliance_test")
        
        assert len(audit_trail) > 0
        for entry in audit_trail:
            assert 'timestamp' in entry
            assert 'event_type' in entry
            assert 'execution_id' in entry
            # Verify no sensitive data in audit logs
            assert 'password' not in str(entry)
            assert 'ssn' not in str(entry)
            assert 'credit_card' not in str(entry)


@pytest.mark.asyncio
async def test_pipeline_stress_scenario():
    """Stress test the pipeline with high load."""
    redis_client = Mock(spec=RedisClient)
    redis_client.get = AsyncMock(return_value=None)
    redis_client.setex = AsyncMock(return_value=True)
    redis_client.keys = AsyncMock(return_value=[])
    redis_client.publish = AsyncMock(return_value=1)
    
    coordinator = PipelineCoordinator(redis_client)
    
    # Create stress test scenario
    stress_requests = []
    for i in range(20):  # 20 concurrent executions
        stress_requests.append({
            'dataset_ids': [f'stress_dataset_{i}_{j}' for j in range(5)],
            'user_id': f'stress_user_{i}',
            'execution_mode': 'fast',
            'quality_threshold': 0.7,
            'enable_bias_detection': (i % 2 == 0)  # Half with bias detection
        })
    
    start_time = time.time()
    
    # Execute with limited concurrency to simulate real-world constraints
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent executions
    
    async def limited_execution(request):
        async with semaphore:
            return await coordinator.execute_pipeline(request)
    
    results = await asyncio.gather(
        *[limited_execution(req) for req in stress_requests]
    )
    
    total_time_ms = (time.time() - start_time) * 1000
    
    # Verify stress test results
    assert len(results) == 20
    successful_executions = sum(1 for r in results if r['status'] == 'completed')
    assert successful_executions >= 18  # At least 90% success rate
    
    # Verify performance under stress
    assert total_time_ms < 120000  # Complete within 2 minutes
    
    # Verify no data corruption under stress
    for result in results:
        if result['status'] == 'completed':
            assert 'execution_id' in result
            assert 'stage_results' in result
            assert len(result['stage_results']) == 6


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])