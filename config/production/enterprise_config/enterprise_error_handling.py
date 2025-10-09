#!/usr/bin/env python3
"""
Enterprise Error Handling and Recovery System

Provides comprehensive error handling with:
- Automatic retry mechanisms
- Circuit breaker patterns
- Error classification and routing
- Recovery strategies
- Error reporting and analytics
"""

import time
import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    PROCESSING = "processing"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False
    resolution_strategy: Optional[str] = None

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RetryStrategy:
    """Configurable retry strategy."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if we should retry based on attempt count and error type."""
        if attempt >= self.max_attempts:
            return False
        
        # Don't retry certain types of errors
        non_retryable_errors = (ValueError, TypeError, KeyError)
        if isinstance(error, non_retryable_errors):
            return False
        
        return True

class EnterpriseErrorHandler:
    """Enterprise-grade error handling system."""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_stats = defaultdict(int)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, RetryStrategy] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Default retry strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Set up default retry strategies for different error types."""
        self.retry_strategies.update({
            'network': RetryStrategy(max_attempts=3, base_delay=1.0),
            'database': RetryStrategy(max_attempts=2, base_delay=0.5),
            'filesystem': RetryStrategy(max_attempts=2, base_delay=0.1),
            'processing': RetryStrategy(max_attempts=1, base_delay=0.0),
            'default': RetryStrategy(max_attempts=2, base_delay=1.0)
        })
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network-related errors
        if any(keyword in error_message for keyword in 
               ['connection', 'timeout', 'network', 'socket', 'dns']):
            return ErrorCategory.NETWORK
        
        # Database-related errors
        if any(keyword in error_message for keyword in 
               ['database', 'sql', 'connection pool', 'deadlock']):
            return ErrorCategory.DATABASE
        
        # Filesystem-related errors
        if any(keyword in error_message for keyword in 
               ['file not found', 'permission denied', 'disk', 'directory']):
            return ErrorCategory.FILESYSTEM
        
        # Processing-related errors
        if any(keyword in error_message for keyword in 
               ['processing', 'validation', 'format', 'parsing']):
            return ErrorCategory.PROCESSING
        
        # Configuration-related errors
        if any(keyword in error_message for keyword in 
               ['config', 'setting', 'parameter', 'environment']):
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity."""
        error_type = type(error).__name__
        
        # Critical errors that can crash the system
        critical_errors = ['SystemExit', 'KeyboardInterrupt', 'MemoryError']
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_severity_errors = ['ConnectionError', 'DatabaseError', 'SecurityError']
        if error_type in high_severity_errors:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_severity_errors = ['ValueError', 'TypeError', 'FileNotFoundError']
        if error_type in medium_severity_errors:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def handle_error(self, error: Exception, component: str = "unknown", 
                    context: Dict[str, Any] = None) -> ErrorInfo:
        """Handle an error with full processing."""
        context = context or {}
        
        # Generate unique error ID
        error_id = f"{component}_{int(time.time())}_{id(error)}"
        
        # Classify and assess error
        category = self.classify_error(error, context)
        severity = self.determine_severity(error, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            component=component,
            traceback=traceback.format_exc(),
            context=context
        )
        
        # Store error
        self.error_history.append(error_info)
        self.error_stats[f"{category.value}_{severity.value}"] += 1
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, 
                       f"Error in {component}: {error_info.error_message}",
                       extra={'error_id': error_id, 'category': category.value})
        
        return error_info
    
    def with_retry(self, component: str = "default", strategy_name: str = "default"):
        """Decorator for automatic retry with error handling."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                strategy = self.retry_strategies.get(strategy_name, 
                                                   self.retry_strategies['default'])
                
                last_error = None
                for attempt in range(strategy.max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        error_info = self.handle_error(e, component, 
                                                     {'attempt': attempt + 1})
                        error_info.retry_count = attempt + 1
                        
                        if not strategy.should_retry(attempt, e):
                            break
                        
                        if attempt < strategy.max_attempts - 1:
                            delay = strategy.get_delay(attempt)
                            self.logger.info(f"Retrying {component} in {delay}s (attempt {attempt + 1})")
                            time.sleep(delay)
                
                # All retries exhausted
                self.logger.error(f"All retries exhausted for {component}")
                raise last_error
            
            return wrapper
        return decorator
    
    def with_circuit_breaker(self, component: str, failure_threshold: int = 5, 
                           recovery_timeout: int = 60):
        """Decorator for circuit breaker protection."""
        def decorator(func: Callable):
            if component not in self.circuit_breakers:
                self.circuit_breakers[component] = CircuitBreaker(
                    failure_threshold, recovery_timeout)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                circuit_breaker = self.circuit_breakers[component]
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    self.handle_error(e, component, {'circuit_breaker_state': circuit_breaker.state})
                    raise
            
            return wrapper
        return decorator
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Count by category and severity
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
            component_counts[error.component] += 1
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / max(hours, 1),  # errors per hour
            'by_category': dict(category_counts),
            'by_severity': dict(severity_counts),
            'by_component': dict(component_counts),
            'circuit_breaker_states': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors for debugging."""
        recent = list(self.error_history)[-limit:]
        return [{
            'error_id': e.error_id,
            'timestamp': e.timestamp.isoformat(),
            'component': e.component,
            'error_type': e.error_type,
            'message': e.error_message,
            'severity': e.severity.value,
            'category': e.category.value,
            'retry_count': e.retry_count
        } for e in recent]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on error handling system."""
        recent_stats = self.get_error_statistics(hours=1)
        
        # Determine health status
        critical_errors = recent_stats['by_severity'].get('critical', 0)
        high_errors = recent_stats['by_severity'].get('high', 0)
        error_rate = recent_stats['error_rate']
        
        if critical_errors > 0:
            status = 'critical'
            message = f'{critical_errors} critical errors in last hour'
        elif high_errors > 5:
            status = 'degraded'
            message = f'{high_errors} high-severity errors in last hour'
        elif error_rate > 10:
            status = 'degraded'
            message = f'High error rate: {error_rate:.1f} errors/hour'
        else:
            status = 'healthy'
            message = 'Error handling system operating normally'
        
        return {
            'status': status,
            'message': message,
            'statistics': recent_stats,
            'recent_errors': self.get_recent_errors(5)
        }

# Global error handler instance
_enterprise_error_handler = None

def get_error_handler() -> EnterpriseErrorHandler:
    """Get global enterprise error handler."""
    global _enterprise_error_handler
    if _enterprise_error_handler is None:
        _enterprise_error_handler = EnterpriseErrorHandler()
    return _enterprise_error_handler

def handle_error(error: Exception, component: str = "unknown", 
                context: Dict[str, Any] = None) -> ErrorInfo:
    """Handle an error using the global error handler."""
    return get_error_handler().handle_error(error, component, context)

def with_retry(component: str = "default", strategy: str = "default"):
    """Decorator for automatic retry."""
    return get_error_handler().with_retry(component, strategy)

def with_circuit_breaker(component: str, failure_threshold: int = 5, 
                        recovery_timeout: int = 60):
    """Decorator for circuit breaker protection."""
    return get_error_handler().with_circuit_breaker(component, failure_threshold, recovery_timeout)

if __name__ == "__main__":
    # Test the error handling system
    error_handler = EnterpriseErrorHandler()
    
    # Test error classification
    try:
        raise ConnectionError("Network connection failed")
    except Exception as e:
        error_info = error_handler.handle_error(e, "test_component")
        print(f"✅ Error classified: {error_info.category.value}, severity: {error_info.severity.value}")
    
    # Test retry decorator
    @error_handler.with_retry("test_retry", "network")
    def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Random network error")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"✅ Retry test result: {result}")
    except Exception as e:
        print(f"❌ Retry test failed: {e}")
    
    # Show statistics
    stats = error_handler.get_error_statistics(hours=1)
    print(f"✅ Error statistics: {stats}")
    
    print("✅ Enterprise error handling system ready")
