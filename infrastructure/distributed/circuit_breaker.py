#!/usr/bin/env python3
"""
Circuit Breaker System for Pixelated Empathy AI
Implements circuit breaker patterns for external dependencies and fault isolation
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered

class FailureType(Enum):
    """Types of failures that can trigger circuit breaker"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CUSTOM = "custom"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout_seconds: int = 60   # Time before trying half-open
    success_threshold: int = 3           # Successes to close from half-open
    timeout_seconds: float = 30.0        # Request timeout
    monitoring_window_seconds: int = 300 # Window for failure rate calculation
    failure_rate_threshold: float = 0.5  # Failure rate to trigger opening
    slow_call_threshold_seconds: float = 10.0  # Slow call threshold
    slow_call_rate_threshold: float = 0.3      # Slow call rate threshold

@dataclass
class CallResult:
    """Result of a circuit breaker call"""
    success: bool
    duration_seconds: float
    error: Optional[Exception] = None
    failure_type: Optional[FailureType] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class CircuitBreaker:
    """Circuit breaker implementation for external dependencies"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_time = datetime.utcnow()
        
        # Call history for monitoring
        self.call_history: List[CallResult] = []
        self.lock = threading.Lock()
        
        # Fallback function
        self.fallback_function: Optional[Callable] = None
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_circuit_opens = 0
    
    def set_fallback(self, fallback_function: Callable):
        """Set fallback function to call when circuit is open"""
        self.fallback_function = fallback_function
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit should be opened
            if self.state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    self._open_circuit()
            
            # Check if circuit should transition to half-open
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._half_open_circuit()
        
        # Handle different circuit states
        if self.state == CircuitState.OPEN:
            return await self._handle_open_circuit(func, *args, **kwargs)
        
        elif self.state == CircuitState.HALF_OPEN:
            return await self._handle_half_open_circuit(func, *args, **kwargs)
        
        else:  # CLOSED
            return await self._handle_closed_circuit(func, *args, **kwargs)
    
    async def _handle_closed_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle call when circuit is closed (normal operation)"""
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Record successful call
            call_result = CallResult(
                success=True,
                duration_seconds=duration
            )
            
            self._record_call_result(call_result)
            
            with self.lock:
                self.total_successes += 1
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            call_result = CallResult(
                success=False,
                duration_seconds=duration,
                error=e,
                failure_type=FailureType.TIMEOUT
            )
            
            self._record_call_result(call_result)
            
            with self.lock:
                self.total_failures += 1
                self.total_timeouts += 1
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
            
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            failure_type = self._classify_exception(e)
            
            call_result = CallResult(
                success=False,
                duration_seconds=duration,
                error=e,
                failure_type=failure_type
            )
            
            self._record_call_result(call_result)
            
            with self.lock:
                self.total_failures += 1
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
            
            raise
    
    async def _handle_open_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle call when circuit is open (failing fast)"""
        
        logger.warning(f"Circuit breaker {self.name} is OPEN - failing fast")
        
        # Try fallback if available
        if self.fallback_function:
            try:
                return await self.fallback_function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback function failed for {self.name}: {e}")
        
        # Raise circuit breaker exception
        raise CircuitBreakerOpenException(
            f"Circuit breaker {self.name} is OPEN. Service is currently unavailable."
        )
    
    async def _handle_half_open_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle call when circuit is half-open (testing recovery)"""
        
        logger.info(f"Circuit breaker {self.name} is HALF-OPEN - testing service recovery")
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Record successful call
            call_result = CallResult(
                success=True,
                duration_seconds=duration
            )
            
            self._record_call_result(call_result)
            
            with self.lock:
                self.total_successes += 1
                self.success_count += 1
                
                # Check if we should close the circuit
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            failure_type = self._classify_exception(e)
            
            call_result = CallResult(
                success=False,
                duration_seconds=duration,
                error=e,
                failure_type=failure_type
            )
            
            self._record_call_result(call_result)
            
            with self.lock:
                self.total_failures += 1
                # Go back to open on any failure in half-open state
                self._open_circuit()
            
            raise
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate in monitoring window
        recent_calls = self._get_recent_calls()
        if len(recent_calls) >= 10:  # Minimum calls for rate calculation
            failure_rate = sum(1 for call in recent_calls if not call.success) / len(recent_calls)
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        # Check slow call rate
        if len(recent_calls) >= 10:
            slow_calls = sum(1 for call in recent_calls 
                           if call.duration_seconds >= self.config.slow_call_threshold_seconds)
            slow_call_rate = slow_calls / len(recent_calls)
            if slow_call_rate >= self.config.slow_call_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit should attempt reset to half-open"""
        
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout_seconds
    
    def _open_circuit(self):
        """Open the circuit"""
        
        self.state = CircuitState.OPEN
        self.state_changed_time = datetime.utcnow()
        self.total_circuit_opens += 1
        
        logger.warning(f"Circuit breaker {self.name} opened due to failures")
    
    def _half_open_circuit(self):
        """Set circuit to half-open state"""
        
        self.state = CircuitState.HALF_OPEN
        self.state_changed_time = datetime.utcnow()
        self.success_count = 0
        
        logger.info(f"Circuit breaker {self.name} set to HALF-OPEN for testing")
    
    def _close_circuit(self):
        """Close the circuit (normal operation)"""
        
        self.state = CircuitState.CLOSED
        self.state_changed_time = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
        
        logger.info(f"Circuit breaker {self.name} closed - service recovered")
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """Classify exception type for failure tracking"""
        
        if isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, ConnectionError):
            return FailureType.CONNECTION_ERROR
        elif hasattr(exception, 'status_code'):
            if exception.status_code == 429:
                return FailureType.RATE_LIMIT
            elif exception.status_code >= 500:
                return FailureType.SERVICE_UNAVAILABLE
            else:
                return FailureType.HTTP_ERROR
        else:
            return FailureType.CUSTOM
    
    def _record_call_result(self, call_result: CallResult):
        """Record call result for monitoring"""
        
        with self.lock:
            self.call_history.append(call_result)
            
            # Keep only recent history
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.monitoring_window_seconds * 2)
            self.call_history = [
                call for call in self.call_history 
                if call.timestamp > cutoff_time
            ]
    
    def _get_recent_calls(self) -> List[CallResult]:
        """Get calls within monitoring window"""
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.monitoring_window_seconds)
        return [call for call in self.call_history if call.timestamp > cutoff_time]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        
        recent_calls = self._get_recent_calls()
        
        # Calculate rates
        failure_rate = 0.0
        success_rate = 0.0
        avg_response_time = 0.0
        slow_call_rate = 0.0
        
        if recent_calls:
            failures = sum(1 for call in recent_calls if not call.success)
            successes = len(recent_calls) - failures
            
            failure_rate = failures / len(recent_calls)
            success_rate = successes / len(recent_calls)
            
            response_times = [call.duration_seconds for call in recent_calls]
            avg_response_time = statistics.mean(response_times)
            
            slow_calls = sum(1 for call in recent_calls 
                           if call.duration_seconds >= self.config.slow_call_threshold_seconds)
            slow_call_rate = slow_calls / len(recent_calls)
        
        return {
            "name": self.name,
            "state": self.state.value,
            "state_changed_time": self.state_changed_time.isoformat(),
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_timeouts": self.total_timeouts,
            "total_circuit_opens": self.total_circuit_opens,
            "recent_metrics": {
                "calls_in_window": len(recent_calls),
                "failure_rate": round(failure_rate, 3),
                "success_rate": round(success_rate, 3),
                "avg_response_time_seconds": round(avg_response_time, 3),
                "slow_call_rate": round(slow_call_rate, 3)
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        
        with self.lock:
            self._close_circuit()
            self.call_history.clear()
        
        logger.info(f"Circuit breaker {self.name} manually reset")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config or self.default_config
            )
        
        return self.circuit_breakers[name]
    
    def call_with_circuit_breaker(self, service_name: str, func: Callable, 
                                *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        
        circuit_breaker = self.get_circuit_breaker(service_name)
        return circuit_breaker.call(func, *args, **kwargs)
    
    def set_fallback(self, service_name: str, fallback_function: Callable):
        """Set fallback function for a service"""
        
        circuit_breaker = self.get_circuit_breaker(service_name)
        circuit_breaker.set_fallback(fallback_function)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        
        return {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()

# Example usage and testing
async def example_circuit_breaker():
    """Example of using circuit breaker system"""
    
    # Initialize circuit breaker manager
    cb_manager = CircuitBreakerManager()
    
    # Simulate external service calls
    call_count = 0
    
    async def unreliable_service():
        """Simulate an unreliable external service"""
        nonlocal call_count
        call_count += 1
        
        # Simulate failures for first 10 calls
        if call_count <= 10:
            if call_count % 3 == 0:
                raise ConnectionError("Service unavailable")
            elif call_count % 4 == 0:
                await asyncio.sleep(2)  # Slow response
                raise Exception("Internal server error")
        
        # Simulate recovery after 10 calls
        await asyncio.sleep(0.1)  # Normal response time
        return {"status": "success", "data": f"response_{call_count}"}
    
    async def fallback_service():
        """Fallback service when main service is down"""
        return {"status": "fallback", "data": "cached_response"}
    
    # Set up circuit breaker with fallback
    cb_manager.set_fallback("external_api", fallback_service)
    
    # Test circuit breaker behavior
    results = []
    
    for i in range(20):
        try:
            result = await cb_manager.call_with_circuit_breaker(
                "external_api", 
                unreliable_service
            )
            results.append({"call": i+1, "success": True, "result": result})
            
        except Exception as e:
            results.append({"call": i+1, "success": False, "error": str(e)})
        
        # Small delay between calls
        await asyncio.sleep(0.1)
    
    # Print results
    print("Circuit Breaker Test Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        if result["success"]:
            print(f"  Call {result['call']}: {status} {result['result']['status']}")
        else:
            print(f"  Call {result['call']}: {status} {result['error']}")
    
    # Get metrics
    metrics = cb_manager.get_all_metrics()
    print(f"\nCircuit Breaker Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(example_circuit_breaker())
