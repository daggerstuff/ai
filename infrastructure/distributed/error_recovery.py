#!/usr/bin/env python3
"""
Automatic Error Recovery and Retry System for Pixelated Empathy AI
Intelligent retry mechanisms with exponential backoff and failure classification
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"

class ErrorSeverity(Enum):
    """Error severity levels"""
    TRANSIENT = "transient"      # Temporary, likely to recover
    PERSISTENT = "persistent"    # Consistent failure, may need intervention
    FATAL = "fatal"             # Unrecoverable error

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter_factor: float = 0.1
    timeout_seconds: Optional[float] = None
    retryable_exceptions: List[type] = None
    non_retryable_exceptions: List[type] = None

class ErrorRecoveryManager:
    """Manages automatic error recovery and retry mechanisms"""
    
    def __init__(self):
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.default_config = RetryConfig()
    
    def register_retry_config(self, operation_name: str, config: RetryConfig):
        """Register retry configuration for an operation"""
        self.retry_configs[operation_name] = config
    
    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register recovery handler for specific error types"""
        self.recovery_handlers[error_type] = handler
    
    async def execute_with_retry(self, operation_name: str, func: Callable, 
                               *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        
        config = self.retry_configs.get(operation_name, self.default_config)
        
        last_exception = None
        attempt = 0
        
        while attempt < config.max_attempts:
            attempt += 1
            
            try:
                # Execute with timeout if configured
                if config.timeout_seconds:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=config.timeout_seconds
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Success - record and return
                self._record_success(operation_name, attempt)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e, config):
                    self._record_failure(operation_name, attempt, e, retryable=False)
                    raise
                
                # Record retryable failure
                self._record_failure(operation_name, attempt, e, retryable=True)
                
                # Try recovery handler
                await self._attempt_recovery(e)
                
                # If not last attempt, wait before retry
                if attempt < config.max_attempts:
                    delay = self._calculate_delay(attempt, config)
                    logger.info(f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt}/{config.max_attempts})")
                    await asyncio.sleep(delay)
        
        # All attempts failed
        logger.error(f"All retry attempts failed for {operation_name}")
        raise last_exception
    
    def _is_retryable_error(self, exception: Exception, config: RetryConfig) -> bool:
        """Determine if error is retryable"""
        
        # Check non-retryable exceptions first
        if config.non_retryable_exceptions:
            for exc_type in config.non_retryable_exceptions:
                if isinstance(exception, exc_type):
                    return False
        
        # Check retryable exceptions
        if config.retryable_exceptions:
            for exc_type in config.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False  # Not in retryable list
        
        # Default retryable errors
        retryable_types = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        
        # Check for HTTP errors (if available)
        if hasattr(exception, 'status_code'):
            # Retry on server errors and rate limits
            return exception.status_code >= 500 or exception.status_code == 429
        
        return isinstance(exception, retryable_types)
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry"""
        
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay_seconds
        
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay_seconds * (2 ** (attempt - 1))
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay_seconds * attempt
        
        elif config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = config.base_delay_seconds * (2 ** (attempt - 1))
            jitter = base_delay * config.jitter_factor * random.random()
            delay = base_delay + jitter
        
        else:
            delay = config.base_delay_seconds
        
        # Cap at max delay
        return min(delay, config.max_delay_seconds)
    
    async def _attempt_recovery(self, exception: Exception):
        """Attempt automatic recovery for the error"""
        
        error_type = type(exception).__name__
        
        if error_type in self.recovery_handlers:
            try:
                recovery_handler = self.recovery_handlers[error_type]
                await recovery_handler(exception)
                logger.info(f"Recovery handler executed for {error_type}")
            except Exception as recovery_error:
                logger.error(f"Recovery handler failed for {error_type}: {recovery_error}")
    
    def _record_success(self, operation_name: str, attempt: int):
        """Record successful operation"""
        
        self.error_history.append({
            "operation": operation_name,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "attempt": attempt,
            "total_attempts": attempt
        })
        
        # Keep only recent history
        self._cleanup_history()
    
    def _record_failure(self, operation_name: str, attempt: int, 
                       exception: Exception, retryable: bool):
        """Record failed operation"""
        
        self.error_history.append({
            "operation": operation_name,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "attempt": attempt,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "retryable": retryable
        })
        
        self._cleanup_history()
    
    def _cleanup_history(self):
        """Clean up old error history"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.error_history = [
            entry for entry in self.error_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics"""
        
        total_operations = len(self.error_history)
        successful_operations = sum(1 for entry in self.error_history if entry["success"])
        failed_operations = total_operations - successful_operations
        
        # Group by operation
        operation_stats = {}
        for entry in self.error_history:
            op_name = entry["operation"]
            if op_name not in operation_stats:
                operation_stats[op_name] = {"total": 0, "success": 0, "failed": 0}
            
            operation_stats[op_name]["total"] += 1
            if entry["success"]:
                operation_stats[op_name]["success"] += 1
            else:
                operation_stats[op_name]["failed"] += 1
        
        # Calculate success rates
        for stats in operation_stats.values():
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        # Error type distribution
        error_types = {}
        for entry in self.error_history:
            if not entry["success"] and "error_type" in entry:
                error_type = entry["error_type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "overall_success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "operation_statistics": operation_stats,
            "error_type_distribution": error_types,
            "registered_configs": len(self.retry_configs),
            "registered_handlers": len(self.recovery_handlers)
        }

# Example usage
async def example_error_recovery():
    """Example of using error recovery system"""
    
    recovery_manager = ErrorRecoveryManager()
    
    # Configure retry for different operations
    recovery_manager.register_retry_config("api_call", RetryConfig(
        max_attempts=5,
        base_delay_seconds=1.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[ConnectionError, TimeoutError]
    ))
    
    recovery_manager.register_retry_config("database_query", RetryConfig(
        max_attempts=3,
        base_delay_seconds=0.5,
        strategy=RetryStrategy.JITTERED_BACKOFF
    ))
    
    # Register recovery handlers
    async def connection_recovery_handler(exception: Exception):
        """Recovery handler for connection errors"""
        logger.info("Attempting connection recovery...")
        await asyncio.sleep(0.1)  # Simulate recovery action
    
    recovery_manager.register_recovery_handler("ConnectionError", connection_recovery_handler)
    
    # Test functions
    call_count = 0
    
    async def unreliable_api_call():
        """Simulate unreliable API call"""
        nonlocal call_count
        call_count += 1
        
        if call_count <= 3:
            raise ConnectionError("API temporarily unavailable")
        
        return {"status": "success", "data": "api_response"}
    
    async def unreliable_db_query():
        """Simulate unreliable database query"""
        if random.random() < 0.3:  # 30% failure rate
            raise TimeoutError("Database query timeout")
        
        return {"rows": [{"id": 1, "name": "test"}]}
    
    # Test error recovery
    print("Testing Error Recovery System:")
    
    try:
        result = await recovery_manager.execute_with_retry("api_call", unreliable_api_call)
        print(f"✅ API call succeeded: {result}")
    except Exception as e:
        print(f"❌ API call failed: {e}")
    
    # Test multiple database queries
    for i in range(5):
        try:
            result = await recovery_manager.execute_with_retry("database_query", unreliable_db_query)
            print(f"✅ DB query {i+1} succeeded")
        except Exception as e:
            print(f"❌ DB query {i+1} failed: {e}")
    
    # Get statistics
    stats = recovery_manager.get_error_statistics()
    print(f"\nError Recovery Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(example_error_recovery())
