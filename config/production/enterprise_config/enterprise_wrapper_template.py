#!/usr/bin/env python3
"""
Enterprise Wrapper Template

Use this template to wrap existing functions with enterprise features.
"""

import time
import functools
from typing import Any, Callable

# Enterprise imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))

from enterprise_config import get_config
from enterprise_logging import get_logger, LogPerformance
from enterprise_error_handling import handle_error, with_retry, with_circuit_breaker

def enterprise_wrapper(component_name: str = "unknown", 
                      enable_retry: bool = True,
                      enable_circuit_breaker: bool = False,
                      enable_performance_logging: bool = True):
    """
    Decorator to add enterprise features to any function.
    
    Args:
        component_name: Name of the component for logging
        enable_retry: Enable automatic retry on failures
        enable_circuit_breaker: Enable circuit breaker protection
        enable_performance_logging: Enable performance logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = get_config()
            logger = get_logger(component_name)
            
            # Performance logging
            if enable_performance_logging:
                with LogPerformance(logger, f"{func.__name__}", 
                                  component=component_name):
                    try:
                        if enable_retry:
                            @with_retry(component_name)
                            def retry_func():
                                return func(*args, **kwargs)
                            return retry_func()
                        else:
                            return func(*args, **kwargs)
                    except Exception as e:
                        handle_error(e, component_name, {
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        })
                        raise
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    handle_error(e, component_name)
                    raise
        
        return wrapper
    return decorator

# Example usage:
@enterprise_wrapper("example_component", enable_retry=True, enable_performance_logging=True)
def example_function(data):
    """Example function with enterprise features."""
    # Your existing code here
    return processed_data

if __name__ == "__main__":
    print("✅ Enterprise wrapper template ready")
    print("Use @enterprise_wrapper() decorator on your functions")
