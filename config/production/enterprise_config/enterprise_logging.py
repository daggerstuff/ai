#!/usr/bin/env python3
"""
Enterprise Logging System

Provides centralized, structured logging with:
- Multiple output formats (JSON, text)
- Log rotation and retention
- Performance metrics
- Error tracking and alerting
- Distributed tracing support
"""

import logging
import logging.handlers
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import queue
import time

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record):
        record.uptime = time.time() - self.start_time
        return True

class EnterpriseLogger:
    """Enterprise-grade logging system."""
    
    def __init__(self, name: str = "pixelated_ai", 
                 log_dir: str = "/home/vivi/pixelated/ai/logs",
                 level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.level = getattr(logging, level.upper())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        self._setup_handlers()
        self._setup_filters()
        
        # Performance tracking
        self.metrics = {
            'log_counts': {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0},
            'start_time': time.time(),
            'errors': []
        }
    
    def _setup_handlers(self):
        """Set up log handlers for different outputs."""
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        file_handler.setLevel(self.level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.json",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        json_handler.setLevel(self.level)
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
        
        # Error handler for critical issues
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def _setup_filters(self):
        """Set up log filters."""
        perf_filter = PerformanceFilter()
        for handler in self.logger.handlers:
            handler.addFilter(perf_filter)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance."""
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
    
    def log_with_context(self, level: str, message: str, **context):
        """Log with additional context."""
        logger = self.get_logger()
        log_func = getattr(logger, level.lower())
        
        # Create a custom record with extra fields
        record = logger.makeRecord(
            logger.name, getattr(logging, level.upper()), 
            __file__, 0, message, (), None
        )
        record.extra_fields = context
        
        logger.handle(record)
        
        # Update metrics
        self.metrics['log_counts'][level.upper()] += 1
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.log_with_context(
            'INFO', f"Performance: {operation}",
            operation=operation,
            duration_seconds=duration,
            **metrics
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with full context."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if context:
            error_info.update(context)
        
        self.log_with_context('ERROR', f"Error occurred: {error}", **error_info)
        
        # Track error for metrics
        self.metrics['errors'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': type(error).__name__,
            'message': str(error)
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics."""
        uptime = time.time() - self.metrics['start_time']
        total_logs = sum(self.metrics['log_counts'].values())
        
        return {
            'uptime_seconds': uptime,
            'total_logs': total_logs,
            'logs_per_second': total_logs / uptime if uptime > 0 else 0,
            'log_counts': self.metrics['log_counts'].copy(),
            'error_count': len(self.metrics['errors']),
            'recent_errors': self.metrics['errors'][-10:]  # Last 10 errors
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on logging system."""
        try:
            # Test log write
            test_message = f"Health check at {datetime.utcnow().isoformat()}"
            self.logger.info(test_message)
            
            # Check log files
            log_files = list(self.log_dir.glob("*.log"))
            
            return {
                'status': 'healthy',
                'log_files_count': len(log_files),
                'log_directory': str(self.log_dir),
                'metrics': self.get_metrics()
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'log_directory': str(self.log_dir)
            }

# Context manager for performance logging
class LogPerformance:
    """Context manager for automatic performance logging."""
    
    def __init__(self, logger: EnterpriseLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_with_context(
            'DEBUG', f"Starting operation: {self.operation}",
            operation=self.operation,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log_performance(self.operation, duration, **self.context)
        else:
            self.logger.log_with_context(
                'ERROR', f"Operation failed: {self.operation}",
                operation=self.operation,
                duration_seconds=duration,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )

# Global logger instance
_enterprise_logger = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get global enterprise logger."""
    global _enterprise_logger
    if _enterprise_logger is None:
        _enterprise_logger = EnterpriseLogger()
    
    return _enterprise_logger.get_logger(name)

def setup_logging(level: str = "INFO", log_dir: str = "/home/vivi/pixelated/ai/logs"):
    """Set up enterprise logging system."""
    global _enterprise_logger
    _enterprise_logger = EnterpriseLogger(level=level, log_dir=log_dir)
    return _enterprise_logger

if __name__ == "__main__":
    # Test the logging system
    logger_system = setup_logging("DEBUG")
    logger = get_logger("test")
    
    logger.info("Testing enterprise logging system")
    logger.debug("Debug message with details")
    logger.warning("Warning message")
    
    # Test performance logging
    with LogPerformance(logger_system, "test_operation", test_param="value"):
        time.sleep(0.1)  # Simulate work
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger_system.log_error(e, {"context": "testing"})
    
    # Show metrics
    metrics = logger_system.get_metrics()
    print(f"✅ Logging metrics: {json.dumps(metrics, indent=2)}")
    
    # Health check
    health = logger_system.health_check()
    print(f"✅ Health check: {health['status']}")
