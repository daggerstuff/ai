"""
Logging utilities for TechDeck Flask service.

This module provides structured logging with HIPAA++ compliance,
audit trail capabilities, and performance monitoring.
"""


import logging
import json
import sys
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from ..config import TechDeckServiceConfig


class HIPAACompliantFormatter(logging.Formatter):
    """
    Custom formatter for HIPAA-compliant logging.

    Ensures sensitive data is properly sanitized and audit trails
    are maintained according to healthcare compliance requirements.
    """

    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'key', 'ssn', 'email',
        'phone', 'address', 'dob', 'date_of_birth', 'medical_record',
        'phi', 'pii', 'credit_card', 'bank_account'
    }

    def format(self, record):
        """Format log record with HIPAA compliance."""
        # Sanitize sensitive data from log message
        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            record.msg = self._sanitize_data(record.msg)

        # Sanitize extra data
        if hasattr(record, 'extra_data'):
            record.extra_data = self._sanitize_data(record.extra_data)

        return super().format(record)

    def _sanitize_data(self, data: Any) -> Any:
        """
        Recursively sanitize sensitive data from log data.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return {
                key: (
                    '[REDACTED]'
                    if any(
                        sensitive in key.lower()
                        for sensitive in self.SENSITIVE_FIELDS
                    )
                    else self._sanitize_data(value)
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(config: TechDeckServiceConfig) -> None:
    """
    Setup comprehensive logging configuration.

    Args:
        config: Service configuration object
    """
    # Create logs directory if it doesn't exist
    if config.LOG_FILE_PATH:
        log_dir = os.path.dirname(config.LOG_FILE_PATH)
        os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    _extracted_from_setup_logging_42(console_handler, config, root_logger)
    # File handler if path is specified
    if config.LOG_FILE_PATH:
        file_handler = RotatingFileHandler(
            config.LOG_FILE_PATH,
            maxBytes=config.LOG_MAX_FILE_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        _extracted_from_setup_logging_42(file_handler, config, root_logger)
    # Audit log handler for HIPAA compliance
    if config.ENABLE_AUDIT_LOGGING:
        _extracted_from_setup_logging_56(config)


# TODO Rename this here and in `setup_logging`
def _extracted_from_setup_logging_42(arg0, config, root_logger):
    arg0.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

    file_formatter = (
        JSONFormatter()
        if config.LOG_FORMAT == 'json'
        else HIPAACompliantFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    arg0.setFormatter(file_formatter)
    root_logger.addHandler(arg0)


# TODO Rename this here and in `setup_logging`
def _extracted_from_setup_logging_56(config):
    audit_handler = TimedRotatingFileHandler(
        config.LOG_FILE_PATH.replace('.log', '_audit.log') if config.LOG_FILE_PATH else 'audit.log',
        when='midnight',
        interval=1,
        backupCount=config.AUDIT_LOG_RETENTION_DAYS,
        encoding='utf-8'
    )
    audit_handler.setLevel(logging.INFO)
    audit_formatter = JSONFormatter()
    audit_handler.setFormatter(audit_formatter)

    # Create audit logger
    audit_logger = logging.getLogger('audit')
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance with specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_request_logger() -> logging.Logger:
    """Return a logger specifically for request-scoped logging (compat shim)."""
    return get_logger('request')


def log_operation(
    operation: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = 'success',
    details: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO
) -> None:
    """
    Log operation with audit trail information.

    Args:
        operation: Operation name
        user_id: User ID for audit trail
        request_id: Request ID for tracking
        status: Operation status (success/failure)
        details: Additional operation details
        level: Log level
    """
    logger = get_logger('audit')

    audit_entry = {
        'event_type': 'operation',
        'operation': operation,
        'status': status,
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'request_id': request_id,
        'details': details or {}
    }

    logger.log(level, audit_entry)


def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log security-related events for compliance.

    Args:
        event_type: Type of security event
        user_id: User ID
        request_id: Request ID
        ip_address: Client IP address
        user_agent: User agent string
        details: Additional event details
    """
    logger = get_logger('security')

    security_entry = {
        'event_type': 'security',
        'security_event': event_type,
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'request_id': request_id,
        'ip_address': ip_address,
        'user_agent': user_agent,
        'details': details or {}
    }

    logger.warning(security_entry)


def log_performance_metric(metric_name: str, value: Optional[float] = None, unit: Optional[str] = None,
                           user_id: Optional[str] = None, request_id: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None):
    """Compatibility helper.

    This function serves two purposes:
    - When called directly as log_performance_metric(name, value, unit, ...)
      it logs the metric entry immediately (legacy usage).
    - When used as a decorator factory @log_performance_metric('metric_name')
      it returns a decorator that measures execution time and logs a metric
      entry with the measured duration in milliseconds.
    """
    logger = get_logger('performance')

    # Direct logging usage
    if value is not None and unit is not None:
        metric_entry = {
            'event_type': 'metric',
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'request_id': request_id,
            'tags': tags or {}
        }
        logger.info(metric_entry)
        return None

    # Decorator factory usage
    def decorator(func):
        import functools
        import inspect

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000.0
            metric_entry = {
                'event_type': 'metric',
                'metric_name': metric_name,
                'value': duration_ms,
                'unit': 'ms',
                'timestamp': datetime.utcnow().isoformat(),
                'tags': tags or {}
            }
            logger.info(metric_entry)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000.0
            metric_entry = {
                'event_type': 'metric',
                'metric_name': metric_name,
                'value': duration_ms,
                'unit': 'ms',
                'timestamp': datetime.utcnow().isoformat(),
                'tags': tags or {}
            }
            logger.info(metric_entry)
            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_error_with_context(
    error: Exception,
    operation: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log error with comprehensive context for debugging.

    Args:
        error: Exception instance
        operation: Operation where error occurred
        user_id: User ID
        request_id: Request ID
        context: Additional context information
    """
    logger = get_logger('error')

    error_entry = {
        'event_type': 'error',
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'request_id': request_id,
        'context': context or {}
    }

    logger.error(error_entry, exc_info=True)


class PerformanceTimer:
    """
    Context manager for measuring operation performance.

    Usage:
        with PerformanceTimer('dataset_processing', user_id='user123') as timer:
            # Perform operation
            process_dataset()

        # Timer automatically logs performance metric
    """

    def __init__(
        self,
        operation: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize performance timer.

        Args:
            operation: Operation name
            user_id: User ID
            request_id: Request ID
            tags: Additional tags
        """
        self.operation = operation
        self.user_id = user_id
        self.request_id = request_id
        self.tags = tags or {}
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log metric."""
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds() * 1000  # Convert to milliseconds

        # Add duration to tags
        self.tags['duration_ms'] = str(duration)

        # Log performance metric
        log_performance_metric(
            metric_name=f'{self.operation}_duration',
            value=duration,
            unit='milliseconds',
            user_id=self.user_id,
            request_id=self.request_id,
            tags=self.tags
        )

        # Log if operation took too long (>50ms for critical operations)
        if duration > 50 and self.operation in ['dataset_processing', 'pipeline_execution', 'bias_detection']:
            logger = get_logger('performance')
            logger.warning(
                f"Operation {self.operation} took {duration:.2f}ms, exceeding 50ms threshold",
                extra={
                    'operation': self.operation,
                    'duration_ms': duration,
                    'threshold_ms': 50,
                    'user_id': self.user_id,
                    'request_id': self.request_id
                }
            )


# Convenience functions for common logging scenarios
def log_dataset_operation(
    operation: str,
    dataset_id: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = 'success',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log dataset-related operations."""
    log_operation(
        operation=f'dataset_{operation}',
        user_id=user_id,
        request_id=request_id,
        status=status,
        details={'dataset_id': dataset_id, **(details or {})}
    )


def log_pipeline_operation(
    operation: str,
    pipeline_id: str,
    stage: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = 'success',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log pipeline-related operations."""
    pipeline_details = {'pipeline_id': pipeline_id}
    if stage:
        pipeline_details['stage'] = stage

    log_operation(
        operation=f'pipeline_{operation}',
        user_id=user_id,
        request_id=request_id,
        status=status,
        details={**pipeline_details, **(details or {})}
    )


def log_bias_detection(
    bias_score: float,
    threshold: float,
    passed: bool,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log bias detection results."""
    log_operation(
        operation='bias_detection',
        user_id=user_id,
        request_id=request_id,
        status='success' if passed else 'failed',
        details={
            'bias_score': bias_score,
            'threshold': threshold,
            'passed': passed,
            **(details or {})
        }
    )


def log_file_upload(
    filename: str,
    file_size: int,
    file_type: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = 'success',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log file upload operations."""
    log_operation(
        operation='file_upload',
        user_id=user_id,
        request_id=request_id,
        status=status,
        details={
            'filename': filename,
            'file_size_bytes': file_size,
            'file_type': file_type,
            **(details or {})
        }
    )
