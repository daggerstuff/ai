"""
Custom error types for TechDeck Flask service.

This module defines comprehensive error types for different failure scenarios
with HIPAA++ compliant error handling and audit logging capabilities.
"""


from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


class TechDeckBaseError(Exception):
    """
    Base error class for all TechDeck service errors.

    Provides consistent error structure with HIPAA++ compliant
    error handling and audit logging capabilities.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize base error with comprehensive error information.

        Args:
            message: Human-readable error message
            error_code: Unique error code for identification
            status_code: HTTP status code
            details: Additional error details
            request_id: Request ID for tracking
            user_id: User ID for audit logging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.request_id = request_id
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc)
        self.error_id = self._generate_error_id()

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for JSON serialization.

        Returns:
            Dictionary representation of error
        """
        return {
            'success': False,
            'error': {
                'code': self.error_code,
                'message': self.message,
                'details': self.details,
                'error_id': self.error_id,
                'request_id': self.request_id,
                'timestamp': self.timestamp.isoformat(),
                'support_reference': self.error_id[:8]
            }
        }

    def to_audit_log(self) -> Dict[str, Any]:
        """
        Convert error to audit log format for HIPAA compliance.

        Returns:
            Audit log dictionary
        """
        return {
            'event_type': 'error',
            'error_code': self.error_code,
            'error_message': self.message,
            'error_id': self.error_id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'details': self._sanitize_for_audit(self.details)
        }

    def _sanitize_for_audit(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize error details for audit logging (remove PII/sensitive data).

        Args:
            details: Original error details

        Returns:
            Sanitized details
        """
        sensitive_keys = ['password', 'token', 'secret', 'key', 'ssn', 'email']
        sanitized = {}

        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value

        return sanitized


class AuthenticationError(TechDeckBaseError):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='AUTHENTICATION_FAILED',
            status_code=401,
            **kwargs
        )


class AuthorizationError(TechDeckBaseError):
    """Authorization/permission-related errors."""

    def __init__(self, message: str, required_permissions: Optional[List[str]] = None, **kwargs):
        details = {'required_permissions': required_permissions or []}
        super().__init__(
            message=message,
            error_code='AUTHORIZATION_FAILED',
            status_code=403,
            details=details,
            **kwargs
        )


class ValidationError(TechDeckBaseError):
    """Input validation errors."""

    def __init__(self, message: str, field_errors: Optional[Dict[str, str]] = None, **kwargs):
        details = {'field_errors': field_errors or {}}
        super().__init__(
            message=message,
            error_code='VALIDATION_ERROR',
            status_code=400,
            details=details,
            **kwargs
        )


class ResourceNotFoundError(TechDeckBaseError):
    """Resource not found (404) errors."""

    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, **kwargs):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        super().__init__(
            message=message,
            error_code='NOT_FOUND',
            status_code=404,
            details=details,
            **kwargs
        )


class DatasetError(TechDeckBaseError):
    """Dataset-related errors."""

    def __init__(self, message: str, dataset_id: Optional[str] = None, **kwargs):
        details = {'dataset_id': dataset_id} if dataset_id else {}
        super().__init__(
            message=message,
            error_code='DATASET_ERROR',
            status_code=400,
            details=details,
            **kwargs
        )


class PipelineError(TechDeckBaseError):
    """Pipeline processing errors."""

    def __init__(self, message: str, pipeline_id: Optional[str] = None, stage: Optional[str] = None, **kwargs):
        details = {}
        if pipeline_id:
            details['pipeline_id'] = pipeline_id
        if stage:
            details['failed_stage'] = stage
        super().__init__(
            message=message,
            error_code='PIPELINE_ERROR',
            status_code=500,
            details=details,
            **kwargs
        )


# Backwards-compatible name expected by other modules
class PipelineExecutionError(PipelineError):
    pass


# Backwards-compatible State Management error expected by state manager
class StateManagementError(TechDeckBaseError):
    """Errors related to state management and persistence."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='STATE_MANAGEMENT_ERROR',
            status_code=500,
            **kwargs
        )


class BiasDetectionError(TechDeckBaseError):
    """Bias detection errors."""

    def __init__(self, message: str, bias_score: Optional[float] = None, threshold: Optional[float] = None, **kwargs):
        details = {}
        if bias_score is not None:
            details['bias_score'] = bias_score
        if threshold is not None:
            details['threshold'] = threshold
        super().__init__(
            message=message,
            error_code='BIAS_DETECTION_ERROR',
            status_code=400,
            details=details,
            **kwargs
        )


class FileUploadError(TechDeckBaseError):
    """File upload errors."""

    def __init__(self, message: str, filename: Optional[str] = None, file_size: Optional[int] = None, **kwargs):
        details = {}
        if filename:
            details['filename'] = filename
        if file_size is not None:
            details['file_size_bytes'] = file_size
        super().__init__(
            message=message,
            error_code='FILE_UPLOAD_ERROR',
            status_code=413,
            details=details,
            **kwargs
        )


class RateLimitError(TechDeckBaseError):
    """Rate limiting errors."""

    def __init__(self, message: str, retry_after: Optional[int] = None, limit: Optional[int] = None, **kwargs):
        details = {}
        if retry_after is not None:
            details['retry_after_seconds'] = retry_after
        if limit is not None:
            details['rate_limit'] = limit
        super().__init__(
            message=message,
            error_code='RATE_LIMIT_EXCEEDED',
            status_code=429,
            details=details,
            **kwargs
        )


class DatabaseError(TechDeckBaseError):
    """Database operation errors."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = {'operation': operation} if operation else {}
        super().__init__(
            message=message,
            error_code='DATABASE_ERROR',
            status_code=500,
            details=details,
            **kwargs
        )


class RedisError(TechDeckBaseError):
    """Redis operation errors."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = {'operation': operation} if operation else {}
        super().__init__(
            message=message,
            error_code='REDIS_ERROR',
            status_code=500,
            details=details,
            **kwargs
        )


class EventBusError(TechDeckBaseError):
    """Errors raised by the event bus and handlers."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='EVENT_BUS_ERROR',
            status_code=500,
            **kwargs
        )


class WebSocketError(TechDeckBaseError):
    """WebSocket communication errors."""

    def __init__(self, message: str, connection_id: Optional[str] = None, **kwargs):
        details = {'connection_id': connection_id} if connection_id else {}
        super().__init__(
            message=message,
            error_code='WEBSOCKET_ERROR',
            status_code=500,
            details=details,
            **kwargs
        )


class IntegrationError(TechDeckBaseError):
    """External service integration errors."""

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        details = {'service': service_name} if service_name else {}
        super().__init__(
            message=message,
            error_code='INTEGRATION_ERROR',
            status_code=503,
            details=details,
            **kwargs
        )


class CircuitBreakerError(TechDeckBaseError):
    """Circuit breaker errors for fault tolerance."""

    def __init__(self, message: str, service_name: Optional[str] = None, failure_count: Optional[int] = None, **kwargs):
        details = {}
        if service_name:
            details['service'] = service_name
        if failure_count is not None:
            details['failure_count'] = failure_count
        super().__init__(
            message=message,
            error_code='CIRCUIT_BREAKER_OPEN',
            status_code=503,
            details=details,
            **kwargs
        )


class RetryExhaustedError(TechDeckBaseError):
    """Retry mechanism exhausted errors."""

    def __init__(self, message: str, max_attempts: Optional[int] = None, **kwargs):
        details = {'max_attempts': max_attempts} if max_attempts else {}
        super().__init__(
            message=message,
            error_code='RETRY_EXHAUSTED',
            status_code=500,
            details=details,
            **kwargs
        )


class TimeoutError(TechDeckBaseError):
    """Operation timeout errors."""

    def __init__(self, message: str, timeout_seconds: Optional[int] = None, **kwargs):
        details = {'timeout_seconds': timeout_seconds} if timeout_seconds else {}
        super().__init__(
            message=message,
            error_code='OPERATION_TIMEOUT',
            status_code=504,
            details=details,
            **kwargs
        )


class EncryptionError(TechDeckBaseError):
    """Data encryption/decryption errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code='ENCRYPTION_ERROR',
            status_code=500,
            **kwargs
        )


class ConfigurationError(TechDeckBaseError):
    """Configuration-related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = {'config_key': config_key} if config_key else {}
        super().__init__(
            message=message,
            error_code='CONFIGURATION_ERROR',
            status_code=500,
            details=details,
            **kwargs
        )


class ServiceUnavailableError(TechDeckBaseError):
    """Service unavailable errors."""

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        details = {'service': service_name} if service_name else {}
        super().__init__(
            message=message,
            error_code='SERVICE_UNAVAILABLE',
            status_code=503,
            details=details,
            **kwargs
        )


# Error mapping for HTTP status codes
ERROR_STATUS_MAPPING = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthorizationError,
    404: ResourceNotFoundError,
    413: FileUploadError,
    429: RateLimitError,
    500: lambda msg, **kwargs: TechDeckBaseError(msg, 'INTERNAL_ERROR', 500, **kwargs),
    503: ServiceUnavailableError,
    504: TimeoutError
}


def create_error_from_status_code(status_code: int, message: str, **kwargs) -> TechDeckBaseError:
    """
    Create appropriate error based on HTTP status code.

    Args:
        status_code: HTTP status code
        message: Error message
        **kwargs: Additional error parameters

    Returns:
        Appropriate TechDeckBaseError instance
    """
    error_class = ERROR_STATUS_MAPPING.get(status_code, TechDeckBaseError)

    return error_class(message, **kwargs)


class ErrorContext:
    """
    Context manager for error handling with automatic audit logging.
    """

    def __init__(self, operation: str, user_id: Optional[str] = None, request_id: Optional[str] = None):
        """
        Initialize error context.

        Args:
            operation: Operation name for logging
            user_id: User ID for audit logging
            request_id: Request ID for tracking
        """
        self.operation = operation
        self.user_id = user_id
        self.request_id = request_id
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Enter error context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit error context and handle errors."""
        if exc_type:
            if issubclass(exc_type, TechDeckBaseError):
                # Log error for audit purposes
                audit_log = exc_val.to_audit_log()
                audit_log.update({
                    'operation': self.operation,
                    'user_id': self.user_id,
                    'request_id': self.request_id
                })
                self.logger.error(f"Operation failed: {audit_log}")
            else:
                # Log unexpected errors
                self.logger.error(
                    f"Unexpected error in {self.operation}",
                    extra={
                        'error_type': exc_type.__name__,
                        'error_value': str(exc_val),
                        'user_id': self.user_id,
                        'request_id': self.request_id,
                        'operation': self.operation
                    }
                )
            return False  # Don't suppress the exception
        return True
