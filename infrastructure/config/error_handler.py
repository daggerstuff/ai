#!/usr/bin/env python3
"""
Configuration Error Handling System for Pixelated Empathy AI
Provides robust error handling and recovery mechanisms for configuration issues
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorCategory(Enum):
    """Error categories"""
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    RESOURCE = "resource"
    PERMISSION = "permission"


@dataclass
class ConfigError:
    """Configuration error details"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    exception: Optional[Exception] = None
    timestamp: float = None
    recovery_suggestion: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ConfigErrorHandler:
    """Main configuration error handler"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.errors: List[ConfigError] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.CONFIGURATION: [
                self._recover_missing_config,
                self._recover_invalid_format,
                self._recover_default_values
            ],
            ErrorCategory.VALIDATION: [
                self._recover_validation_error,
                self._recover_type_mismatch
            ],
            ErrorCategory.NETWORK: [
                self._recover_network_error,
                self._recover_connection_timeout
            ],
            ErrorCategory.DATABASE: [
                self._recover_database_connection,
                self._recover_database_schema
            ],
            ErrorCategory.SECURITY: [
                self._recover_security_config,
                self._recover_permissions
            ],
            ErrorCategory.RESOURCE: [
                self._recover_resource_limits,
                self._recover_memory_issues
            ],
            ErrorCategory.PERMISSION: [
                self._recover_file_permissions,
                self._recover_directory_access
            ]
        }
    
    def handle_error(self, error: ConfigError) -> bool:
        """Handle a configuration error and attempt recovery"""
        logger.error(f"Configuration error: {error.message}")
        
        # Add to error list
        self.errors.append(error)
        
        # Log error details
        self._log_error_details(error)
        
        # Attempt recovery based on category
        if error.category in self.recovery_strategies:
            for strategy in self.recovery_strategies[error.category]:
                try:
                    if strategy(error):
                        logger.info(f"Successfully recovered from error: {error.message}")
                        return True
                except Exception as e:
                    logger.warning(f"Recovery strategy failed: {e}")
        
        # If critical error and no recovery, exit
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical configuration error with no recovery: {error.message}")
            self._generate_error_report()
            sys.exit(1)
        
        return False
    
    def _log_error_details(self, error: ConfigError):
        """Log detailed error information"""
        details = {
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'field': error.field,
            'value': str(error.value) if error.value is not None else None,
            'timestamp': error.timestamp,
            'recovery_suggestion': error.recovery_suggestion
        }
        
        if error.exception:
            details['exception'] = str(error.exception)
            details['traceback'] = traceback.format_exception(
                type(error.exception),
                error.exception,
                error.exception.__traceback__
            )
        
        logger.error(f"Error details: {json.dumps(details, indent=2)}")
    
    def _recover_missing_config(self, error: ConfigError) -> bool:
        """Recover from missing configuration files"""
        if "not found" in error.message.lower() or "missing" in error.message.lower():
            # Try to create default configuration
            if error.field and error.field.endswith('.yaml'):
                config_path = self.config_dir / error.field
                default_config = self._get_default_config(error.field)
                
                if default_config:
                    try:
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(config_path, 'w') as f:
                            f.write(default_config)
                        logger.info(f"Created default configuration: {config_path}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to create default config: {e}")
        
        return False
    
    def _recover_invalid_format(self, error: ConfigError) -> bool:
        """Recover from invalid configuration format"""
        if error.field and Path(error.field).exists():
            try:
                # Try to backup and recreate
                backup_path = Path(error.field).with_suffix('.backup')
                Path(error.field).rename(backup_path)
                
                # Create new default config
                default_config = self._get_default_config(Path(error.field).name)
                if default_config:
                    with open(error.field, 'w') as f:
                        f.write(default_config)
                    logger.info(f"Recovered invalid config file: {error.field}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to recover invalid config: {e}")
        
        return False
    
    def _recover_default_values(self, error: ConfigError) -> bool:
        """Recover by setting default values"""
        default_values = {
            'LOG_LEVEL': 'INFO',
            'MAX_WORKERS': '4',
            'BATCH_SIZE': '100',
            'DEBUG': 'false',
            'ENVIRONMENT': 'development'
        }
        
        if error.field in default_values:
            os.environ[error.field] = default_values[error.field]
            logger.info(f"Set default value for {error.field}: {default_values[error.field]}")
            return True
        
        return False
    
    def _recover_validation_error(self, error: ConfigError) -> bool:
        """Recover from validation errors"""
        # Try to fix common validation issues
        if error.field and "URL" in error.field:
            # Try to fix URL format
            if error.value and not str(error.value).startswith(('http://', 'https://', 'postgresql://', 'redis://')):
                # Add default scheme
                if 'DATABASE' in error.field:
                    fixed_value = f"postgresql://{error.value}"
                elif 'REDIS' in error.field:
                    fixed_value = f"redis://{error.value}"
                else:
                    fixed_value = f"http://{error.value}"
                
                os.environ[error.field] = fixed_value
                logger.info(f"Fixed URL format for {error.field}: {fixed_value}")
                return True
        
        return False
    
    def _recover_type_mismatch(self, error: ConfigError) -> bool:
        """Recover from type mismatch errors"""
        if error.field and error.value is not None:
            # Try to convert to expected type
            try:
                if error.field in ['MAX_WORKERS', 'BATCH_SIZE', 'PORT']:
                    # Should be integer
                    int_value = int(float(str(error.value)))
                    os.environ[error.field] = str(int_value)
                    logger.info(f"Converted {error.field} to integer: {int_value}")
                    return True
                elif error.field in ['DEBUG']:
                    # Should be boolean
                    bool_value = str(error.value).lower() in ['true', '1', 'yes', 'on']
                    os.environ[error.field] = str(bool_value).lower()
                    logger.info(f"Converted {error.field} to boolean: {bool_value}")
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def _recover_network_error(self, error: ConfigError) -> bool:
        """Recover from network-related errors"""
        # Implement network recovery strategies
        return False
    
    def _recover_connection_timeout(self, error: ConfigError) -> bool:
        """Recover from connection timeout errors"""
        # Implement timeout recovery strategies
        return False
    
    def _recover_database_connection(self, error: ConfigError) -> bool:
        """Recover from database connection errors"""
        # Try alternative database configurations
        if "connection" in error.message.lower():
            # Try localhost if remote connection fails
            db_url = os.getenv('DATABASE_URL', '')
            if 'localhost' not in db_url and '127.0.0.1' not in db_url:
                # Try localhost version
                try:
                    from urllib.parse import urlparse, urlunparse
                    parsed = urlparse(db_url)
                    localhost_url = urlunparse(parsed._replace(netloc=f"{parsed.username}:{parsed.password}@localhost:{parsed.port or 5432}"))
                    os.environ['DATABASE_URL'] = localhost_url
                    logger.info(f"Trying localhost database connection")
                    return True
                except Exception:
                    pass
        
        return False
    
    def _recover_database_schema(self, error: ConfigError) -> bool:
        """Recover from database schema errors"""
        # Implement schema recovery strategies
        return False
    
    def _recover_security_config(self, error: ConfigError) -> bool:
        """Recover from security configuration errors"""
        if error.field in ['JWT_SECRET', 'ENCRYPTION_KEY']:
            # Generate secure random values
            import secrets
            import string
            
            alphabet = string.ascii_letters + string.digits
            secure_value = ''.join(secrets.choice(alphabet) for _ in range(64))
            os.environ[error.field] = secure_value
            logger.warning(f"Generated temporary {error.field} - CHANGE IN PRODUCTION!")
            return True
        
        return False
    
    def _recover_permissions(self, error: ConfigError) -> bool:
        """Recover from permission errors"""
        if error.field and Path(error.field).exists():
            try:
                # Try to fix file permissions
                os.chmod(error.field, 0o600)
                logger.info(f"Fixed permissions for {error.field}")
                return True
            except PermissionError:
                logger.error(f"Cannot fix permissions for {error.field} - insufficient privileges")
        
        return False
    
    def _recover_resource_limits(self, error: ConfigError) -> bool:
        """Recover from resource limit errors"""
        # Implement resource limit recovery
        return False
    
    def _recover_memory_issues(self, error: ConfigError) -> bool:
        """Recover from memory-related issues"""
        # Reduce memory-intensive settings
        if 'memory' in error.message.lower():
            # Reduce batch size
            current_batch = int(os.getenv('BATCH_SIZE', '100'))
            if current_batch > 10:
                new_batch = max(10, current_batch // 2)
                os.environ['BATCH_SIZE'] = str(new_batch)
                logger.info(f"Reduced batch size to {new_batch} due to memory constraints")
                return True
        
        return False
    
    def _recover_file_permissions(self, error: ConfigError) -> bool:
        """Recover from file permission errors"""
        return self._recover_permissions(error)
    
    def _recover_directory_access(self, error: ConfigError) -> bool:
        """Recover from directory access errors"""
        if error.field and not Path(error.field).exists():
            try:
                Path(error.field).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created missing directory: {error.field}")
                return True
            except PermissionError:
                logger.error(f"Cannot create directory {error.field} - insufficient privileges")
        
        return False
    
    def _get_default_config(self, filename: str) -> Optional[str]:
        """Get default configuration content for a file"""
        defaults = {
            'database.yaml': """
# Database Configuration
database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
""",
            'redis.yaml': """
# Redis Configuration
redis:
  url: ${REDIS_URL}
  max_connections: 10
  retry_on_timeout: true
  socket_timeout: 5
  socket_connect_timeout: 5
""",
            'security.yaml': """
# Security Configuration
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
  authentication:
    jwt_expiry: 3600
    require_2fa: false
  rate_limiting:
    enabled: true
    requests_per_minute: 60
""",
            'monitoring.yaml': """
# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
  logging:
    level: INFO
    format: json
  health_checks:
    enabled: true
    interval: 30
""",
            'backup.yaml': """
# Backup Configuration
backup:
  enabled: true
  schedule: "0 2 * * *"
  retention: "30d"
  compression: true
  encryption: true
"""
        }
        
        return defaults.get(filename)
    
    def _generate_error_report(self):
        """Generate comprehensive error report"""
        report_path = self.config_dir / 'error_report.json'
        
        report = {
            'timestamp': time.time(),
            'total_errors': len(self.errors),
            'errors_by_category': {},
            'errors_by_severity': {},
            'errors': []
        }
        
        # Categorize errors
        for error in self.errors:
            # By category
            category = error.category.value
            if category not in report['errors_by_category']:
                report['errors_by_category'][category] = 0
            report['errors_by_category'][category] += 1
            
            # By severity
            severity = error.severity.value
            if severity not in report['errors_by_severity']:
                report['errors_by_severity'][severity] = 0
            report['errors_by_severity'][severity] += 1
            
            # Error details
            report['errors'].append({
                'category': category,
                'severity': severity,
                'message': error.message,
                'field': error.field,
                'value': str(error.value) if error.value is not None else None,
                'timestamp': error.timestamp,
                'recovery_suggestion': error.recovery_suggestion,
                'exception': str(error.exception) if error.exception else None
            })
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Error report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate error report: {e}")
    
    @contextmanager
    def error_context(self, category: ErrorCategory, field: str = None):
        """Context manager for handling errors in a specific context"""
        try:
            yield
        except Exception as e:
            error = ConfigError(
                category=category,
                severity=ErrorSeverity.HIGH,
                message=str(e),
                field=field,
                exception=e,
                recovery_suggestion="Check configuration and try again"
            )
            self.handle_error(error)
            raise
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        if not self.errors:
            return {'total': 0, 'by_category': {}, 'by_severity': {}}
        
        summary = {
            'total': len(self.errors),
            'by_category': {},
            'by_severity': {}
        }
        
        for error in self.errors:
            # Count by category
            category = error.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        return summary


# Convenience functions for common error types
def handle_config_error(message: str, field: str = None, value: Any = None, 
                       severity: ErrorSeverity = ErrorSeverity.HIGH) -> bool:
    """Handle a configuration error"""
    handler = ConfigErrorHandler()
    error = ConfigError(
        category=ErrorCategory.CONFIGURATION,
        severity=severity,
        message=message,
        field=field,
        value=value
    )
    return handler.handle_error(error)


def handle_validation_error(message: str, field: str = None, value: Any = None) -> bool:
    """Handle a validation error"""
    handler = ConfigErrorHandler()
    error = ConfigError(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.MEDIUM,
        message=message,
        field=field,
        value=value
    )
    return handler.handle_error(error)


def handle_security_error(message: str, field: str = None, 
                         severity: ErrorSeverity = ErrorSeverity.CRITICAL) -> bool:
    """Handle a security error"""
    handler = ConfigErrorHandler()
    error = ConfigError(
        category=ErrorCategory.SECURITY,
        severity=severity,
        message=message,
        field=field,
        recovery_suggestion="Review security configuration immediately"
    )
    return handler.handle_error(error)


if __name__ == '__main__':
    # Example usage
    handler = ConfigErrorHandler()
    
    # Test error handling
    test_error = ConfigError(
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.MEDIUM,
        message="Test configuration error",
        field="test_field",
        value="test_value"
    )
    
    handler.handle_error(test_error)
    print(f"Error summary: {handler.get_error_summary()}")
