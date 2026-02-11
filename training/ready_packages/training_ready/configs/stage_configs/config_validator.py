#!/usr/bin/env python3
"""
Configuration Validation System for Pixelated Empathy AI
Validates all configuration files and environment variables
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a configuration validation"""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""
    results: List[ValidationResult] = field(default_factory=list)
    
    def add_error(self, message: str, field: str = None, value: Any = None, suggestion: str = None):
        """Add an error to the report"""
        self.results.append(ValidationResult(
            level=ValidationLevel.ERROR,
            message=message,
            field=field,
            value=value,
            suggestion=suggestion
        ))
    
    def add_warning(self, message: str, field: str = None, value: Any = None, suggestion: str = None):
        """Add a warning to the report"""
        self.results.append(ValidationResult(
            level=ValidationLevel.WARNING,
            message=message,
            field=field,
            value=value,
            suggestion=suggestion
        ))
    
    def add_info(self, message: str, field: str = None, value: Any = None):
        """Add an info message to the report"""
        self.results.append(ValidationResult(
            level=ValidationLevel.INFO,
            message=message,
            field=field,
            value=value
        ))
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains errors"""
        return any(r.level == ValidationLevel.ERROR for r in self.results)
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains warnings"""
        return any(r.level == ValidationLevel.WARNING for r in self.results)
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of validation results"""
        summary = {level.value: 0 for level in ValidationLevel}
        for result in self.results:
            summary[result.level.value] += 1
        return summary


class ConfigValidator:
    """Main configuration validator"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.report = ValidationReport()
    
    def validate_all(self) -> ValidationReport:
        """Validate all configuration aspects"""
        logger.info("Starting comprehensive configuration validation...")
        
        # Reset report
        self.report = ValidationReport()
        
        # Validate different aspects
        self._validate_environment_variables()
        self._validate_database_config()
        self._validate_redis_config()
        self._validate_security_config()
        self._validate_monitoring_config()
        self._validate_file_permissions()
        self._validate_network_config()
        self._validate_resource_limits()
        self._validate_backup_config()
        
        # Log summary
        summary = self.report.get_summary()
        logger.info(f"Validation complete: {summary}")
        
        return self.report
    
    def _validate_environment_variables(self):
        """Validate required environment variables"""
        logger.info("Validating environment variables...")
        
        required_vars = {
            'DATABASE_URL': self._validate_database_url,
            'REDIS_URL': self._validate_redis_url,
            'JWT_SECRET': self._validate_jwt_secret,
            'ENCRYPTION_KEY': self._validate_encryption_key,
            'LOG_LEVEL': self._validate_log_level,
            'ENVIRONMENT': self._validate_environment,
        }
        
        optional_vars = {
            'MAX_WORKERS': self._validate_max_workers,
            'BATCH_SIZE': self._validate_batch_size,
            'DEBUG': self._validate_debug_flag,
            'SENTRY_DSN': self._validate_sentry_dsn,
        }
        
        # Check required variables
        for var_name, validator in required_vars.items():
            value = os.getenv(var_name)
            if not value:
                self.report.add_error(
                    f"Required environment variable '{var_name}' is not set",
                    field=var_name,
                    suggestion=f"Set {var_name} environment variable"
                )
            else:
                validator(value, var_name)
        
        # Check optional variables
        for var_name, validator in optional_vars.items():
            value = os.getenv(var_name)
            if value:
                validator(value, var_name)
    
    def _validate_database_url(self, value: str, field: str):
        """Validate database URL format"""
        try:
            parsed = urlparse(value)
            if not parsed.scheme:
                self.report.add_error(
                    f"Database URL missing scheme",
                    field=field,
                    suggestion="Use format: postgresql://user:pass@host:port/db"
                )
            elif parsed.scheme not in ['postgresql', 'postgres']:
                self.report.add_warning(
                    f"Unexpected database scheme: {parsed.scheme}",
                    field=field,
                    suggestion="Consider using PostgreSQL for production"
                )
            
            if not parsed.hostname:
                self.report.add_error(
                    f"Database URL missing hostname",
                    field=field
                )
            
            if not parsed.path or parsed.path == '/':
                self.report.add_error(
                    f"Database URL missing database name",
                    field=field
                )
                
        except Exception as e:
            self.report.add_error(
                f"Invalid database URL format: {e}",
                field=field
            )
    
    def _validate_redis_url(self, value: str, field: str):
        """Validate Redis URL format"""
        try:
            parsed = urlparse(value)
            if not parsed.scheme:
                self.report.add_error(
                    f"Redis URL missing scheme",
                    field=field,
                    suggestion="Use format: redis://[:password@]host:port[/db]"
                )
            elif parsed.scheme not in ['redis', 'rediss']:
                self.report.add_error(
                    f"Invalid Redis scheme: {parsed.scheme}",
                    field=field,
                    suggestion="Use 'redis://' or 'rediss://' for SSL"
                )
            
            if not parsed.hostname:
                self.report.add_error(
                    f"Redis URL missing hostname",
                    field=field
                )
                
        except Exception as e:
            self.report.add_error(
                f"Invalid Redis URL format: {e}",
                field=field
            )
    
    def _validate_jwt_secret(self, value: str, field: str):
        """Validate JWT secret strength"""
        if len(value) < 32:
            self.report.add_error(
                f"JWT secret too short (minimum 32 characters)",
                field=field,
                value=f"Length: {len(value)}",
                suggestion="Generate a longer, more secure secret"
            )
        elif len(value) < 64:
            self.report.add_warning(
                f"JWT secret could be longer for better security",
                field=field,
                value=f"Length: {len(value)}",
                suggestion="Consider using 64+ character secret"
            )
        
        # Check for common weak patterns
        if value.lower() in ['secret', 'password', 'changeme', 'default']:
            self.report.add_error(
                f"JWT secret uses common weak value",
                field=field,
                suggestion="Generate a cryptographically secure random secret"
            )
    
    def _validate_encryption_key(self, value: str, field: str):
        """Validate encryption key"""
        if len(value) < 32:
            self.report.add_error(
                f"Encryption key too short (minimum 32 characters)",
                field=field,
                value=f"Length: {len(value)}"
            )
        
        # Check if it's base64 encoded (common for encryption keys)
        try:
            import base64
            base64.b64decode(value)
            if len(base64.b64decode(value)) < 32:
                self.report.add_warning(
                    f"Decoded encryption key may be too short",
                    field=field
                )
        except Exception:
            # Not base64, check raw length
            pass
    
    def _validate_log_level(self, value: str, field: str):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if value.upper() not in valid_levels:
            self.report.add_error(
                f"Invalid log level: {value}",
                field=field,
                suggestion=f"Use one of: {', '.join(valid_levels)}"
            )
        elif value.upper() == 'DEBUG':
            env = os.getenv('ENVIRONMENT', '').lower()
            if env in ['production', 'prod']:
                self.report.add_warning(
                    f"DEBUG log level in production environment",
                    field=field,
                    suggestion="Use INFO or WARNING for production"
                )
    
    def _validate_environment(self, value: str, field: str):
        """Validate environment setting"""
        valid_envs = ['development', 'dev', 'staging', 'production', 'prod', 'test']
        if value.lower() not in valid_envs:
            self.report.add_warning(
                f"Unexpected environment value: {value}",
                field=field,
                suggestion=f"Consider using: {', '.join(valid_envs)}"
            )
    
    def _validate_max_workers(self, value: str, field: str):
        """Validate max workers setting"""
        try:
            workers = int(value)
            if workers < 1:
                self.report.add_error(
                    f"Max workers must be positive",
                    field=field,
                    value=workers
                )
            elif workers > 32:
                self.report.add_warning(
                    f"Very high worker count may cause resource issues",
                    field=field,
                    value=workers,
                    suggestion="Consider CPU core count when setting workers"
                )
        except ValueError:
            self.report.add_error(
                f"Max workers must be an integer",
                field=field,
                value=value
            )
    
    def _validate_batch_size(self, value: str, field: str):
        """Validate batch size setting"""
        try:
            batch_size = int(value)
            if batch_size < 1:
                self.report.add_error(
                    f"Batch size must be positive",
                    field=field,
                    value=batch_size
                )
            elif batch_size > 1000:
                self.report.add_warning(
                    f"Large batch size may cause memory issues",
                    field=field,
                    value=batch_size,
                    suggestion="Consider memory constraints when setting batch size"
                )
        except ValueError:
            self.report.add_error(
                f"Batch size must be an integer",
                field=field,
                value=value
            )
    
    def _validate_debug_flag(self, value: str, field: str):
        """Validate debug flag"""
        if value.lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
            self.report.add_warning(
                f"Debug flag should be boolean-like",
                field=field,
                value=value,
                suggestion="Use 'true', 'false', '1', or '0'"
            )
        
        if value.lower() in ['true', '1', 'yes']:
            env = os.getenv('ENVIRONMENT', '').lower()
            if env in ['production', 'prod']:
                self.report.add_warning(
                    f"Debug enabled in production environment",
                    field=field,
                    suggestion="Disable debug in production"
                )
    
    def _validate_sentry_dsn(self, value: str, field: str):
        """Validate Sentry DSN format"""
        try:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.hostname:
                self.report.add_error(
                    f"Invalid Sentry DSN format",
                    field=field,
                    suggestion="Check Sentry project settings for correct DSN"
                )
        except Exception as e:
            self.report.add_error(
                f"Invalid Sentry DSN: {e}",
                field=field
            )
    
    def _validate_database_config(self):
        """Validate database configuration files"""
        logger.info("Validating database configuration...")
        
        # Check for database config files
        db_config_files = [
            'database.yaml',
            'database.json',
            'db_config.yaml'
        ]
        
        for config_file in db_config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                self._validate_config_file(config_path)
    
    def _validate_redis_config(self):
        """Validate Redis configuration"""
        logger.info("Validating Redis configuration...")
        
        redis_config = self.config_dir / 'redis.yaml'
        if redis_config.exists():
            self._validate_config_file(redis_config)
    
    def _validate_security_config(self):
        """Validate security configuration"""
        logger.info("Validating security configuration...")
        
        security_config = self.config_dir / 'security.yaml'
        if security_config.exists():
            try:
                with open(security_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check security settings
                if 'encryption' in config:
                    if not config['encryption'].get('enabled', False):
                        self.report.add_warning(
                            "Encryption is disabled",
                            field="encryption.enabled",
                            suggestion="Enable encryption for production"
                        )
                
                if 'authentication' in config:
                    auth_config = config['authentication']
                    if auth_config.get('require_2fa', False) is False:
                        env = os.getenv('ENVIRONMENT', '').lower()
                        if env in ['production', 'prod']:
                            self.report.add_warning(
                                "2FA not required in production",
                                field="authentication.require_2fa",
                                suggestion="Enable 2FA for production security"
                            )
                
            except Exception as e:
                self.report.add_error(
                    f"Error reading security config: {e}",
                    field="security.yaml"
                )
    
    def _validate_monitoring_config(self):
        """Validate monitoring configuration"""
        logger.info("Validating monitoring configuration...")
        
        monitoring_config = self.config_dir / 'monitoring.yaml'
        if monitoring_config.exists():
            self._validate_config_file(monitoring_config)
    
    def _validate_file_permissions(self):
        """Validate file permissions for security"""
        logger.info("Validating file permissions...")
        
        sensitive_files = [
            '.env',
            'secrets.yaml',
            'private.key',
            'ssl.key'
        ]
        
        for filename in sensitive_files:
            filepath = self.config_dir / filename
            if filepath.exists():
                stat_info = filepath.stat()
                mode = oct(stat_info.st_mode)[-3:]
                
                # Check if file is readable by others
                if int(mode[2]) > 0:
                    self.report.add_warning(
                        f"Sensitive file '{filename}' is readable by others",
                        field=f"permissions.{filename}",
                        value=f"Mode: {mode}",
                        suggestion="Set permissions to 600 or 640"
                    )
    
    def _validate_network_config(self):
        """Validate network configuration"""
        logger.info("Validating network configuration...")
        
        # Check common network settings
        bind_host = os.getenv('BIND_HOST', '0.0.0.0')
        if bind_host == '0.0.0.0':
            env = os.getenv('ENVIRONMENT', '').lower()
            if env in ['production', 'prod']:
                self.report.add_warning(
                    "Binding to all interfaces (0.0.0.0) in production",
                    field="BIND_HOST",
                    suggestion="Consider binding to specific interface for security"
                )
        
        # Check port configuration
        port = os.getenv('PORT', '8000')
        try:
            port_num = int(port)
            if port_num < 1024 and os.getuid() != 0:
                self.report.add_warning(
                    f"Port {port_num} requires root privileges",
                    field="PORT",
                    suggestion="Use port >= 1024 or run as root"
                )
        except (ValueError, AttributeError):
            pass
    
    def _validate_resource_limits(self):
        """Validate resource limit configurations"""
        logger.info("Validating resource limits...")
        
        # Check memory limits
        max_memory = os.getenv('MAX_MEMORY')
        if max_memory:
            try:
                # Parse memory value (e.g., "2G", "512M")
                if max_memory.endswith('G'):
                    memory_gb = float(max_memory[:-1])
                    if memory_gb < 1:
                        self.report.add_warning(
                            f"Low memory limit: {max_memory}",
                            field="MAX_MEMORY",
                            suggestion="Consider increasing memory for better performance"
                        )
                elif max_memory.endswith('M'):
                    memory_mb = float(max_memory[:-1])
                    if memory_mb < 512:
                        self.report.add_warning(
                            f"Very low memory limit: {max_memory}",
                            field="MAX_MEMORY",
                            suggestion="Increase memory limit for stable operation"
                        )
            except ValueError:
                self.report.add_error(
                    f"Invalid memory limit format: {max_memory}",
                    field="MAX_MEMORY",
                    suggestion="Use format like '2G' or '512M'"
                )
    
    def _validate_backup_config(self):
        """Validate backup configuration"""
        logger.info("Validating backup configuration...")
        
        backup_config = self.config_dir / 'backup.yaml'
        if backup_config.exists():
            try:
                with open(backup_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config.get('enabled', False):
                    env = os.getenv('ENVIRONMENT', '').lower()
                    if env in ['production', 'prod']:
                        self.report.add_error(
                            "Backups disabled in production",
                            field="backup.enabled",
                            suggestion="Enable backups for production data protection"
                        )
                
                # Check backup schedule
                schedule = config.get('schedule')
                if schedule:
                    # Basic cron validation
                    if not re.match(r'^[\d\*\-,/]+\s+[\d\*\-,/]+\s+[\d\*\-,/]+\s+[\d\*\-,/]+\s+[\d\*\-,/]+$', schedule):
                        self.report.add_warning(
                            f"Invalid cron schedule format: {schedule}",
                            field="backup.schedule",
                            suggestion="Use valid cron format (e.g., '0 2 * * *')"
                        )
                
            except Exception as e:
                self.report.add_error(
                    f"Error reading backup config: {e}",
                    field="backup.yaml"
                )
    
    def _validate_config_file(self, filepath: Path):
        """Validate a configuration file"""
        try:
            with open(filepath, 'r') as f:
                if filepath.suffix in ['.yaml', '.yml']:
                    yaml.safe_load(f)
                elif filepath.suffix == '.json':
                    json.load(f)
                
            self.report.add_info(
                f"Configuration file '{filepath.name}' is valid",
                field=str(filepath)
            )
            
        except yaml.YAMLError as e:
            self.report.add_error(
                f"Invalid YAML in '{filepath.name}': {e}",
                field=str(filepath)
            )
        except json.JSONDecodeError as e:
            self.report.add_error(
                f"Invalid JSON in '{filepath.name}': {e}",
                field=str(filepath)
            )
        except Exception as e:
            self.report.add_error(
                f"Error reading '{filepath.name}': {e}",
                field=str(filepath)
            )
    
    def print_report(self, report: ValidationReport = None):
        """Print validation report in a readable format"""
        if report is None:
            report = self.report
        
        print("\n" + "="*80)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*80)
        
        summary = report.get_summary()
        print(f"\nSUMMARY:")
        print(f"  Errors:   {summary['error']}")
        print(f"  Warnings: {summary['warning']}")
        print(f"  Info:     {summary['info']}")
        
        if report.results:
            print(f"\nDETAILS:")
            for result in report.results:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[result.level.value]
                print(f"\n{icon} {result.level.value.upper()}: {result.message}")
                
                if result.field:
                    print(f"   Field: {result.field}")
                if result.value is not None:
                    print(f"   Value: {result.value}")
                if result.suggestion:
                    print(f"   Suggestion: {result.suggestion}")
        
        print("\n" + "="*80)
        
        if report.has_errors:
            print("❌ VALIDATION FAILED - Please fix errors before proceeding")
            return False
        elif report.has_warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS - Review warnings for production")
            return True
        else:
            print("✅ VALIDATION PASSED - Configuration is valid")
            return True


def main():
    """Main entry point for configuration validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Pixelated Empathy AI configuration")
    parser.add_argument(
        '--config-dir',
        default=None,
        help="Configuration directory path"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output report in JSON format"
    )
    parser.add_argument(
        '--fail-on-warnings',
        action='store_true',
        help="Exit with error code if warnings are found"
    )
    
    args = parser.parse_args()
    
    # Create validator and run validation
    validator = ConfigValidator(args.config_dir)
    report = validator.validate_all()
    
    if args.json:
        # Output JSON report
        json_report = {
            'summary': report.get_summary(),
            'results': [
                {
                    'level': r.level.value,
                    'message': r.message,
                    'field': r.field,
                    'value': r.value,
                    'suggestion': r.suggestion
                }
                for r in report.results
            ]
        }
        print(json.dumps(json_report, indent=2))
    else:
        # Print human-readable report
        success = validator.print_report(report)
        
        # Exit with appropriate code
        if not success:
            sys.exit(1)
        elif args.fail_on_warnings and report.has_warnings:
            sys.exit(2)
        else:
            sys.exit(0)


if __name__ == '__main__':
    main()
