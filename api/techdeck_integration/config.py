"""
Configuration management for TechDeck-Python Pipeline Integration Service.

This module provides comprehensive configuration management with environment-based
settings, validation, and secure handling of sensitive data for HIPAA++ compliance.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TechDeckServiceConfig:
    """
    Comprehensive configuration for TechDeck Flask service.

    This configuration class manages all service settings with proper
    validation, environment-based loading, and secure handling of
    sensitive data for HIPAA++ compliance.
    """

    # Flask Configuration
    SECRET_KEY: str = field(default_factory=lambda: os.environ.get('SECRET_KEY', 'dev-secret-key'))
    HOST: str = field(default_factory=lambda: os.environ.get('FLASK_HOST', '0.0.0.0'))
    PORT: int = field(default_factory=lambda: int(os.environ.get('FLASK_PORT', '5000')))
    DEBUG: bool = field(default_factory=lambda: os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
    TESTING: bool = field(default_factory=lambda: os.environ.get('FLASK_TESTING', 'False').lower() == 'true')

    # Security Configuration
    JWT_SECRET_KEY: str = field(default_factory=lambda: os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key'))
    JWT_ALGORITHM: str = field(default_factory=lambda: os.environ.get('JWT_ALGORITHM', 'HS256'))
    JWT_ACCESS_TOKEN_EXPIRES: int = field(default_factory=lambda: int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', '3600')))
    JWT_REFRESH_TOKEN_EXPIRES: int = field(default_factory=lambda: int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES', '86400')))

    # Rate Limiting Configuration
    RATE_LIMIT_STORAGE_URL: str = field(default_factory=lambda: os.environ.get('RATE_LIMIT_STORAGE_URL', 'redis://localhost:6379'))
    RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: int(os.environ.get('RATE_LIMIT_PER_MINUTE', '60')))
    RATE_LIMIT_PER_HOUR: int = field(default_factory=lambda: int(os.environ.get('RATE_LIMIT_PER_HOUR', '1000')))
    RATE_LIMIT_PER_DAY: int = field(default_factory=lambda: int(os.environ.get('RATE_LIMIT_PER_DAY', '10000')))

    # Database Configuration
    MONGODB_URI: str = field(default_factory=lambda: os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/techdeck'))
    MONGODB_DATABASE: str = field(default_factory=lambda: os.environ.get('MONGODB_DATABASE', 'techdeck'))
    MONGODB_MAX_POOL_SIZE: int = field(default_factory=lambda: int(os.environ.get('MONGODB_MAX_POOL_SIZE', '100')))
    MONGODB_MIN_POOL_SIZE: int = field(default_factory=lambda: int(os.environ.get('MONGODB_MIN_POOL_SIZE', '10')))

    # Redis Configuration
    REDIS_URL: str = field(default_factory=lambda: os.environ.get('REDIS_URL', 'redis://localhost:6379'))
    REDIS_DB: int = field(default_factory=lambda: int(os.environ.get('REDIS_DB', '0')))
    REDIS_PASSWORD: Optional[str] = field(default_factory=lambda: os.environ.get('REDIS_PASSWORD'))
    REDIS_SOCKET_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('REDIS_SOCKET_TIMEOUT', '5')))
    REDIS_CONNECTION_POOL_SIZE: int = field(default_factory=lambda: int(os.environ.get('REDIS_CONNECTION_POOL_SIZE', '50')))

    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = field(default_factory=lambda: int(os.environ.get('MAX_FILE_SIZE_MB', '100')))
    UPLOAD_FOLDER: str = field(default_factory=lambda: os.environ.get('UPLOAD_FOLDER', '/tmp/techdeck_uploads'))
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda:
        os.environ.get('ALLOWED_EXTENSIONS', 'csv,json,jsonl,parquet').split(','))
    FILE_UPLOAD_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('FILE_UPLOAD_TIMEOUT', '300')))

    # Pipeline Configuration
    PIPELINE_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('PIPELINE_TIMEOUT', '1800')))
    PIPELINE_MAX_CONCURRENT: int = field(default_factory=lambda: int(os.environ.get('PIPELINE_MAX_CONCURRENT', '5')))
    PIPELINE_RETRY_ATTEMPTS: int = field(default_factory=lambda: int(os.environ.get('PIPELINE_RETRY_ATTEMPTS', '3')))
    PIPELINE_RETRY_DELAY: int = field(default_factory=lambda: int(os.environ.get('PIPELINE_RETRY_DELAY', '5')))

    # Bias Detection Configuration
    BIAS_DETECTION_ENABLED: bool = field(default_factory=lambda:
        os.environ.get('BIAS_DETECTION_ENABLED', 'True').lower() == 'true')
    BIAS_DETECTION_THRESHOLD: float = field(default_factory=lambda:
        float(os.environ.get('BIAS_DETECTION_THRESHOLD', '0.7')))
    BIAS_DETECTION_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('BIAS_DETECTION_TIMEOUT', '30')))

    # WebSocket Configuration
    WEBSOCKET_ENABLED: bool = field(default_factory=lambda:
        os.environ.get('WEBSOCKET_ENABLED', 'True').lower() == 'true')
    WEBSOCKET_PING_INTERVAL: int = field(default_factory=lambda: int(os.environ.get('WEBSOCKET_PING_INTERVAL', '30')))
    WEBSOCKET_PING_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('WEBSOCKET_PING_TIMEOUT', '10')))
    WEBSOCKET_MAX_CONNECTIONS: int = field(default_factory=lambda: int(os.environ.get('WEBSOCKET_MAX_CONNECTIONS', '1000')))

    # Logging Configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.environ.get('LOG_LEVEL', 'INFO'))
    LOG_FORMAT: str = field(default_factory=lambda: os.environ.get('LOG_FORMAT', 'json'))
    LOG_FILE_PATH: Optional[str] = field(default_factory=lambda: os.environ.get('LOG_FILE_PATH'))
    LOG_MAX_FILE_SIZE_MB: int = field(default_factory=lambda: int(os.environ.get('LOG_MAX_FILE_SIZE_MB', '100')))
    LOG_BACKUP_COUNT: int = field(default_factory=lambda: int(os.environ.get('LOG_BACKUP_COUNT', '5')))

    # Performance Configuration
    RESPONSE_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('RESPONSE_TIMEOUT', '50')))
    CACHE_TTL_SECONDS: int = field(default_factory=lambda: int(os.environ.get('CACHE_TTL_SECONDS', '3600')))
    ENABLE_COMPRESSION: bool = field(default_factory=lambda:
        os.environ.get('ENABLE_COMPRESSION', 'True').lower() == 'true')

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda:
        os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:4321').split(','))
    ALLOWED_METHODS: List[str] = field(default_factory=lambda:
        os.environ.get('ALLOWED_METHODS', 'GET,POST,PUT,DELETE,OPTIONS').split(','))
    ALLOWED_HEADERS: List[str] = field(default_factory=lambda:
        os.environ.get('ALLOWED_HEADERS', 'Content-Type,Authorization,X-Request-ID').split(','))

    # Security Headers
    ENABLE_SECURITY_HEADERS: bool = field(default_factory=lambda:
        os.environ.get('ENABLE_SECURITY_HEADERS', 'True').lower() == 'true')
    CONTENT_SECURITY_POLICY: str = field(default_factory=lambda:
        os.environ.get('CONTENT_SECURITY_POLICY', "default-src 'self'"))

    # HIPAA Compliance Configuration
    ENABLE_AUDIT_LOGGING: bool = field(default_factory=lambda:
        os.environ.get('ENABLE_AUDIT_LOGGING', 'True').lower() == 'true')
    AUDIT_LOG_RETENTION_DAYS: int = field(default_factory=lambda: int(os.environ.get('AUDIT_LOG_RETENTION_DAYS', '90')))
    ENCRYPT_SENSITIVE_DATA: bool = field(default_factory=lambda:
        os.environ.get('ENCRYPT_SENSITIVE_DATA', 'True').lower() == 'true')
    DATA_RETENTION_DAYS: int = field(default_factory=lambda: int(os.environ.get('DATA_RETENTION_DAYS', '365')))

    # Integration Configuration
    PIPELINE_ORCHESTRATOR_URL: str = field(default_factory=lambda:
        os.environ.get('PIPELINE_ORCHESTRATOR_URL', 'http://localhost:8080'))
    BIAS_DETECTION_SERVICE_URL: str = field(default_factory=lambda:
        os.environ.get('BIAS_DETECTION_SERVICE_URL', 'http://localhost:8081'))
    ANALYTICS_SERVICE_URL: str = field(default_factory=lambda:
        os.environ.get('ANALYTICS_SERVICE_URL', 'http://localhost:8082'))

    # Health Check Configuration
    HEALTH_CHECK_TIMEOUT: int = field(default_factory=lambda: int(os.environ.get('HEALTH_CHECK_TIMEOUT', '5')))
    HEALTH_CHECK_INTERVAL: int = field(default_factory=lambda: int(os.environ.get('HEALTH_CHECK_INTERVAL', '30')))

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_ENABLED: bool = field(default_factory=lambda:
        os.environ.get('CIRCUIT_BREAKER_ENABLED', 'True').lower() == 'true')
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = field(default_factory=lambda:
        int(os.environ.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5')))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = field(default_factory=lambda:
        int(os.environ.get('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60')))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_required_settings()
        self._validate_file_paths()
        self._validate_numeric_ranges()
        self._setup_logging()

    def _validate_required_settings(self) -> None:
        """Validate required configuration settings."""
        required_settings = [
            'SECRET_KEY',
            'JWT_SECRET_KEY',
            'MONGODB_URI',
            'REDIS_URL'
        ]

        missing_settings = []
        for setting in required_settings:
            value = getattr(self, setting)
            if not value or value == 'dev-secret-key':
                missing_settings.append(setting)

        if missing_settings:
            raise ValueError(f"Missing required configuration settings: {missing_settings}")

    def _validate_file_paths(self) -> None:
        """Validate file paths and create directories if needed."""
        # Create upload directory if it doesn't exist
        upload_path = Path(self.UPLOAD_FOLDER)
        try:
            upload_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create upload directory {upload_path}: {e}")

        # Validate log file path if specified
        if self.LOG_FILE_PATH:
            log_path = Path(self.LOG_FILE_PATH).parent
            try:
                log_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create log directory {log_path}: {e}")

    def _validate_numeric_ranges(self) -> None:
        """Validate numeric configuration ranges."""
        numeric_validations = {
            'PORT': (1, 65535),
            'JWT_ACCESS_TOKEN_EXPIRES': (60, 86400),
            'JWT_REFRESH_TOKEN_EXPIRES': (3600, 604800),
            'RATE_LIMIT_PER_MINUTE': (1, 1000),
            'RATE_LIMIT_PER_HOUR': (1, 10000),
            'RATE_LIMIT_PER_DAY': (1, 100000),
            'MAX_FILE_SIZE_MB': (1, 1000),
            'PIPELINE_TIMEOUT': (60, 7200),
            'PIPELINE_MAX_CONCURRENT': (1, 50),
            'PIPELINE_RETRY_ATTEMPTS': (0, 10),
            'BIAS_DETECTION_THRESHOLD': (0.0, 1.0),
            'RESPONSE_TIMEOUT': (10, 5000),
            'CACHE_TTL_SECONDS': (60, 86400),
            'WEBSOCKET_MAX_CONNECTIONS': (10, 10000),
            'HEALTH_CHECK_TIMEOUT': (1, 60),
            'HEALTH_CHECK_INTERVAL': (10, 300),
            'CIRCUIT_BREAKER_FAILURE_THRESHOLD': (1, 20),
            'CIRCUIT_BREAKER_RECOVERY_TIMEOUT': (30, 600)
        }

        for setting, (min_val, max_val) in numeric_validations.items():
            value = getattr(self, setting)
            if not (min_val <= value <= max_val):
                raise ValueError(f"Configuration {setting} must be between {min_val} and {max_val}, got {value}")

    def _setup_logging(self) -> None:
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @classmethod
    def from_env_file(cls, env_file_path: str) -> 'TechDeckServiceConfig':
        """
        Load configuration from environment file.

        Args:
            env_file_path: Path to .env file

        Returns:
            TechDeckServiceConfig instance

        Raises:
            FileNotFoundError: If env file doesn't exist
            ValueError: If env file contains invalid configuration
        """
        from dotenv import load_dotenv

        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f"Environment file not found: {env_file_path}")

        # Load environment variables from file
        load_dotenv(env_file_path, override=True)

        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        config_dict = {}
        sensitive_keys = ['SECRET_KEY', 'JWT_SECRET_KEY', 'REDIS_PASSWORD']

        for key, value in self.__dict__.items():
            if key in sensitive_keys:
                config_dict[key] = '[REDACTED]' if value else None
            else:
                config_dict[key] = value

        return config_dict

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis connection configuration."""
        return {
            'host': self.REDIS_URL.split('://')[1].split(':')[0],
            'port': int(self.REDIS_URL.split(':')[-1]),
            'db': self.REDIS_DB,
            'password': self.REDIS_PASSWORD,
            'socket_timeout': self.REDIS_SOCKET_TIMEOUT,
            'connection_pool_size': self.REDIS_CONNECTION_POOL_SIZE
        }

    def get_mongodb_config(self) -> Dict[str, Any]:
        """Get MongoDB connection configuration."""
        return {
            'uri': self.MONGODB_URI,
            'database': self.MONGODB_DATABASE,
            'maxPoolSize': self.MONGODB_MAX_POOL_SIZE,
            'minPoolSize': self.MONGODB_MIN_POOL_SIZE
        }


# Environment-specific configuration classes
class DevelopmentConfig(TechDeckServiceConfig):
    """Development environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.RATE_LIMIT_PER_MINUTE = 1000  # Higher limits for development


class ProductionConfig(TechDeckServiceConfig):
    """Production environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.DEBUG = False
        self.TESTING = False
        self.LOG_LEVEL = 'INFO'
        self.ENABLE_SECURITY_HEADERS = True
        self.ENCRYPT_SENSITIVE_DATA = True

        # Validate production-specific requirements
        if self.SECRET_KEY == 'dev-secret-key':
            raise ValueError("SECRET_KEY must be set for production environment")
        if self.JWT_SECRET_KEY == 'jwt-secret-key':
            raise ValueError("JWT_SECRET_KEY must be set for production environment")


class TestingConfig(TechDeckServiceConfig):
    """Testing environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.TESTING = True
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.RATE_LIMIT_PER_MINUTE = 10000  # No rate limiting for tests
        self.ENABLE_AUDIT_LOGGING = False  # Disable audit logging for tests


# Configuration factory
def get_config(env: Optional[str] = None) -> TechDeckServiceConfig:
    """
    Get configuration for specified environment.

    Args:
        env: Environment name ('development', 'production', 'testing').
             If None, uses FLASK_ENV environment variable.

    Returns:
        TechDeckServiceConfig instance for the specified environment.
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development').lower()

    config_classes = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }

    config_class = config_classes.get(env, TechDeckServiceConfig)
    return config_class()


# Backwards-compatible alias expected by some modules
Config = TechDeckServiceConfig
