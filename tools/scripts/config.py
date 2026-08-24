"""
Configuration management for Pixelated AI CLI

This module handles configuration files, environment variables, and profiles
for the CLI tool, ensuring HIPAA++ compliance and secure credential management.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator


# Register custom YAML tags used by NVIDIA NeMo Microservices Platform
def override_constructor(loader, node):
    """Handle !override tag by returning the underlying data"""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


yaml.SafeLoader.add_constructor("!override", override_constructor)
yaml.SafeLoader.add_constructor("!merge", override_constructor)
yaml.Loader.add_constructor("!override", override_constructor)
yaml.Loader.add_constructor("!merge", override_constructor)


logger = logging.getLogger(__name__)


class APIConfig(BaseModel):
    """API configuration settings"""

    base_url: str = Field(default="http://localhost:8000", description="Base API URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @validator("timeout")
    def validate_timeout(self, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @validator("max_retries")
    def validate_retries(self, v):
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v


class AuthConfig(BaseModel):
    """Authentication configuration settings"""

    jwt_token: str | None = Field(default=None, description="JWT authentication token")
    refresh_token: str | None = Field(default=None, description="JWT refresh token")
    token_expiry: str | None = Field(default=None, description="Token expiry timestamp")
    client_id: str | None = Field(default=None, description="OAuth client ID")
    client_secret: str | None = Field(default=None, description="OAuth client secret")
    auth_url: str = Field(default="http://localhost:8000/auth", description="Authentication endpoint")

    @validator("jwt_token", "refresh_token", "client_secret")
    def validate_sensitive_data(self, v):
        # Ensure sensitive data is handled securely
        return v


class PipelineConfig(BaseModel):
    """Pipeline configuration settings"""

    default_timeout: int = Field(default=3600, description="Default pipeline timeout in seconds")
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent pipeline jobs")
    checkpoint_interval: int = Field(default=300, description="Checkpoint interval in seconds")
    enable_bias_detection: bool = Field(default=True, description="Enable bias detection")
    enable_fhe_encryption: bool = Field(default=True, description="Enable FHE encryption")
    audit_logging: bool = Field(default=True, description="Enable audit logging")

    @validator("default_timeout", "max_concurrent_jobs", "checkpoint_interval")
    def validate_positive_values(self, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration settings"""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: str | None = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Maximum log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")

    @validator("level")
    def validate_log_level(self, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration settings"""

    encrypt_credentials: bool = Field(default=True, description="Encrypt stored credentials")
    key_file: str = Field(default=".cli_key", description="Encryption key file")
    validate_inputs: bool = Field(default=True, description="Validate all inputs")
    sanitize_outputs: bool = Field(default=True, description="Sanitize outputs")
    rate_limit: int = Field(default=100, description="Rate limit per minute")

    @validator("rate_limit")
    def validate_rate_limit(self, v):
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        return v


class CLIConfig(BaseModel):
    """Main CLI configuration class"""

    api: APIConfig = Field(default_factory=APIConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        """Pydantic configuration"""

        validate_assignment = True
        use_enum_values = True

    def __init__(self, config_file: Path | None = None, profile: str = "default"):
        """Initialize configuration with file and profile support"""
        super().__init__()
        self._config_file = config_file
        self._profile = profile
        self._config_dir = Path.home() / ".pixelated-ai"
        self._encryption_key = None

        # Ensure config directory exists
        self._config_dir.mkdir(exist_ok=True)

        # Load configuration
        self._load_configuration()

        # Setup encryption if enabled
        if self.security.encrypt_credentials:
            self._setup_encryption()

    @property
    def profile(self) -> str:
        """Get current profile name"""
        return self._profile

    @property
    def config_dir(self) -> Path:
        """Get configuration directory path"""
        return self._config_dir

    @property
    def api_base_url(self) -> str:
        """Get API base URL"""
        return self.api.base_url

    @api_base_url.setter
    def api_base_url(self, value: str) -> None:
        self.api.base_url = value

    @property
    def timeout(self) -> int:
        """Get request timeout in seconds"""
        return self.api.timeout

    @timeout.setter
    def timeout(self, value: int) -> None:
        self.api.timeout = value

    @property
    def max_retries(self) -> int:
        """Get maximum number of retries"""
        return self.api.max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self.api.max_retries = value

    @property
    def log_level(self) -> str:
        """Get logging level"""
        return self.logging.level

    @log_level.setter
    def log_level(self, value: str) -> None:
        self.logging.level = value

    @property
    def auth(self) -> AuthConfig:
        """Get auth configuration"""
        return self._auth

    @auth.setter
    def auth(self, value: AuthConfig) -> None:
        self._auth = value

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "profile": self.profile,
            "api_base_url": self.api_base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CLIConfig":
        """Create configuration from dictionary."""
        instance = cls(profile=data.get("profile", "default"))
        if "api_base_url" in data:
            instance.api_base_url = data["api_base_url"]
        if "timeout" in data:
            instance.timeout = data["timeout"]
        if "max_retries" in data:
            instance.max_retries = data["max_retries"]
        if "log_level" in data:
            instance.log_level = data["log_level"]
        return instance

    def save(self, encrypt: bool = False) -> None:
        """Save configuration to file."""
        self.save_configuration(profile=None)

    def validate(self) -> bool:
        """Validate auth configuration."""
        return self.auth is not None

    def _load_configuration(self) -> None:
        """Load configuration from files and environment variables"""
        # Load from default locations
        config_sources = []

        # 1. Load from specified config file
        if self._config_file and self._config_file.exists():
            config_sources.append(self._config_file)

        # 2. Load from profile-specific config
        profile_config = self._config_dir / f"config.{self._profile}.yaml"
        if profile_config.exists():
            config_sources.append(profile_config)

        # 3. Load from default config
        default_config = self._config_dir / "config.yaml"
        if default_config.exists():
            config_sources.append(default_config)

        # Load configuration from files
        for config_file in config_sources:
            try:
                self._load_from_file(config_file)
                logger.info(f"Loaded configuration from {config_file}")
                break  # Use first valid config file
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")

        # Override with environment variables
        self._load_from_env()

    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            if config_data:
                # Update configuration with file data
                for section, values in config_data.items():
                    if hasattr(self, section):
                        section_obj = getattr(self, section)
                        for key, value in values.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)
        except Exception as e:
            raise ValueError(f"Invalid configuration file {config_file}: {e}") from e

    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            "PIXELATED_API_URL": ("api", "base_url"),
            "PIXELATED_API_TIMEOUT": ("api", "timeout"),
            "PIXELATED_JWT_TOKEN": ("auth", "jwt_token"),
            "PIXELATED_CLIENT_ID": ("auth", "client_id"),
            "PIXELATED_CLIENT_SECRET": ("auth", "client_secret"),
            "PIXELATED_LOG_LEVEL": ("logging", "level"),
            "PIXELATED_ENCRYPT_CREDENTIALS": ("security", "encrypt_credentials"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert string values to appropriate types
                    if key in [
                        "timeout",
                        "max_retries",
                        "max_concurrent_jobs",
                        "checkpoint_interval",
                        "rate_limit",
                        "max_file_size",
                        "backup_count",
                    ]:
                        value = int(value)
                    elif key in ["retry_delay"]:
                        value = float(value)
                    elif key in [
                        "verify_ssl",
                        "enable_bias_detection",
                        "enable_fhe_encryption",
                        "audit_logging",
                        "encrypt_credentials",
                        "validate_inputs",
                        "sanitize_outputs",
                    ]:
                        value = value.lower() in ("true", "1", "yes", "on")

                    section_obj = getattr(self, section)
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={value}: {e}")

    def _setup_encryption(self) -> None:
        """Setup encryption for sensitive data"""
        key_file = self._config_dir / self.security.key_file

        if key_file.exists():
            # Load existing key
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_file, 0o600)

        self._encryption_key = key
        self._cipher = Fernet(key)

    def save_configuration(self, profile: str | None = None) -> None:
        """Save current configuration to file"""
        profile = profile or self._profile
        config_file = self._config_dir / f"config.{profile}.yaml"

        try:
            config_dict = self.dict()

            # Encrypt sensitive data if enabled
            if self.security.encrypt_credentials and self._cipher:
                for field in ["jwt_token", "refresh_token", "client_secret"]:
                    if value := config_dict["auth"].get(field):
                        encrypted_value = self._cipher.encrypt(value.encode()).decode()
                        config_dict["auth"][field] = f"encrypted:{encrypted_value}"

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}") from e

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value"""
        if not encrypted_value.startswith("encrypted:"):
            return encrypted_value

        if not self._cipher:
            raise ValueError("Encryption not initialized")

        try:
            encrypted_data = encrypted_value[10:].encode()  # Remove "encrypted:" prefix
            decrypted_data = self._cipher.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt value: {e}") from e

    def validate_configuration(self) -> bool:
        """Validate current configuration"""
        try:
            # Validate API connectivity
            if not self.api.base_url:
                raise ValueError("API base URL is required")

            # Validate authentication settings
            if self.auth.jwt_token and len(self.auth.jwt_token) < 10:
                raise ValueError("Invalid JWT token format")

            # Validate pipeline settings
            if self.pipeline.max_concurrent_jobs <= 0:
                raise ValueError("Max concurrent jobs must be positive")

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_active_profile(self) -> dict[str, Any]:
        """Get active profile configuration"""
        return self.dict()

    def switch_profile(self, profile: str) -> None:
        """Switch to a different configuration profile"""
        profile_config = self._config_dir / f"config.{profile}.yaml"

        if not profile_config.exists():
            raise ValueError(f"Profile '{profile}' not found")

        # Load new profile
        self._profile = profile
        self._load_configuration()

        logger.info(f"Switched to profile: {profile}")

    def create_profile(self, profile: str, base_profile: str | None = None) -> None:
        """Create a new configuration profile"""
        if base_profile:
            # Copy from base profile
            base_config = self._config_dir / f"config.{base_profile}.yaml"
            if base_config.exists():
                new_config = self._config_dir / f"config.{profile}.yaml"
                new_config.write_text(base_config.read_text())
                logger.info(f"Created profile '{profile}' from '{base_profile}'")
            else:
                raise ValueError(f"Base profile '{base_profile}' not found")
        else:
            # Create with current configuration
            self.save_configuration(profile)

    def list_profiles(self) -> list:
        """List available configuration profiles"""
        profiles = []
        for config_file in self._config_dir.glob("config.*.yaml"):
            profile_name = config_file.stem.replace("config.", "")
            profiles.append(profile_name)
        return sorted(profiles)


# Default configuration template
DEFAULT_CONFIG = """
# Pixelated AI CLI Configuration
# This file contains configuration settings for the CLI tool

api:
  base_url: "http://localhost:8000"
  timeout: 30
  max_retries: 3
  retry_delay: 1.0
  verify_ssl: true

auth:
  jwt_token: null
  refresh_token: null
  token_expiry: null
  client_id: null
  client_secret: null
  auth_url: "http://localhost:8000/auth"

pipeline:
  default_timeout: 3600
  max_concurrent_jobs: 5
  checkpoint_interval: 300
  enable_bias_detection: true
  enable_fhe_encryption: true
  audit_logging: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: null
  max_file_size: 10485760
  backup_count: 5

security:
  encrypt_credentials: true
  key_file: ".cli_key"
  validate_inputs: true
  sanitize_outputs: true
  rate_limit: 100
"""


class ConfigProfile(BaseModel):
    """A named configuration profile with pipeline-specific settings."""

    name: str = Field(..., description="Profile name")
    api_base_url: str = Field(default="http://localhost:8000", description="API base URL")
    auth_endpoint: str = Field(default="/auth", description="Auth endpoint")
    pipeline_endpoint: str = Field(default="/api/pipelines", description="Pipeline endpoint")
    timeout: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Max retries")

    @validator("name")
    def validate_name(self, v):
        if not v or not v.strip():
            raise ValueError("Profile name cannot be empty")
        return v

    @validator("api_base_url")
    def validate_url(self, v):
        if v and not (v.startswith(("http://", "https://"))):
            raise ValueError(f"Invalid URL: {v}")
        return v

    def validate(self) -> bool:
        try:
            self.name  # triggers validators
            return True
        except Exception:
            return False


# Module-level config singleton
_config_instance: CLIConfig | None = None


def get_config(config_file=None, profile: str = "default") -> CLIConfig:
    """Get or create the global CLIConfig instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = CLIConfig(config_file=config_file, profile=profile)
    return _config_instance


def save_config(config: CLIConfig | None = None, profile: str | None = None) -> None:
    """Save the configuration to disk."""
    cfg = config or _config_instance
    if cfg is None:
        raise CLIConfigError("No configuration to save")
    cfg.save_configuration(profile=profile)


__all__ = [
    "DEFAULT_CONFIG",
    "APIConfig",
    "AuthConfig",
    "CLIConfig",
    "CLIConfigError",
    "ConfigProfile",
    "LoggingConfig",
    "PipelineConfig",
    "SecurityConfig",
    "get_config",
    "save_config",
]
