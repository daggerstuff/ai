#!/usr/bin/env python3
"""
Enterprise Configuration Management System

Provides centralized configuration management for all components with:
- Environment-based configuration
- Secrets management
- Validation and defaults
- Hot reloading capabilities
- Enterprise-grade security
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class EnterpriseConfig:
    """Enterprise configuration container."""
    
    # Core system settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Data processing settings
    batch_size: int = 1000
    max_workers: int = 4
    memory_limit_gb: int = 8
    processing_timeout: int = 3600
    
    # Quality validation settings
    quality_threshold: float = 0.7
    enable_real_validation: bool = True
    validation_sample_rate: float = 0.1
    
    # Database settings
    database_url: Optional[str] = None
    connection_pool_size: int = 10
    query_timeout: int = 30
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    
    # Security settings
    api_key_required: bool = True
    rate_limit_per_minute: int = 1000
    enable_encryption: bool = True
    
    # File paths
    data_path: str = "/home/vivi/pixelated/ai/data"
    logs_path: str = "/home/vivi/pixelated/ai/logs"
    cache_path: str = "/home/vivi/pixelated/ai/cache"
    
    # Feature flags
    enable_distributed_processing: bool = False
    enable_auto_scaling: bool = False
    enable_backup: bool = True

class EnterpriseConfigManager:
    """Manages enterprise configuration with environment overrides and validation."""
    
    def __init__(self, config_dir: str = "/home/vivi/pixelated/ai/enterprise_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.yaml"
        self.secrets_file = self.config_dir / "secrets.yaml"
        self.env_file = self.config_dir / ".env"
        
        self._config: Optional[EnterpriseConfig] = None
        self._config_hash: Optional[str] = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> EnterpriseConfig:
        """Load configuration from files and environment."""
        
        # Start with defaults
        config_dict = {}
        
        # Load from YAML config file
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                config_dict.update(file_config)
        
        # Load environment-specific overrides
        env = os.getenv('PIXELATED_ENV', 'production')
        env_config_file = self.config_dir / f"config.{env}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                config_dict.update(env_config)
        
        # Load secrets (if available)
        if self.secrets_file.exists():
            with open(self.secrets_file, 'r') as f:
                secrets = yaml.safe_load(f) or {}
                config_dict.update(secrets)
        
        # Override with environment variables
        env_overrides = self._load_env_overrides()
        config_dict.update(env_overrides)
        
        # Create config object
        self._config = EnterpriseConfig(**config_dict)
        
        # Calculate config hash for change detection
        self._config_hash = self._calculate_config_hash(config_dict)
        
        self.logger.info(f"Configuration loaded for environment: {self._config.environment}")
        return self._config
    
    def get_config(self) -> EnterpriseConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if files have changed."""
        current_hash = self._calculate_current_hash()
        if current_hash != self._config_hash:
            self.logger.info("Configuration files changed, reloading...")
            self.load_config()
            return True
        return False
    
    def save_config(self, config: EnterpriseConfig):
        """Save configuration to file."""
        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_')
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info("Configuration saved to file")
    
    def validate_config(self, config: EnterpriseConfig) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate paths exist
        for path_attr in ['data_path', 'logs_path', 'cache_path']:
            path_value = getattr(config, path_attr)
            if not Path(path_value).exists():
                try:
                    Path(path_value).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create {path_attr}: {e}")
        
        # Validate numeric ranges
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if not 0 <= config.quality_threshold <= 1:
            errors.append("quality_threshold must be between 0 and 1")
        
        if not 0 <= config.validation_sample_rate <= 1:
            errors.append("validation_sample_rate must be between 0 and 1")
        
        # Log validation results
        if errors:
            for error in errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        # Define environment variable mappings
        env_mappings = {
            'PIXELATED_ENV': 'environment',
            'PIXELATED_DEBUG': ('debug_mode', bool),
            'PIXELATED_LOG_LEVEL': 'log_level',
            'PIXELATED_BATCH_SIZE': ('batch_size', int),
            'PIXELATED_MAX_WORKERS': ('max_workers', int),
            'PIXELATED_MEMORY_LIMIT': ('memory_limit_gb', int),
            'PIXELATED_QUALITY_THRESHOLD': ('quality_threshold', float),
            'PIXELATED_DATABASE_URL': 'database_url',
            'PIXELATED_API_KEY_REQUIRED': ('api_key_required', bool),
            'PIXELATED_ENABLE_METRICS': ('enable_metrics', bool),
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_key, tuple):
                    key, type_func = config_key
                    try:
                        if type_func == bool:
                            overrides[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            overrides[key] = type_func(env_value)
                    except ValueError:
                        self.logger.warning(f"Invalid value for {env_var}: {env_value}")
                else:
                    overrides[config_key] = env_value
        
        return overrides
    
    def _calculate_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _calculate_current_hash(self) -> str:
        """Calculate hash of current configuration files."""
        combined_content = ""
        
        for file_path in [self.config_file, self.secrets_file]:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    combined_content += f.read()
        
        # Add environment variables
        env_vars = {k: v for k, v in os.environ.items() if k.startswith('PIXELATED_')}
        combined_content += json.dumps(env_vars, sort_keys=True)
        
        return hashlib.md5(combined_content.encode()).hexdigest()
    
    def create_default_configs(self):
        """Create default configuration files."""
        
        # Create main config file
        default_config = {
            'environment': 'production',
            'debug_mode': False,
            'log_level': 'INFO',
            'batch_size': 1000,
            'max_workers': 4,
            'memory_limit_gb': 8,
            'quality_threshold': 0.7,
            'enable_real_validation': True,
            'enable_metrics': True,
            'enable_backup': True
        }
        
        if not self.config_file.exists():
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        # Create development config
        dev_config_file = self.config_dir / "config.development.yaml"
        if not dev_config_file.exists():
            dev_config = {
                'environment': 'development',
                'debug_mode': True,
                'log_level': 'DEBUG',
                'batch_size': 100,
                'validation_sample_rate': 1.0
            }
            with open(dev_config_file, 'w') as f:
                yaml.dump(dev_config, f, default_flow_style=False, indent=2)
        
        # Create environment file template
        if not self.env_file.exists():
            env_template = """# Pixelated AI Enterprise Configuration
# Copy this file to .env and customize for your environment

# Environment (production, development, testing)
PIXELATED_ENV=production

# Debug mode (true/false)
PIXELATED_DEBUG=false

# Logging level (DEBUG, INFO, WARNING, ERROR)
PIXELATED_LOG_LEVEL=INFO

# Processing settings
PIXELATED_BATCH_SIZE=1000
PIXELATED_MAX_WORKERS=4
PIXELATED_MEMORY_LIMIT=8

# Quality settings
PIXELATED_QUALITY_THRESHOLD=0.7

# Database (optional)
# PIXELATED_DATABASE_URL=postgresql://user:pass@localhost/pixelated

# Security
PIXELATED_API_KEY_REQUIRED=true

# Monitoring
PIXELATED_ENABLE_METRICS=true
"""
            with open(self.env_file, 'w') as f:
                f.write(env_template)
        
        self.logger.info("Default configuration files created")

# Global configuration manager instance
_config_manager = None

def get_config() -> EnterpriseConfig:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()
        _config_manager.create_default_configs()
    
    return _config_manager.get_config()

def reload_config() -> EnterpriseConfig:
    """Reload configuration from files."""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()
    
    return _config_manager.load_config()

if __name__ == "__main__":
    # Test the configuration system
    manager = EnterpriseConfigManager()
    manager.create_default_configs()
    
    config = manager.load_config()
    print(f"✅ Configuration loaded for environment: {config.environment}")
    print(f"   Debug mode: {config.debug_mode}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Quality threshold: {config.quality_threshold}")
    
    if manager.validate_config(config):
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
