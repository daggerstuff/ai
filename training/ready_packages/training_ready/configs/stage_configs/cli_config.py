"""
Configuration management for CLI.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore


class ConfigManager:
    """Manages configuration for the research system."""

    DEFAULT_CONFIG_PATH = Path.home() / ".journal_research" / "config.yaml"
    DEFAULT_CONFIG = {
        "orchestrator": {
            "max_retries": 3,
            "retry_delay_seconds": 1.0,
            "progress_history_limit": 100,
            "parallel_evaluation": False,
            "parallel_integration_planning": False,
            "max_workers": 4,
            "session_storage_path": None,
            "visualization_max_points": 100,
            "fallback_on_failure": True,
        },
        "discovery": {
            "pubmed": {
                "api_key": None,
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                "search_limit": 100,
            },
            "doaj": {
                "base_url": "https://doaj.org/api/v2",
            },
            "repositories": {
                "dryad": {"base_url": "https://datadryad.org/api/v2"},
                "zenodo": {"base_url": "https://zenodo.org/api"},
                "clinical_trials": {"base_url": "https://clinicaltrials.gov/api/v2"},
            },
        },
        "evaluation": {
            "therapeutic_relevance_weight": 0.35,
            "data_structure_quality_weight": 0.25,
            "training_integration_weight": 0.20,
            "ethical_accessibility_weight": 0.20,
            "high_priority_threshold": 7.5,
            "medium_priority_threshold": 5.0,
        },
        "acquisition": {
            "storage_base_path": "data/acquired_datasets",
            "encryption_enabled": False,
            "download_timeout": 3600,
            "max_retries": 3,
            "chunk_size": 8192,
            "resume_downloads": True,
        },
        "integration": {
            "target_format": "chatml",
            "default_complexity": "medium",
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }

    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        """Initialize config manager with optional config path."""
        # Convert string to Path if needed
        if config_path is not None and isinstance(config_path, str):
            config_path = Path(config_path)
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        # Only create parent directory if it's writable
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # If we can't create the directory, that's okay - we'll handle it in load/save
            pass

    def load(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    if YAML_AVAILABLE and self.config_path.suffix in (".yaml", ".yml"):
                        assert yaml is not None  # Type guard for type checker
                        config = yaml.safe_load(f) or {}
                    else:
                        # Fall back to JSON
                        config = json.load(f) or {}
                # Merge with defaults to ensure all keys exist
                merged = self._merge_config(self.DEFAULT_CONFIG, config)
                return self._apply_legacy_aliases(merged)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                return self._apply_legacy_aliases(deepcopy(self.DEFAULT_CONFIG))
        return self._apply_legacy_aliases(deepcopy(self.DEFAULT_CONFIG))

    def save(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            if YAML_AVAILABLE and self.config_path.suffix in (".yaml", ".yml"):
                assert yaml is not None  # Type guard for type checker
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                # Fall back to JSON
                json.dump(config, f, indent=2)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path."""
        config = self.load()
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value by dot-separated path."""
        config = self.load()
        keys = key_path.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        self.save(config)

    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config into default config."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_legacy_aliases(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure legacy top-level aliases exist for backward compatibility."""
        # Maintain top-level storage_base_path alias
        acquisition_config = config.get("acquisition", {})
        storage_base_path = acquisition_config.get("storage_base_path")
        if storage_base_path and "storage_base_path" not in config:
            config["storage_base_path"] = storage_base_path

        # Maintain top-level logging directory alias
        logging_config = config.get("logging", {})
        log_file = logging_config.get("file")
        if log_file and "log_file" not in config:
            config["log_file"] = log_file

        return config

    def load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        env_prefix = "JOURNAL_RESEARCH_"

        # Map environment variables to config paths
        env_mappings = {
            "PUBMED_API_KEY": "discovery.pubmed.api_key",
            "STORAGE_PATH": "acquisition.storage_base_path",
            "LOG_LEVEL": "logging.level",
            "MAX_RETRIES": "orchestrator.max_retries",
            "MAX_WORKERS": "orchestrator.max_workers",
        }

        for env_var, config_path in env_mappings.items():
            env_key = env_prefix + env_var
            if env_key in os.environ:
                overrides[config_path] = os.environ[env_key]

        return overrides

    def apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        overrides = self.load_env_overrides()
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            target[keys[-1]] = value
        return config


# Global config manager instance
_config_manager = ConfigManager()


def load_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Load configuration with environment overrides."""
    # Convert string to Path if needed
    if config_path is not None and isinstance(config_path, str):
        config_path = Path(config_path)
    manager = ConfigManager(config_path) if config_path else _config_manager
    config = manager.load()
    config = manager.apply_env_overrides(config)
    return config


def save_config(config: Dict[str, Any], config_path: Optional[Union[Path, str]] = None) -> None:
    """Save configuration to file."""
    # Convert string to Path if needed
    if config_path is not None and isinstance(config_path, str):
        config_path = Path(config_path)
    manager = ConfigManager(config_path) if config_path else _config_manager
    manager.save(config)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by dot-separated path."""
    return _config_manager.get(key_path, default)

