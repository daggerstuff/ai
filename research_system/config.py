"""
Configuration management for the journal dataset research system.

Loads configuration from YAML file with environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration loader with environment variable support."""
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        self._config = self._substitute_env_vars(raw_config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            obj: Configuration object (dict, list, str, or other)
        
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj
    
    def _substitute_env_var_string(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: String potentially containing ${VAR_NAME} or ${VAR_NAME:default}
        
        Returns:
            String with environment variables substituted
        """
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)
        
        return re.sub(pattern, replace_var, value)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "api_endpoints.pubmed.base_url")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Examples:
            >>> config.get("api_endpoints.pubmed.base_url")
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            >>> config.get("storage.logs")
            "ai/research_system/logs"
        """
        keys = key_path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_api_endpoint(self, service: str) -> dict[str, str]:
        """
        Get API endpoint configuration for a service.
        
        Args:
            service: Service name (pubmed, doaj, dryad, zenodo, clinicaltrials)
        
        Returns:
            Dictionary with endpoint configuration
        """
        return self.get(f"api_endpoints.{service}", {})
    
    def get_search_keywords(self, dataset_type: str) -> list[str]:
        """
        Get search keywords for a dataset type.
        
        Args:
            dataset_type: Type of dataset (therapy_transcripts, clinical_outcomes, etc.)
        
        Returns:
            List of search keywords
        """
        return self.get(f"search_keywords.{dataset_type}", [])
    
    def get_all_search_keywords(self) -> dict[str, list[str]]:
        """
        Get all search keywords organized by dataset type.
        
        Returns:
            Dictionary mapping dataset types to keyword lists
        """
        return self.get("search_keywords", {})
    
    def get_mesh_terms(self) -> list[str]:
        """
        Get MeSH terms for PubMed searches.
        
        Returns:
            List of MeSH terms
        """
        return self.get("mesh_terms", [])
    
    def get_storage_path(self, path_type: str) -> Path:
        """
        Get storage path for a specific type.
        
        Args:
            path_type: Type of storage (acquired_datasets, logs, reports, temp_downloads)
        
        Returns:
            Path object for the storage location
        """
        path_str = self.get(f"storage.{path_type}", "")
        return Path(path_str) if path_str else Path(".")
    
    def get_weekly_targets(self, week: int) -> dict[str, int]:
        """
        Get weekly targets for a specific week.
        
        Args:
            week: Week number (1, 2, etc.)
        
        Returns:
            Dictionary of targets for the week
        """
        return self.get(f"targets.week_{week}", {})
    
    def get_evaluation_weights(self) -> dict[str, float]:
        """
        Get evaluation dimension weights.
        
        Returns:
            Dictionary mapping dimension names to weights
        """
        return self.get("evaluation_weights", {})
    
    def get_priority_thresholds(self) -> dict[str, float]:
        """
        Get priority tier thresholds.
        
        Returns:
            Dictionary mapping tier names to score thresholds
        """
        return self.get("priority_thresholds", {})
    
    def get_rate_limit(self, service: str) -> float:
        """
        Get rate limit for a service in requests per second.
        
        Args:
            service: Service name
        
        Returns:
            Requests per second limit
        """
        key = f"rate_limits.{service}_requests_per_second"
        return self.get(key, 1.0)
    
    def get_retry_config(self) -> dict[str, int]:
        """
        Get retry configuration.
        
        Returns:
            Dictionary with retry settings
        """
        return self.get("retry", {})
    
    def get_download_config(self) -> dict[str, Any]:
        """
        Get download configuration.
        
        Returns:
            Dictionary with download settings
        """
        return self.get("download", {})
    
    def get_security_config(self) -> dict[str, Any]:
        """
        Get security configuration.
        
        Returns:
            Dictionary with security settings
        """
        return self.get("security", {})
    
    def get_logging_config(self) -> dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dictionary with logging settings
        """
        return self.get("logging", {})
    
    def ensure_storage_paths(self) -> None:
        """Create all configured storage directories if they don't exist."""
        storage_types = ["acquired_datasets", "logs", "reports", "temp_downloads"]
        
        for storage_type in storage_types:
            path = self.get_storage_path(storage_type)
            path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config_instance: Config | None = None


def get_config(config_path: str | None = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file. Only used on first call.
    
    Returns:
        Global Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: str | None = None) -> Config:
    """
    Reload configuration from file.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Reloaded Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
