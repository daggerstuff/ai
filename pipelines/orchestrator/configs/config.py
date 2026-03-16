"""
Centralized configuration for the Pixelated Empathy AI dataset pipeline.
Provides an enterprise-grade, unified configuration management system.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataLoaderConfig:
    """Configuration for the data loader and acquisition."""
    huggingface_datasets: dict[str, str] = field(default_factory=lambda: {
        "mental_health_counseling": "Amod/mental_health_counseling_conversations",
        "psych8k": "EmoCareAI/Psych8k",
        # Mental Health Investigation Resources (Phase 1)
        "mental_health_snli": "iqrakiran/customized-mental-health-snli2",
        "mental_health_preprocessed": "typosonlr/MentalHealthPreProcessed",
        "depression_detection": "ShreyaR/DepressionDetection",
    })
    download_path: str = "ai/datasets/external"
    cache_dir: str = "ai/datasets/cache"
    huggingface_cache_dir: str = "ai/datasets/huggingface_cache"
    max_retries: int = 3

@dataclass
class StandardizationConfig:
    """Configuration for the DataStandardizer."""
    max_workers: int = 8
    batch_size: int = 200
    enable_monitoring: bool = True
    output_dir: str = "ai/datasets/standardized"

@dataclass
class LoggingConfig:
    """Configuration for the logging system."""
    log_level: str = "INFO"
    log_file: str = "logs/dataset_pipeline.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

@dataclass
class Config:
    """Root configuration class for the entire pipeline."""
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the config to a dictionary."""
        return {
            "data_loader": self.data_loader.__dict__,
            "standardization": self.standardization.__dict__,
            "logging": self.logging.__dict__,
        }

# Singleton instance to be used across the application
config = Config()

def get_config() -> Config:
    """Returns the singleton config instance."""
    return config

# Example usage:
# from config import get_config
# config = get_config()
# print(config.standardization.batch_size)
