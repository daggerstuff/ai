"""
Tests for the centralized configuration system.
"""

from ai.dataset_pipeline.config import (
    Config,
    DataLoaderConfig,
    LoggingConfig,
    StandardizationConfig,
    get_config,
)


class TestConfigClasses:
    """Test cases for configuration dataclasses."""

    def test_data_loader_config_defaults(self):
        """Test DataLoaderConfig default values."""
        config = DataLoaderConfig()

        assert "mental_health_counseling" in config.huggingface_datasets
        assert "psych8k" in config.huggingface_datasets
        assert config.download_path == "ai/datasets/external"
        assert config.cache_dir == "ai/datasets/cache"
        assert config.max_retries == 3

    def test_data_loader_config_custom(self):
        """Test DataLoaderConfig with custom values."""
        custom_datasets = {"test_dataset": "test/path"}
        config = DataLoaderConfig(
            huggingface_datasets=custom_datasets,
            download_path="custom/path",
            cache_dir="custom/cache",
            max_retries=5
        )

        assert config.huggingface_datasets == custom_datasets
        assert config.download_path == "custom/path"
        assert config.cache_dir == "custom/cache"
        assert config.max_retries == 5

    def test_standardization_config_defaults(self):
        """Test StandardizationConfig default values."""
        config = StandardizationConfig()

        assert config.max_workers == 8
        assert config.batch_size == 200
        assert config.enable_monitoring is True
        assert config.output_dir == "ai/datasets/standardized"

    def test_standardization_config_custom(self):
        """Test StandardizationConfig with custom values."""
        config = StandardizationConfig(
            max_workers=16,
            batch_size=500,
            enable_monitoring=False,
            output_dir="custom/output"
        )

        assert config.max_workers == 16
        assert config.batch_size == 500
        assert config.enable_monitoring is False
        assert config.output_dir == "custom/output"

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()

        assert config.log_level == "INFO"
        assert config.log_file == "logs/dataset_pipeline.log"
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5

    def test_logging_config_custom(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            log_level="DEBUG",
            log_file="custom/logs.log",
            max_bytes=5 * 1024 * 1024,
            backup_count=3
        )

        assert config.log_level == "DEBUG"
        assert config.log_file == "custom/logs.log"
        assert config.max_bytes == 5 * 1024 * 1024
        assert config.backup_count == 3


class TestRootConfig:
    """Test cases for the root Config class."""

    def test_config_defaults(self):
        """Test Config default values."""
        config = Config()

        assert isinstance(config.data_loader, DataLoaderConfig)
        assert isinstance(config.standardization, StandardizationConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_config_to_dict(self):
        """Test Config serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert "data_loader" in config_dict
        assert "standardization" in config_dict
        assert "logging" in config_dict
        assert isinstance(config_dict["data_loader"], dict)
        assert isinstance(config_dict["standardization"], dict)
        assert isinstance(config_dict["logging"], dict)

    def test_get_config_singleton(self):
        """Test that get_config returns a singleton instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2
        assert isinstance(config1, Config)
