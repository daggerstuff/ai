from training.configs.stage_configs.storage_config import (
    StorageBackend,
    StorageConfig,
    get_storage_config,
    set_storage_config,
)


def test_set_storage_config():
    """Test setting the global storage config."""
    # Save the original config to avoid side effects in other tests
    original_config = get_storage_config()

    try:
        # Create a new distinct config
        new_config = StorageConfig(backend=StorageBackend.S3, s3_bucket="test-bucket")

        # Set the new config
        set_storage_config(new_config)

        # Verify the global config has been mutated
        assert get_storage_config() is new_config
        assert get_storage_config().backend == StorageBackend.S3
        assert get_storage_config().s3_bucket == "test-bucket"
    finally:
        # Restore the original config
        set_storage_config(original_config)
