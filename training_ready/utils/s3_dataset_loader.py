"""
Back-compat shim for ai.training_ready.utils.s3_dataset_loader

This module now lives at ai.utils.s3_dataset_loader. This shim re-exports the
public API to avoid breaking existing imports. Please migrate imports to:

    from ai.utils.s3_dataset_loader import S3DatasetLoader, get_s3_dataset_path, load_dataset_from_s3
"""
from ai.utils.s3_dataset_loader import *  # noqa: F401,F403

# Explicit export list to aid static analyzers
from ai.utils.s3_dataset_loader import (
    S3DatasetLoader,
    get_s3_dataset_path,
    load_dataset_from_s3,
)

__all__ = [
    "S3DatasetLoader",
    "get_s3_dataset_path",
    "load_dataset_from_s3",
]
