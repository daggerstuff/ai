"""
Back-compat shim for ai.training.utils.s3_dataset_loader

This module now lives at ai.tools.utilities.s3_dataset_loader. This shim re-exports the
public API to avoid breaking existing imports.
"""

from ai.tools.utilities.s3_dataset_loader import S3DatasetLoader

__all__ = [
    "S3DatasetLoader",
]
