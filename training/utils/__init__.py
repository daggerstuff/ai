"""
Training Ready Utilities
"""

# Back-compat shim: re-export from consolidated ai.utils package
from ai.core.utils.ngc_resources import NGCResourceDownloader, download_nemo_quickstart
from ai.core.utils.s3_dataset_loader import (
    S3DatasetLoader,
    get_s3_dataset_path,
    load_dataset_from_s3,
)

__all__ = [
    "NGCResourceDownloader",
    "S3DatasetLoader",
    "download_nemo_quickstart",
    "get_s3_dataset_path",
    "load_dataset_from_s3",
]
