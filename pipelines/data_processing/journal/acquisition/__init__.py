"""
Access & Acquisition Manager

Handle dataset access requests and secure acquisition.
"""

from ai.pipelines.data_processing.journal.acquisition.acquisition_manager import (
    AccessAcquisitionManager,
    AcquisitionConfig,
    DownloadProgress,
)

__all__ = [
    "AccessAcquisitionManager",
    "AcquisitionConfig",
    "DownloadProgress",
]
