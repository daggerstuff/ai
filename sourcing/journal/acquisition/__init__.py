"""
Access & Acquisition Manager

Handle dataset access requests and secure acquisition.
"""

from ai.sourcing.journal.acquisition.acquisition_manager import (
    AccessAcquisitionManager,
    AcquisitionConfig,
    DownloadProgress,
)

__all__ = [
    "AccessAcquisitionManager",
    "AcquisitionConfig",
    "DownloadProgress",
]
