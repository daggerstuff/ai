"""
Back-compat shim for ai.training.ready_packages.utils.ngc_resources

This module now lives at ai.utils.ngc_resources. This shim re-exports the public API.
Migrate imports to:

    from ai.utils.ngc_resources import NGCResourceDownloader, download_nemo_quickstart
"""
from ai.utils.ngc_resources import *  # noqa: F401,F403

from ai.utils.ngc_resources import (
    NGCResourceDownloader,
    download_nemo_quickstart,
)

__all__ = [
    "NGCResourceDownloader",
    "download_nemo_quickstart",
]
