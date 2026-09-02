"""
Back-compat shim for ai.training.utils.ngc_resources

This module now lives at ai.tools.utilities.ngc_resources. This shim re-exports the public API.
Migrate imports to:

    from ai.tools.utilities.utils.ngc_resources import NGCResourceDownloader, download_nemo_quickstart
"""
from ai.tools.utilities.utils.ngc_resources import *  # noqa: F401,F403

from ai.tools.utilities.utils.ngc_resources import (
    NGCResourceDownloader,
    download_nemo_quickstart,
)

__all__ = [
    "NGCResourceDownloader",
    "download_nemo_quickstart",
]
