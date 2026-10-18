"""Compatibility utilities for NGC resource downloads.

This module intentionally mirrors the project-level :mod:`ai.tools.utilities.ngc_resources`
interface while remaining import-safe in environments where the NGC CLI is
unavailable.
"""

from __future__ import annotations

from pathlib import Path

from ai.tools.utilities.ngc_resources import (
    NGCResourceDownloader as _ResolvedNGCResourceDownloader,
    download_nemo_quickstart as _resolved_download_nemo_quickstart,
)


class NGCResourceDownloader(_ResolvedNGCResourceDownloader):
    """Compatibility subclass that preserves the historical symbol location."""

    def download(self, *args, **kwargs):
        """Backward-compatible alias for quickstart downloads.

        The historical project code sometimes calls ``download(...)`` without the
        strict ``download_nemo_quickstart(...)`` naming. We support that path as
        a best-effort no-op shim that forwards to
        ``download_nemo_quickstart``.
        """
        version = kwargs.pop("version", None)
        output_dir = kwargs.pop("output_dir", None)
        if output_dir is not None and isinstance(output_dir, (str, Path)):
            output_dir = Path(output_dir)
        return self.download_nemo_quickstart(version=version, output_dir=output_dir)

    def download_nemo_quickstart(self, version: str | None = None, output_dir: Path | None = None):
        return super().download_nemo_quickstart(version=version, output_dir=output_dir)


class NgcResources:
    """Facade for historical entrypoints.

    This class intentionally keeps a very small API; it is used in a few legacy
    scripts that only require a convenience object with helper methods.
    """

    def __init__(self, output_base: Path | None = None, api_key: str | None = None):
        self._downloader = NGCResourceDownloader(
            output_base=output_base,
            api_key=api_key,
        )

    def download_nemo_quickstart(self, version: str | None = None, output_dir: Path | None = None):
        return self._downloader.download_nemo_quickstart(version=version, output_dir=output_dir)


def download_nemo_quickstart(version: str | None = None, output_dir: Path | None = None):
    """Compatibility function forwarding to the concrete implementation."""
    return _resolved_download_nemo_quickstart(version=version, output_dir=output_dir)


__all__ = ["NGCResourceDownloader", "NgcResources", "download_nemo_quickstart"]
