"""Compatibility shim for legacy imports in ``ai.core.sourcing.youtube.processor``."""

from __future__ import annotations

import os
from typing import Any


def _load_core_processor():
    """Import the actual processor module lazily to avoid hard dependency errors."""
    from ai.sourcing.youtube.models import (
        ChannelHunterConfig,
        ChannelQualityThresholds,
    )
    from ai.sourcing.youtube.processor import (
        ChannelProcessor,
        run_pipeline,
    )

    return ChannelProcessor, ChannelQualityThresholds, ChannelHunterConfig, run_pipeline


def _require_dependency() -> tuple[type, type, type, Any]:
    try:
        return _load_core_processor()
    except Exception as exc:
        raise RuntimeError(
            "ai.sourcing.youtube.processor is unavailable because optional dependencies are missing. "
            "Install project requirements to use this pipeline."
        ) from exc


class Processor:
    """Backward-compatible processor alias for the package-level YouTube processor."""

    _processor: Any

    def __init__(self, api_key: str | None = None, hunter_config: Any = None, quality_thresholds: Any = None, **kwargs):
        """Initialize the compatibility wrapper.

        The wrapper forwards calls to the concrete ``ChannelProcessor`` once its optional
        dependencies are available.
        """
        # Preserve permissive behavior for callers constructing with no args.
        if api_key is None:
            api_key = os.environ.get("YOUTUBE_API_KEY") or os.environ.get("YOUTUBE_DATA_API_KEY")
        if not api_key:
            raise ValueError("Missing YouTube API key. Set YOUTUBE_API_KEY or pass api_key explicitly.")
        processor_cls, quality_thresholds_cls, hunter_config_cls, _ = _require_dependency()
        if hunter_config is not None and not isinstance(hunter_config, hunter_config_cls):
            hunter_config = None
        if quality_thresholds is not None and not isinstance(quality_thresholds, quality_thresholds_cls):
            quality_thresholds = None
        self._processor = processor_cls(api_key, hunter_config=hunter_config, quality_thresholds=quality_thresholds)
        self._extra_kwargs = kwargs

    def __getattr__(self, item: str) -> Any:
        return getattr(self._processor, item)

    def run_discovery(self, *args, **kwargs):
        return self._processor.run_discovery(*args, **kwargs)

    def generate_report(self, *args, **kwargs):
        return self._processor.generate_report(*args, **kwargs)

    def export_channels(self, *args, **kwargs):
        return self._processor.export_channels(*args, **kwargs)


def run_pipeline(api_key: str, *args, **kwargs):
    """Compatibility wrapper around :func:`ai.sourcing.youtube.processor.run_pipeline`."""
    _, _, _, run_pipeline_impl = _require_dependency()
    return run_pipeline_impl(api_key, *args, **kwargs)


__all__ = ["Processor", "run_pipeline"]
