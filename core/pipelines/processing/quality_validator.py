"""Processing quality validator compatibility shim with production defaults."""

from __future__ import annotations

from typing import Any

from ai.core.pipelines.quality.quality_validator import QualityResult, QualityValidator as _CoreQualityValidator


class QualityValidator(_CoreQualityValidator):
    """Adapter around the canonical :mod:`ai.core.pipelines.quality.quality_validator`."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Keep compatibility with legacy constructor style that accepted no arguments.
        super().__init__(config=config)


__all__ = ["QualityResult", "QualityValidator"]
