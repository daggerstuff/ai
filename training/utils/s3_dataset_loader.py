"""Back-compat shim for ``ai.training.utils.s3_dataset_loader``.

The canonical loader lives at ``ai.tools.utilities.core.utils.s3_dataset_loader``,
whose ``S3DatasetLoader`` methods take a ``(bucket, key)`` pair. Training scripts
call ``load_json(s3_path)`` and ``stream_jsonl(s3_path)`` with a single
``s3://bucket/key`` (or local path) argument. This module adapts that older
single-arg call shape to the canonical loader, and re-exports the module-level
``get_s3_dataset_path`` / ``load_dataset_from_s3`` helpers unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ai.tools.utilities.core.utils.s3_dataset_loader import (
    S3DatasetLoader as _CanonicalS3DatasetLoader,
    get_s3_dataset_path,
    load_dataset_from_s3,
)


class S3DatasetLoader(_CanonicalS3DatasetLoader):
    """Canonical loader with single-``s3://path`` convenience methods."""

    def load_json(self, s3_path: str) -> Any:
        """Load a JSON object from an ``s3://bucket/key`` or local path."""
        return super().load_json("", s3_path)

    def stream_jsonl(self, s3_path: str) -> Iterator[dict[str, Any]]:
        """Stream JSONL records from an ``s3://bucket/key`` or local path."""
        return super().stream_jsonl("", s3_path)


__all__ = [
    "S3DatasetLoader",
    "get_s3_dataset_path",
    "load_dataset_from_s3",
]