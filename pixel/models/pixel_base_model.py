"""Minimal stub for the removed legacy PixelBaseModel.

The original ``PixelBaseModel`` (an enhanced Qwen3-30B with EQ/persona/clinical
heads) was removed as legacy. This stub preserves the small API surface used by
``api/pixel_inference_service.py`` so the service remains importable; it performs
no real inference.
"""

from typing import Any

from torch import nn


class PixelBaseModel(nn.Module):
    """No-op placeholder for the removed Pixel base model."""

    def __init__(self, qwen3_config: dict[str, Any] | None = None):
        super().__init__()
        self.qwen3_config = qwen3_config or {}

    @classmethod
    def load(cls, path: str, qwen3_config: dict[str, Any] | None = None) -> "PixelBaseModel":
        """Return a fresh model instance (loading is unsupported in the stub)."""
        return cls(qwen3_config)

    def save(self, path: str) -> None:
        """No-op save (unsupported in the stub)."""

    def forward(self, x: Any, history: Any = None) -> dict[str, Any]:
        """Return an empty EQ-outputs dict (no real inference)."""
        return {"eq_outputs": {}}
