"""Stage-based dataset orchestration for Pixelated Empathy training pipeline."""

from __future__ import annotations

from .stage_organizer import StageConfig, StageManifest, StageOrganizer

__all__ = ["StageOrganizer", "StageConfig", "StageManifest"]
