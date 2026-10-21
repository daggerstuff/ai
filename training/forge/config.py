"""Configuration definitions and environment resolution for CoreWeave Forge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ForgeConfig:
    """Central configuration for CoreWeave Forge services."""

    # API Base URLs
    training_api_base: str = field(
        default_factory=lambda: os.getenv(
            "COREWEAVE_FORGE_TRAINING_URL",
            os.getenv("WANDB_TRAINING_URL", "https://api.training.wandb.ai/v1"),
        )
    )
    inference_api_base: str = field(
        default_factory=lambda: os.getenv(
            "COREWEAVE_FORGE_INFERENCE_URL",
            os.getenv("WANDB_INFERENCE_URL", "https://api.inference.wandb.ai/v1"),
        )
    )
    hivemind_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "HIVEMIND_SERVER_ENDPOINT", "https://hivemind.wandb.tools"
        )
    )

    # Scopes
    entity: str = field(
        default_factory=lambda: os.getenv(
            "COREWEAVE_ENTITY", os.getenv("WANDB_ENTITY", "wutang")
        )
    )
    project: str = field(
        default_factory=lambda: os.getenv(
            "COREWEAVE_PROJECT",
            os.getenv("WANDB_PROJECT", "pixelated-empathy-kan28"),
        )
    )

    # API Keys & Secrets
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "COREWEAVE_FORGE_API_KEY", os.getenv("WANDB_API_KEY", "")
        )
    )

    # Storage and Quota Management (GB)
    storage_quota_gb: float = 200.0
    storage_warn_threshold_pct: float = 85.0

    # Paths
    local_checkpoint_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "output"
        / "wandb_local_checkpoints"
    )

    # Model Catalog Defaults
    default_base_model: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
    approved_models: frozenset[str] = frozenset(
        {
            "google/gemma-4-26B-A4B-it",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        }
    )

    def is_artifact_uri(self, model_identifier: str) -> bool:
        """Return True if model_identifier is a wandb/forge artifact URI."""
        return model_identifier.startswith("wandb-artifact:///")

    def resolve_inference_endpoint(self, model_identifier: str) -> str:
        """Resolve the correct inference endpoint for a given model.

        Trained checkpoint artifacts must be routed to the Post-Training API completions endpoint,
        while foundation base models are routed to Serverless Inference.
        """
        if self.is_artifact_uri(model_identifier):
            return f"{self.training_api_base}/chat/completions"
        return f"{self.inference_api_base}/chat/completions"
