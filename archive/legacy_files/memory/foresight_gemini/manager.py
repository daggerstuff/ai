from __future__ import annotations

"""
Gemini-oriented memory manager backed by the shared local memory service.

This module preserves the historical import path while removing the old cloud
split. Gemini-specific code now writes into the same local
SQLite-backed service used by the rest of the repository.
"""

from typing import Any

from pydantic import BaseModel, Field

from ai.memory.local_foresight_manager import LocalForesightMemoryManager


class GeminiForesightConfig(BaseModel):
    """Configuration for Gemini integrations using local shared memory."""

    gemini_api_key: str = Field(..., description="Gemini/Google API key")
    model_name: str = Field("gemini-1.5-pro", description="Gemini model to use")
    user_id: str = Field("default_user", description="Default user ID for memory")
    db_path: str = Field(..., description="Path to the shared local memory database")
    bank_id: str = Field("pixelated", description="Shared memory bank identifier")
    foresight_api_key: str | None = Field(
        default=None,
        description="Deprecated; local shared memory is the only supported backend",
    )
    memory_config: dict[str, Any] | None = Field(
        default=None,
        description="Deprecated; local shared memory is the only supported backend",
    )


class GeminiForesightManager(LocalForesightMemoryManager):
    """
    Backward-compatible Gemini manager using the local shared memory service.

    The Gemini integration keeps its provider/model metadata, but all durable
    memory goes through the repository's single supported backend.
    """

    def __init__(self, config: GeminiForesightConfig):
        self.config = config
        super().__init__(db_path=config.db_path, bank_id=config.bank_id)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
    ) -> str:
        merged = self._metadata_dict(metadata)
        merged.setdefault("provider", "gemini")
        merged.setdefault("model_name", self.config.model_name)
        return super().add_memory(
            content=content,
            user_id=user_id,
            metadata=merged,
            category=category,
        )

    def build_provider_metadata(self) -> dict[str, str]:
        return {
            "provider": "gemini",
            "model_name": self.config.model_name,
        }
