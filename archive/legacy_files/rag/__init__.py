"""Pixelated empathy RAG package exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nemotron_rag import (
        DocumentMetadata,
        NemotronRAGConfig,
        RAGResponse,
        TherapeuticRAGPipeline,
        create_rag_pipeline,
    )


__all__ = (
    "DocumentMetadata",
    "NemotronRAGConfig",
    "RAGResponse",
    "TherapeuticRAGPipeline",
    "create_rag_pipeline",
)


def __getattr__(name: str):
    """Lazily expose RAG symbols from :mod:`ai.rag.nemotron_rag`.

    This keeps `import ai.rag` cheap and avoids importing optional heavy
    dependencies (such as FAISS and native extensions) during test collection.
    """
    if name in __all__:
        module = import_module("ai.rag.nemotron_rag")
        return getattr(module, name)
    raise AttributeError(f"module 'ai.rag' has no attribute {name!r}")


def __dir__():
    return list(__all__)
