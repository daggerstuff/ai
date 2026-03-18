"""
NVIDIA Nemotron RAG Pipeline for Pixelated Empathy.

Provides retrieval-augmented generation capabilities for therapeutic
knowledge management using NVIDIA NIM inference services.

Components:
- nemotron_rag: Core RAG pipeline with Nemotron models
"""

from .nemotron_rag import (
    DocumentMetadata,
    NemotronRAGConfig,
    RAGResponse,
    TherapeuticRAGPipeline,
    create_rag_pipeline,
)

__all__ = [
    "NemotronRAGConfig",
    "TherapeuticRAGPipeline",
    "DocumentMetadata",
    "RAGResponse",
    "create_rag_pipeline",
]
