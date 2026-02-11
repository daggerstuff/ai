"""
Embedding Agent API - Vector embedding service for clinical knowledge.

This module provides a FastAPI-based embedding service that wraps the
ClinicalKnowledgeEmbedder for text-to-vector conversion and similarity search.
"""

from .service import EmbeddingAgentService
from .models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    EmbeddingAgentConfig,
    EmbeddingAgentStatus,
)
from .app import create_app, embedding_router

__all__ = [
    "EmbeddingAgentService",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "BatchEmbeddingRequest",
    "BatchEmbeddingResponse",
    "SimilaritySearchRequest",
    "SimilaritySearchResponse",
    "EmbeddingAgentConfig",
    "EmbeddingAgentStatus",
    "create_app",
    "embedding_router",
]

