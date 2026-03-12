"""
Embedding Agent API - Vector embedding service for clinical knowledge.

This module provides a FastAPI-based embedding service that wraps the
ClinicalKnowledgeEmbedder for text-to-vector conversion and similarity search.
"""

from .app import create_app, embedding_router
from .models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingAgentConfig,
    EmbeddingAgentStatus,
    EmbeddingRequest,
    EmbeddingResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from .service import EmbeddingAgentService

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
