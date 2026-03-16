"""
Pydantic models for the Embedding Agent API.

Defines request/response schemas for embedding operations including:
- Single text embedding
- Batch text embedding
- Similarity search
- Agent status and configuration
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    MINILM_L6_V2 = "all-MiniLM-L6-v2"
    MINILM_L12_V2 = "all-MiniLM-L12-v2"
    MPNET_BASE_V2 = "all-mpnet-base-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"


class KnowledgeType(str, Enum):
    """Types of knowledge for categorization."""

    DSM5 = "dsm5"
    PDM2 = "pdm2"
    CLINICAL = "clinical"
    THERAPEUTIC_TECHNIQUE = "therapeutic_technique"
    THERAPEUTIC_CONVERSATION = "therapeutic_conversation"
    GENERAL = "general"


class EmbeddingAgentConfig(BaseModel):
    """Configuration for the embedding agent."""

    model_name: EmbeddingModel = Field(
        default=EmbeddingModel.MINILM_L6_V2,
        description="The embedding model to use"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of the embedding vectors"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation"
    )
    max_text_length: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Maximum text length to process"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache generated embeddings"
    )
    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU for embedding generation"
    )


class EmbeddingRequest(BaseModel):
    """Request model for single text embedding."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to embed"
    )
    knowledge_type: Optional[KnowledgeType] = Field(
        default=KnowledgeType.GENERAL,
        description="Type of knowledge for categorization"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata to associate with the embedding"
    )
    model: Optional[EmbeddingModel] = Field(
        default=None,
        description="Override the default embedding model"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class EmbeddingResponse(BaseModel):
    """Response model for single text embedding."""

    embedding: List[float] = Field(
        ...,
        description="The embedding vector"
    )
    embedding_id: str = Field(
        ...,
        description="Unique identifier for this embedding"
    )
    model_used: str = Field(
        ...,
        description="The model used to generate the embedding"
    )
    dimension: int = Field(
        ...,
        description="Dimension of the embedding vector"
    )
    text_hash: str = Field(
        ...,
        description="Hash of the input text for caching"
    )
    cached: bool = Field(
        default=False,
        description="Whether the embedding was retrieved from cache"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate the embedding in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of embedding creation"
    )


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to embed"
    )
    knowledge_types: Optional[List[KnowledgeType]] = Field(
        default=None,
        description="Knowledge types for each text (must match texts length)"
    )
    metadata_list: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata for each text (must match texts length)"
    )
    model: Optional[EmbeddingModel] = Field(
        default=None,
        description="Override the default embedding model"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Ensure all texts are valid."""
        validated = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
            validated.append(text.strip())
        return validated

    @field_validator("knowledge_types")
    @classmethod
    def validate_knowledge_types(
        cls, v: Optional[List[KnowledgeType]], info
    ) -> Optional[List[KnowledgeType]]:
        """Ensure knowledge_types matches texts length."""
        if v is not None and "texts" in info.data:
            if len(v) != len(info.data["texts"]):
                raise ValueError(
                    "knowledge_types length must match texts length"
                )
        return v


class BatchEmbeddingItem(BaseModel):
    """Individual item in batch embedding response."""

    index: int = Field(..., description="Index in the original request")
    embedding: List[float] = Field(..., description="The embedding vector")
    embedding_id: str = Field(..., description="Unique identifier")
    text_hash: str = Field(..., description="Hash of the input text")
    cached: bool = Field(default=False, description="Whether from cache")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding."""

    embeddings: List[BatchEmbeddingItem] = Field(
        ...,
        description="List of embedding results"
    )
    total_count: int = Field(..., description="Total number of embeddings")
    cached_count: int = Field(
        default=0,
        description="Number of embeddings retrieved from cache"
    )
    generated_count: int = Field(
        default=0,
        description="Number of newly generated embeddings"
    )
    model_used: str = Field(..., description="The model used")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Query text to search for similar items"
    )
    query_embedding: Optional[List[float]] = Field(
        default=None,
        description="Pre-computed query embedding (optional)"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar items to return"
    )
    knowledge_types: Optional[List[KnowledgeType]] = Field(
        default=None,
        description="Filter by knowledge types"
    )
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results"
    )


class SimilarityMatch(BaseModel):
    """Individual similarity search result."""

    item_id: str = Field(..., description="ID of the matching item")
    content: str = Field(..., description="Content of the matching item")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score"
    )
    knowledge_type: KnowledgeType = Field(
        ...,
        description="Type of the matching knowledge"
    )
    source: str = Field(..., description="Source of the knowledge item")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search."""

    matches: List[SimilarityMatch] = Field(
        ...,
        description="List of similar items"
    )
    query_embedding_id: str = Field(
        ...,
        description="ID of the query embedding used"
    )
    total_searched: int = Field(
        ...,
        description="Total number of items searched"
    )
    processing_time_ms: float = Field(
        ...,
        description="Search processing time in milliseconds"
    )
    model_used: str = Field(..., description="Embedding model used")


class EmbeddingAgentStatus(BaseModel):
    """Status information for the embedding agent."""

    status: str = Field(
        default="healthy",
        description="Current agent status"
    )
    model_loaded: bool = Field(
        default=False,
        description="Whether the embedding model is loaded"
    )
    model_name: str = Field(
        ...,
        description="Name of the loaded model"
    )
    embedding_dimension: int = Field(
        ...,
        description="Dimension of embeddings"
    )
    cache_size: int = Field(
        default=0,
        description="Number of cached embeddings"
    )
    knowledge_items_count: int = Field(
        default=0,
        description="Number of indexed knowledge items"
    )
    gpu_available: bool = Field(
        default=False,
        description="Whether GPU is available"
    )
    gpu_memory_used_mb: Optional[float] = Field(
        default=None,
        description="GPU memory used in MB"
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="Agent uptime in seconds"
    )
    requests_processed: int = Field(
        default=0,
        description="Total requests processed"
    )
    average_response_time_ms: float = Field(
        default=0.0,
        description="Average response time in milliseconds"
    )
    last_request_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last request"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    model_loaded: bool = Field(default=False)
    cache_available: bool = Field(default=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

