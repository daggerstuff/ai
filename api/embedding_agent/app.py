"""
FastAPI Application for Embedding Agent Service.

Provides REST API endpoints for:
- Text embedding (single and batch)
- Similarity search
- Health checks and status monitoring
- Cache management
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingAgentConfig,
    EmbeddingAgentStatus,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    HealthCheckResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from .service import EmbeddingAgentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instance
_embedding_service: Optional[EmbeddingAgentService] = None


def get_embedding_service() -> EmbeddingAgentService:
    """Get or create the embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        config = EmbeddingAgentConfig(
            model_name=EmbeddingModel(
                os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            ),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_text_length=int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
            cache_embeddings=os.getenv("EMBEDDING_CACHE", "true").lower() == "true",
            use_gpu=os.getenv("EMBEDDING_USE_GPU", "false").lower() == "true",
        )
        project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))
        _embedding_service = EmbeddingAgentService(
            config=config,
            project_root=project_root,
        )
    return _embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Embedding Agent Service")
    service = get_embedding_service()

    # Optionally load knowledge base on startup
    if os.getenv("EMBEDDING_LOAD_KNOWLEDGE", "false").lower() == "true":
        logger.info("Loading knowledge base on startup...")
        count = service.load_knowledge_base()
        logger.info(f"Loaded {count} knowledge items")

    yield

    # Shutdown
    logger.info("Shutting down Embedding Agent Service")
    if _embedding_service is not None:
        _embedding_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Embedding Agent API",
    description=(
        "Vector embedding service for clinical knowledge. "
        "Provides text-to-vector conversion and similarity search capabilities."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Router for embedding endpoints (can be included in other apps)
from fastapi import APIRouter

embedding_router = APIRouter(prefix="/api/v1/embeddings", tags=["Embeddings"])


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__} if os.getenv("DEBUG") else None,
        ).model_dump(),
    )


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
@embedding_router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.

    Returns the current health status of the embedding service.
    """
    service = get_embedding_service()
    status_info = service.get_status()

    return HealthCheckResponse(
        status="healthy" if status_info.model_loaded else "degraded",
        version="1.0.0",
        model_loaded=status_info.model_loaded,
        cache_available=True,
    )


@app.get("/status", response_model=EmbeddingAgentStatus, tags=["Health"])
@embedding_router.get("/status", response_model=EmbeddingAgentStatus)
async def get_status() -> EmbeddingAgentStatus:
    """
    Get detailed status of the embedding agent.

    Returns comprehensive status including:
    - Model information
    - Cache statistics
    - Performance metrics
    - Resource usage
    """
    service = get_embedding_service()
    return service.get_status()


# Embedding endpoints
@embedding_router.post(
    "/embed",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
@app.post(
    "/api/v1/embeddings/embed",
    response_model=EmbeddingResponse,
    tags=["Embeddings"],
)
async def embed_text(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embedding for a single text.

    Converts text into a dense vector representation using the configured
    embedding model. Results are cached for faster subsequent requests.

    **Request Body:**
    - `text`: Text to embed (1-10000 characters)
    - `knowledge_type`: Optional categorization (dsm5, pdm2, clinical, etc.)
    - `metadata`: Optional metadata to associate with the embedding
    - `model`: Optional model override

    **Response:**
    - `embedding`: Vector representation (list of floats)
    - `embedding_id`: Unique identifier
    - `dimension`: Vector dimension
    - `cached`: Whether result was from cache
    - `processing_time_ms`: Processing time in milliseconds
    """
    try:
        service = get_embedding_service()
        return service.embed_text(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in embed_text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embedding",
        )


@embedding_router.post(
    "/embed/batch",
    response_model=BatchEmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
@app.post(
    "/api/v1/embeddings/embed/batch",
    response_model=BatchEmbeddingResponse,
    tags=["Embeddings"],
)
async def embed_batch(request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
    """
    Generate embeddings for multiple texts in batch.

    Efficiently processes multiple texts in a single request.
    Cached embeddings are returned immediately, while new ones
    are generated in optimized batches.

    **Request Body:**
    - `texts`: List of texts to embed (1-100 items)
    - `knowledge_types`: Optional list of types for each text
    - `metadata_list`: Optional metadata for each text
    - `model`: Optional model override

    **Response:**
    - `embeddings`: List of embedding results
    - `total_count`: Total texts processed
    - `cached_count`: Number from cache
    - `generated_count`: Number newly generated
    - `processing_time_ms`: Total processing time
    """
    try:
        service = get_embedding_service()
        return service.embed_batch(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in embed_batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate batch embeddings",
        )


@embedding_router.post(
    "/search",
    response_model=SimilaritySearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
@app.post(
    "/api/v1/embeddings/search",
    response_model=SimilaritySearchResponse,
    tags=["Embeddings"],
)
async def search_similar(
    request: SimilaritySearchRequest,
) -> SimilaritySearchResponse:
    """
    Search for similar items in the knowledge base.

    Finds knowledge items most similar to the query using
    cosine similarity between embedding vectors.

    **Request Body:**
    - `query`: Query text to search
    - `query_embedding`: Pre-computed embedding (optional)
    - `top_k`: Number of results to return (1-100)
    - `knowledge_types`: Filter by knowledge types
    - `min_similarity`: Minimum similarity threshold
    - `include_metadata`: Whether to include metadata

    **Response:**
    - `matches`: List of similar items with scores
    - `total_searched`: Number of items searched
    - `processing_time_ms`: Search time
    """
    try:
        service = get_embedding_service()
        return service.search_similar(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in search_similar: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform similarity search",
        )


# Knowledge management endpoints
@embedding_router.post(
    "/knowledge/load",
    tags=["Knowledge"],
    responses={
        200: {"description": "Knowledge loaded successfully"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
@app.post("/api/v1/embeddings/knowledge/load", tags=["Knowledge"])
async def load_knowledge() -> Dict[str, Any]:
    """
    Load clinical knowledge into the service.

    Loads and indexes all available clinical knowledge items
    (DSM-5, PDM-2, therapeutic conversations, etc.) for
    similarity search.

    **Response:**
    - `success`: Whether loading succeeded
    - `items_loaded`: Number of knowledge items loaded
    - `message`: Status message
    """
    try:
        service = get_embedding_service()
        count = service.load_knowledge_base()
        return {
            "success": True,
            "items_loaded": count,
            "message": f"Successfully loaded {count} knowledge items",
        }
    except Exception as e:
        logger.error(f"Error loading knowledge: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load knowledge base",
        )


# Cache management endpoints
@embedding_router.delete(
    "/cache",
    tags=["Cache"],
    responses={
        200: {"description": "Cache cleared successfully"},
    },
)
@app.delete("/api/v1/embeddings/cache", tags=["Cache"])
async def clear_cache() -> Dict[str, Any]:
    """
    Clear the embedding cache.

    Removes all cached embeddings. Use this if you need to
    regenerate embeddings or free up memory.

    **Response:**
    - `success`: Whether clearing succeeded
    - `items_cleared`: Number of cached items removed
    """
    service = get_embedding_service()
    count = service.clear_cache()
    return {
        "success": True,
        "items_cleared": count,
        "message": f"Cleared {count} cached embeddings",
    }


@embedding_router.get(
    "/cache/stats",
    tags=["Cache"],
)
@app.get("/api/v1/embeddings/cache/stats", tags=["Cache"])
async def cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    **Response:**
    - `cache_size`: Number of cached embeddings
    - `cache_enabled`: Whether caching is enabled
    """
    service = get_embedding_service()
    status_info = service.get_status()
    return {
        "cache_size": status_info.cache_size,
        "cache_enabled": service.config.cache_embeddings,
    }


# Configuration endpoint
@embedding_router.get(
    "/config",
    tags=["Configuration"],
)
@app.get("/api/v1/embeddings/config", tags=["Configuration"])
async def get_config() -> Dict[str, Any]:
    """
    Get current embedding agent configuration.

    Returns the current configuration including model settings,
    batch sizes, and feature flags.
    """
    service = get_embedding_service()
    return {
        "model_name": service.config.model_name.value,
        "embedding_dimension": service.config.embedding_dimension,
        "batch_size": service.config.batch_size,
        "max_text_length": service.config.max_text_length,
        "normalize_embeddings": service.config.normalize_embeddings,
        "cache_embeddings": service.config.cache_embeddings,
        "use_gpu": service.config.use_gpu,
    }


# Model info endpoint
@embedding_router.get(
    "/models",
    tags=["Models"],
)
@app.get("/api/v1/embeddings/models", tags=["Models"])
async def list_models() -> Dict[str, Any]:
    """
    List available embedding models.

    Returns all supported embedding models with their
    characteristics.
    """
    models = []
    for model in EmbeddingModel:
        dim = EmbeddingAgentService.MODEL_DIMENSIONS.get(model, 384)
        models.append({
            "id": model.value,
            "name": model.name,
            "dimension": dim,
            "description": _get_model_description(model),
        })
    return {"models": models}


def _get_model_description(model: EmbeddingModel) -> str:
    """Get description for an embedding model."""
    descriptions = {
        EmbeddingModel.MINILM_L6_V2: "Fast, lightweight model with good quality (384 dim)",
        EmbeddingModel.MINILM_L12_V2: "Slightly larger MiniLM variant (384 dim)",
        EmbeddingModel.MPNET_BASE_V2: "Higher quality, larger model (768 dim)",
        EmbeddingModel.BGE_SMALL: "BAAI small English model (384 dim)",
        EmbeddingModel.BGE_BASE: "BAAI base English model (768 dim)",
        EmbeddingModel.CLINICAL_BERT: "Clinical domain-specific BERT (768 dim)",
    }
    return descriptions.get(model, "Embedding model")


# Include router in app
app.include_router(embedding_router)


def create_app(
    config: Optional[EmbeddingAgentConfig] = None,
    project_root: Optional[Path] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Optional custom configuration
        project_root: Optional project root path

    Returns:
        Configured FastAPI application
    """
    global _embedding_service

    if config is not None or project_root is not None:
        # Create service with custom config
        _embedding_service = EmbeddingAgentService(
            config=config,
            project_root=project_root,
        )

    return app


# Main entry point
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("EMBEDDING_AGENT_PORT", "8001"))
    host = os.getenv("EMBEDDING_AGENT_HOST", "0.0.0.0")

    uvicorn.run(
        "ai.api.embedding_agent.app:app",
        host=host,
        port=port,
        reload=os.getenv("EMBEDDING_AGENT_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )

