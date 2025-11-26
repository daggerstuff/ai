"""
Embedding Agent Service - Core business logic for embedding operations.

This service wraps the ClinicalKnowledgeEmbedder and provides:
- Single and batch text embedding
- Similarity search across clinical knowledge
- Caching and performance optimization
- GPU acceleration support
"""

import hashlib
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .models import (
    BatchEmbeddingItem,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingAgentConfig,
    EmbeddingAgentStatus,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResponse,
    KnowledgeType,
    SimilarityMatch,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)

# Try to import the clinical knowledge embedder
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pixel"))
    from data.clinical_knowledge_embedder import (
        ClinicalKnowledgeEmbedder,
        EmbeddingConfig,
        KnowledgeItem,
    )
    CLINICAL_EMBEDDER_AVAILABLE = True
except ImportError:
    CLINICAL_EMBEDDER_AVAILABLE = False
    ClinicalKnowledgeEmbedder = None
    EmbeddingConfig = None
    KnowledgeItem = None

logger = logging.getLogger(__name__)


class EmbeddingAgentService:
    """
    Core service for embedding operations.

    Provides text embedding, batch embedding, and similarity search
    capabilities with caching, GPU support, and performance tracking.
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        EmbeddingModel.MINILM_L6_V2: 384,
        EmbeddingModel.MINILM_L12_V2: 384,
        EmbeddingModel.MPNET_BASE_V2: 768,
        EmbeddingModel.BGE_SMALL: 384,
        EmbeddingModel.BGE_BASE: 768,
        EmbeddingModel.CLINICAL_BERT: 768,
    }

    def __init__(
        self,
        config: Optional[EmbeddingAgentConfig] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the embedding agent service.

        Args:
            config: Configuration for the embedding agent
            project_root: Root path for loading clinical knowledge
        """
        self.config = config or EmbeddingAgentConfig()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent

        # Threading lock for thread-safe operations
        self._lock = Lock()

        # Statistics tracking
        self._start_time = datetime.utcnow()
        self._requests_processed = 0
        self._total_response_time_ms = 0.0
        self._last_request_at: Optional[datetime] = None

        # Embedding cache
        self._embedding_cache: Dict[str, Tuple[List[float], str]] = {}

        # Initialize models
        self._embedding_model: Optional[SentenceTransformer] = None
        self._clinical_embedder: Optional[ClinicalKnowledgeEmbedder] = None
        self._faiss_index = None
        self._knowledge_items: List[Any] = []

        # Load model on initialization
        self._initialize_model()

        logger.info(
            f"EmbeddingAgentService initialized with model: {self.config.model_name}"
        )

    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. "
                "Running in mock mode."
            )
            return

        try:
            device = "cuda" if self.config.use_gpu else "cpu"

            # Check CUDA availability if GPU requested
            if self.config.use_gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning(
                            "GPU requested but CUDA not available. "
                            "Falling back to CPU."
                        )
                        device = "cpu"
                except ImportError:
                    logger.warning(
                        "PyTorch not available. Cannot check GPU. "
                        "Using CPU."
                    )
                    device = "cpu"

            self._embedding_model = SentenceTransformer(
                self.config.model_name.value,
                device=device
            )

            # Update dimension from actual model
            actual_dim = self._embedding_model.get_sentence_embedding_dimension()
            if actual_dim != self.config.embedding_dimension:
                logger.info(
                    f"Updating embedding dimension from {self.config.embedding_dimension} "
                    f"to {actual_dim} based on loaded model"
                )
                self.config.embedding_dimension = actual_dim

            logger.info(
                f"Loaded embedding model: {self.config.model_name.value} "
                f"on {device} with dimension {actual_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._embedding_model = None

    def _initialize_clinical_embedder(self) -> None:
        """Initialize the clinical knowledge embedder if available."""
        if not CLINICAL_EMBEDDER_AVAILABLE:
            logger.warning("ClinicalKnowledgeEmbedder not available")
            return

        try:
            embedding_config = EmbeddingConfig(
                model_name=self.config.model_name.value,
                batch_size=self.config.batch_size,
                max_length=self.config.max_text_length,
                normalize_embeddings=self.config.normalize_embeddings,
                cache_embeddings=self.config.cache_embeddings,
                embedding_dimension=self.config.embedding_dimension,
            )

            self._clinical_embedder = ClinicalKnowledgeEmbedder(
                config=embedding_config,
                project_root=self.project_root,
            )

            logger.info("ClinicalKnowledgeEmbedder initialized")

        except Exception as e:
            logger.error(f"Failed to initialize clinical embedder: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for caching purposes."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _generate_embedding_id(self) -> str:
        """Generate a unique embedding ID."""
        return f"emb_{uuid.uuid4().hex[:16]}"

    def _update_stats(self, response_time_ms: float) -> None:
        """Update service statistics."""
        with self._lock:
            self._requests_processed += 1
            self._total_response_time_ms += response_time_ms
            self._last_request_at = datetime.utcnow()

    def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embedding for a single text.

        Args:
            request: The embedding request

        Returns:
            EmbeddingResponse with the generated embedding
        """
        start_time = time.time()

        text = request.text
        text_hash = self._get_text_hash(text)
        model_name = request.model or self.config.model_name

        # Check cache first
        cache_key = f"{model_name.value}:{text_hash}"
        if self.config.cache_embeddings and cache_key in self._embedding_cache:
            embedding, embedding_id = self._embedding_cache[cache_key]
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time)

            return EmbeddingResponse(
                embedding=embedding,
                embedding_id=embedding_id,
                model_used=model_name.value,
                dimension=len(embedding),
                text_hash=text_hash,
                cached=True,
                processing_time_ms=processing_time,
            )

        # Generate embedding
        embedding = self._generate_embedding(text, model_name)
        embedding_id = self._generate_embedding_id()

        # Cache the embedding
        if self.config.cache_embeddings:
            self._embedding_cache[cache_key] = (embedding, embedding_id)

        processing_time = (time.time() - start_time) * 1000
        self._update_stats(processing_time)

        return EmbeddingResponse(
            embedding=embedding,
            embedding_id=embedding_id,
            model_used=model_name.value,
            dimension=len(embedding),
            text_hash=text_hash,
            cached=False,
            processing_time_ms=processing_time,
        )

    def _generate_embedding(
        self,
        text: str,
        model: EmbeddingModel,
    ) -> List[float]:
        """
        Generate embedding using the model.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            List of floats representing the embedding
        """
        if self._embedding_model is None:
            # Generate mock embedding
            return self._generate_mock_embedding(text)

        try:
            # Truncate text if necessary
            truncated_text = text[:self.config.max_text_length]

            # Generate embedding
            embedding = self._embedding_model.encode(
                truncated_text,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )

            # Convert to list
            if hasattr(embedding, "tolist"):
                return embedding.tolist()
            return list(embedding)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing."""
        text_hash = self._get_text_hash(text)
        hash_int = int(text_hash[:8], 16)

        if NUMPY_AVAILABLE:
            np.random.seed(hash_int)
            embedding = np.random.normal(0, 1, self.config.embedding_dimension)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()
        else:
            # Simple deterministic mock
            import random
            random.seed(hash_int)
            embedding = [
                random.gauss(0, 1) for _ in range(self.config.embedding_dimension)
            ]
            # Simple normalization
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            return embedding

    def embed_batch(self, request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
        """
        Generate embeddings for multiple texts.

        Args:
            request: The batch embedding request

        Returns:
            BatchEmbeddingResponse with all embeddings
        """
        start_time = time.time()

        model_name = request.model or self.config.model_name
        embeddings: List[BatchEmbeddingItem] = []
        cached_count = 0

        # Process texts
        texts_to_embed: List[Tuple[int, str]] = []

        for i, text in enumerate(request.texts):
            text_hash = self._get_text_hash(text)
            cache_key = f"{model_name.value}:{text_hash}"

            # Check cache
            if self.config.cache_embeddings and cache_key in self._embedding_cache:
                embedding, embedding_id = self._embedding_cache[cache_key]
                embeddings.append(BatchEmbeddingItem(
                    index=i,
                    embedding=embedding,
                    embedding_id=embedding_id,
                    text_hash=text_hash,
                    cached=True,
                ))
                cached_count += 1
            else:
                texts_to_embed.append((i, text))

        # Batch generate embeddings for uncached texts
        if texts_to_embed:
            batch_embeddings = self._batch_generate_embeddings(
                [t[1] for t in texts_to_embed],
                model_name,
            )

            for (idx, text), emb in zip(texts_to_embed, batch_embeddings):
                text_hash = self._get_text_hash(text)
                embedding_id = self._generate_embedding_id()
                cache_key = f"{model_name.value}:{text_hash}"

                # Cache
                if self.config.cache_embeddings:
                    self._embedding_cache[cache_key] = (emb, embedding_id)

                embeddings.append(BatchEmbeddingItem(
                    index=idx,
                    embedding=emb,
                    embedding_id=embedding_id,
                    text_hash=text_hash,
                    cached=False,
                ))

        # Sort by original index
        embeddings.sort(key=lambda x: x.index)

        processing_time = (time.time() - start_time) * 1000
        self._update_stats(processing_time)

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            total_count=len(request.texts),
            cached_count=cached_count,
            generated_count=len(request.texts) - cached_count,
            model_used=model_name.value,
            dimension=self.config.embedding_dimension,
            processing_time_ms=processing_time,
        )

    def _batch_generate_embeddings(
        self,
        texts: List[str],
        model: EmbeddingModel,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if self._embedding_model is None:
            return [self._generate_mock_embedding(t) for t in texts]

        try:
            # Truncate texts
            truncated_texts = [t[:self.config.max_text_length] for t in texts]

            # Batch encode
            embeddings = self._embedding_model.encode(
                truncated_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )

            # Convert to list of lists
            result = []
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    result.append(emb.tolist())
                else:
                    result.append(list(emb))
            return result

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return [self._generate_mock_embedding(t) for t in texts]

    def search_similar(
        self,
        request: SimilaritySearchRequest,
    ) -> SimilaritySearchResponse:
        """
        Search for similar items in the knowledge base.

        Args:
            request: The similarity search request

        Returns:
            SimilaritySearchResponse with matching items
        """
        start_time = time.time()

        # Get or generate query embedding
        if request.query_embedding:
            query_embedding = request.query_embedding
            query_embedding_id = self._generate_embedding_id()
        else:
            embed_request = EmbeddingRequest(text=request.query)
            embed_response = self.embed_text(embed_request)
            query_embedding = embed_response.embedding
            query_embedding_id = embed_response.embedding_id

        # Search in knowledge items
        matches = self._search_knowledge_items(
            query_embedding=query_embedding,
            top_k=request.top_k,
            knowledge_types=request.knowledge_types,
            min_similarity=request.min_similarity,
            include_metadata=request.include_metadata,
        )

        processing_time = (time.time() - start_time) * 1000
        self._update_stats(processing_time)

        return SimilaritySearchResponse(
            matches=matches,
            query_embedding_id=query_embedding_id,
            total_searched=len(self._knowledge_items),
            processing_time_ms=processing_time,
            model_used=self.config.model_name.value,
        )

    def _search_knowledge_items(
        self,
        query_embedding: List[float],
        top_k: int,
        knowledge_types: Optional[List[KnowledgeType]],
        min_similarity: float,
        include_metadata: bool,
    ) -> List[SimilarityMatch]:
        """Search through indexed knowledge items."""
        if not self._knowledge_items:
            # Return empty if no knowledge loaded
            return []

        matches = []
        query_np = None

        if NUMPY_AVAILABLE:
            query_np = np.array(query_embedding)
            if self.config.normalize_embeddings:
                norm = np.linalg.norm(query_np)
                if norm > 0:
                    query_np = query_np / norm

        for item in self._knowledge_items:
            # Filter by knowledge type if specified
            if knowledge_types:
                item_type = getattr(item, "knowledge_type", "general")
                if item_type not in [kt.value for kt in knowledge_types]:
                    continue

            # Get item embedding
            item_embedding = getattr(item, "embedding", None)
            if item_embedding is None:
                continue

            # Calculate similarity
            if NUMPY_AVAILABLE and query_np is not None:
                item_np = np.array(item_embedding)
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(item_np)
                    if norm > 0:
                        item_np = item_np / norm
                similarity = float(np.dot(query_np, item_np))
            else:
                # Simple cosine similarity
                dot_product = sum(
                    a * b for a, b in zip(query_embedding, item_embedding)
                )
                norm_q = sum(x**2 for x in query_embedding) ** 0.5
                norm_i = sum(x**2 for x in item_embedding) ** 0.5
                similarity = dot_product / (norm_q * norm_i) if norm_q * norm_i > 0 else 0.0

            # Apply threshold
            if similarity >= min_similarity:
                match = SimilarityMatch(
                    item_id=getattr(item, "id", str(uuid.uuid4())),
                    content=getattr(item, "content", "")[:500],  # Truncate content
                    similarity_score=similarity,
                    knowledge_type=KnowledgeType(
                        getattr(item, "knowledge_type", "general")
                    ),
                    source=getattr(item, "source", "unknown"),
                    metadata=getattr(item, "metadata", {}) if include_metadata else None,
                )
                matches.append(match)

        # Sort by similarity and take top_k
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]

    def load_knowledge_base(self) -> int:
        """
        Load clinical knowledge into the service.

        Returns:
            Number of knowledge items loaded
        """
        if not CLINICAL_EMBEDDER_AVAILABLE:
            logger.warning("Clinical embedder not available")
            return 0

        if self._clinical_embedder is None:
            self._initialize_clinical_embedder()

        if self._clinical_embedder is None:
            return 0

        try:
            # Extract and embed knowledge
            knowledge_items, _ = self._clinical_embedder.process_all_knowledge()
            self._knowledge_items = knowledge_items

            logger.info(f"Loaded {len(knowledge_items)} knowledge items")
            return len(knowledge_items)

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return 0

    def get_status(self) -> EmbeddingAgentStatus:
        """
        Get the current status of the embedding agent.

        Returns:
            EmbeddingAgentStatus with current metrics
        """
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        avg_response_time = (
            self._total_response_time_ms / self._requests_processed
            if self._requests_processed > 0 else 0.0
        )

        # Check GPU memory if available
        gpu_memory = None
        gpu_available = False
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            except ImportError:
                pass

        return EmbeddingAgentStatus(
            status="healthy" if self._embedding_model is not None else "degraded",
            model_loaded=self._embedding_model is not None,
            model_name=self.config.model_name.value,
            embedding_dimension=self.config.embedding_dimension,
            cache_size=len(self._embedding_cache),
            knowledge_items_count=len(self._knowledge_items),
            gpu_available=gpu_available,
            gpu_memory_used_mb=gpu_memory,
            uptime_seconds=uptime,
            requests_processed=self._requests_processed,
            average_response_time_ms=avg_response_time,
            last_request_at=self._last_request_at,
        )

    def clear_cache(self) -> int:
        """
        Clear the embedding cache.

        Returns:
            Number of cached items cleared
        """
        with self._lock:
            count = len(self._embedding_cache)
            self._embedding_cache.clear()
            logger.info(f"Cleared {count} cached embeddings")
            return count

    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Shutting down EmbeddingAgentService")
        self._embedding_model = None
        self._clinical_embedder = None
        self._embedding_cache.clear()
        self._knowledge_items.clear()

