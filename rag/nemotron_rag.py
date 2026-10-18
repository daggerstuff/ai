"""
NVIDIA Nemotron RAG Pipeline for Therapeutic Knowledge.

Implements a complete RAG (Retrieval-Augmented Generation) pipeline using
NVIDIA NIM models for mental health therapeutic support.

Features:
- Document ingestion with Nemotron-Embed-VL (2048-dim embeddings)
- FAISS vector store for semantic search
- Retrieval with optional category filtering
- Response generation with Nemotron-Super using retrieved context
- Citation tracking for evidence-based responses

Usage:
    config = NemotronRAGConfig(api_key=os.environ.get("NVIDIA_API_KEY"))
    pipeline = TherapeuticRAGPipeline(config)

    # Ingest documents
    await pipeline.ingest_document(
        document="Therapy protocol text...",
        metadata={"category": "treatment_protocols", "source": "APA Guidelines"},
        doc_id="proto-001"
    )

    # Retrieve and generate
    response = await pipeline.query("What are CBT techniques for anxiety?")
"""

import asyncio
import hashlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

faiss: Any | None = None
FAISS_AVAILABLE = False

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _load_faiss() -> Any | None:
    """Lazily import FAISS to avoid importing native extension at module import time."""
    global faiss, FAISS_AVAILABLE
    if FAISS_AVAILABLE and faiss is not None:
        return faiss

    try:
        with warnings.catch_warnings():
            # Keep warning policy strict, but isolate known upstream SWIG warnings so
            # importing FAISS on CPython 3.13 does not crash test runs.
            warnings.simplefilter("error", DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=(
                    r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) "
                    r"has no __module__ attribute"
                ),
            )
            import faiss as _faiss

        faiss = _faiss
        FAISS_AVAILABLE = True
        return faiss
    except Exception as exc:  # pragma: no cover - defensive for optional dependency failures
        logger.debug("FAISS import failed during lazy load: %s", exc)
        faiss = None
        FAISS_AVAILABLE = False
        return None


# =============================================================================
# Enums and Constants
# =============================================================================


class KnowledgeCategory(StrEnum):
    """Categories of therapeutic knowledge."""

    TREATMENT_PROTOCOLS = "treatment_protocols"
    CRISIS_PROTOCOLS = "crisis_protocols"
    PSYCHOEDUCATION = "psychoeducation"
    SESSION_HISTORY = "session_history"
    RESEARCH_PAPERS = "research_papers"
    CLIENT_RESOURCES = "client_resources"


class IndexType(StrEnum):
    """FAISS index types for different scales."""

    FLAT = "flat"  # <10K documents, real-time updates
    IVF = "ivf"  # 10K-1M documents, hourly batch
    HNSW = "hnsw"  # >1M documents, daily batch


# Default knowledge base configuration
THERAPEUTIC_KNOWLEDGE_BASE = {
    KnowledgeCategory.TREATMENT_PROTOCOLS: {
        "description": "Evidence-based treatment guidelines",
        "sources": ["APA Guidelines", "NICE Guidelines", "Research Papers"],
        "update_frequency": "monthly",
    },
    KnowledgeCategory.CRISIS_PROTOCOLS: {
        "description": "Emergency response procedures",
        "sources": ["Crisis hotlines", "Safety protocols", "Referral networks"],
        "update_frequency": "weekly",
    },
    KnowledgeCategory.PSYCHOEDUCATION: {
        "description": "Client educational materials",
        "sources": ["Worksheets", "Articles", "Videos"],
        "update_frequency": "quarterly",
    },
    KnowledgeCategory.SESSION_HISTORY: {
        "description": "Client progress and notes",
        "sources": ["Session transcripts", "Progress notes", "Treatment plans"],
        "update_frequency": "real-time",
    },
}

# NVIDIA NIM model identifiers for therapeutic RAG pipeline
NIM_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
NIM_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
NIM_GENERATION_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
NIM_FAST_MODEL = "nvidia/llama-3.1-nemotron-nano-4b-v1.1"

# Alternative models for different use cases
NIM_REASONING_MODEL = "nvidia/deepseek-v3"
NIM_BALANCED_MODEL = "meta/llama-3.1-70b-instruct"
NIM_MULTILINGUAL_MODEL = "qwen/qwen-2.5-72b-instruct"
NIM_SAFETY_MODEL = "nvidia/nemotron-guard-0.5"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_DIMENSION = 2048  # Nemotron-Embed-VL produces 2048-dim vectors


# =============================================================================
# Configuration Classes
# =============================================================================


class QueryComplexity(StrEnum):
    """Query complexity levels for model selection in RAG."""

    SIMPLE = "simple"  # Factual lookup, single document retrieval
    MODERATE = "moderate"  # Multi-document synthesis, standard queries
    COMPLEX = "complex"  # Nuanced therapeutic reasoning, crisis context
    CRISIS = "crisis"  # Safety-critical queries requiring highest accuracy


class NemotronRAGConfig(BaseModel):
    """Configuration for Nemotron RAG pipeline.

    Attributes:
        api_key: NVIDIA API key for NIM access
        base_url: NIM API endpoint
        embedding_model: Model for document embeddings
        generation_model: Model for response synthesis
        embedding_dimension: Embedding vector size
        retrieval_top_k: Initial retrieval candidates
        reranking_top_n: Final reranked results
        index_type: FAISS index type
        n_clusters: IVF clusters (if using IVF index)
        cache_ttl: Cache time-to-live in seconds
        complexity_model_mapping: Mapping of query complexity to models
    """

    api_key: str = Field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    base_url: str = NIM_BASE_URL

    # Model selection - using ModelTier enum values from expanded catalog
    embedding_model: str = NIM_EMBEDDING_MODEL
    reranking_model: str = NIM_RERANK_MODEL
    generation_model: str = NIM_GENERATION_MODEL
    fast_model: str = NIM_FAST_MODEL

    # Additional models from expanded NVIDIA NIM catalog
    reasoning_model: str = NIM_REASONING_MODEL
    balanced_model: str = NIM_BALANCED_MODEL
    multilingual_model: str = NIM_MULTILINGUAL_MODEL
    safety_model: str = NIM_SAFETY_MODEL

    # Complexity-based model selection
    complexity_model_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            QueryComplexity.SIMPLE.value: NIM_FAST_MODEL,
            QueryComplexity.MODERATE.value: NIM_BALANCED_MODEL,
            QueryComplexity.COMPLEX.value: NIM_GENERATION_MODEL,
            QueryComplexity.CRISIS.value: NIM_SAFETY_MODEL,
        }
    )

    # Retrieval settings
    embedding_dimension: int = EMBEDDING_DIMENSION
    retrieval_top_k: int = 20
    reranking_top_n: int = 5

    # Vector store settings
    index_type: IndexType = IndexType.IVF
    n_clusters: int = 100

    # Cache settings
    cache_ttl: int = 3600  # 1 hour

    # Generation settings
    max_context_length: int = 4096
    generation_temperature: float = 0.7
    max_generation_tokens: int = 2048

    model_config = ConfigDict(use_enum_values=True)


class DocumentMetadata(BaseModel):
    """Metadata for ingested documents.

    Attributes:
        doc_id: Unique document identifier
        category: Knowledge category
        source: Source identifier (e.g., "APA Guidelines")
        title: Document title
        created_at: Ingestion timestamp
        updated_at: Last update timestamp
        tags: Optional tags for filtering
        url: Optional source URL
        author: Optional author information
    """

    doc_id: str
    category: KnowledgeCategory
    source: str
    title: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)
    url: str | None = None
    author: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class RAGResponse(BaseModel):
    """Response from RAG query.

    Attributes:
        response: Generated response text
        sources: List of source metadata
        model: Generation model used
        retrieved_count: Number of documents retrieved
        latency_ms: Total query latency
        citations: Extracted citations
    """

    response: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    model: str
    retrieved_count: int = 0
    latency_ms: float = 0.0
    citations: list[str] = Field(default_factory=list)


# =============================================================================
# Document Store
# =============================================================================


@dataclass
class Document:
    """Internal document representation."""

    doc_id: str
    content: str
    metadata: DocumentMetadata
    embedding: np.ndarray | None = None
    embedding_id: int = -1


@dataclass
class DocumentStore:
    """In-memory document storage with FAISS indexing."""

    documents: dict[str, Document] = field(default_factory=dict)
    index: Any | None = None  # faiss.Index
    is_trained: bool = False

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def add(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def get_by_index(self, idx: int) -> Document | None:
        """Get document by embedding index."""
        for doc in self.documents.values():
            if doc.embedding_id == idx:
                return doc
        return None


# =============================================================================
# Main RAG Pipeline
# =============================================================================


class TherapeuticRAGPipeline:
    """
    RAG pipeline for therapeutic knowledge retrieval.

    Features:
    - Document ingestion with automatic categorization
    - Semantic search with FAISS vector store
    - Citation-aware response generation
    - Integration with session memory
    - Crisis-aware query handling

    Example:
        config = NemotronRAGConfig(api_key=os.environ.get("NVIDIA_API_KEY"))
        pipeline = TherapeuticRAGPipeline(config)

        # Ingest documents
        await pipeline.ingest_document(
            document="Treatment protocol text...",
            metadata={"category": "treatment_protocols", "source": "APA"},
            doc_id="protocol-001"
        )

        # Query the knowledge base
        response = await pipeline.query(
            "What are effective treatments for generalized anxiety?"
        )
    """

    def __init__(self, config: NemotronRAGConfig):
        """Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self.store = DocumentStore()

        logger.info(f"Initialized TherapeuticRAGPipeline with {config.index_type} index")

    def _initialize_vector_store(self) -> None:
        """Initialize FAISS vector index based on configuration."""
        if self.store.index is not None:
            return

        faiss_lib = _load_faiss()
        if faiss_lib is None:
            logger.warning(
                "FAISS not available. Using in-memory search only. Install faiss-cpu or faiss-gpu for vector search."
            )
            return

        dim = self.config.embedding_dimension

        if self.config.index_type == IndexType.FLAT:
            # Simple flat index for small corpora
            self.store.index = faiss_lib.IndexFlatIP(dim)
            self.store.is_trained = True

        elif self.config.index_type == IndexType.IVF:
            # IVF index for medium corpora
            quantizer = faiss_lib.IndexFlatIP(dim)
            self.store.index = faiss_lib.IndexIVFFlat(
                quantizer,
                dim,
                self.config.n_clusters,
                faiss_lib.METRIC_INNER_PRODUCT,
            )
            self.store.is_trained = False

        elif self.config.index_type == IndexType.HNSW:
            # HNSW index for large corpora
            self.store.index = faiss_lib.IndexHNSWFlat(dim, 32)
            self.store.is_trained = True

        logger.debug(f"Initialized {self.config.index_type} FAISS index")

    async def ingest_document(self, document: str, metadata: dict[str, Any], doc_id: str | None = None) -> str:
        """
        Ingest a document into the knowledge base.

        Args:
            document: Document text content
            metadata: Metadata dict with category, source, etc.
            doc_id: Optional unique identifier (auto-generated if not provided)

        Returns:
            Document ID
        """
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = hashlib.sha256(document.encode()).hexdigest()[:16]

        # Create metadata object
        if isinstance(metadata.get("category"), str):
            category = KnowledgeCategory(metadata["category"])
        else:
            category = metadata.get("category", KnowledgeCategory.PSYCHOEDUCATION)

        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            category=category,
            source=metadata.get("source", "unknown"),
            title=metadata.get("title"),
            tags=metadata.get("tags", []),
            url=metadata.get("url"),
            author=metadata.get("author"),
        )

        # Generate embedding
        embedding = await self._generate_embedding(document, input_type="passage")

        # Create document
        doc = Document(
            doc_id=doc_id, content=document, metadata=doc_metadata, embedding=embedding, embedding_id=len(self.store)
        )

        # Add to store
        self.store.add(doc)

        # Add to FAISS index
        self._add_to_index(embedding)

        logger.debug(f"Ingested document {doc_id} into category {category}")
        return doc_id

    async def _generate_embedding(self, text: str, input_type: str = "query") -> np.ndarray:
        """Generate embedding for text using Nemotron-Embed-VL."""
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
            encoding_format="float",
            extra_body={"input_type": input_type},
        )

        embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Normalize for inner product search
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm

        return embedding

    def _classify_query_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity for model selection.

        Uses simple heuristics to determine the appropriate model tier:
        - SIMPLE: Direct factual queries, keyword lookups
        - MODERATE: Multi-concept queries, synthesis needs
        - COMPLEX: Therapeutic reasoning, emotional context
        - CRISIS: Safety keywords, crisis indicators

        Args:
            query: User query text

        Returns:
            QueryComplexity level
        """
        query_lower = query.lower()

        # Crisis indicators - safety-critical (check first)
        crisis_keywords = [
            "suicide",
            "kill myself",
            "end my life",
            "hurt myself",
            "self-harm",
            "overdose",
            "crisis",
            "emergency",
            "danger",
            "unsafe",
            "hopeless",
            "want to die",
            "can't go on",
            "ending my life",
            "end it all",
            "take my life",
        ]
        if any(kw in query_lower for kw in crisis_keywords):
            return QueryComplexity.CRISIS

        # Complex therapeutic reasoning indicators
        complex_keywords = [
            "why do i feel",
            "understand my",
            "relationship between",
            "pattern in my",
            "trauma",
            "underlying",
            "deeper issue",
            "connection between",
            "emotional root",
            "psychological",
            "therapy process",
            "treatment approach",
            "nuanced",
            "subtle",
            "complex",
            " vs ",
            "versus",
        ]
        if any(kw in query_lower for kw in complex_keywords):
            return QueryComplexity.COMPLEX

        # Moderate complexity - synthesis and multi-concept
        moderate_keywords = [
            "how does",
            "what are the differences",
            "multiple",
            "various",
            "several",
            "both",
            "combination",
            "together",
            "integrate",
            "compare",
            "comparison",
        ]
        if any(kw in query_lower for kw in moderate_keywords):
            return QueryComplexity.MODERATE

        # Default to simple for direct queries
        return QueryComplexity.SIMPLE

    def _select_model_for_query(self, query: str) -> str:
        """Select the appropriate model based on query complexity.

        Args:
            query: User query text

        Returns:
            Model identifier from ModelTier enum
        """
        complexity = self._classify_query_complexity(query)
        return self.config.complexity_model_mapping.get(complexity.value, self.config.generation_model)

    def _add_to_index(self, embedding: np.ndarray) -> None:
        """Add embedding to FAISS index."""
        self._initialize_vector_store()
        faiss_lib = _load_faiss()
        if faiss_lib is None or self.store.index is None:
            return

        # Train IVF index if needed
        if self.config.index_type == IndexType.IVF and not self.store.is_trained:
            if len(self.store) >= self.config.n_clusters:
                # Gather all embeddings for training
                embeddings = np.vstack(
                    [doc.embedding for doc in self.store.documents.values() if doc.embedding is not None]
                )
                self.store.index.train(embeddings)
                self.store.is_trained = True
                logger.info("Trained IVF index with {len(embeddings)} vectors")
            return  # Don't add yet, will add after training

        if self.store.is_trained or self.config.index_type != IndexType.IVF:
            self.store.index.add(embedding)

    async def retrieve(
        self, query: str, category: KnowledgeCategory | None = None, n_results: int | None = None
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            category: Optional filter by knowledge category
            n_results: Number of results (default from config)

        Returns:
            List of relevant documents
        """
        n_results = n_results or self.config.reranking_top_n

        # Generate query embedding
        query_embedding = await self._generate_embedding(query, input_type="query")

        # Search vector store
        self._initialize_vector_store()
        if self.store.index is not None and self.store.is_trained and _load_faiss() is not None:
            candidates = self._search_faiss(query_embedding, category)
        else:
            candidates = self._search_memory(query_embedding, category)

        # Rerank candidates (simplified - use similarity scores)
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        return [c["document"] for c in ranked[:n_results]]

    def _search_faiss(self, query_embedding: np.ndarray, category: KnowledgeCategory | None) -> list[dict]:
        """Search using FAISS index."""
        distances, indices = self.store.index.search(query_embedding, self.config.retrieval_top_k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            doc = self.store.get_by_index(idx)
            if doc is None:
                continue

            # Filter by category if specified
            if category and doc.metadata.category != category:
                continue

            candidates.append({"document": doc, "score": float(dist)})

        return candidates

    def _search_memory(self, query_embedding: np.ndarray, category: KnowledgeCategory | None) -> list[dict]:
        """Fallback in-memory search when FAISS is unavailable."""
        candidates = []

        for doc in self.store.documents.values():
            # Filter by category
            if category and doc.metadata.category != category:
                continue

            if doc.embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(np.dot(query_embedding, doc.embedding.T)[0, 0])
            candidates.append({"document": doc, "score": similarity})

        return candidates

    async def query(
        self,
        query: str,
        category: KnowledgeCategory | None = None,
        _user_id: str | None = None,
        include_citations: bool = True,
    ) -> RAGResponse:
        """
        Query the knowledge base and generate a response.

        Uses intelligent model selection based on query complexity:
        - SIMPLE queries → Fast model (Nemotron-Nano-4B)
        - MODERATE queries → Balanced model (Llama-3.1-70B)
        - COMPLEX queries → Generation model (Nemotron-Super-49B)
        - CRISIS queries → Safety model (Nemotron-Safety-Guard)

        Args:
            query: User query
            category: Optional filter by category
            user_id: Optional user identifier for logging
            include_citations: Whether to include citations

        Returns:
            RAGResponse with generated text and sources
        """
        start_time = datetime.now(UTC)

        # Select appropriate model based on query complexity
        selected_model = self._select_model_for_query(query)
        logger.debug(f"Selected model '{selected_model}' for query complexity")

        # Retrieve relevant documents
        context_docs = await self.retrieve(query, category)

        if not context_docs:
            # No relevant documents found
            response_text = await self._generate_without_context(query, selected_model)
            latency = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return RAGResponse(
                response=response_text,
                sources=[],
                model=selected_model,
                retrieved_count=0,
                latency_ms=latency,
                citations=[],
            )

        # Build context string
        context_text = self._build_context(context_docs)

        # Generate response with context
        response_text = await self._generate_with_context(query, context_text)

        # Extract citations if requested
        citations = []
        if include_citations:
            citations = [doc.metadata.source for doc in context_docs]

        latency = (datetime.now(UTC) - start_time).total_seconds() * 1000

        return RAGResponse(
            response=response_text,
            sources=[doc.metadata.model_dump() for doc in context_docs],
            model=selected_model,
            retrieved_count=len(context_docs),
            latency_ms=latency,
            citations=citations,
        )

    def _build_context(self, documents: list[Document]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(documents):
            source = doc.metadata.source
            title = doc.metadata.title or "Untitled"
            content_preview = doc.content[:500]

            context_parts.append(f"[Source {i + 1}: {source} - {title}]\n{content_preview}")

        return "\n\n".join(context_parts)

    async def _generate_with_context(self, query: str, context: str, model: str | None = None) -> str:
        """Generate response using retrieved context.

        Args:
            query: User query
            context: Retrieved context documents
            model: Model to use (defaults to config.generation_model)

        Returns:
            Generated response text
        """
        model = model or self.config.generation_model
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._get_rag_system_prompt()},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=self.config.generation_temperature,
            max_tokens=self.config.max_generation_tokens,
        )

        return response.choices[0].message.content

    async def _generate_without_context(self, query: str, model: str | None = None) -> str:
        """Generate response when no relevant context is found.

        Args:
            query: User query
            model: Model to use (defaults to config.generation_model)

        Returns:
            Generated response text
        """
        model = model or self.config.generation_model
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._get_rag_system_prompt()},
                {
                    "role": "user",
                    "content": f"No relevant context was found in the knowledge base.\n\nQuestion: {query}",
                },
            ],
            temperature=self.config.generation_temperature,
            max_tokens=self.config.max_generation_tokens,
        )

        return response.choices[0].message.content

    def _get_rag_system_prompt(self) -> str:
        """System prompt for RAG responses."""
        return """You are Antigravity, a therapeutic AI companion with access to a comprehensive knowledge base of mental health resources.

When responding:
1. Prioritize evidence-based information from the provided context
2. Cite sources when making specific claims (e.g., "According to [Source]...")
3. Acknowledge when information is general guidance vs. specific clinical recommendations
4. Encourage professional consultation for clinical concerns
5. Maintain an empathetic, non-judgmental tone

If the context doesn't contain relevant information:
- Clearly state this limitation
- Provide general guidance with appropriate caveats
- Suggest consulting a mental health professional

Important safety considerations:
- If the user expresses thoughts of self-harm or suicide, prioritize safety resources
- Never diagnose or prescribe treatments
- Recommend professional help for serious concerns
- Validate emotions while maintaining appropriate boundaries"""

    async def batch_ingest(self, documents: list[dict[str, Any]]) -> list[str]:
        """
        Batch ingest multiple documents.

        Args:
            documents: List of dicts with 'document', 'metadata', 'doc_id' keys

        Returns:
            List of ingested document IDs
        """
        tasks = [
            self.ingest_document(document=doc["document"], metadata=doc.get("metadata", {}), doc_id=doc.get("doc_id"))
            for doc in documents
        ]

        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_documents": len(self.store),
            "index_type": self.config.index_type,
            "index_trained": self.store.is_trained,
            "embedding_dimension": self.config.embedding_dimension,
            "categories": {
                cat.value: sum(1 for doc in self.store.documents.values() if doc.metadata.category == cat)
                for cat in KnowledgeCategory
            },
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_rag_pipeline(
    api_key: str | None = None, index_type: IndexType = IndexType.IVF, **kwargs
) -> TherapeuticRAGPipeline:
    """
    Create a RAG pipeline with sensible defaults.

    Args:
        api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
        index_type: FAISS index type
        **kwargs: Additional configuration options

    Returns:
        Configured TherapeuticRAGPipeline instance
    """
    config = NemotronRAGConfig(api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""), index_type=index_type, **kwargs)

    return TherapeuticRAGPipeline(config)


# =============================================================================
# Convenience Functions
# =============================================================================


async def embed_text(text: str, api_key: str | None = None) -> np.ndarray:
    """
    Quick embedding generation for a single text.

    Args:
        text: Text to embed
        api_key: Optional API key

    Returns:
        Normalized embedding vector
    """
    client = AsyncOpenAI(base_url=NIM_BASE_URL, api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""))

    response = await client.embeddings.create(
        model=NIM_EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
        extra_body={"input_type": "query"},
    )

    embedding = np.array([response.data[0].embedding], dtype=np.float32)

    embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    if embedding_norm.size and float(embedding_norm.item()) > 0:
        embedding = embedding / embedding_norm

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b.T)[0, 0])
