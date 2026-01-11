"""
FAISS Knowledge Index Builder

Builds optimized FAISS indexes for clinical knowledge retrieval during training
with multiple index types, performance optimization, and comprehensive search capabilities.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning(
        "FAISS not available. Install: uv pip install faiss-cpu numpy")

from .clinical_knowledge_embedder import (
    ClinicalKnowledgeEmbedder,
    EmbeddingConfig,
    KnowledgeItem,
)


class IndexType(Enum):
    """FAISS index types for different use cases."""
    FLAT = "Flat"  # Exact search, best quality
    IVF_FLAT = "IVF_Flat"  # Inverted file with flat quantizer
    IVF_PQ = "IVF_PQ"  # Inverted file with product quantization
    HNSW = "HNSW"  # Hierarchical Navigable Small World
    LSH = "LSH"  # Locality Sensitive Hashing


@dataclass
class IndexConfig:
    """Configuration for FAISS index building."""
    index_type: IndexType = IndexType.IVF_FLAT
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    m: int = 8  # Number of subquantizers for PQ
    nbits: int = 8  # Number of bits per subquantizer
    hnsw_m: int = 16  # Number of connections for HNSW
    efConstruction: int = 200  # Construction parameter for HNSW
    efSearch: int = 50  # Search parameter for HNSW
    use_gpu: bool = False  # Use GPU acceleration if available
    normalize_vectors: bool = True  # Normalize vectors before indexing
    train_size_ratio: float = 0.1  # Ratio of data to use for training


@dataclass
class SearchResult:
    """Result from FAISS similarity search."""
    knowledge_item: KnowledgeItem
    score: float
    distance: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexStats:
    """Statistics about the FAISS index."""
    total_vectors: int
    index_type: str
    dimension: int
    is_trained: bool
    memory_usage_mb: float
    build_time_seconds: float
    search_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class FAISSKnowledgeIndex:
    """FAISS-based index for clinical knowledge retrieval."""

    def __init__(
            self,
            config: Optional[IndexConfig] = None,
            project_root: Optional[Path] = None):
        """Initialize the FAISS index builder."""
        self.config = config or IndexConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # FAISS index and related data
        self.index: Optional[faiss.Index] = None
        self.knowledge_items: List[KnowledgeItem] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.id_to_index_map: Dict[str, int] = {}
        self.index_to_id_map: Dict[int, str] = {}

        # Performance tracking
        self.stats: Optional[IndexStats] = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if not FAISS_AVAILABLE:
            self.logger.warning(
                "FAISS not available. Index will work in mock mode.")

    def load_knowledge_embeddings(
            self,
            embedder: Optional[ClinicalKnowledgeEmbedder] = None,
            embeddings_path: Optional[Path] = None) -> bool:
        """Load knowledge embeddings from embedder or file."""
        if embedder is not None:
            # Load from embedder
            self.knowledge_items = embedder.knowledge_items
            self.embeddings_matrix = embedder.embeddings_matrix
            self.logger.info(
                f"Loaded {len(self.knowledge_items)} items from embedder")
            return True

        elif embeddings_path is not None:
            # Load from file
            try:
                with open(embeddings_path, 'rb') as f:
                    data = pickle.load(f)

                self.knowledge_items = data["knowledge_items"]
                self.embeddings_matrix = data["embeddings_matrix"]
                self.logger.info(
                    f"Loaded {len(self.knowledge_items)} items from {embeddings_path}")
                return True

            except Exception as e:
                self.logger.error(
                    f"Failed to load embeddings from {embeddings_path}: {e}")
                return False

        else:
            # Try to load from default embedder
            embedder = ClinicalKnowledgeEmbedder(
                project_root=self.project_root)
            default_path = self.project_root / "ai" / "pixel" / \
                "data" / "clinical_knowledge_embeddings.pkl"

            if embedder.load_embeddings(default_path):
                self.knowledge_items = embedder.knowledge_items
                self.embeddings_matrix = embedder.embeddings_matrix
                self.logger.info(
                    f"Loaded {len(self.knowledge_items)} items from default path")
                return True
            else:
                self.logger.warning(
                    "No embeddings found. Creating mock data for testing.")
                self._create_mock_data()
                return True

    def _create_mock_data(self):
        """Create mock data for testing when real data isn't available."""

        # Create mock embedder and generate data
        config = EmbeddingConfig(embedding_dimension=384)
        embedder = ClinicalKnowledgeEmbedder(config, self.project_root)

        # Generate mock knowledge items
        mock_items = embedder._create_mock_knowledge_items()
        mock_items = embedder._generate_mock_embeddings(mock_items)

        self.knowledge_items = mock_items

        if FAISS_AVAILABLE:
            self.embeddings_matrix = np.vstack(
                [item.embedding for item in mock_items])
        else:
            self.embeddings_matrix = [item.embedding for item in mock_items]

        self.logger.info(f"Created {len(mock_items)} mock knowledge items")

    def _prepare_embeddings_matrix(self) -> np.ndarray:
        """Prepare embeddings matrix for FAISS indexing."""
        if not FAISS_AVAILABLE:
            return self.embeddings_matrix

        if self.embeddings_matrix is None:
            raise ValueError("No embeddings matrix available")

        # Ensure matrix is float32 (required by FAISS)
        if isinstance(self.embeddings_matrix, list):
            matrix = np.array(self.embeddings_matrix, dtype=np.float32)
        else:
            matrix = self.embeddings_matrix.astype(np.float32)

        # Normalize vectors if requested
        if self.config.normalize_vectors:
            faiss.normalize_L2(matrix)

        return matrix

    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if not FAISS_AVAILABLE:
            return MockFAISSIndex(dimension)

        if self.config.index_type == IndexType.FLAT:
            # Exact search index
            index = faiss.IndexFlatL2(dimension)

        elif self.config.index_type == IndexType.IVF_FLAT:
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            index.nprobe = self.config.nprobe

        elif self.config.index_type == IndexType.IVF_PQ:
            # IVF with product quantization
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, self.config.nlist,
                                     self.config.m, self.config.nbits)
            index.nprobe = self.config.nprobe

        elif self.config.index_type == IndexType.HNSW:
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
            index.hnsw.efConstruction = self.config.efConstruction
            index.hnsw.efSearch = self.config.efSearch

        elif self.config.index_type == IndexType.LSH:
            # Locality Sensitive Hashing
            index = faiss.IndexLSH(dimension, 256)  # 256 hash bits

        else:
            raise ValueError(
                f"Unsupported index type: {self.config.index_type}")

        self.logger.info(
            f"Created {self.config.index_type.value} index with dimension {dimension}")
        return index

    def build_index(self) -> bool:
        """Build the FAISS index from loaded embeddings."""
        if not self.knowledge_items or self.embeddings_matrix is None:
            self.logger.error("No knowledge items or embeddings available")
            return False

        start_time = time.time()

        try:
            # Prepare embeddings matrix
            embeddings = self._prepare_embeddings_matrix()
            dimension = len(
                embeddings[0]) if isinstance(
                embeddings,
                list) else embeddings.shape[1]

            # Create index
            self.index = self._create_faiss_index(dimension)

            # Train index if necessary
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.logger.info("Training index...")

                if FAISS_AVAILABLE:
                    # Use subset of data for training if dataset is large
                    train_size = max(
                        1000, int(
                            len(embeddings) * self.config.train_size_ratio))
                    train_data = embeddings[:train_size] if len(
                        embeddings) > train_size else embeddings
                    self.index.train(train_data)
                else:
                    self.index.train(embeddings)

                self.logger.info("Index training completed")

            # Add vectors to index
            self.logger.info(f"Adding {len(embeddings)} vectors to index...")
            if FAISS_AVAILABLE:
                self.index.add(embeddings)
            else:
                self.index.add(embeddings)

            # Create ID mappings
            self._create_id_mappings()

            # Calculate statistics
            build_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage()

            self.stats = IndexStats(
                total_vectors=len(self.knowledge_items),
                index_type=self.config.index_type.value,
                dimension=dimension,
                is_trained=getattr(self.index, 'is_trained', True),
                memory_usage_mb=memory_usage,
                build_time_seconds=build_time
            )

            self.logger.info(f"Index built successfully in {build_time:.2f}s")
            self.logger.info(
                f"Index stats: {self.stats.total_vectors} vectors, {memory_usage:.1f}MB")

            return True

        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            return False

    def _create_id_mappings(self):
        """Create mappings between knowledge item IDs and index positions."""
        self.id_to_index_map = {
            item.id: i for i, item in enumerate(
                self.knowledge_items)}
        self.index_to_id_map = {
            i: item.id for i, item in enumerate(
                self.knowledge_items)}

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the index in MB."""
        if not FAISS_AVAILABLE or self.index is None:
            # Rough estimate for mock index
            return len(self.knowledge_items) * 384 * 4 / \
                (1024 * 1024)  # 4 bytes per float32

        try:
            # Get actual memory usage from FAISS
            return self.index.sa_code_size() / (1024 * 1024)
        except BaseException:
            # Fallback estimation
            return len(self.knowledge_items) * \
                self.embeddings_matrix.shape[1] * 4 / (1024 * 1024)

    def search(self,
               query_embedding: Union[np.ndarray,
                                      List[float]],
               k: int = 10,
               return_distances: bool = True) -> List[SearchResult]:
        """Search for similar knowledge items."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        start_time = time.time()

        try:
            # Prepare query embedding
            if isinstance(query_embedding, list):
                if FAISS_AVAILABLE:
                    query = np.array([query_embedding], dtype=np.float32)
                else:
                    query = [query_embedding]
            else:
                if FAISS_AVAILABLE:
                    query = query_embedding.reshape(1, -1).astype(np.float32)
                else:
                    query = query_embedding.tolist()

            # Normalize query if configured
            if self.config.normalize_vectors and FAISS_AVAILABLE:
                faiss.normalize_L2(query)

            # Perform search
            if FAISS_AVAILABLE:
                distances, indices = self.index.search(query, k)
                distances = distances[0]  # First query result
                indices = indices[0]
            else:
                distances, indices = self.index.search(query, k)

            # Create search results
            results = []
            for rank, (idx, distance) in enumerate(zip(indices, distances)):
                if idx >= 0 and idx < len(self.knowledge_items):  # Valid index
                    knowledge_item = self.knowledge_items[idx]

                    # Convert distance to similarity score (higher is better)
                    score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

                    result = SearchResult(
                        knowledge_item=knowledge_item,
                        score=score,
                        distance=float(distance),
                        rank=rank,
                        metadata={
                            "index_position": int(idx),
                            "search_time_ms": (time.time() - start_time) * 1000
                        }
                    )
                    results.append(result)

            # Update search time statistics
            search_time_ms = (time.time() - start_time) * 1000
            if self.stats:
                self.stats.search_time_ms = search_time_ms

            self.logger.debug(
                f"Search completed in {search_time_ms:.2f}ms, found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def search_by_text(
            self,
            query_text: str,
            k: int = 10,
            embedder: Optional[ClinicalKnowledgeEmbedder] = None) -> List[SearchResult]:
        """Search using text query (requires embedder for encoding)."""
        if embedder is None:
            # Create default embedder
            embedder = ClinicalKnowledgeEmbedder(
                project_root=self.project_root)

        # Generate embedding for query text
        if FAISS_AVAILABLE and embedder.embedding_model is not None:
            query_embedding = embedder.embedding_model.encode([query_text])[0]
        else:
            # Use mock embedding
            hash_int = hash(query_text) % 1000000
            query_embedding = [
                (hash_int + i) %
                1000 / 1000.0 for i in range(384)]

        return self.search(query_embedding, k)

    def filter_search(self,
                      query_embedding: Union[np.ndarray,
                                             List[float]],
                      k: int = 10,
                      filters: Optional[Dict[str,
                                             Any]] = None) -> List[SearchResult]:
        """Search with metadata filters."""
        # Get initial search results
        # Get more to allow for filtering
        initial_results = self.search(query_embedding, k * 2)

        if not filters:
            return initial_results[:k]

        # Apply filters
        filtered_results = []
        for result in initial_results:
            item = result.knowledge_item

            # Check each filter condition
            matches = True
            for filter_key, filter_value in filters.items():
                if filter_key == "knowledge_type":
                    if item.knowledge_type != filter_value:
                        matches = False
                        break
                elif filter_key == "source":
                    if item.source != filter_value:
                        matches = False
                        break
                elif filter_key in item.metadata:
                    if item.metadata[filter_key] != filter_value:
                        matches = False
                        break

            if matches:
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break

        return filtered_results

    def save_index(self, output_path: Optional[Path] = None) -> Path:
        """Save the FAISS index and metadata to disk."""
        if output_path is None:
            output_path = self.project_root / "ai" / \
                "pixel" / "data" / "faiss_knowledge_index"

        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save FAISS index
            if FAISS_AVAILABLE and self.index is not None:
                index_file = output_path / "faiss.index"
                faiss.write_index(self.index, str(index_file))
            else:
                # Save mock index
                index_file = output_path / "mock_index.pkl"
                with open(index_file, 'wb') as f:
                    pickle.dump(self.index, f)

            # Save metadata
            metadata = {
                "config": self.config,
                "knowledge_items": self.knowledge_items,
                "id_to_index_map": self.id_to_index_map,
                "index_to_id_map": self.index_to_id_map,
                "stats": self.stats,
                "created_at": datetime.now()
            }

            metadata_file = output_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            # Save configuration as JSON for easy inspection
            config_file = output_path / "config.json"
            config_dict = {
                "index_type": self.config.index_type.value,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe,
                "dimension": self.stats.dimension if self.stats else 384,
                "total_vectors": len(self.knowledge_items),
                "created_at": datetime.now().isoformat()
            }

            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Index saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, input_path: Optional[Path] = None) -> bool:
        """Load FAISS index and metadata from disk."""
        if input_path is None:
            input_path = self.project_root / "ai" / \
                "pixel" / "data" / "faiss_knowledge_index"

        if not input_path.exists():
            self.logger.warning(f"Index directory not found: {input_path}")
            return False

        try:
            # Load metadata
            metadata_file = input_path / "metadata.pkl"
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            self.config = metadata["config"]
            self.knowledge_items = metadata["knowledge_items"]
            self.id_to_index_map = metadata["id_to_index_map"]
            self.index_to_id_map = metadata["index_to_id_map"]
            self.stats = metadata["stats"]

            # Load FAISS index
            if FAISS_AVAILABLE:
                index_file = input_path / "faiss.index"
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                else:
                    # Try loading mock index
                    mock_index_file = input_path / "mock_index.pkl"
                    with open(mock_index_file, 'rb') as f:
                        self.index = pickle.load(f)
            else:
                # Load mock index
                mock_index_file = input_path / "mock_index.pkl"
                with open(mock_index_file, 'rb') as f:
                    self.index = pickle.load(f)

            self.logger.info(f"Index loaded from {input_path}")
            self.logger.info(
                f"Loaded {len(self.knowledge_items)} knowledge items")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False

    def get_index_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the index."""
        info = {
            "is_built": self.index is not None,
            "knowledge_items_count": len(self.knowledge_items),
            "config": {
                "index_type": self.config.index_type.value,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe,
                "normalize_vectors": self.config.normalize_vectors
            }
        }

        if self.stats:
            info["stats"] = {
                "total_vectors": self.stats.total_vectors,
                "dimension": self.stats.dimension,
                "is_trained": self.stats.is_trained,
                "memory_usage_mb": self.stats.memory_usage_mb,
                "build_time_seconds": self.stats.build_time_seconds,
                "search_time_ms": self.stats.search_time_ms,
                "created_at": self.stats.created_at.isoformat()
            }

        if self.index is not None and FAISS_AVAILABLE:
            info["faiss_info"] = {
                "ntotal": self.index.ntotal,
                "d": self.index.d,
                "is_trained": getattr(self.index, 'is_trained', True)
            }

        return info

    def benchmark_search_performance(
            self, num_queries: int = 100, k: int = 10) -> Dict[str, float]:
        """Benchmark search performance with random queries."""
        if self.index is None:
            raise ValueError("Index not built")

        if not self.knowledge_items:
            raise ValueError("No knowledge items available")

        self.logger.info(
            f"Benchmarking search performance with {num_queries} queries...")

        # Generate random query embeddings
        dimension = len(self.knowledge_items[0].embedding)

        search_times = []

        for i in range(num_queries):
            # Create random query
            if FAISS_AVAILABLE:
                query = np.random.normal(0, 1, dimension).astype(np.float32)
            else:
                query = [np.random.normal(0, 1) for _ in range(dimension)]

            # Time the search
            start_time = time.time()
            results = self.search(query, k)
            search_time = (time.time() - start_time) * 1000  # Convert to ms

            search_times.append(search_time)

        # Calculate statistics
        benchmark_results = {
            "avg_search_time_ms": np.mean(search_times) if FAISS_AVAILABLE else sum(search_times) / len(search_times),
            "min_search_time_ms": min(search_times),
            "max_search_time_ms": max(search_times),
            "std_search_time_ms": np.std(search_times) if FAISS_AVAILABLE else 0.0,
            "queries_per_second": 1000 / (sum(search_times) / len(search_times)),
            "total_benchmark_time_s": sum(search_times) / 1000
        }

        self.logger.info(
            f"Benchmark completed: {benchmark_results['avg_search_time_ms']:.2f}ms avg search time")
        return benchmark_results


class MockFAISSIndex:
    """Mock FAISS index for testing when FAISS is not available."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ntotal = 0
        self.d = dimension
        self.is_trained = True

    def add(self, vectors):
        """Add vectors to mock index."""
        if isinstance(vectors, list):
            self.vectors.extend(vectors)
        else:
            self.vectors.extend(vectors.tolist())
        self.ntotal = len(self.vectors)

    def search(self, query, k):
        """Mock search implementation."""
        if isinstance(query, list):
            query_vec = query[0] if len(query) > 0 else query
        else:
            query_vec = query.tolist()[0] if len(
                query.shape) > 1 else query.tolist()

        # Calculate simple distances (mock implementation)
        distances = []
        for i, vec in enumerate(self.vectors):
            # Simple Euclidean distance
            dist = sum((a - b) ** 2 for a, b in zip(query_vec, vec)) ** 0.5
            distances.append((dist, i))

        # Sort by distance and return top k
        distances.sort()
        top_k = distances[:k]

        result_distances = [d[0] for d in top_k]
        result_indices = [d[1] for d in top_k]

        return result_distances, result_indices

    def train(self, vectors):
        """Mock training (no-op)."""
        pass


def main():
    """Test the FAISS knowledge index."""
    print("Testing FAISS Knowledge Index")

    # Create configuration
    config = IndexConfig(
        index_type=IndexType.IVF_FLAT,
        nlist=50,
        nprobe=5
    )

    # Initialize index
    faiss_index = FAISSKnowledgeIndex(config)

    # Load knowledge embeddings
    success = faiss_index.load_knowledge_embeddings()
    if not success:
        print("Failed to load knowledge embeddings")
        return

    print(f"Loaded {len(faiss_index.knowledge_items)} knowledge items")

    # Build index
    if faiss_index.build_index():
        print("Index built successfully")

        # Get index info
        info = faiss_index.get_index_info()
        print("Index Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test search
        if faiss_index.knowledge_items:
            # Use first item's embedding as query
            query_embedding = faiss_index.knowledge_items[0].embedding
            results = faiss_index.search(query_embedding, k=3)

            print(f"\nSearch Results ({len(results)} found):")
            for result in results:
                print(f"  Rank {result.rank}: {result.knowledge_item.id}")
                print(
                    f"    Score: {result.score:.4f}, Distance: {result.distance:.4f}")
                print(f"    Content: {result.knowledge_item.content[:100]}...")

        # Benchmark performance
        try:
            benchmark = faiss_index.benchmark_search_performance(
                num_queries=10, k=5)
            print("\nBenchmark Results:")
            for key, value in benchmark.items():
                print(f"  {key}: {value:.4f}")
        except Exception as e:
            print(f"Benchmark failed: {e}")

        # Test save/load
        try:
            save_path = faiss_index.save_index()
            print(f"\nIndex saved to: {save_path}")

            # Test loading
            new_index = FAISSKnowledgeIndex(config)
            if new_index.load_index(save_path):
                print("Index loaded successfully")
            else:
                print("Failed to load index")

        except Exception as e:
            print(f"Save/load test failed: {e}")

    else:
        print("Failed to build index")


if __name__ == "__main__":
    main()
