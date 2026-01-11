"""
Unit Tests for FAISS Knowledge Index

Tests FAISS index building, search functionality, and performance optimization.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .clinical_knowledge_embedder import KnowledgeItem
from .faiss_knowledge_index import (
    FAISS_AVAILABLE,
    FAISSKnowledgeIndex,
    IndexConfig,
    IndexStats,
    IndexType,
    MockFAISSIndex,
    SearchResult,
)


class TestIndexConfig:
    """Test index configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IndexConfig()
        
        assert config.index_type == IndexType.IVF_FLAT
        assert config.nlist == 100
        assert config.nprobe == 10
        assert config.m == 8
        assert config.nbits == 8
        assert config.hnsw_m == 16
        assert config.efConstruction == 200
        assert config.efSearch == 50
        assert config.use_gpu is False
        assert config.normalize_vectors is True
        assert config.train_size_ratio == 0.1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = IndexConfig(
            index_type=IndexType.HNSW,
            nlist=200,
            nprobe=20,
            hnsw_m=32,
            use_gpu=True,
            normalize_vectors=False
        )
        
        assert config.index_type == IndexType.HNSW
        assert config.nlist == 200
        assert config.nprobe == 20
        assert config.hnsw_m == 32
        assert config.use_gpu is True
        assert config.normalize_vectors is False


class TestIndexType:
    """Test index type enumeration."""
    
    def test_index_types(self):
        """Test all index types are available."""
        assert IndexType.FLAT.value == "Flat"
        assert IndexType.IVF_FLAT.value == "IVF_Flat"
        assert IndexType.IVF_PQ.value == "IVF_PQ"
        assert IndexType.HNSW.value == "HNSW"
        assert IndexType.LSH.value == "LSH"


class TestSearchResult:
    """Test search result data structure."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        knowledge_item = KnowledgeItem(
            id="test_item",
            content="Test content",
            knowledge_type="dsm5"
        )
        
        result = SearchResult(
            knowledge_item=knowledge_item,
            score=0.95,
            distance=0.05,
            rank=0,
            metadata={"test": "value"}
        )
        
        assert result.knowledge_item == knowledge_item
        assert result.score == 0.95
        assert result.distance == 0.05
        assert result.rank == 0
        assert result.metadata["test"] == "value"


class TestIndexStats:
    """Test index statistics."""
    
    def test_index_stats_creation(self):
        """Test creating index statistics."""
        stats = IndexStats(
            total_vectors=1000,
            index_type="IVF_Flat",
            dimension=384,
            is_trained=True,
            memory_usage_mb=50.5,
            build_time_seconds=12.3
        )
        
        assert stats.total_vectors == 1000
        assert stats.index_type == "IVF_Flat"
        assert stats.dimension == 384
        assert stats.is_trained is True
        assert stats.memory_usage_mb == 50.5
        assert stats.build_time_seconds == 12.3
        assert isinstance(stats.created_at, datetime)


class TestMockFAISSIndex:
    """Test mock FAISS index functionality."""
    
    def test_mock_index_creation(self):
        """Test creating a mock FAISS index."""
        index = MockFAISSIndex(384)
        
        assert index.dimension == 384
        assert index.d == 384
        assert index.ntotal == 0
        assert index.is_trained is True
        assert index.vectors == []
    
    def test_mock_index_add_vectors(self):
        """Test adding vectors to mock index."""
        index = MockFAISSIndex(3)
        
        # Add list of vectors
        vectors = [[1, 2, 3], [4, 5, 6]]
        index.add(vectors)
        
        assert index.ntotal == 2
        assert len(index.vectors) == 2
        assert index.vectors[0] == [1, 2, 3]
        assert index.vectors[1] == [4, 5, 6]
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_mock_index_add_numpy_vectors(self):
        """Test adding numpy vectors to mock index."""
        index = MockFAISSIndex(3)
        
        vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        index.add(vectors)
        
        assert index.ntotal == 2
        assert len(index.vectors) == 2
    
    def test_mock_index_search(self):
        """Test search functionality in mock index."""
        index = MockFAISSIndex(3)
        
        # Add some vectors
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        index.add(vectors)
        
        # Search for similar vector
        query = [[1, 0, 0]]  # Should be closest to first vector
        distances, indices = index.search(query, k=2)
        
        assert len(distances) == 2
        assert len(indices) == 2
        assert indices[0] == 0  # First vector should be closest
        assert distances[0] < distances[1]  # First distance should be smaller
    
    def test_mock_index_train(self):
        """Test training mock index (no-op)."""
        index = MockFAISSIndex(3)
        
        # Training should not raise an error
        vectors = [[1, 2, 3], [4, 5, 6]]
        index.train(vectors)
        
        # Should still be trained
        assert index.is_trained is True


class TestFAISSKnowledgeIndex:
    """Test FAISS knowledge index functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = IndexConfig(
            index_type=IndexType.FLAT,  # Use simple index for testing
            nlist=10,
            nprobe=5
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_faiss_index_initialization(self):
        """Test FAISS index initialization."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        assert faiss_index.config == self.config
        assert faiss_index.project_root == self.temp_dir
        assert faiss_index.index is None
        assert faiss_index.knowledge_items == []
        assert faiss_index.embeddings_matrix is None
        assert faiss_index.stats is None
    
    def test_create_mock_data(self):
        """Test creation of mock data."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index._create_mock_data()
        
        assert len(faiss_index.knowledge_items) > 0
        assert faiss_index.embeddings_matrix is not None
        
        # Check that all items have embeddings
        for item in faiss_index.knowledge_items:
            assert item.embedding is not None
    
    def test_load_knowledge_embeddings_mock(self):
        """Test loading knowledge embeddings with mock data."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        # Should create mock data when no real data is available
        success = faiss_index.load_knowledge_embeddings()
        
        assert success is True
        assert len(faiss_index.knowledge_items) > 0
        assert faiss_index.embeddings_matrix is not None
    
    def test_load_knowledge_embeddings_from_embedder(self):
        """Test loading knowledge embeddings from embedder."""
        # Create mock embedder
        embedder = Mock()
        embedder.knowledge_items = [
            KnowledgeItem(id="test1", content="content1", embedding=[1, 2, 3]),
            KnowledgeItem(id="test2", content="content2", embedding=[4, 5, 6])
        ]
        embedder.embeddings_matrix = [[1, 2, 3], [4, 5, 6]]
        
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        success = faiss_index.load_knowledge_embeddings(embedder=embedder)
        
        assert success is True
        assert len(faiss_index.knowledge_items) == 2
        assert faiss_index.embeddings_matrix == [[1, 2, 3], [4, 5, 6]]
    
    def test_build_index_no_data(self):
        """Test building index with no data."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        success = faiss_index.build_index()
        
        assert success is False
    
    def test_build_index_with_mock_data(self):
        """Test building index with mock data."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()  # Load mock data
        
        success = faiss_index.build_index()
        
        assert success is True
        assert faiss_index.index is not None
        assert faiss_index.stats is not None
        assert faiss_index.stats.total_vectors == len(faiss_index.knowledge_items)
        assert len(faiss_index.id_to_index_map) == len(faiss_index.knowledge_items)
        assert len(faiss_index.index_to_id_map) == len(faiss_index.knowledge_items)
    
    def test_create_id_mappings(self):
        """Test creation of ID mappings."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.knowledge_items = [
            KnowledgeItem(id="item1", content="content1"),
            KnowledgeItem(id="item2", content="content2"),
            KnowledgeItem(id="item3", content="content3")
        ]
        
        faiss_index._create_id_mappings()
        
        assert faiss_index.id_to_index_map["item1"] == 0
        assert faiss_index.id_to_index_map["item2"] == 1
        assert faiss_index.id_to_index_map["item3"] == 2
        
        assert faiss_index.index_to_id_map[0] == "item1"
        assert faiss_index.index_to_id_map[1] == "item2"
        assert faiss_index.index_to_id_map[2] == "item3"
    
    def test_search_without_index(self):
        """Test search without built index."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        with pytest.raises(ValueError, match="Index not built"):
            faiss_index.search([1, 2, 3])
    
    def test_search_with_built_index(self):
        """Test search with built index."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        # Use first item's embedding as query
        query_embedding = faiss_index.knowledge_items[0].embedding
        results = faiss_index.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, SearchResult) for result in results)
        
        if results:
            # First result should be the exact match (distance 0 or very small)
            assert results[0].distance <= 0.1
            assert results[0].rank == 0
    
    def test_search_by_text(self):
        """Test text-based search."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        results = faiss_index.search_by_text("depression symptoms", k=3)
        
        assert isinstance(results, list)
        assert all(isinstance(result, SearchResult) for result in results)
    
    def test_filter_search(self):
        """Test search with filters."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        # Search with knowledge type filter
        query_embedding = faiss_index.knowledge_items[0].embedding
        filters = {"knowledge_type": "dsm5"}
        
        results = faiss_index.filter_search(query_embedding, k=2, filters=filters)
        
        assert isinstance(results, list)
        # All results should match the filter
        for result in results:
            assert result.knowledge_item.knowledge_type == "dsm5"
    
    def test_get_index_info_empty(self):
        """Test getting index info when empty."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        info = faiss_index.get_index_info()
        
        assert info["is_built"] is False
        assert info["knowledge_items_count"] == 0
        assert "config" in info
        assert info["config"]["index_type"] == self.config.index_type.value
    
    def test_get_index_info_built(self):
        """Test getting index info when built."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        info = faiss_index.get_index_info()
        
        assert info["is_built"] is True
        assert info["knowledge_items_count"] > 0
        assert "stats" in info
        assert "config" in info
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        # Build index
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        # Save index
        save_path = self.temp_dir / "test_index"
        saved_path = faiss_index.save_index(save_path)
        
        assert saved_path == save_path
        assert save_path.exists()
        assert (save_path / "metadata.pkl").exists()
        assert (save_path / "config.json").exists()
        
        # Load index in new instance
        new_faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        success = new_faiss_index.load_index(save_path)
        
        assert success is True
        assert len(new_faiss_index.knowledge_items) == len(faiss_index.knowledge_items)
        assert new_faiss_index.index is not None
        assert new_faiss_index.stats is not None
    
    def test_load_index_not_found(self):
        """Test loading index when directory doesn't exist."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        non_existent_path = self.temp_dir / "non_existent"
        success = faiss_index.load_index(non_existent_path)
        
        assert success is False
    
    def test_benchmark_search_performance(self):
        """Test search performance benchmarking."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index.load_knowledge_embeddings()
        faiss_index.build_index()
        
        benchmark_results = faiss_index.benchmark_search_performance(num_queries=5, k=3)
        
        assert "avg_search_time_ms" in benchmark_results
        assert "min_search_time_ms" in benchmark_results
        assert "max_search_time_ms" in benchmark_results
        assert "queries_per_second" in benchmark_results
        assert "total_benchmark_time_s" in benchmark_results
        
        assert benchmark_results["avg_search_time_ms"] > 0
        assert benchmark_results["queries_per_second"] > 0
    
    def test_benchmark_without_index(self):
        """Test benchmarking without built index."""
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        with pytest.raises(ValueError, match="Index not built"):
            faiss_index.benchmark_search_performance()
    
    def test_different_index_types(self):
        """Test building different types of indexes."""
        index_types = [IndexType.FLAT, IndexType.IVF_FLAT]
        
        for index_type in index_types:
            config = IndexConfig(index_type=index_type, nlist=5)
            faiss_index = FAISSKnowledgeIndex(config, self.temp_dir)
            faiss_index.load_knowledge_embeddings()
            
            success = faiss_index.build_index()
            assert success is True, f"Failed to build {index_type.value} index"
            
            # Test search
            query_embedding = faiss_index.knowledge_items[0].embedding
            results = faiss_index.search(query_embedding, k=2)
            assert len(results) <= 2


class TestIntegration:
    """Integration tests for the complete FAISS index system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = IndexConfig(index_type=IndexType.FLAT, nlist=10)
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_index_workflow(self):
        """Test complete index building and search workflow."""
        # Initialize index
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        # Load knowledge (will use mock data)
        success = faiss_index.load_knowledge_embeddings()
        assert success is True
        
        # Build index
        success = faiss_index.build_index()
        assert success is True
        
        # Perform searches
        query_embedding = faiss_index.knowledge_items[0].embedding
        results = faiss_index.search(query_embedding, k=3)
        assert len(results) <= 3
        
        # Test text search
        text_results = faiss_index.search_by_text("test query", k=2)
        assert len(text_results) <= 2
        
        # Test filtered search
        filtered_results = faiss_index.filter_search(
            query_embedding, k=2, filters={"knowledge_type": "dsm5"}
        )
        assert isinstance(filtered_results, list)
        
        # Get index info
        info = faiss_index.get_index_info()
        assert info["is_built"] is True
        
        # Benchmark performance
        benchmark = faiss_index.benchmark_search_performance(num_queries=3, k=2)
        assert benchmark["avg_search_time_ms"] > 0
    
    def test_index_persistence(self):
        """Test that index persists across instances."""
        # Build and save index
        faiss_index1 = FAISSKnowledgeIndex(self.config, self.temp_dir)
        faiss_index1.load_knowledge_embeddings()
        faiss_index1.build_index()
        
        save_path = faiss_index1.save_index()
        
        # Load index in new instance
        faiss_index2 = FAISSKnowledgeIndex(self.config, self.temp_dir)
        success = faiss_index2.load_index(save_path)
        
        assert success is True
        assert len(faiss_index2.knowledge_items) == len(faiss_index1.knowledge_items)
        
        # Test that searches work the same
        query_embedding = faiss_index1.knowledge_items[0].embedding
        
        results1 = faiss_index1.search(query_embedding, k=3)
        results2 = faiss_index2.search(query_embedding, k=3)
        
        assert len(results1) == len(results2)
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.knowledge_item.id == r2.knowledge_item.id
            assert abs(r1.distance - r2.distance) < 0.001  # Allow for small floating point differences
    
    def test_large_scale_mock_performance(self):
        """Test performance with larger mock dataset."""
        # Create larger mock dataset
        faiss_index = FAISSKnowledgeIndex(self.config, self.temp_dir)
        
        # Create many mock items
        mock_items = []
        for i in range(100):
            item = KnowledgeItem(
                id=f"mock_item_{i}",
                content=f"Mock content for item {i}",
                knowledge_type="dsm5" if i % 2 == 0 else "pdm2",
                embedding=[float(j + i) for j in range(384)]
            )
            mock_items.append(item)
        
        faiss_index.knowledge_items = mock_items
        if FAISS_AVAILABLE:
            faiss_index.embeddings_matrix = np.array([item.embedding for item in mock_items])
        else:
            faiss_index.embeddings_matrix = [item.embedding for item in mock_items]
        
        # Build index
        success = faiss_index.build_index()
        assert success is True
        
        # Test search performance
        query_embedding = mock_items[0].embedding
        results = faiss_index.search(query_embedding, k=10)
        
        assert len(results) == 10
        assert results[0].distance <= results[-1].distance  # Results should be sorted by distance
        
        # Benchmark with larger dataset
        benchmark = faiss_index.benchmark_search_performance(num_queries=10, k=5)
        assert benchmark["queries_per_second"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
