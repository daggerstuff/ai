"""
Unit Tests for Clinical Knowledge Embedder

Tests vector embedding generation for psychology knowledge items.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .clinical_knowledge_embedder import (
    ClinicalKnowledgeEmbedder,
    EmbeddingConfig,
    KnowledgeItem,
)


class TestEmbeddingConfig:
    """Test embedding configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.normalize_embeddings is True
        assert config.cache_embeddings is True
        assert config.embedding_dimension == 384
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            batch_size=16,
            max_length=256,
            normalize_embeddings=False,
            cache_embeddings=False,
            embedding_dimension=768
        )
        
        assert config.model_name == "custom-model"
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.normalize_embeddings is False
        assert config.cache_embeddings is False
        assert config.embedding_dimension == 768


class TestKnowledgeItem:
    """Test knowledge item data structure."""
    
    def test_knowledge_item_creation(self):
        """Test creating a knowledge item."""
        item = KnowledgeItem(
            id="test_item",
            content="Test content for psychology knowledge",
            metadata={"source": "test", "category": "depression"},
            knowledge_type="dsm5",
            source="test_source"
        )
        
        assert item.id == "test_item"
        assert item.content == "Test content for psychology knowledge"
        assert item.metadata["source"] == "test"
        assert item.knowledge_type == "dsm5"
        assert item.source == "test_source"
        assert hasattr(item, 'content_hash')
        assert isinstance(item.created_at, datetime)
    
    def test_knowledge_item_content_hash(self):
        """Test content hash generation."""
        item1 = KnowledgeItem(id="1", content="Same content")
        item2 = KnowledgeItem(id="2", content="Same content")
        item3 = KnowledgeItem(id="3", content="Different content")
        
        assert item1.content_hash == item2.content_hash
        assert item1.content_hash != item3.content_hash
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_knowledge_item_with_embedding(self):
        """Test knowledge item with embedding."""
        embedding = np.random.normal(0, 1, 384).astype(np.float32)
        item = KnowledgeItem(
            id="test_item",
            content="Test content",
            embedding=embedding
        )
        
        assert item.embedding is not None
        assert item.embedding.shape == (384,)


class TestClinicalKnowledgeEmbedder:
    """Test clinical knowledge embedder functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=2,
            cache_embeddings=True,
            embedding_dimension=384
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        assert embedder.config == self.config
        assert embedder.project_root == self.temp_dir
        assert embedder.psychology_loader is not None
        assert embedder.knowledge_items == []
        assert embedder.embeddings_matrix is None
    
    @patch('ai.pixel.data.clinical_knowledge_embedder.DEPENDENCIES_AVAILABLE', False)
    def test_embedder_without_dependencies(self):
        """Test embedder behavior without dependencies."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        assert embedder.embedding_model is None
    
    def test_create_mock_knowledge_items(self):
        """Test creation of mock knowledge items."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        mock_items = embedder._create_mock_knowledge_items()
        
        assert len(mock_items) == 3
        assert any(item.knowledge_type == "dsm5" for item in mock_items)
        assert any(item.knowledge_type == "therapeutic_technique" for item in mock_items)
        assert any(item.knowledge_type == "pdm2" for item in mock_items)
        
        for item in mock_items:
            assert item.id is not None
            assert item.content is not None
            assert item.source == "mock_data"
    
    def test_generate_mock_embeddings(self):
        """Test generation of mock embeddings."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        knowledge_items = embedder._create_mock_knowledge_items()
        
        items_with_embeddings = embedder._generate_mock_embeddings(knowledge_items)
        
        assert len(items_with_embeddings) == len(knowledge_items)
        for item in items_with_embeddings:
            assert item.embedding is not None
            assert len(item.embedding) == self.config.embedding_dimension
    
    def test_extract_conversation_content(self):
        """Test extraction of conversation content."""
        from .therapeutic_conversation_schema import (
            ClinicalContext,
            ConversationRole,
            TherapeuticConversation,
        )
        
        # Create mock conversation
        conversation = TherapeuticConversation(
            title="Test Therapy Session",
            clinical_context=ClinicalContext(
                dsm5_categories=["Major Depressive Disorder"],
                presenting_concerns=["Persistent sadness", "Loss of interest"]
            )
        )
        
        conversation.add_turn(
            ConversationRole.CLIENT,
            "I've been feeling really down lately.",
            clinical_rationale="Client expressing depressive symptoms"
        )
        
        conversation.add_turn(
            ConversationRole.THERAPIST,
            "I hear that you're struggling. Can you tell me more about these feelings?",
            clinical_rationale="Empathetic response with open-ended question"
        )
        
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        content = embedder._extract_conversation_content(conversation)
        
        assert "Test Therapy Session" in content
        assert "Major Depressive Disorder" in content
        assert "Persistent sadness" in content
        assert "Client:" in content
        assert "Therapist:" in content
        assert "Clinical Rationale:" in content
    
    @patch.object(ClinicalKnowledgeEmbedder, '_create_mock_knowledge_items')
    def test_extract_psychology_knowledge_items_with_mock(self, mock_create_mock):
        """Test extraction of psychology knowledge items using mock data."""
        mock_items = [
            KnowledgeItem(id="mock1", content="Mock content 1", knowledge_type="dsm5"),
            KnowledgeItem(id="mock2", content="Mock content 2", knowledge_type="pdm2")
        ]
        mock_create_mock.return_value = mock_items
        
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Mock the psychology loader methods to raise exceptions (simulating missing data)
        embedder.psychology_loader.load_dsm5_knowledge = Mock(side_effect=Exception("No data"))
        embedder.psychology_loader.load_pdm2_knowledge = Mock(side_effect=Exception("No data"))
        embedder.psychology_loader.load_therapeutic_conversations = Mock(side_effect=Exception("No data"))
        embedder.psychology_loader.load_clinical_datasets = Mock(side_effect=Exception("No data"))
        
        knowledge_items = embedder.extract_psychology_knowledge_items()
        
        assert len(knowledge_items) == 2
        assert knowledge_items == mock_items
        mock_create_mock.assert_called_once()
    
    def test_create_embeddings_matrix_empty_items(self):
        """Test creating embeddings matrix with empty items."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        with pytest.raises(ValueError, match="No knowledge items provided"):
            embedder.create_embeddings_matrix([])
    
    def test_create_embeddings_matrix_no_embeddings(self):
        """Test creating embeddings matrix with items without embeddings."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        items = [KnowledgeItem(id="test", content="test content")]
        
        with pytest.raises(ValueError, match="No knowledge items have embeddings"):
            embedder.create_embeddings_matrix(items)
    
    def test_create_embeddings_matrix_with_embeddings(self):
        """Test creating embeddings matrix with valid embeddings."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Create items with mock embeddings
        items = embedder._create_mock_knowledge_items()
        items = embedder._generate_mock_embeddings(items)
        
        matrix = embedder.create_embeddings_matrix(items)
        
        if NUMPY_AVAILABLE:
            assert matrix.shape == (len(items), self.config.embedding_dimension)
        else:
            assert len(matrix) == len(items)
            assert len(matrix[0]) == self.config.embedding_dimension
        
        assert embedder.embeddings_matrix is not None
        assert len(embedder.knowledge_items) == len(items)
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Create and process mock items
        items = embedder._create_mock_knowledge_items()
        items = embedder._generate_mock_embeddings(items)
        embedder.knowledge_items = items
        embedder.create_embeddings_matrix(items)
        
        # Save embeddings
        output_path = self.temp_dir / "test_embeddings.pkl"
        saved_path = embedder.save_embeddings(output_path)
        
        assert saved_path == output_path
        assert output_path.exists()
        
        # Create new embedder and load embeddings
        new_embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        success = new_embedder.load_embeddings(output_path)
        
        assert success is True
        assert len(new_embedder.knowledge_items) == len(items)
        assert new_embedder.embeddings_matrix is not None
    
    def test_load_embeddings_file_not_found(self):
        """Test loading embeddings when file doesn't exist."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        non_existent_path = self.temp_dir / "non_existent.pkl"
        success = embedder.load_embeddings(non_existent_path)
        
        assert success is False
    
    def test_get_embedding_stats_empty(self):
        """Test getting embedding statistics with no items."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        stats = embedder.get_embedding_stats()
        
        assert "error" in stats
        assert stats["error"] == "No knowledge items loaded"
    
    def test_get_embedding_stats_with_items(self):
        """Test getting embedding statistics with items."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Create items with embeddings
        items = embedder._create_mock_knowledge_items()
        items = embedder._generate_mock_embeddings(items)
        embedder.knowledge_items = items
        
        stats = embedder.get_embedding_stats()
        
        assert stats["total_items"] == 3
        assert stats["embedding_dimension"] == self.config.embedding_dimension
        assert stats["items_with_embeddings"] == 3
        assert "knowledge_types" in stats
        assert "sources" in stats
        
        # Check knowledge type counts
        assert stats["knowledge_types"]["dsm5"] == 1
        assert stats["knowledge_types"]["therapeutic_technique"] == 1
        assert stats["knowledge_types"]["pdm2"] == 1
    
    def test_embeddings_cache_functionality(self):
        """Test embeddings caching functionality."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Create test item
        item = KnowledgeItem(id="test", content="test content")
        
        # Mock embedding generation
        mock_embedding = [0.1] * self.config.embedding_dimension
        embedder.embeddings_cache[item.content_hash] = mock_embedding
        
        # Generate embeddings (should use cache)
        items_with_embeddings = embedder.generate_embeddings([item])
        
        assert len(items_with_embeddings) == 1
        assert items_with_embeddings[0].embedding == mock_embedding
    
    @patch.object(ClinicalKnowledgeEmbedder, 'extract_psychology_knowledge_items')
    @patch.object(ClinicalKnowledgeEmbedder, 'generate_embeddings')
    @patch.object(ClinicalKnowledgeEmbedder, 'create_embeddings_matrix')
    @patch.object(ClinicalKnowledgeEmbedder, 'save_embeddings')
    def test_process_all_knowledge_pipeline(self, mock_save, mock_create_matrix, 
                                          mock_generate, mock_extract):
        """Test complete knowledge processing pipeline."""
        # Setup mocks
        mock_items = [KnowledgeItem(id="test", content="test")]
        mock_matrix = [[0.1] * 384]
        
        mock_extract.return_value = mock_items
        mock_generate.return_value = mock_items
        mock_create_matrix.return_value = mock_matrix
        mock_save.return_value = Path("test_path")
        
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Run pipeline
        items, matrix = embedder.process_all_knowledge()
        
        # Verify all steps were called
        mock_extract.assert_called_once()
        mock_generate.assert_called_once_with(mock_items)
        mock_create_matrix.assert_called_once_with(mock_items)
        mock_save.assert_called_once()
        
        assert items == mock_items
        assert matrix == mock_matrix


class TestIntegration:
    """Integration tests for the complete embedder system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EmbeddingConfig(batch_size=2, embedding_dimension=384)
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_mock_processing(self):
        """Test end-to-end processing with mock data."""
        embedder = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        
        # Process knowledge (will use mock data due to missing dependencies)
        knowledge_items, embeddings_matrix = embedder.process_all_knowledge()
        
        # Verify results
        assert len(knowledge_items) > 0
        assert embeddings_matrix is not None
        
        # Check that embeddings file was created
        embeddings_file = self.temp_dir / "ai" / "pixel" / "data" / "clinical_knowledge_embeddings.pkl"
        assert embeddings_file.exists()
        
        # Verify statistics
        stats = embedder.get_embedding_stats()
        assert stats["total_items"] > 0
        assert stats["items_with_embeddings"] == stats["total_items"]
    
    def test_embedder_persistence(self):
        """Test that embedder state persists across instances."""
        # Create and process with first embedder
        embedder1 = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        items1, matrix1 = embedder1.process_all_knowledge()
        
        # Create second embedder and load data
        embedder2 = ClinicalKnowledgeEmbedder(self.config, self.temp_dir)
        success = embedder2.load_embeddings()
        
        assert success is True
        assert len(embedder2.knowledge_items) == len(items1)
        
        # Verify loaded data matches original
        for orig_item, loaded_item in zip(items1, embedder2.knowledge_items):
            assert orig_item.id == loaded_item.id
            assert orig_item.content == loaded_item.content
            assert orig_item.knowledge_type == loaded_item.knowledge_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
