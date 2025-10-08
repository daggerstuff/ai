"""
Unit Tests for Real-Time Knowledge Retrieval

Tests real-time knowledge retrieval functionality for training integration.
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from .realtime_knowledge_retrieval import (
    RealtimeKnowledgeRetrieval,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStats,
    RetrievalMode,
    TrainingPhase,
    create_training_retrieval_system,
    create_inference_retrieval_system
)
from .clinical_similarity_search import EnhancedSearchResult
from .clinical_knowledge_embedder import KnowledgeItem


class TestRetrievalMode:
    """Test retrieval mode enumeration."""
    
    def test_retrieval_modes(self):
        """Test all retrieval modes are available."""
        assert RetrievalMode.SYNCHRONOUS.value == "synchronous"
        assert RetrievalMode.ASYNCHRONOUS.value == "asynchronous"
        assert RetrievalMode.BATCH.value == "batch"
        assert RetrievalMode.CACHED_ONLY.value == "cached_only"


class TestTrainingPhase:
    """Test training phase enumeration."""
    
    def test_training_phases(self):
        """Test all training phases are available."""
        assert TrainingPhase.INITIALIZATION.value == "initialization"
        assert TrainingPhase.FORWARD_PASS.value == "forward_pass"
        assert TrainingPhase.BACKWARD_PASS.value == "backward_pass"
        assert TrainingPhase.VALIDATION.value == "validation"
        assert TrainingPhase.CHECKPOINT.value == "checkpoint"


class TestRetrievalRequest:
    """Test retrieval request data structure."""
    
    def test_default_retrieval_request(self):
        """Test creating a retrieval request with defaults."""
        request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        assert request.query_text == "depression symptoms"
        assert request.training_phase == TrainingPhase.FORWARD_PASS
        assert request.batch_id is None
        assert request.step_id is None
        assert request.context_metadata == {}
        assert request.max_results == 5
        assert request.min_relevance == 0.6
        assert request.timeout_seconds == 1.0
        assert request.priority == 1
        assert request.callback is None
        assert request.request_id.startswith("req_")
        assert isinstance(request.created_at, datetime)
    
    def test_custom_retrieval_request(self):
        """Test creating a retrieval request with custom parameters."""
        callback = Mock()
        
        request = RetrievalRequest(
            query_text="anxiety treatment",
            training_phase=TrainingPhase.VALIDATION,
            batch_id="batch_123",
            step_id=456,
            context_metadata={"domain": "anxiety"},
            max_results=10,
            min_relevance=0.8,
            timeout_seconds=2.0,
            priority=2,
            callback=callback,
            request_id="custom_req_id"
        )
        
        assert request.query_text == "anxiety treatment"
        assert request.training_phase == TrainingPhase.VALIDATION
        assert request.batch_id == "batch_123"
        assert request.step_id == 456
        assert request.context_metadata["domain"] == "anxiety"
        assert request.max_results == 10
        assert request.min_relevance == 0.8
        assert request.timeout_seconds == 2.0
        assert request.priority == 2
        assert request.callback == callback
        assert request.request_id == "custom_req_id"


class TestRetrievalResponse:
    """Test retrieval response data structure."""
    
    def test_retrieval_response_creation(self):
        """Test creating a retrieval response."""
        knowledge_item = KnowledgeItem(
            id="test_item",
            content="Test content",
            knowledge_type="dsm5"
        )
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=knowledge_item,
            similarity_score=0.9,
            relevance_score=0.8,
            combined_score=0.85,
            rank=0,
            relevance_explanation="High relevance"
        )
        
        response = RetrievalResponse(
            request_id="test_request",
            results=[enhanced_result],
            retrieval_time_ms=150.5,
            cache_hit=True,
            metadata={"test": "value"}
        )
        
        assert response.request_id == "test_request"
        assert len(response.results) == 1
        assert response.results[0] == enhanced_result
        assert response.retrieval_time_ms == 150.5
        assert response.cache_hit is True
        assert response.error is None
        assert response.metadata["test"] == "value"
        assert isinstance(response.completed_at, datetime)


class TestRetrievalStats:
    """Test retrieval statistics."""
    
    def test_default_retrieval_stats(self):
        """Test creating retrieval statistics with defaults."""
        stats = RetrievalStats()
        
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.avg_retrieval_time_ms == 0.0
        assert stats.total_retrieval_time_ms == 0.0
        assert isinstance(stats.requests_by_phase, dict)
        assert stats.errors == 0
        assert stats.timeouts == 0
        assert isinstance(stats.last_reset, datetime)


class TestRealtimeKnowledgeRetrieval:
    """Test real-time knowledge retrieval functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock similarity search
        self.mock_similarity_search = Mock()
        self.mock_similarity_search.search.return_value = self._create_mock_enhanced_results()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_enhanced_results(self) -> list:
        """Create mock enhanced search results."""
        knowledge_items = [
            KnowledgeItem(
                id="depression_item",
                content="Major depressive disorder symptoms",
                knowledge_type="dsm5"
            ),
            KnowledgeItem(
                id="therapy_item",
                content="Cognitive behavioral therapy techniques",
                knowledge_type="therapeutic_technique"
            )
        ]
        
        return [
            EnhancedSearchResult(
                knowledge_item=item,
                similarity_score=0.9 - i * 0.1,
                relevance_score=0.8 - i * 0.1,
                combined_score=0.85 - i * 0.1,
                rank=i,
                relevance_explanation=f"Relevant result {i}"
            )
            for i, item in enumerate(knowledge_items)
        ]
    
    def test_realtime_retrieval_initialization(self):
        """Test initialization of real-time knowledge retrieval."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.SYNCHRONOUS,
            cache_size=100,
            project_root=self.temp_dir
        )
        
        assert retrieval_system.similarity_search == self.mock_similarity_search
        assert retrieval_system.mode == RetrievalMode.SYNCHRONOUS
        assert retrieval_system.cache_size == 100
        assert retrieval_system.project_root == self.temp_dir
        assert isinstance(retrieval_system.stats, RetrievalStats)
        assert retrieval_system.cache == {}
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        request1 = RetrievalRequest(
            query_text="Depression Symptoms",
            training_phase=TrainingPhase.FORWARD_PASS,
            max_results=5,
            min_relevance=0.6
        )
        
        request2 = RetrievalRequest(
            query_text="depression symptoms",  # Different case
            training_phase=TrainingPhase.FORWARD_PASS,
            max_results=5,
            min_relevance=0.6
        )
        
        key1 = retrieval_system._generate_cache_key(request1)
        key2 = retrieval_system._generate_cache_key(request2)
        
        assert key1 == key2  # Should be case-insensitive
        assert "depression symptoms" in key1
        assert "5" in key1
        assert "0.60" in key1
    
    def test_cache_operations(self):
        """Test cache add and get operations."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        # Test adding to cache
        cache_key = "test_key"
        response = RetrievalResponse(
            request_id="test_request",
            results=[],
            retrieval_time_ms=100,
            cache_hit=False
        )
        
        retrieval_system._add_to_cache(cache_key, response)
        
        # Test getting from cache
        cached_response = retrieval_system._get_from_cache(cache_key)
        
        assert cached_response is not None
        assert cached_response.request_id == "test_request"
        assert cached_response.retrieval_time_ms == 100
        
        # Test cache miss
        missing_response = retrieval_system._get_from_cache("non_existent_key")
        assert missing_response is None
    
    def test_synchronous_retrieval(self):
        """Test synchronous knowledge retrieval."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.SYNCHRONOUS
        )
        
        request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        response = retrieval_system.retrieve(request)
        
        assert isinstance(response, RetrievalResponse)
        assert response.request_id == request.request_id
        assert len(response.results) > 0
        assert response.cache_hit is False  # First request
        assert response.error is None
        
        # Verify similarity search was called
        self.mock_similarity_search.search.assert_called_once()
    
    def test_synchronous_retrieval_with_cache_hit(self):
        """Test synchronous retrieval with cache hit."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.SYNCHRONOUS
        )
        
        request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        # First request
        response1 = retrieval_system.retrieve(request)
        assert response1.cache_hit is False
        
        # Second identical request should hit cache
        response2 = retrieval_system.retrieve(request)
        assert response2.cache_hit is True
        
        # Similarity search should only be called once
        assert self.mock_similarity_search.search.call_count == 1
    
    def test_asynchronous_retrieval(self):
        """Test asynchronous knowledge retrieval."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.ASYNCHRONOUS
        )
        
        callback_results = []
        
        def test_callback(response):
            callback_results.append(response)
        
        request = RetrievalRequest(
            query_text="anxiety treatment",
            training_phase=TrainingPhase.VALIDATION,
            callback=test_callback
        )
        
        request_id = retrieval_system.retrieve(request)
        
        assert isinstance(request_id, str)
        assert request_id == request.request_id
        
        # Wait for async processing
        time.sleep(0.1)
        
        # Check that callback was called
        assert len(callback_results) == 1
        assert isinstance(callback_results[0], RetrievalResponse)
        assert callback_results[0].request_id == request.request_id
    
    def test_cached_only_retrieval(self):
        """Test cached-only retrieval mode."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.CACHED_ONLY
        )
        
        request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        # First request should miss cache
        response1 = retrieval_system.retrieve(request)
        assert response1.cache_hit is False
        assert response1.error == "Not found in cache"
        assert len(response1.results) == 0
        
        # Add to cache manually
        cache_key = retrieval_system._generate_cache_key(request)
        cached_response = RetrievalResponse(
            request_id="cached_request",
            results=self._create_mock_enhanced_results(),
            retrieval_time_ms=50,
            cache_hit=True
        )
        retrieval_system._add_to_cache(cache_key, cached_response)
        
        # Second request should hit cache
        response2 = retrieval_system.retrieve(request)
        assert response2.cache_hit is True
        assert len(response2.results) > 0
    
    def test_batch_retrieval(self):
        """Test batch retrieval mode."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.BATCH,
            batch_size=2,
            batch_timeout_ms=50
        )
        
        callback_results = []
        
        def test_callback(response):
            callback_results.append(response)
        
        # Submit multiple requests
        requests = []
        for i in range(3):
            request = RetrievalRequest(
                query_text=f"test query {i}",
                training_phase=TrainingPhase.FORWARD_PASS,
                callback=test_callback
            )
            requests.append(request)
            request_id = retrieval_system.retrieve(request)
            assert isinstance(request_id, str)
        
        # Wait for batch processing
        time.sleep(0.2)
        
        # Check that callbacks were called
        assert len(callback_results) >= 2  # At least some should be processed
    
    def test_training_phase_to_search_context_mapping(self):
        """Test mapping of training phases to search contexts."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        from .clinical_similarity_search import SearchContext
        
        # Test mappings
        assert retrieval_system._map_training_phase_to_search_context(
            TrainingPhase.INITIALIZATION) == SearchContext.TRAINING
        assert retrieval_system._map_training_phase_to_search_context(
            TrainingPhase.FORWARD_PASS) == SearchContext.TRAINING
        assert retrieval_system._map_training_phase_to_search_context(
            TrainingPhase.BACKWARD_PASS) == SearchContext.TRAINING
        assert retrieval_system._map_training_phase_to_search_context(
            TrainingPhase.VALIDATION) == SearchContext.VALIDATION
        assert retrieval_system._map_training_phase_to_search_context(
            TrainingPhase.CHECKPOINT) == SearchContext.RESEARCH
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.SYNCHRONOUS
        )
        
        request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        # Initial stats
        initial_stats = retrieval_system.get_stats()
        assert initial_stats["total_requests"] == 0
        assert initial_stats["cache_hits"] == 0
        
        # Make request
        response = retrieval_system.retrieve(request)
        
        # Check updated stats
        updated_stats = retrieval_system.get_stats()
        assert updated_stats["total_requests"] == 1
        assert updated_stats["cache_misses"] == 1
        assert updated_stats["requests_by_phase"]["forward_pass"] == 1
        assert updated_stats["avg_retrieval_time_ms"] > 0
        
        # Make same request again (should hit cache)
        response2 = retrieval_system.retrieve(request)
        
        final_stats = retrieval_system.get_stats()
        assert final_stats["total_requests"] == 2
        assert final_stats["cache_hits"] == 1
        assert final_stats["cache_hit_rate_percent"] == 50.0
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            cache_size=2  # Small cache for testing
        )
        
        # Add items to exceed cache size
        for i in range(5):
            cache_key = f"key_{i}"
            response = RetrievalResponse(
                request_id=f"req_{i}",
                results=[],
                retrieval_time_ms=10,
                cache_hit=False
            )
            retrieval_system._add_to_cache(cache_key, response)
        
        # Trigger cleanup
        retrieval_system._cleanup_cache()
        
        # Cache should be reduced to max size
        assert len(retrieval_system.cache) <= retrieval_system.cache_size
    
    def test_clear_cache(self):
        """Test cache clearing."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        # Add items to cache
        cache_key = "test_key"
        response = RetrievalResponse(
            request_id="test_request",
            results=[],
            retrieval_time_ms=10,
            cache_hit=False
        )
        retrieval_system._add_to_cache(cache_key, response)
        
        assert len(retrieval_system.cache) == 1
        
        # Clear cache
        retrieval_system.clear_cache()
        
        assert len(retrieval_system.cache) == 0
        assert len(retrieval_system.cache_access_times) == 0
    
    def test_reset_stats(self):
        """Test statistics reset."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        # Make some requests to generate stats
        request = RetrievalRequest(
            query_text="test query",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        retrieval_system.retrieve(request)
        
        # Verify stats exist
        stats = retrieval_system.get_stats()
        assert stats["total_requests"] > 0
        
        # Reset stats
        retrieval_system.reset_stats()
        
        # Verify stats are reset
        reset_stats = retrieval_system.get_stats()
        assert reset_stats["total_requests"] == 0
        assert reset_stats["cache_hits"] == 0
        assert reset_stats["cache_misses"] == 0
    
    def test_pending_request_management(self):
        """Test pending request management."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search,
            mode=RetrievalMode.ASYNCHRONOUS
        )
        
        request = RetrievalRequest(
            query_text="test query",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        # Submit async request
        request_id = retrieval_system.retrieve(request)
        
        # Check pending request
        pending_request = retrieval_system.get_pending_request(request_id)
        assert pending_request is not None
        assert pending_request.request_id == request_id
        
        # Cancel request
        cancelled = retrieval_system.cancel_request(request_id)
        assert cancelled is True
        
        # Check request is no longer pending
        pending_request = retrieval_system.get_pending_request(request_id)
        assert pending_request is None
    
    def test_shutdown(self):
        """Test system shutdown."""
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=self.mock_similarity_search
        )
        
        # Add some data
        request = RetrievalRequest(
            query_text="test query",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        retrieval_system.retrieve(request)
        
        # Shutdown
        retrieval_system.shutdown()
        
        # Verify cleanup
        assert len(retrieval_system.cache) == 0
        assert len(retrieval_system.pending_requests) == 0


class TestConvenienceFunctions:
    """Test convenience functions for creating retrieval systems."""
    
    def test_create_training_retrieval_system(self):
        """Test creating training-optimized retrieval system."""
        system = create_training_retrieval_system()
        
        assert system.mode == RetrievalMode.ASYNCHRONOUS
        assert system.cache_size == 1000
        assert system.batch_size == 20
        assert system.batch_timeout_ms == 50
        assert system.max_workers == 8
    
    def test_create_inference_retrieval_system(self):
        """Test creating inference-optimized retrieval system."""
        system = create_inference_retrieval_system()
        
        assert system.mode == RetrievalMode.SYNCHRONOUS
        assert system.cache_size == 500
        assert system.batch_size == 5
        assert system.batch_timeout_ms == 10
        assert system.max_workers == 2


class TestIntegration:
    """Integration tests for real-time knowledge retrieval."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_retrieval_workflow(self):
        """Test complete retrieval workflow."""
        # Create mock similarity search
        mock_similarity_search = Mock()
        mock_similarity_search.search.return_value = [
            EnhancedSearchResult(
                knowledge_item=KnowledgeItem(
                    id="test_item",
                    content="Test clinical knowledge",
                    knowledge_type="dsm5"
                ),
                similarity_score=0.9,
                relevance_score=0.8,
                combined_score=0.85,
                rank=0,
                relevance_explanation="High clinical relevance"
            )
        ]
        
        # Create retrieval system
        retrieval_system = RealtimeKnowledgeRetrieval(
            similarity_search=mock_similarity_search,
            mode=RetrievalMode.SYNCHRONOUS,
            project_root=self.temp_dir
        )
        
        # Test multiple retrieval scenarios
        scenarios = [
            ("depression symptoms", TrainingPhase.FORWARD_PASS),
            ("anxiety treatment", TrainingPhase.VALIDATION),
            ("therapy techniques", TrainingPhase.BACKWARD_PASS),
            ("clinical assessment", TrainingPhase.CHECKPOINT)
        ]
        
        for query_text, phase in scenarios:
            request = RetrievalRequest(
                query_text=query_text,
                training_phase=phase,
                max_results=3,
                min_relevance=0.5
            )
            
            response = retrieval_system.retrieve(request)
            
            # Verify response
            assert isinstance(response, RetrievalResponse)
            assert response.request_id == request.request_id
            assert len(response.results) > 0
            assert response.error is None
            assert response.retrieval_time_ms >= 0
        
        # Verify statistics
        stats = retrieval_system.get_stats()
        assert stats["total_requests"] == len(scenarios)
        assert stats["total_requests"] > 0
        
        # Test cache functionality
        first_request = RetrievalRequest(
            query_text="depression symptoms",
            training_phase=TrainingPhase.FORWARD_PASS
        )
        
        # This should hit cache
        cached_response = retrieval_system.retrieve(first_request)
        assert cached_response.cache_hit is True
        
        # Cleanup
        retrieval_system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
