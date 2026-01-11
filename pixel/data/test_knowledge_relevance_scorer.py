"""
Unit Tests for Knowledge Relevance Scorer

Tests advanced scoring and ranking functionality for clinical knowledge relevance.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from .clinical_knowledge_embedder import KnowledgeItem
from .clinical_similarity_search import EnhancedSearchResult, SearchContext, SearchQuery
from .knowledge_relevance_scorer import (
    KnowledgeRelevanceScorer,
    RankedResult,
    RankingStrategy,
    RelevanceScore,
    ScoringAlgorithm,
    ScoringConfig,
)


class TestScoringAlgorithm:
    """Test scoring algorithm enumeration."""
    
    def test_scoring_algorithms(self):
        """Test all scoring algorithms are available."""
        assert ScoringAlgorithm.TF_IDF.value == "tf_idf"
        assert ScoringAlgorithm.BM25.value == "bm25"
        assert ScoringAlgorithm.COSINE_SIMILARITY.value == "cosine_similarity"
        assert ScoringAlgorithm.CLINICAL_WEIGHTED.value == "clinical_weighted"
        assert ScoringAlgorithm.HYBRID.value == "hybrid"


class TestRankingStrategy:
    """Test ranking strategy enumeration."""
    
    def test_ranking_strategies(self):
        """Test all ranking strategies are available."""
        assert RankingStrategy.RELEVANCE_ONLY.value == "relevance_only"
        assert RankingStrategy.RECENCY_WEIGHTED.value == "recency_weighted"
        assert RankingStrategy.AUTHORITY_WEIGHTED.value == "authority_weighted"
        assert RankingStrategy.DIVERSITY_AWARE.value == "diversity_aware"
        assert RankingStrategy.CONTEXT_ADAPTIVE.value == "context_adaptive"


class TestScoringConfig:
    """Test scoring configuration."""
    
    def test_default_scoring_config(self):
        """Test default configuration values."""
        config = ScoringConfig()
        
        assert config.algorithm == ScoringAlgorithm.HYBRID
        assert config.ranking_strategy == RankingStrategy.CONTEXT_ADAPTIVE
        assert config.clinical_weight == 0.4
        assert config.semantic_weight == 0.3
        assert config.recency_weight == 0.1
        assert config.authority_weight == 0.1
        assert config.diversity_weight == 0.1
        assert config.k1 == 1.2
        assert config.b == 0.75
        assert config.min_score_threshold == 0.1
        assert config.max_results == 50
    
    def test_custom_scoring_config(self):
        """Test custom configuration values."""
        config = ScoringConfig(
            algorithm=ScoringAlgorithm.BM25,
            ranking_strategy=RankingStrategy.RELEVANCE_ONLY,
            clinical_weight=0.6,
            semantic_weight=0.4,
            k1=2.0,
            b=0.5,
            max_results=20
        )
        
        assert config.algorithm == ScoringAlgorithm.BM25
        assert config.ranking_strategy == RankingStrategy.RELEVANCE_ONLY
        assert config.clinical_weight == 0.6
        assert config.semantic_weight == 0.4
        assert config.k1 == 2.0
        assert config.b == 0.5
        assert config.max_results == 20


class TestRelevanceScore:
    """Test relevance score data structure."""
    
    def test_relevance_score_creation(self):
        """Test creating a relevance score."""
        score = RelevanceScore(
            total_score=0.85,
            clinical_score=0.9,
            semantic_score=0.8,
            recency_score=0.7,
            authority_score=0.95,
            diversity_score=0.6,
            algorithm_scores={"bm25": 0.8, "tf_idf": 0.75},
            explanation="High clinical and authority relevance",
            confidence=0.9
        )
        
        assert score.total_score == 0.85
        assert score.clinical_score == 0.9
        assert score.semantic_score == 0.8
        assert score.recency_score == 0.7
        assert score.authority_score == 0.95
        assert score.diversity_score == 0.6
        assert score.algorithm_scores["bm25"] == 0.8
        assert score.algorithm_scores["tf_idf"] == 0.75
        assert score.explanation == "High clinical and authority relevance"
        assert score.confidence == 0.9


class TestRankedResult:
    """Test ranked result data structure."""
    
    def test_ranked_result_creation(self):
        """Test creating a ranked result."""
        knowledge_item = KnowledgeItem(
            id="test_item",
            content="Test content",
            knowledge_type="dsm5"
        )
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=knowledge_item,
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test result"
        )
        
        relevance_score = RelevanceScore(
            total_score=0.85,
            clinical_score=0.9,
            semantic_score=0.8,
            recency_score=0.7,
            authority_score=0.95,
            diversity_score=0.6
        )
        
        ranked_result = RankedResult(
            result=enhanced_result,
            relevance_score=relevance_score,
            final_rank=0,
            original_rank=2,
            rank_change=2,
            ranking_factors={"clinical": 0.9, "authority": 0.95}
        )
        
        assert ranked_result.result == enhanced_result
        assert ranked_result.relevance_score == relevance_score
        assert ranked_result.final_rank == 0
        assert ranked_result.original_rank == 2
        assert ranked_result.rank_change == 2
        assert ranked_result.ranking_factors["clinical"] == 0.9
        assert ranked_result.ranking_factors["authority"] == 0.95


class TestKnowledgeRelevanceScorer:
    """Test knowledge relevance scorer functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = ScoringConfig(
            algorithm=ScoringAlgorithm.HYBRID,
            ranking_strategy=RankingStrategy.CONTEXT_ADAPTIVE
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_enhanced_results(self) -> list:
        """Create mock enhanced search results for testing."""
        knowledge_items = [
            ("dsm5_depression", "Major depressive disorder diagnostic criteria symptoms", "dsm5", "dsm5_source"),
            ("therapy_cbt", "Cognitive behavioral therapy treatment intervention", "therapeutic_technique", "therapy_source"),
            ("case_study", "Clinical case study depression patient", "case_study", "case_source"),
            ("research_paper", "Recent research on anxiety disorders", "research", "research_source")
        ]
        
        results = []
        for i, (item_id, content, knowledge_type, source) in enumerate(knowledge_items):
            knowledge_item = KnowledgeItem(
                id=item_id,
                content=content,
                knowledge_type=knowledge_type,
                source=source
            )
            
            enhanced_result = EnhancedSearchResult(
                knowledge_item=knowledge_item,
                similarity_score=0.9 - i * 0.1,
                relevance_score=0.8 - i * 0.1,
                combined_score=0.85 - i * 0.1,
                rank=i,
                relevance_explanation=f"Test result {i}",
                clinical_domains=["depression", "anxiety"][i % 2:i % 2 + 1],
                therapeutic_relevance=0.7 - i * 0.1,
                diagnostic_relevance=0.8 - i * 0.1
            )
            
            results.append(enhanced_result)
        
        return results
    
    def test_knowledge_relevance_scorer_initialization(self):
        """Test initialization of knowledge relevance scorer."""
        scorer = KnowledgeRelevanceScorer(self.config, self.temp_dir)
        
        assert scorer.config == self.config
        assert scorer.project_root == self.temp_dir
        assert scorer.clinical_terms is not None
        assert scorer.authority_scores is not None
        assert isinstance(scorer.clinical_terms, dict)
        assert isinstance(scorer.authority_scores, dict)
    
    def test_load_clinical_terms(self):
        """Test loading of clinical terms."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        clinical_terms = scorer.clinical_terms
        
        assert "diagnostic" in clinical_terms
        assert "therapeutic" in clinical_terms
        assert "clinical_domains" in clinical_terms
        assert "modalities" in clinical_terms
        
        # Check specific terms
        assert "diagnosis" in clinical_terms["diagnostic"]
        assert "therapy" in clinical_terms["therapeutic"]
        assert "depression" in clinical_terms["clinical_domains"]
        assert "cbt" in clinical_terms["modalities"]
        
        # Check weights
        assert clinical_terms["diagnostic"]["diagnosis"] == 1.0
        assert clinical_terms["therapeutic"]["therapy"] == 1.0
    
    def test_load_authority_scores(self):
        """Test loading of authority scores."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        authority_scores = scorer.authority_scores
        
        assert "dsm5" in authority_scores
        assert "pdm2" in authority_scores
        assert "clinical_guidelines" in authority_scores
        assert "peer_reviewed" in authority_scores
        
        # Check score hierarchy
        assert authority_scores["dsm5"] == 1.0  # Highest
        assert authority_scores["pdm2"] > authority_scores["peer_reviewed"]
        assert authority_scores["peer_reviewed"] > authority_scores["general"]
    
    def test_calculate_clinical_score(self):
        """Test clinical score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        # Create test result with clinical content
        knowledge_item = KnowledgeItem(
            id="test_item",
            content="Major depressive disorder diagnosis criteria symptoms therapy treatment",
            knowledge_type="dsm5"
        )
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=knowledge_item,
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test"
        )
        
        query = SearchQuery(
            text="depression diagnosis symptoms",
            clinical_domains=["depression"]
        )
        
        clinical_score = scorer._calculate_clinical_score(enhanced_result, query)
        
        assert 0 <= clinical_score <= 1
        assert clinical_score > 0.5  # Should be high due to clinical term matches
    
    def test_calculate_semantic_score(self):
        """Test semantic score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(id="test", content="test"),
            similarity_score=0.8,
            relevance_score=0.6,
            combined_score=0.7,
            rank=0,
            relevance_explanation="Test"
        )
        
        query = SearchQuery(text="test query")
        
        semantic_score = scorer._calculate_semantic_score(enhanced_result, query)
        
        assert 0 <= semantic_score <= 1
        assert semantic_score >= enhanced_result.similarity_score  # Should be at least as high
    
    def test_calculate_recency_score(self):
        """Test recency score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        # Test different knowledge types
        knowledge_types = ["dsm5", "pdm2", "therapeutic_technique", "case_study", "general"]
        expected_scores = [1.0, 0.8, 0.8, 0.6, 0.4]
        
        for knowledge_type, expected_score in zip(knowledge_types, expected_scores):
            enhanced_result = EnhancedSearchResult(
                knowledge_item=KnowledgeItem(
                    id="test",
                    content="test",
                    knowledge_type=knowledge_type
                ),
                similarity_score=0.8,
                relevance_score=0.7,
                combined_score=0.75,
                rank=0,
                relevance_explanation="Test"
            )
            
            recency_score = scorer._calculate_recency_score(enhanced_result)
            assert recency_score == expected_score
    
    def test_calculate_authority_score(self):
        """Test authority score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        # Test DSM-5 source (should have high authority)
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(
                id="test",
                content="test",
                knowledge_type="dsm5",
                source="official_dsm5_source"
            ),
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test"
        )
        
        authority_score = scorer._calculate_authority_score(enhanced_result)
        
        assert 0 <= authority_score <= 1
        assert authority_score > 0.8  # Should be high for DSM-5
    
    def test_calculate_diversity_score(self):
        """Test diversity score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(id="test", content="test"),
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test",
            clinical_domains=["depression", "anxiety"]
        )
        
        query = SearchQuery(
            text="test query",
            clinical_domains=["depression"]
        )
        
        diversity_score = scorer._calculate_diversity_score(enhanced_result, query)
        
        assert 0 <= diversity_score <= 1
        assert diversity_score > 0.5  # Should have some diversity bonus
    
    def test_calculate_tf_idf_score(self):
        """Test TF-IDF score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(
                id="test",
                content="depression symptoms depression treatment depression therapy"
            ),
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test"
        )
        
        query = SearchQuery(text="depression symptoms")
        
        tf_idf_score = scorer._calculate_tf_idf_score(enhanced_result, query)
        
        assert 0 <= tf_idf_score <= 1
        assert tf_idf_score > 0  # Should have some score due to term matches
    
    def test_calculate_bm25_score(self):
        """Test BM25 score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(
                id="test",
                content="depression symptoms major depressive disorder treatment"
            ),
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test"
        )
        
        query = SearchQuery(text="depression symptoms")
        
        bm25_score = scorer._calculate_bm25_score(enhanced_result, query)
        
        assert 0 <= bm25_score <= 1
        assert bm25_score > 0  # Should have some score due to term matches
    
    def test_calculate_clinical_weighted_score(self):
        """Test clinical weighted score calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        enhanced_result = EnhancedSearchResult(
            knowledge_item=KnowledgeItem(
                id="test",
                content="depression diagnosis symptoms therapy"
            ),
            similarity_score=0.8,
            relevance_score=0.7,
            combined_score=0.75,
            rank=0,
            relevance_explanation="Test"
        )
        
        # Clinical query
        clinical_query = SearchQuery(text="depression diagnosis symptoms")
        clinical_weighted_score = scorer._calculate_clinical_weighted_score(enhanced_result, clinical_query)
        
        # Non-clinical query
        general_query = SearchQuery(text="general information")
        general_weighted_score = scorer._calculate_clinical_weighted_score(enhanced_result, general_query)
        
        assert 0 <= clinical_weighted_score <= 1
        assert 0 <= general_weighted_score <= 1
        # Clinical query should weight clinical relevance higher
        # This test might need adjustment based on actual implementation
    
    def test_score_and_rank_basic(self):
        """Test basic scoring and ranking functionality."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        mock_results = self._create_mock_enhanced_results()
        query = SearchQuery(
            text="depression symptoms treatment",
            context=SearchContext.TRAINING,
            clinical_domains=["depression"]
        )
        
        ranked_results = scorer.score_and_rank(mock_results, query)
        
        assert len(ranked_results) == len(mock_results)
        assert all(isinstance(result, RankedResult) for result in ranked_results)
        
        # Check that results are properly ranked
        for i, result in enumerate(ranked_results):
            assert result.final_rank == i
            assert isinstance(result.relevance_score, RelevanceScore)
            assert 0 <= result.relevance_score.total_score <= 1
    
    def test_score_and_rank_empty_results(self):
        """Test scoring and ranking with empty results."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        query = SearchQuery(text="test query")
        ranked_results = scorer.score_and_rank([], query)
        
        assert ranked_results == []
    
    def test_ranking_strategies(self):
        """Test different ranking strategies."""
        strategies = [
            RankingStrategy.RELEVANCE_ONLY,
            RankingStrategy.RECENCY_WEIGHTED,
            RankingStrategy.AUTHORITY_WEIGHTED,
            RankingStrategy.DIVERSITY_AWARE,
            RankingStrategy.CONTEXT_ADAPTIVE
        ]
        
        mock_results = self._create_mock_enhanced_results()
        query = SearchQuery(
            text="depression treatment",
            context=SearchContext.TRAINING
        )
        
        for strategy in strategies:
            config = ScoringConfig(ranking_strategy=strategy)
            scorer = KnowledgeRelevanceScorer(config)
            
            ranked_results = scorer.score_and_rank(mock_results, query)
            
            assert len(ranked_results) == len(mock_results)
            assert all(isinstance(result, RankedResult) for result in ranked_results)
            
            # Check that ranking is applied
            for i, result in enumerate(ranked_results):
                assert result.final_rank == i
    
    def test_context_adaptive_ranking(self):
        """Test context-adaptive ranking for different contexts."""
        scorer = KnowledgeRelevanceScorer(
            ScoringConfig(ranking_strategy=RankingStrategy.CONTEXT_ADAPTIVE)
        )
        
        mock_results = self._create_mock_enhanced_results()
        
        contexts = [
            SearchContext.TRAINING,
            SearchContext.INFERENCE,
            SearchContext.VALIDATION,
            SearchContext.RESEARCH
        ]
        
        for context in contexts:
            query = SearchQuery(
                text="depression treatment",
                context=context
            )
            
            ranked_results = scorer.score_and_rank(mock_results, query)
            
            assert len(ranked_results) == len(mock_results)
            # Different contexts might produce different rankings
            # This is expected behavior
    
    def test_get_scoring_stats(self):
        """Test scoring statistics generation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        mock_results = self._create_mock_enhanced_results()
        query = SearchQuery(
            text="depression treatment",
            clinical_domains=["depression"]
        )
        
        ranked_results = scorer.score_and_rank(mock_results, query)
        stats = scorer.get_scoring_stats(ranked_results)
        
        assert "total_results" in stats
        assert "score_statistics" in stats
        assert "clinical_statistics" in stats
        assert "semantic_statistics" in stats
        assert "ranking_changes" in stats
        assert "knowledge_type_distribution" in stats
        
        # Check specific statistics
        assert stats["total_results"] == len(ranked_results)
        assert "mean_score" in stats["score_statistics"]
        assert "max_score" in stats["score_statistics"]
        assert "min_score" in stats["score_statistics"]
        
        # Check that statistics are reasonable
        assert 0 <= stats["score_statistics"]["mean_score"] <= 1
        assert 0 <= stats["score_statistics"]["max_score"] <= 1
        assert 0 <= stats["score_statistics"]["min_score"] <= 1
    
    def test_get_scoring_stats_empty(self):
        """Test scoring statistics with empty results."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        stats = scorer.get_scoring_stats([])
        
        assert "error" in stats
        assert stats["error"] == "No results to analyze"
    
    def test_generate_score_explanation(self):
        """Test score explanation generation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        # Test high scores
        explanation = scorer._generate_score_explanation(
            clinical_score=0.8,
            semantic_score=0.9,
            recency_score=0.9,
            authority_score=0.9,
            diversity_score=0.6,
            algorithm_scores={"bm25": 0.7, "tf_idf": 0.6}
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "High clinical relevance" in explanation
        assert "Strong semantic match" in explanation
        assert "Authoritative source" in explanation
        assert "Current knowledge" in explanation
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        scorer = KnowledgeRelevanceScorer(self.config)
        
        # Test with multiple algorithm scores
        algorithm_scores = {"bm25": 0.8, "tf_idf": 0.75, "cosine": 0.85}
        confidence = scorer._calculate_confidence(algorithm_scores, 0.8)
        
        assert 0 <= confidence <= 1
        
        # Test with single algorithm score
        single_algorithm_scores = {"bm25": 0.8}
        single_confidence = scorer._calculate_confidence(single_algorithm_scores, 0.8)
        
        assert 0 <= single_confidence <= 1
        
        # Test with empty algorithm scores
        empty_confidence = scorer._calculate_confidence({}, 0.8)
        
        assert empty_confidence == 0.5


class TestIntegration:
    """Integration tests for knowledge relevance scorer."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_scoring_workflow(self):
        """Test complete scoring and ranking workflow."""
        # Test multiple configurations
        configs = [
            ScoringConfig(
                algorithm=ScoringAlgorithm.HYBRID,
                ranking_strategy=RankingStrategy.CONTEXT_ADAPTIVE
            ),
            ScoringConfig(
                algorithm=ScoringAlgorithm.BM25,
                ranking_strategy=RankingStrategy.RELEVANCE_ONLY
            ),
            ScoringConfig(
                algorithm=ScoringAlgorithm.CLINICAL_WEIGHTED,
                ranking_strategy=RankingStrategy.DIVERSITY_AWARE
            )
        ]
        
        # Create comprehensive test data
        knowledge_items = [
            ("dsm5_depression", "Major depressive disorder diagnostic criteria persistent sadness loss interest", "dsm5"),
            ("therapy_cbt", "Cognitive behavioral therapy treatment depression anxiety intervention", "therapeutic_technique"),
            ("pdm2_attachment", "Attachment theory psychodynamic framework secure anxious avoidant", "pdm2"),
            ("case_study_depression", "Clinical case study patient depression symptoms therapy outcome", "case_study"),
            ("research_anxiety", "Recent research anxiety disorders prevalence treatment effectiveness", "research")
        ]
        
        mock_results = []
        for i, (item_id, content, knowledge_type) in enumerate(knowledge_items):
            knowledge_item = KnowledgeItem(
                id=item_id,
                content=content,
                knowledge_type=knowledge_type,
                source=f"{knowledge_type}_source"
            )
            
            enhanced_result = EnhancedSearchResult(
                knowledge_item=knowledge_item,
                similarity_score=0.9 - i * 0.05,
                relevance_score=0.8 - i * 0.05,
                combined_score=0.85 - i * 0.05,
                rank=i,
                relevance_explanation=f"Result {i}",
                clinical_domains=["depression", "anxiety"][i % 2:i % 2 + 1]
            )
            
            mock_results.append(enhanced_result)
        
        # Test different query scenarios
        queries = [
            SearchQuery(
                text="depression symptoms diagnosis",
                context=SearchContext.TRAINING,
                clinical_domains=["depression"]
            ),
            SearchQuery(
                text="therapy treatment intervention",
                context=SearchContext.INFERENCE,
                clinical_domains=["depression", "anxiety"]
            ),
            SearchQuery(
                text="clinical assessment evaluation",
                context=SearchContext.VALIDATION
            )
        ]
        
        for config in configs:
            scorer = KnowledgeRelevanceScorer(config, self.temp_dir)
            
            for query in queries:
                # Score and rank results
                ranked_results = scorer.score_and_rank(mock_results, query)
                
                # Verify results
                assert len(ranked_results) == len(mock_results)
                assert all(isinstance(result, RankedResult) for result in ranked_results)
                
                # Verify ranking
                for i, result in enumerate(ranked_results):
                    assert result.final_rank == i
                    assert isinstance(result.relevance_score, RelevanceScore)
                    assert 0 <= result.relevance_score.total_score <= 1
                    assert result.relevance_score.explanation != ""
                    assert 0 <= result.relevance_score.confidence <= 1
                
                # Verify that scores are in descending order (for relevance-based ranking)
                if config.ranking_strategy == RankingStrategy.RELEVANCE_ONLY:
                    for i in range(len(ranked_results) - 1):
                        assert (ranked_results[i].relevance_score.total_score >= 
                               ranked_results[i + 1].relevance_score.total_score)
                
                # Get and verify statistics
                stats = scorer.get_scoring_stats(ranked_results)
                assert stats["total_results"] == len(ranked_results)
                assert 0 <= stats["score_statistics"]["mean_score"] <= 1
                assert len(stats["knowledge_type_distribution"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
