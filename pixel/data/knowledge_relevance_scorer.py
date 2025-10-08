"""
Knowledge Relevance Scoring and Ranking

Advanced scoring and ranking system for clinical knowledge relevance
with multiple scoring algorithms, contextual weighting, and adaptive ranking.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import math
from datetime import datetime
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .clinical_similarity_search import EnhancedSearchResult, SearchQuery, SearchContext
from .clinical_knowledge_embedder import KnowledgeItem


class ScoringAlgorithm(Enum):
    """Scoring algorithms for knowledge relevance."""
    TF_IDF = "tf_idf"  # Term Frequency-Inverse Document Frequency
    BM25 = "bm25"  # Best Matching 25
    COSINE_SIMILARITY = "cosine_similarity"  # Cosine similarity
    CLINICAL_WEIGHTED = "clinical_weighted"  # Clinical domain weighted scoring
    HYBRID = "hybrid"  # Combination of multiple algorithms


class RankingStrategy(Enum):
    """Ranking strategies for search results."""
    RELEVANCE_ONLY = "relevance_only"  # Pure relevance score
    RECENCY_WEIGHTED = "recency_weighted"  # Weight by knowledge recency
    AUTHORITY_WEIGHTED = "authority_weighted"  # Weight by source authority
    DIVERSITY_AWARE = "diversity_aware"  # Promote result diversity
    CONTEXT_ADAPTIVE = "context_adaptive"  # Adapt to search context


@dataclass
class ScoringConfig:
    """Configuration for relevance scoring."""
    algorithm: ScoringAlgorithm = ScoringAlgorithm.HYBRID
    ranking_strategy: RankingStrategy = RankingStrategy.CONTEXT_ADAPTIVE
    clinical_weight: float = 0.4
    semantic_weight: float = 0.3
    recency_weight: float = 0.1
    authority_weight: float = 0.1
    diversity_weight: float = 0.1
    k1: float = 1.2  # BM25 parameter
    b: float = 0.75  # BM25 parameter
    min_score_threshold: float = 0.1
    max_results: int = 50


@dataclass
class RelevanceScore:
    """Detailed relevance score breakdown."""
    total_score: float
    clinical_score: float
    semantic_score: float
    recency_score: float
    authority_score: float
    diversity_score: float
    algorithm_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    confidence: float = 0.0


@dataclass
class RankedResult:
    """Search result with detailed ranking information."""
    result: EnhancedSearchResult
    relevance_score: RelevanceScore
    final_rank: int
    original_rank: int
    rank_change: int = 0
    ranking_factors: Dict[str, float] = field(default_factory=dict)


class KnowledgeRelevanceScorer:
    """Advanced knowledge relevance scoring and ranking system."""

    def __init__(
            self,
            config: Optional[ScoringConfig] = None,
            project_root: Optional[Path] = None):
        """Initialize the knowledge relevance scorer."""
        self.config = config or ScoringConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Clinical knowledge mappings
        self.clinical_terms = self._load_clinical_terms()
        self.authority_scores = self._load_authority_scores()
        self.term_frequencies = defaultdict(int)
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize scoring components
        self._initialize_scoring_components()

    def _load_clinical_terms(self) -> Dict[str, Dict[str, float]]:
        """Load clinical terms with importance weights."""
        return {
            "diagnostic": {
                "diagnosis": 1.0,
                "diagnostic": 1.0,
                "criteria": 0.9,
                "symptoms": 0.8,
                "disorder": 0.9,
                "condition": 0.7,
                "syndrome": 0.8,
                "pathology": 0.9,
                "assessment": 0.7,
                "evaluation": 0.6,
                "screening": 0.6},
            "therapeutic": {
                "therapy": 1.0,
                "treatment": 1.0,
                "intervention": 0.9,
                "therapeutic": 0.9,
                "counseling": 0.8,
                "psychotherapy": 1.0,
                "rehabilitation": 0.7,
                "management": 0.6,
                "care": 0.5,
                "support": 0.4},
            "clinical_domains": {
                "depression": 1.0,
                "anxiety": 1.0,
                "trauma": 1.0,
                "ptsd": 1.0,
                "bipolar": 0.9,
                "schizophrenia": 0.9,
                "personality": 0.8,
                "substance": 0.8,
                "addiction": 0.8,
                "eating": 0.7},
            "modalities": {
                "cbt": 1.0,
                "dbt": 1.0,
                "emdr": 0.9,
                "psychodynamic": 0.8,
                "humanistic": 0.7,
                "behavioral": 0.8,
                "cognitive": 0.9,
                "mindfulness": 0.7,
                "exposure": 0.8}}

    def _load_authority_scores(self) -> Dict[str, float]:
        """Load authority scores for different knowledge sources."""
        return {
            "dsm5": 1.0,  # Highest authority
            "pdm2": 0.9,
            "clinical_guidelines": 0.8,
            "peer_reviewed": 0.7,
            "therapeutic_technique": 0.6,
            "case_study": 0.5,
            "general": 0.3
        }

    def _initialize_scoring_components(self):
        """Initialize scoring algorithm components."""
        # This would typically load pre-computed statistics
        # For now, we'll use default values
        self.avg_doc_length = 100  # Average document length
        self.total_documents = 1000  # Total number of documents

    def score_and_rank(self, results: List[EnhancedSearchResult],
                       query: SearchQuery) -> List[RankedResult]:
        """Score and rank search results based on relevance."""
        if not results:
            return []

        # Calculate relevance scores
        scored_results = []
        for i, result in enumerate(results):
            relevance_score = self._calculate_relevance_score(result, query)

            ranked_result = RankedResult(
                result=result,
                relevance_score=relevance_score,
                final_rank=0,  # Will be set after ranking
                original_rank=i
            )
            scored_results.append(ranked_result)

        # Apply ranking strategy
        ranked_results = self._apply_ranking_strategy(scored_results, query)

        # Update final ranks
        for i, ranked_result in enumerate(ranked_results):
            ranked_result.final_rank = i
            ranked_result.rank_change = ranked_result.original_rank - i

        return ranked_results

    def _calculate_relevance_score(self, result: EnhancedSearchResult,
                                   query: SearchQuery) -> RelevanceScore:
        """Calculate detailed relevance score for a result."""
        # Calculate individual score components
        clinical_score = self._calculate_clinical_score(result, query)
        semantic_score = self._calculate_semantic_score(result, query)
        recency_score = self._calculate_recency_score(result)
        authority_score = self._calculate_authority_score(result)
        diversity_score = self._calculate_diversity_score(result, query)

        # Calculate algorithm-specific scores
        algorithm_scores = {}
        if self.config.algorithm in [
                ScoringAlgorithm.TF_IDF,
                ScoringAlgorithm.HYBRID]:
            algorithm_scores["tf_idf"] = self._calculate_tf_idf_score(
                result, query)

        if self.config.algorithm in [
                ScoringAlgorithm.BM25,
                ScoringAlgorithm.HYBRID]:
            algorithm_scores["bm25"] = self._calculate_bm25_score(
                result, query)

        if self.config.algorithm in [
                ScoringAlgorithm.COSINE_SIMILARITY,
                ScoringAlgorithm.HYBRID]:
            algorithm_scores["cosine"] = result.similarity_score

        if self.config.algorithm in [
                ScoringAlgorithm.CLINICAL_WEIGHTED,
                ScoringAlgorithm.HYBRID]:
            algorithm_scores["clinical_weighted"] = self._calculate_clinical_weighted_score(
                result, query)

        # Combine scores based on configuration
        total_score = (
            clinical_score * self.config.clinical_weight +
            semantic_score * self.config.semantic_weight +
            recency_score * self.config.recency_weight +
            authority_score * self.config.authority_weight +
            diversity_score * self.config.diversity_weight
        )

        # Generate explanation
        explanation = self._generate_score_explanation(
            clinical_score, semantic_score, recency_score,
            authority_score, diversity_score, algorithm_scores
        )

        # Calculate confidence
        confidence = self._calculate_confidence(algorithm_scores, total_score)

        return RelevanceScore(
            total_score=total_score,
            clinical_score=clinical_score,
            semantic_score=semantic_score,
            recency_score=recency_score,
            authority_score=authority_score,
            diversity_score=diversity_score,
            algorithm_scores=algorithm_scores,
            explanation=explanation,
            confidence=confidence
        )

    def _calculate_clinical_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate clinical relevance score."""
        content = result.knowledge_item.content.lower()
        query_text = query.text.lower()

        score = 0.0
        total_weight = 0.0

        # Check clinical terms
        for category, terms in self.clinical_terms.items():
            category_score = 0.0
            category_weight = 0.0

            for term, weight in terms.items():
                if term in query_text and term in content:
                    category_score += weight * 2  # Both query and content
                elif term in query_text or term in content:
                    category_score += weight  # Only one

                category_weight += weight

            if category_weight > 0:
                normalized_category_score = category_score / category_weight
                score += normalized_category_score
                total_weight += 1.0

        # Normalize by number of categories
        if total_weight > 0:
            score = score / total_weight

        # Boost for exact clinical domain matches
        if query.clinical_domains:
            for domain in query.clinical_domains:
                if domain.lower() in content:
                    score += 0.2

        return min(score, 1.0)

    def _calculate_semantic_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate semantic similarity score."""
        # Use the existing similarity score from the result
        base_score = result.similarity_score

        # Enhance with relevance score
        relevance_boost = result.relevance_score * 0.3

        # Combine scores
        semantic_score = base_score + relevance_boost

        return min(semantic_score, 1.0)

    def _calculate_recency_score(self, result: EnhancedSearchResult) -> float:
        """Calculate recency score based on knowledge freshness."""
        # For now, use a default recency score
        # In practice, this would be based on publication date, update date,
        # etc.

        knowledge_type = result.knowledge_item.knowledge_type

        # DSM-5 and clinical guidelines are considered current
        if knowledge_type in ["dsm5", "clinical_guidelines"]:
            return 1.0
        elif knowledge_type in ["pdm2", "therapeutic_technique"]:
            return 0.8
        elif knowledge_type in ["case_study", "research"]:
            return 0.6
        else:
            return 0.4

    def _calculate_authority_score(
            self, result: EnhancedSearchResult) -> float:
        """Calculate authority score based on source credibility."""
        knowledge_type = result.knowledge_item.knowledge_type
        source = result.knowledge_item.source

        # Get base authority score
        authority_score = self.authority_scores.get(knowledge_type, 0.3)

        # Adjust based on source
        if "official" in source.lower() or "dsm" in source.lower():
            authority_score += 0.2
        elif "peer_reviewed" in source.lower() or "journal" in source.lower():
            authority_score += 0.1

        return min(authority_score, 1.0)

    def _calculate_diversity_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate diversity score to promote varied results."""
        # This is a simplified diversity calculation
        # In practice, this would consider the diversity of the entire result
        # set

        knowledge_type = result.knowledge_item.knowledge_type
        clinical_domains = result.clinical_domains

        # Promote diversity in knowledge types
        diversity_score = 0.5

        # Boost for different clinical domains
        if clinical_domains and query.clinical_domains:
            domain_overlap = len(
                set(clinical_domains) & set(
                    query.clinical_domains))
            domain_diversity = len(
                set(clinical_domains) - set(query.clinical_domains))

            if domain_diversity > 0:
                diversity_score += 0.3
            if domain_overlap > 0:
                diversity_score += 0.2

        return min(diversity_score, 1.0)

    def _calculate_tf_idf_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate TF-IDF score."""
        content = result.knowledge_item.content.lower()
        query_terms = query.text.lower().split()

        score = 0.0

        for term in query_terms:
            # Term frequency in document
            tf = content.count(term) / len(content.split())

            # Inverse document frequency (simplified)
            # In practice, this would use pre-computed IDF values
            idf = math.log(self.total_documents /
                           max(1, self.document_frequencies.get(term, 1)))

            score += tf * idf

        # Normalize by query length
        if len(query_terms) > 0:
            score = score / len(query_terms)

        return min(score, 1.0)

    def _calculate_bm25_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate BM25 score."""
        content = result.knowledge_item.content.lower()
        query_terms = query.text.lower().split()
        doc_length = len(content.split())

        score = 0.0

        for term in query_terms:
            # Term frequency in document
            tf = content.count(term)

            # BM25 formula
            idf = math.log(self.total_documents /
                           max(1, self.document_frequencies.get(term, 1)))

            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * \
                (1 - self.config.b + self.config.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return min(score / max(1, len(query_terms)), 1.0)

    def _calculate_clinical_weighted_score(
            self,
            result: EnhancedSearchResult,
            query: SearchQuery) -> float:
        """Calculate clinical domain weighted score."""
        clinical_score = self._calculate_clinical_score(result, query)
        semantic_score = result.similarity_score

        # Weight clinical relevance higher for clinical queries
        clinical_weight = 0.7 if any(domain in query.text.lower()
                                     for domain_terms in self.clinical_terms.values()
                                     for domain in domain_terms.keys()) else 0.3

        return clinical_score * clinical_weight + \
            semantic_score * (1 - clinical_weight)

    def _generate_score_explanation(self,
                                    clinical_score: float,
                                    semantic_score: float,
                                    recency_score: float,
                                    authority_score: float,
                                    diversity_score: float,
                                    algorithm_scores: Dict[str,
                                                           float]) -> str:
        """Generate human-readable explanation of the score."""
        explanations = []

        if clinical_score > 0.7:
            explanations.append("High clinical relevance")
        elif clinical_score > 0.4:
            explanations.append("Moderate clinical relevance")

        if semantic_score > 0.8:
            explanations.append("Strong semantic match")
        elif semantic_score > 0.5:
            explanations.append("Good semantic match")

        if authority_score > 0.8:
            explanations.append("Authoritative source")

        if recency_score > 0.8:
            explanations.append("Current knowledge")

        # Add algorithm-specific explanations
        if "bm25" in algorithm_scores and algorithm_scores["bm25"] > 0.6:
            explanations.append("Strong keyword match")

        if "tf_idf" in algorithm_scores and algorithm_scores["tf_idf"] > 0.5:
            explanations.append("Relevant term frequency")

        return "; ".join(
            explanations) if explanations else "Basic relevance match"

    def _calculate_confidence(
            self, algorithm_scores: Dict[str, float], total_score: float) -> float:
        """Calculate confidence in the relevance score."""
        if not algorithm_scores:
            return 0.5

        # Confidence based on agreement between algorithms
        scores = list(algorithm_scores.values())
        if len(scores) > 1:
            score_variance = np.var(scores) if NUMPY_AVAILABLE else sum(
                (s - total_score)**2 for s in scores) / len(scores)
            confidence = 1.0 - min(score_variance, 1.0)
        else:
            confidence = 0.7  # Default confidence for single algorithm

        # Boost confidence for high scores
        if total_score > 0.8:
            confidence += 0.1

        return min(confidence, 1.0)

    def _apply_ranking_strategy(self, scored_results: List[RankedResult],
                                query: SearchQuery) -> List[RankedResult]:
        """Apply ranking strategy to scored results."""
        if self.config.ranking_strategy == RankingStrategy.RELEVANCE_ONLY:
            return self._rank_by_relevance(scored_results)
        elif self.config.ranking_strategy == RankingStrategy.RECENCY_WEIGHTED:
            return self._rank_by_recency_weighted(scored_results)
        elif self.config.ranking_strategy == RankingStrategy.AUTHORITY_WEIGHTED:
            return self._rank_by_authority_weighted(scored_results)
        elif self.config.ranking_strategy == RankingStrategy.DIVERSITY_AWARE:
            return self._rank_by_diversity_aware(scored_results)
        elif self.config.ranking_strategy == RankingStrategy.CONTEXT_ADAPTIVE:
            return self._rank_by_context_adaptive(scored_results, query)
        else:
            return self._rank_by_relevance(scored_results)

    def _rank_by_relevance(
            self,
            scored_results: List[RankedResult]) -> List[RankedResult]:
        """Rank by total relevance score only."""
        return sorted(
            scored_results,
            key=lambda x: x.relevance_score.total_score,
            reverse=True)

    def _rank_by_recency_weighted(
            self, scored_results: List[RankedResult]) -> List[RankedResult]:
        """Rank with recency weighting."""
        def recency_weighted_score(result):
            return (result.relevance_score.total_score * 0.7 +
                    result.relevance_score.recency_score * 0.3)

        return sorted(scored_results, key=recency_weighted_score, reverse=True)

    def _rank_by_authority_weighted(
            self, scored_results: List[RankedResult]) -> List[RankedResult]:
        """Rank with authority weighting."""
        def authority_weighted_score(result):
            return (result.relevance_score.total_score * 0.6 +
                    result.relevance_score.authority_score * 0.4)

        return sorted(
            scored_results,
            key=authority_weighted_score,
            reverse=True)

    def _rank_by_diversity_aware(
            self, scored_results: List[RankedResult]) -> List[RankedResult]:
        """Rank with diversity awareness."""
        # Simple diversity-aware ranking
        # In practice, this would use more sophisticated algorithms like MMR

        ranked_results = []
        remaining_results = scored_results.copy()
        seen_types = set()

        # First pass: select diverse knowledge types
        for result in sorted(
                remaining_results,
                key=lambda x: x.relevance_score.total_score,
                reverse=True):
            knowledge_type = result.result.knowledge_item.knowledge_type
            if knowledge_type not in seen_types or len(ranked_results) < 3:
                ranked_results.append(result)
                seen_types.add(knowledge_type)
                remaining_results.remove(result)

        # Second pass: add remaining results by relevance
        remaining_sorted = sorted(
            remaining_results,
            key=lambda x: x.relevance_score.total_score,
            reverse=True)
        ranked_results.extend(remaining_sorted)

        return ranked_results

    def _rank_by_context_adaptive(self, scored_results: List[RankedResult],
                                  query: SearchQuery) -> List[RankedResult]:
        """Rank adaptively based on search context."""
        if query.context == SearchContext.TRAINING:
            # For training, prioritize diverse, authoritative sources
            return self._rank_by_diversity_aware(scored_results)
        elif query.context == SearchContext.INFERENCE:
            # For inference, prioritize high relevance and authority
            return self._rank_by_authority_weighted(scored_results)
        elif query.context == SearchContext.VALIDATION:
            # For validation, prioritize pure relevance
            return self._rank_by_relevance(scored_results)
        else:  # RESEARCH
            # For research, prioritize recency and diversity
            return self._rank_by_recency_weighted(scored_results)

    def get_scoring_stats(
            self, ranked_results: List[RankedResult]) -> Dict[str, Any]:
        """Get statistics about the scoring and ranking."""
        if not ranked_results:
            return {"error": "No results to analyze"}

        scores = [r.relevance_score.total_score for r in ranked_results]
        clinical_scores = [
            r.relevance_score.clinical_score for r in ranked_results]
        semantic_scores = [
            r.relevance_score.semantic_score for r in ranked_results]

        stats = {
            "total_results": len(ranked_results),
            "score_statistics": {
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "score_range": max(scores) - min(scores)},
            "clinical_statistics": {
                "mean_clinical_score": sum(clinical_scores) / len(clinical_scores),
                "high_clinical_relevance": sum(
                    1 for s in clinical_scores if s > 0.7),
                "moderate_clinical_relevance": sum(
                    1 for s in clinical_scores if 0.4 < s <= 0.7)},
            "semantic_statistics": {
                "mean_semantic_score": sum(semantic_scores) / len(semantic_scores),
                "high_semantic_match": sum(
                    1 for s in semantic_scores if s > 0.8)},
            "ranking_changes": {
                "promoted": sum(
                    1 for r in ranked_results if r.rank_change > 0),
                "demoted": sum(
                    1 for r in ranked_results if r.rank_change < 0),
                "unchanged": sum(
                    1 for r in ranked_results if r.rank_change == 0)},
            "knowledge_type_distribution": {}}

        # Knowledge type distribution
        type_counts = defaultdict(int)
        for result in ranked_results:
            knowledge_type = result.result.knowledge_item.knowledge_type
            type_counts[knowledge_type] += 1

        stats["knowledge_type_distribution"] = dict(type_counts)

        return stats


def main():
    """Test the knowledge relevance scorer."""
    print("Testing Knowledge Relevance Scorer")

    # Create scorer with different configurations
    configs = [
        ScoringConfig(
            algorithm=ScoringAlgorithm.HYBRID,
            ranking_strategy=RankingStrategy.CONTEXT_ADAPTIVE),
        ScoringConfig(
            algorithm=ScoringAlgorithm.BM25,
            ranking_strategy=RankingStrategy.RELEVANCE_ONLY),
        ScoringConfig(
            algorithm=ScoringAlgorithm.CLINICAL_WEIGHTED,
            ranking_strategy=RankingStrategy.DIVERSITY_AWARE)]

    for i, config in enumerate(configs):
        print(
            f"\n{i+1}. Testing configuration: {config.algorithm.value} + {config.ranking_strategy.value}")

        scorer = KnowledgeRelevanceScorer(config)

        # Create mock search results
        from .clinical_similarity_search import SearchQuery, SearchContext

        mock_results = []
        knowledge_items = [
            ("dsm5_depression",
             "Major depressive disorder diagnostic criteria and symptoms",
             "dsm5"),
            ("therapy_cbt",
             "Cognitive behavioral therapy techniques for depression",
             "therapeutic_technique"),
            ("case_study",
             "Clinical case study of depression treatment",
             "case_study")]

        for item_id, content, knowledge_type in knowledge_items:
            from .clinical_knowledge_embedder import KnowledgeItem
            from .clinical_similarity_search import EnhancedSearchResult

            knowledge_item = KnowledgeItem(
                id=item_id,
                content=content,
                knowledge_type=knowledge_type,
                source="test_source"
            )

            enhanced_result = EnhancedSearchResult(
                knowledge_item=knowledge_item,
                similarity_score=0.8,
                relevance_score=0.7,
                combined_score=0.75,
                rank=0,
                relevance_explanation="Test result",
                clinical_domains=["depression"]
            )

            mock_results.append(enhanced_result)

        # Create test query
        query = SearchQuery(
            text="depression symptoms and treatment",
            context=SearchContext.TRAINING,
            clinical_domains=["depression"]
        )

        # Score and rank results
        try:
            ranked_results = scorer.score_and_rank(mock_results, query)

            print(f"  Ranked {len(ranked_results)} results:")
            for result in ranked_results:
                print(
                    f"    {result.final_rank + 1}. {result.result.knowledge_item.id}")
                print(
                    f"       Score: {result.relevance_score.total_score:.3f}")
                print(
                    f"       Explanation: {result.relevance_score.explanation}")
                print(f"       Rank change: {result.rank_change:+d}")

            # Get statistics
            stats = scorer.get_scoring_stats(ranked_results)
            print(f"  Statistics:")
            print(
                f"    Mean score: {stats['score_statistics']['mean_score']:.3f}")
            print(
                f"    High clinical relevance: {stats['clinical_statistics']['high_clinical_relevance']}")
            print(
                f"    Ranking changes: +{stats['ranking_changes']['promoted']}, -{stats['ranking_changes']['demoted']}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n✅ Knowledge Relevance Scorer testing completed!")


if __name__ == "__main__":
    main()
