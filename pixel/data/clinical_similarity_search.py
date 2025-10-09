"""
Clinical Knowledge Similarity Search

Implements advanced similarity search for relevant clinical knowledge retrieval
with semantic matching, contextual relevance, and clinical domain expertise.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
from enum import Enum
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .faiss_knowledge_index import FAISSKnowledgeIndex, SearchResult, IndexConfig
from .clinical_knowledge_embedder import ClinicalKnowledgeEmbedder, KnowledgeItem


class SearchContext(Enum):
    """Context types for clinical knowledge search."""
    TRAINING = "training"  # During model training
    INFERENCE = "inference"  # During model inference
    VALIDATION = "validation"  # During model validation
    RESEARCH = "research"  # For research purposes


class RelevanceType(Enum):
    """Types of clinical relevance."""
    DIAGNOSTIC = "diagnostic"  # Diagnostic criteria and symptoms
    THERAPEUTIC = "therapeutic"  # Treatment and intervention
    ASSESSMENT = "assessment"  # Clinical assessment tools
    THEORETICAL = "theoretical"  # Theoretical frameworks
    CASE_STUDY = "case_study"  # Clinical case examples


@dataclass
class SearchQuery:
    """Structured search query for clinical knowledge."""
    text: str
    context: SearchContext = SearchContext.TRAINING
    relevance_types: List[RelevanceType] = field(default_factory=list)
    # dsm5, pdm2, etc.
    knowledge_types: List[str] = field(default_factory=list)
    # depression, anxiety, etc.
    clinical_domains: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)  # cbt, dbt, etc.
    severity_levels: List[str] = field(
        default_factory=list)  # mild, moderate, severe
    max_results: int = 10
    min_relevance_score: float = 0.5
    include_metadata: bool = True


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with clinical relevance scoring."""
    knowledge_item: KnowledgeItem
    similarity_score: float
    relevance_score: float
    combined_score: float
    rank: int
    relevance_explanation: str
    clinical_domains: List[str] = field(default_factory=list)
    therapeutic_relevance: float = 0.0
    diagnostic_relevance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClinicalSimilaritySearch:
    """Advanced similarity search for clinical knowledge."""

    def __init__(self, faiss_index: Optional[FAISSKnowledgeIndex] = None,
                 embedder: Optional[ClinicalKnowledgeEmbedder] = None,
                 project_root: Optional[Path] = None):
        """Initialize the clinical similarity search system."""
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Initialize components
        self.faiss_index = faiss_index
        self.embedder = embedder

        # Clinical knowledge mappings
        self.clinical_terms = self._load_clinical_terms()
        self.diagnostic_keywords = self._load_diagnostic_keywords()
        self.therapeutic_keywords = self._load_therapeutic_keywords()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize if components not provided
        if self.faiss_index is None:
            self._initialize_faiss_index()

        if self.embedder is None:
            self._initialize_embedder()

    def _initialize_faiss_index(self):
        """Initialize FAISS index if not provided."""
        try:
            self.faiss_index = FAISSKnowledgeIndex(
                project_root=self.project_root)

            # Try to load existing index
            if not self.faiss_index.load_index():
                # Build new index if none exists
                self.logger.info("Building new FAISS index...")
                self.faiss_index.load_knowledge_embeddings()
                self.faiss_index.build_index()

        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            self.faiss_index = None

    def _initialize_embedder(self):
        """Initialize embedder if not provided."""
        try:
            self.embedder = ClinicalKnowledgeEmbedder(
                project_root=self.project_root)
        except Exception as e:
            self.logger.error(f"Failed to initialize embedder: {e}")
            self.embedder = None

    def _load_clinical_terms(self) -> Dict[str, List[str]]:
        """Load clinical terminology mappings."""
        return {
            "depression": [
                "major depressive disorder",
                "mdd",
                "depressive episode",
                "dysthymia",
                "persistent depressive disorder",
                "seasonal affective disorder",
                "sad",
                "postpartum depression",
                "melancholic features",
                "atypical depression"],
            "anxiety": [
                "generalized anxiety disorder",
                "gad",
                "panic disorder",
                "agoraphobia",
                "social anxiety disorder",
                "specific phobia",
                "separation anxiety",
                "selective mutism",
                "anxiety disorders",
                "panic attacks"],
            "trauma": [
                "post-traumatic stress disorder",
                "ptsd",
                "acute stress disorder",
                "complex ptsd",
                "trauma-related disorders",
                "dissociative disorders",
                "adjustment disorders",
                "trauma exposure",
                "traumatic events"],
            "personality": [
                "borderline personality disorder",
                "bpd",
                "narcissistic personality disorder",
                "antisocial personality disorder",
                "avoidant personality disorder",
                "dependent personality disorder",
                "obsessive-compulsive personality disorder",
                "paranoid personality disorder",
                "schizoid personality disorder"],
            "psychosis": [
                "schizophrenia",
                "brief psychotic disorder",
                "delusional disorder",
                "schizoaffective disorder",
                "substance-induced psychotic disorder",
                "psychotic features",
                "hallucinations",
                "delusions"]}

    def _load_diagnostic_keywords(self) -> Set[str]:
        """Load diagnostic-related keywords."""
        return {
            "diagnosis",
            "diagnostic",
            "criteria",
            "symptoms",
            "symptom",
            "disorder",
            "condition",
            "syndrome",
            "pathology",
            "clinical presentation",
            "assessment",
            "evaluation",
            "screening",
            "differential diagnosis",
            "comorbidity",
            "prevalence",
            "etiology",
            "risk factors"}

    def _load_therapeutic_keywords(self) -> Set[str]:
        """Load therapy-related keywords."""
        return {
            "therapy",
            "treatment",
            "intervention",
            "therapeutic",
            "counseling",
            "psychotherapy",
            "cbt",
            "dbt",
            "emdr",
            "psychodynamic",
            "humanistic",
            "behavioral",
            "cognitive",
            "mindfulness",
            "exposure",
            "systematic desensitization",
            "cognitive restructuring",
            "behavioral activation",
            "interpersonal therapy"}

    def search(self, query: Union[str, SearchQuery]
               ) -> List[EnhancedSearchResult]:
        """Perform enhanced similarity search for clinical knowledge."""
        if isinstance(query, str):
            query = SearchQuery(text=query)

        if self.faiss_index is None:
            self.logger.error("FAISS index not available")
            return []

        try:
            # Get initial similarity results
            similarity_results = self._get_similarity_results(query)

            # Enhance results with clinical relevance
            enhanced_results = self._enhance_results_with_relevance(
                similarity_results, query)

            # Apply filters
            filtered_results = self._apply_filters(enhanced_results, query)

            # Re-rank by combined score
            final_results = self._rerank_results(filtered_results, query)

            self.logger.info(
                f"Search completed: {len(final_results)} results for '{query.text[:50]}...'")
            return final_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def _get_similarity_results(
            self, query: SearchQuery) -> List[SearchResult]:
        """Get initial similarity results from FAISS index."""
        if self.embedder is None:
            # Use text-based search if embedder not available
            return self.faiss_index.search_by_text(
                query.text, k=query.max_results * 2)

        # Generate query embedding
        try:
            if hasattr(
                    self.embedder,
                    'embedding_model') and self.embedder.embedding_model:
                query_embedding = self.embedder.embedding_model.encode([query.text])[
                    0]
            else:
                # Use mock embedding
                query_embedding = self._generate_mock_embedding(query.text)

            return self.faiss_index.search(
                query_embedding, k=query.max_results * 2)

        except Exception as e:
            self.logger.warning(
                f"Failed to generate embedding, using text search: {e}")
            return self.faiss_index.search_by_text(
                query.text, k=query.max_results * 2)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for testing."""
        # Create deterministic embedding based on text hash
        text_hash = hash(text) % 1000000
        return [(text_hash + i) % 1000 / 1000.0 for i in range(384)]

    def _enhance_results_with_relevance(
            self,
            results: List[SearchResult],
            query: SearchQuery) -> List[EnhancedSearchResult]:
        """Enhance search results with clinical relevance scoring."""
        enhanced_results = []

        for result in results:
            # Calculate clinical relevance
            relevance_score, explanation = self._calculate_clinical_relevance(
                result.knowledge_item, query
            )

            # Calculate domain relevance
            clinical_domains = self._extract_clinical_domains(
                result.knowledge_item.content)

            # Calculate therapeutic and diagnostic relevance
            therapeutic_relevance = self._calculate_therapeutic_relevance(
                result.knowledge_item.content, query.text
            )
            diagnostic_relevance = self._calculate_diagnostic_relevance(
                result.knowledge_item.content, query.text
            )

            # Combine similarity and relevance scores
            combined_score = self._combine_scores(
                result.score, relevance_score, query.context)

            enhanced_result = EnhancedSearchResult(
                knowledge_item=result.knowledge_item,
                similarity_score=result.score,
                relevance_score=relevance_score,
                combined_score=combined_score,
                rank=result.rank,
                relevance_explanation=explanation,
                clinical_domains=clinical_domains,
                therapeutic_relevance=therapeutic_relevance,
                diagnostic_relevance=diagnostic_relevance,
                metadata=result.metadata
            )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _calculate_clinical_relevance(self, knowledge_item: KnowledgeItem,
                                      query: SearchQuery) -> Tuple[float, str]:
        """Calculate clinical relevance score and explanation."""
        content = knowledge_item.content.lower()
        query_text = query.text.lower()

        relevance_factors = []
        explanations = []

        # Check for direct keyword matches
        query_words = set(query_text.split())
        content_words = set(content.split())
        keyword_overlap = len(query_words.intersection(
            content_words)) / len(query_words)

        if keyword_overlap > 0.3:
            relevance_factors.append(keyword_overlap * 0.4)
            explanations.append(
                f"High keyword overlap ({keyword_overlap:.2f})")

        # Check clinical domain relevance
        domain_relevance = 0.0
        matched_domains = []

        for domain, terms in self.clinical_terms.items():
            domain_score = 0.0
            for term in terms:
                if term in query_text and term in content:
                    domain_score += 1.0
                elif term in query_text or term in content:
                    domain_score += 0.5

            if domain_score > 0:
                domain_relevance += domain_score / len(terms)
                matched_domains.append(domain)

        if domain_relevance > 0:
            relevance_factors.append(min(domain_relevance, 1.0) * 0.3)
            explanations.append(
                f"Clinical domain match: {', '.join(matched_domains)}")

        # Check diagnostic relevance
        diagnostic_score = sum(1 for keyword in self.diagnostic_keywords
                               if keyword in query_text and keyword in content)
        if diagnostic_score > 0:
            relevance_factors.append(min(diagnostic_score / 5.0, 1.0) * 0.2)
            explanations.append(
                f"Diagnostic relevance ({diagnostic_score} matches)")

        # Check therapeutic relevance
        therapeutic_score = sum(1 for keyword in self.therapeutic_keywords
                                if keyword in query_text and keyword in content)
        if therapeutic_score > 0:
            relevance_factors.append(min(therapeutic_score / 5.0, 1.0) * 0.2)
            explanations.append(
                f"Therapeutic relevance ({therapeutic_score} matches)")

        # Knowledge type bonus
        if query.knowledge_types and knowledge_item.knowledge_type in query.knowledge_types:
            relevance_factors.append(0.1)
            explanations.append(
                f"Knowledge type match: {knowledge_item.knowledge_type}")

        # Calculate final relevance score
        final_score = sum(relevance_factors) if relevance_factors else 0.1
        final_score = min(final_score, 1.0)  # Cap at 1.0

        explanation = "; ".join(
            explanations) if explanations else "Basic similarity match"

        return final_score, explanation

    def _extract_clinical_domains(self, content: str) -> List[str]:
        """Extract clinical domains from content."""
        content_lower = content.lower()
        domains = []

        for domain, terms in self.clinical_terms.items():
            if any(term in content_lower for term in terms):
                domains.append(domain)

        return domains

    def _calculate_therapeutic_relevance(
            self, content: str, query: str) -> float:
        """Calculate therapeutic relevance score."""
        content_lower = content.lower()
        query_lower = query.lower()

        therapeutic_matches = sum(1 for keyword in self.therapeutic_keywords
                                  if keyword in content_lower and keyword in query_lower)

        return min(therapeutic_matches / 3.0, 1.0)  # Normalize to 0-1

    def _calculate_diagnostic_relevance(
            self, content: str, query: str) -> float:
        """Calculate diagnostic relevance score."""
        content_lower = content.lower()
        query_lower = query.lower()

        diagnostic_matches = sum(1 for keyword in self.diagnostic_keywords
                                 if keyword in content_lower and keyword in query_lower)

        return min(diagnostic_matches / 3.0, 1.0)  # Normalize to 0-1

    def _combine_scores(self, similarity_score: float, relevance_score: float,
                        context: SearchContext) -> float:
        """Combine similarity and relevance scores based on context."""
        if context == SearchContext.TRAINING:
            # For training, prioritize relevance slightly more
            return 0.4 * similarity_score + 0.6 * relevance_score
        elif context == SearchContext.INFERENCE:
            # For inference, balance similarity and relevance
            return 0.5 * similarity_score + 0.5 * relevance_score
        elif context == SearchContext.VALIDATION:
            # For validation, prioritize similarity
            return 0.7 * similarity_score + 0.3 * relevance_score
        else:  # RESEARCH
            # For research, prioritize relevance
            return 0.3 * similarity_score + 0.7 * relevance_score

    def _apply_filters(self, results: List[EnhancedSearchResult],
                       query: SearchQuery) -> List[EnhancedSearchResult]:
        """Apply filters to search results."""
        filtered_results = []

        for result in results:
            # Apply minimum relevance score filter
            if result.combined_score < query.min_relevance_score:
                continue

            # Apply knowledge type filter
            if (query.knowledge_types and
                    result.knowledge_item.knowledge_type not in query.knowledge_types):
                continue

            # Apply clinical domain filter
            if (query.clinical_domains and not any(
                    domain in result.clinical_domains for domain in query.clinical_domains)):
                continue

            # Apply relevance type filter
            if query.relevance_types:
                passes_relevance_filter = False

                for relevance_type in query.relevance_types:
                    if relevance_type == RelevanceType.DIAGNOSTIC and result.diagnostic_relevance > 0.3:
                        passes_relevance_filter = True
                        break
                    elif relevance_type == RelevanceType.THERAPEUTIC and result.therapeutic_relevance > 0.3:
                        passes_relevance_filter = True
                        break
                    # Add other relevance type checks as needed

                if not passes_relevance_filter:
                    continue

            filtered_results.append(result)

        return filtered_results

    def _rerank_results(self, results: List[EnhancedSearchResult],
                        query: SearchQuery) -> List[EnhancedSearchResult]:
        """Re-rank results by combined score and limit to max results."""
        # Sort by combined score (descending)
        sorted_results = sorted(
            results,
            key=lambda x: x.combined_score,
            reverse=True)

        # Update ranks and limit results
        final_results = []
        for i, result in enumerate(sorted_results[:query.max_results]):
            result.rank = i
            final_results.append(result)

        return final_results

    def search_by_clinical_domain(
            self,
            domain: str,
            context: SearchContext = SearchContext.TRAINING,
            max_results: int = 10) -> List[EnhancedSearchResult]:
        """Search for knowledge items by clinical domain."""
        if domain not in self.clinical_terms:
            self.logger.warning(f"Unknown clinical domain: {domain}")
            return []

        # Create comprehensive query for the domain
        domain_terms = self.clinical_terms[domain]
        query_text = " ".join(domain_terms[:5])  # Use top 5 terms

        query = SearchQuery(
            text=query_text,
            context=context,
            clinical_domains=[domain],
            max_results=max_results
        )

        return self.search(query)

    def search_for_training_examples(
            self,
            topic: str,
            modality: str = None,
            max_results: int = 20) -> List[EnhancedSearchResult]:
        """Search for training examples on a specific topic."""
        query = SearchQuery(
            text=topic,
            context=SearchContext.TRAINING,
            relevance_types=[
                RelevanceType.CASE_STUDY,
                RelevanceType.THERAPEUTIC],
            modalities=[modality] if modality else [],
            max_results=max_results,
            min_relevance_score=0.4)

        return self.search(query)

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query."""
        suggestions = []
        partial_lower = partial_query.lower()

        # Suggest clinical domains
        for domain, terms in self.clinical_terms.items():
            if partial_lower in domain or any(
                    partial_lower in term for term in terms):
                suggestions.append(domain.replace("_", " ").title())

        # Suggest common clinical terms
        common_terms = [
            "cognitive behavioral therapy",
            "dialectical behavior therapy",
            "major depressive disorder",
            "generalized anxiety disorder",
            "post-traumatic stress disorder",
            "borderline personality disorder",
            "therapeutic alliance",
            "clinical assessment",
            "treatment planning"]

        for term in common_terms:
            if partial_lower in term.lower():
                suggestions.append(term)

        return list(set(suggestions))[:10]  # Return unique suggestions, max 10


def main():
    """Test the clinical similarity search system."""
    print("Testing Clinical Similarity Search")

    # Initialize search system
    search_system = ClinicalSimilaritySearch()

    if search_system.faiss_index is None:
        print("FAISS index not available, creating mock system...")
        return

    # Test basic search
    print("\n1. Testing basic search...")
    results = search_system.search("depression symptoms and treatment")

    print(f"Found {len(results)} results:")
    for result in results[:3]:
        print(f"  - {result.knowledge_item.id}")
        print(
            f"    Similarity: {result.similarity_score:.3f}, Relevance: {result.relevance_score:.3f}")
        print(f"    Combined: {result.combined_score:.3f}")
        print(f"    Explanation: {result.relevance_explanation}")
        print(f"    Domains: {result.clinical_domains}")

    # Test structured query
    print("\n2. Testing structured query...")
    structured_query = SearchQuery(
        text="cognitive behavioral therapy for anxiety",
        context=SearchContext.TRAINING,
        relevance_types=[RelevanceType.THERAPEUTIC],
        clinical_domains=["anxiety"],
        max_results=5
    )

    structured_results = search_system.search(structured_query)
    print(f"Structured search found {len(structured_results)} results")

    # Test domain search
    print("\n3. Testing domain search...")
    domain_results = search_system.search_by_clinical_domain("depression")
    print(f"Depression domain search found {len(domain_results)} results")

    # Test training examples search
    print("\n4. Testing training examples search...")
    training_results = search_system.search_for_training_examples(
        "panic attacks", "cbt")
    print(f"Training examples search found {len(training_results)} results")

    # Test search suggestions
    print("\n5. Testing search suggestions...")
    suggestions = search_system.get_search_suggestions("depr")
    print(f"Suggestions for 'depr': {suggestions}")

    print("\n✅ Clinical Similarity Search testing completed!")


if __name__ == "__main__":
    main()
