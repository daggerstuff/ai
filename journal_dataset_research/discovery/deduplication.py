"""
Deduplication Logic

Implements deduplication of dataset sources found across multiple platforms
using DOI matching and similarity-based matching.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Set

from ai.journal_dataset_research.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class Deduplicator:
    """Deduplicates dataset sources using multiple strategies."""

    def __init__(
        self,
        doi_match_threshold: float = 1.0,  # Exact match for DOI
        title_similarity_threshold: float = 0.85,
        author_similarity_threshold: float = 0.80,
    ):
        """Initialize deduplicator."""
        self.doi_match_threshold = doi_match_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.author_similarity_threshold = author_similarity_threshold

    def deduplicate(self, sources: List[DatasetSource]) -> List[DatasetSource]:
        """
        Deduplicate a list of dataset sources.

        Args:
            sources: List of DatasetSource objects

        Returns:
            Deduplicated list of DatasetSource objects
        """
        if not sources:
            return []

        # Step 1: Group by DOI (exact match)
        doi_groups = self._group_by_doi(sources)

        # Step 2: For sources without DOI, use similarity matching
        no_doi_sources = [s for s in sources if not s.doi]
        similarity_groups = self._group_by_similarity(no_doi_sources)

        # Step 3: Merge groups and select best representative
        deduplicated = []

        # Process DOI groups
        for doi, group in doi_groups.items():
            if group:
                best = self._select_best_source(group)
                deduplicated.append(best)

        # Process similarity groups (only for sources not already in DOI groups)
        processed_ids: Set[str] = {s.source_id for s in deduplicated}
        for group in similarity_groups:
            # Filter out sources already processed via DOI
            group = [s for s in group if s.source_id not in processed_ids]
            if group:
                best = self._select_best_source(group)
                deduplicated.append(best)
                processed_ids.add(best.source_id)

        logger.info(f"Deduplicated {len(sources)} sources to {len(deduplicated)} unique sources")
        return deduplicated

    def _group_by_doi(self, sources: List[DatasetSource]) -> Dict[str, List[DatasetSource]]:
        """Group sources by normalized DOI."""
        groups: Dict[str, List[DatasetSource]] = {}

        for source in sources:
            if not source.doi:
                continue

            # Normalize DOI
            doi_normalized = self._normalize_doi(source.doi)
            if doi_normalized:
                if doi_normalized not in groups:
                    groups[doi_normalized] = []
                groups[doi_normalized].append(source)

        return groups

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI for comparison."""
        if not doi:
            return ""

        # Remove common prefixes
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        doi = doi.replace("doi:", "").replace("DOI:", "").strip()

        # Remove trailing punctuation
        doi = doi.rstrip(".,;")

        return doi.lower()

    def _group_by_similarity(self, sources: List[DatasetSource]) -> List[List[DatasetSource]]:
        """Group sources by title and author similarity."""
        if not sources:
            return []

        groups: List[List[DatasetSource]] = []
        processed: Set[int] = set()

        for i, source1 in enumerate(sources):
            if i in processed:
                continue

            # Start a new group
            group = [source1]
            processed.add(i)

            # Find similar sources
            for j, source2 in enumerate(sources[i + 1 :], start=i + 1):
                if j in processed:
                    continue

                if self._are_similar(source1, source2):
                    group.append(source2)
                    processed.add(j)

            groups.append(group)

        return groups

    def _are_similar(self, source1: DatasetSource, source2: DatasetSource) -> bool:
        """Check if two sources are similar based on title and authors."""
        # Check title similarity
        title_sim = self._similarity(source1.title.lower(), source2.title.lower())
        if title_sim < self.title_similarity_threshold:
            return False

        # Check author similarity
        if source1.authors and source2.authors:
            author_sim = self._author_similarity(source1.authors, source2.authors)
            if author_sim < self.author_similarity_threshold:
                return False

        return True

    def _similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        if not str1 or not str2:
            return 0.0

        return SequenceMatcher(None, str1, str2).ratio()

    def _author_similarity(self, authors1: List[str], authors2: List[str]) -> float:
        """Calculate similarity between author lists."""
        if not authors1 or not authors2:
            return 0.0

        # Normalize author names
        authors1_normalized = [self._normalize_author_name(a) for a in authors1]
        authors2_normalized = [self._normalize_author_name(a) for a in authors2]

        # Check for overlapping authors
        set1 = set(authors1_normalized)
        set2 = set(authors2_normalized)

        if not set1 or not set2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _normalize_author_name(self, name: str) -> str:
        """Normalize author name for comparison."""
        if not name:
            return ""

        # Remove extra whitespace
        name = " ".join(name.split())

        # Convert to lowercase
        name = name.lower()

        # Remove common suffixes
        suffixes = ["jr", "sr", "ii", "iii", "iv"]
        for suffix in suffixes:
            if name.endswith(f" {suffix}"):
                name = name[: -len(suffix) - 1]

        return name.strip()

    def _select_best_source(self, group: List[DatasetSource]) -> DatasetSource:
        """Select the best representative source from a group."""
        if len(group) == 1:
            return group[0]

        # Score each source
        scored = []
        for source in group:
            score = 0

            # Prefer sources with DOI
            if source.doi:
                score += 10

            # Prefer sources with more complete metadata
            if source.abstract:
                score += 5
            if source.keywords:
                score += 2
            if len(source.authors) > 0:
                score += 3

            # Prefer sources with better data availability
            if source.data_availability == "available":
                score += 5
            elif source.data_availability == "upon_request":
                score += 2

            # Prefer more recent sources
            # (This is implicit - we'll take the first one with highest score)

            scored.append((score, source))

        # Sort by score (descending) and return the best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

