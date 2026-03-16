"""
Dataset deduplication logic for identifying duplicate sources across platforms.

Uses DOI-based matching and title/author similarity for non-DOI sources.
"""

import logging
from typing import Optional

from ..models import DatasetSource
from .metadata_parser import MetadataParser

logger = logging.getLogger(__name__)


class DatasetDeduplicator:
    """Deduplicates dataset sources found across multiple platforms."""
    
    def __init__(self, title_similarity_threshold: float = 0.8):
        """
        Initialize deduplicator.
        
        Args:
            title_similarity_threshold: Minimum similarity for title matching (0-1)
        """
        self.title_similarity_threshold = title_similarity_threshold
        self.parser = MetadataParser()
    
    def deduplicate(self, sources: list[DatasetSource]) -> list[DatasetSource]:
        """
        Remove duplicate sources from list.
        
        Args:
            sources: List of DatasetSource objects
        
        Returns:
            Deduplicated list of DatasetSource objects
        """
        if not sources:
            return []
        
        logger.info(f"Deduplicating {len(sources)} sources")
        
        # Track unique sources
        unique_sources = []
        doi_index: dict[str, DatasetSource] = {}
        title_index: list[tuple[str, DatasetSource]] = []
        
        for source in sources:
            # Check for DOI-based duplicates
            if source.doi:
                normalized_doi = self.parser.normalize_doi(source.doi)
                
                if normalized_doi in doi_index:
                    logger.debug(f"Duplicate DOI found: {normalized_doi}")
                    # Merge metadata from duplicate
                    self._merge_duplicate_metadata(doi_index[normalized_doi], source)
                    continue
                
                doi_index[normalized_doi] = source
                unique_sources.append(source)
                continue
            
            # Check for title/author similarity
            is_duplicate = False
            normalized_title = self.parser.normalize_title(source.title)
            
            for existing_title, existing_source in title_index:
                if self._is_duplicate_by_title_and_authors(
                    source, existing_source, normalized_title, existing_title
                ):
                    logger.debug(f"Duplicate by title/authors: {source.title}")
                    # Merge metadata from duplicate
                    self._merge_duplicate_metadata(existing_source, source)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                title_index.append((normalized_title, source))
        
        logger.info(f"Deduplicated to {len(unique_sources)} unique sources")
        return unique_sources
    
    def _is_duplicate_by_title_and_authors(
        self,
        source1: DatasetSource,
        source2: DatasetSource,
        normalized_title1: str,
        normalized_title2: str
    ) -> bool:
        """
        Check if two sources are duplicates based on title and author similarity.
        
        Args:
            source1: First source
            source2: Second source
            normalized_title1: Normalized title of first source
            normalized_title2: Normalized title of second source
        
        Returns:
            True if sources are duplicates
        """
        # Calculate title similarity
        title_similarity = self.parser.calculate_text_similarity(
            normalized_title1,
            normalized_title2
        )
        
        if title_similarity < self.title_similarity_threshold:
            return False
        
        # Check author overlap
        if source1.authors and source2.authors:
            last_names1 = self.parser.extract_author_last_names(source1.authors)
            last_names2 = self.parser.extract_author_last_names(source2.authors)
            
            if last_names1 and last_names2:
                # Calculate Jaccard similarity of author last names
                intersection = len(last_names1 & last_names2)
                union = len(last_names1 | last_names2)
                author_similarity = intersection / union if union > 0 else 0.0
                
                # Require at least 50% author overlap
                if author_similarity < 0.5:
                    return False
        
        # Check publication year similarity (within 1 year)
        year_diff = abs(
            source1.publication_date.year - source2.publication_date.year
        )
        if year_diff > 1:
            return False
        
        return True
    
    def _merge_duplicate_metadata(
        self,
        primary: DatasetSource,
        duplicate: DatasetSource
    ) -> None:
        """
        Merge metadata from duplicate into primary source.
        
        Updates primary source with additional information from duplicate.
        
        Args:
            primary: Primary source to update
            duplicate: Duplicate source with additional metadata
        """
        # Merge keywords
        existing_keywords = set(kw.lower() for kw in primary.keywords)
        for keyword in duplicate.keywords:
            if keyword.lower() not in existing_keywords:
                primary.keywords.append(keyword)
        
        # Update DOI if primary doesn't have one
        if not primary.doi and duplicate.doi:
            primary.doi = duplicate.doi
        
        # Update abstract if primary has shorter one
        if len(duplicate.abstract) > len(primary.abstract):
            primary.abstract = duplicate.abstract
        
        # Merge authors
        existing_authors = set(
            self.parser.normalize_author_name(author)
            for author in primary.authors
        )
        for author in duplicate.authors:
            normalized = self.parser.normalize_author_name(author)
            if normalized not in existing_authors:
                primary.authors.append(author)
        
        # Update data availability if duplicate has better info
        availability_priority = {
            'available': 3,
            'upon_request': 2,
            'restricted': 1,
            'unknown': 0
        }
        
        if (availability_priority.get(duplicate.data_availability, 0) >
            availability_priority.get(primary.data_availability, 0)):
            primary.data_availability = duplicate.data_availability
        
        logger.debug(f"Merged metadata from duplicate into {primary.source_id}")
    
    def find_duplicates_by_doi(
        self,
        sources: list[DatasetSource],
        doi: str
    ) -> list[DatasetSource]:
        """
        Find all sources with matching DOI.
        
        Args:
            sources: List of sources to search
            doi: DOI to match
        
        Returns:
            List of matching sources
        """
        normalized_doi = self.parser.normalize_doi(doi)
        if not normalized_doi:
            return []
        
        matches = []
        for source in sources:
            if source.doi:
                source_doi = self.parser.normalize_doi(source.doi)
                if source_doi == normalized_doi:
                    matches.append(source)
        
        return matches
    
    def find_similar_sources(
        self,
        target: DatasetSource,
        sources: list[DatasetSource],
        min_similarity: float = 0.7
    ) -> list[tuple[DatasetSource, float]]:
        """
        Find sources similar to target based on title and authors.
        
        Args:
            target: Target source to find similarities for
            sources: List of sources to search
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (source, similarity_score) tuples, sorted by similarity
        """
        normalized_target_title = self.parser.normalize_title(target.title)
        target_last_names = self.parser.extract_author_last_names(target.authors)
        
        similar = []
        
        for source in sources:
            if source.source_id == target.source_id:
                continue
            
            # Calculate title similarity
            normalized_title = self.parser.normalize_title(source.title)
            title_sim = self.parser.calculate_text_similarity(
                normalized_target_title,
                normalized_title
            )
            
            # Calculate author similarity
            author_sim = 0.0
            if target.authors and source.authors:
                source_last_names = self.parser.extract_author_last_names(source.authors)
                if target_last_names and source_last_names:
                    intersection = len(target_last_names & source_last_names)
                    union = len(target_last_names | source_last_names)
                    author_sim = intersection / union if union > 0 else 0.0
            
            # Combined similarity (weighted average)
            combined_sim = 0.7 * title_sim + 0.3 * author_sim
            
            if combined_sim >= min_similarity:
                similar.append((source, combined_sim))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar
    
    def create_unified_source(
        self,
        duplicates: list[DatasetSource]
    ) -> Optional[DatasetSource]:
        """
        Create a unified source from multiple duplicates.
        
        Combines metadata from all duplicates into a single source.
        
        Args:
            duplicates: List of duplicate sources
        
        Returns:
            Unified DatasetSource or None if list is empty
        """
        if not duplicates:
            return None
        
        # Use first source as base
        unified = duplicates[0]
        
        # Merge metadata from remaining duplicates
        for duplicate in duplicates[1:]:
            self._merge_duplicate_metadata(unified, duplicate)
        
        return unified
    
    def group_by_similarity(
        self,
        sources: list[DatasetSource],
        similarity_threshold: float = 0.8
    ) -> list[list[DatasetSource]]:
        """
        Group sources by similarity.
        
        Args:
            sources: List of sources to group
            similarity_threshold: Minimum similarity for grouping
        
        Returns:
            List of source groups
        """
        if not sources:
            return []
        
        groups: list[list[DatasetSource]] = []
        assigned: set[str] = set()
        
        for source in sources:
            if source.source_id in assigned:
                continue
            
            # Create new group with this source
            group = [source]
            assigned.add(source.source_id)
            
            # Find similar sources
            similar = self.find_similar_sources(
                source,
                sources,
                similarity_threshold
            )
            
            for similar_source, _ in similar:
                if similar_source.source_id not in assigned:
                    group.append(similar_source)
                    assigned.add(similar_source.source_id)
            
            groups.append(group)
        
        return groups
    
    def get_deduplication_stats(
        self,
        original_count: int,
        deduplicated_count: int
    ) -> dict[str, any]:
        """
        Calculate deduplication statistics.
        
        Args:
            original_count: Original number of sources
            deduplicated_count: Number after deduplication
        
        Returns:
            Dictionary with statistics
        """
        duplicates_removed = original_count - deduplicated_count
        duplicate_rate = (duplicates_removed / original_count * 100) if original_count > 0 else 0
        
        return {
            'original_count': original_count,
            'deduplicated_count': deduplicated_count,
            'duplicates_removed': duplicates_removed,
            'duplicate_rate_percent': round(duplicate_rate, 2)
        }
