"""
Unified search interface for all dataset sources.

Provides a single interface to search across PubMed, DOAJ, and repositories.
"""

import logging
from typing import Any

from ai.sourcing.research_system.config import get_config
from ai.sourcing.research_system.models import DatasetSource
from ai.sourcing.research_system.source_discovery.deduplication import DatasetDeduplicator
from ai.sourcing.research_system.source_discovery.doaj_client import DOAJClient
from ai.sourcing.research_system.source_discovery.pubmed_client import PubMedClient
from ai.sourcing.research_system.source_discovery.repository_clients import (
    ClinicalTrialsClient,
    DryadClient,
    ZenodoClient,
)

logger = logging.getLogger(__name__)


class UnifiedSearchClient:
    """Unified interface for searching all dataset sources."""

    def __init__(self):
        """Initialize unified search client with all source clients."""
        self.config = get_config()

        # Initialize clients
        self.pubmed = PubMedClient()
        self.doaj = DOAJClient()
        self.dryad = DryadClient()
        self.zenodo = ZenodoClient()
        self.clinical_trials = ClinicalTrialsClient()

        # Initialize deduplicator
        self.deduplicator = DatasetDeduplicator()

        logger.info("Initialized unified search client")

    def search_all_sources(
        self, keywords: list[str], max_results_per_source: int = 50, deduplicate: bool = True
    ) -> list[DatasetSource]:
        """
        Search all sources with given keywords.

        Args:
            keywords: Search keywords
            max_results_per_source: Maximum results per source
            deduplicate: Whether to deduplicate results

        Returns:
            List of DatasetSource objects
        """
        logger.info(f"Searching all sources with keywords: {keywords}")

        all_sources = []

        # Search PubMed
        try:
            mesh_terms = self.config.get_mesh_terms()
            pubmed_sources = self.pubmed.search_and_fetch(
                keywords=keywords, mesh_terms=mesh_terms, max_results=max_results_per_source
            )
            all_sources.extend(pubmed_sources)
            logger.info(f"Found {len(pubmed_sources)} sources from PubMed")
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")

        # Search DOAJ
        try:
            doaj_sources = self.doaj.search_therapeutic_content(
                keywords=keywords, max_results=max_results_per_source
            )
            all_sources.extend(doaj_sources)
            logger.info(f"Found {len(doaj_sources)} sources from DOAJ")
        except Exception as e:
            logger.error(f"DOAJ search failed: {e}")

        # Search Dryad
        try:
            dryad_sources = self.dryad.search_therapeutic_datasets(
                keywords=keywords, max_results=max_results_per_source
            )
            all_sources.extend(dryad_sources)
            logger.info(f"Found {len(dryad_sources)} sources from Dryad")
        except Exception as e:
            logger.error(f"Dryad search failed: {e}")

        # Search Zenodo
        try:
            zenodo_sources = self.zenodo.search_therapeutic_datasets(
                keywords=keywords, max_results=max_results_per_source
            )
            all_sources.extend(zenodo_sources)
            logger.info(f"Found {len(zenodo_sources)} sources from Zenodo")
        except Exception as e:
            logger.error(f"Zenodo search failed: {e}")

        # Search ClinicalTrials.gov
        try:
            ct_sources = self.clinical_trials.search_mental_health_studies(
                keywords=keywords, max_results=max_results_per_source
            )
            all_sources.extend(ct_sources)
            logger.info(f"Found {len(ct_sources)} sources from ClinicalTrials.gov")
        except Exception as e:
            logger.error(f"ClinicalTrials.gov search failed: {e}")

        logger.info(f"Total sources found: {len(all_sources)}")

        # Deduplicate if requested
        if deduplicate:
            all_sources = self.deduplicator.deduplicate(all_sources)
            logger.info(f"After deduplication: {len(all_sources)} unique sources")

        return all_sources

    def search_by_dataset_type(
        self, dataset_type: str, max_results_per_source: int = 50, deduplicate: bool = True
    ) -> list[DatasetSource]:
        """
        Search for specific dataset type using configured keywords.

        Args:
            dataset_type: Type of dataset (therapy_transcripts, clinical_outcomes, etc.)
            max_results_per_source: Maximum results per source
            deduplicate: Whether to deduplicate results

        Returns:
            List of DatasetSource objects
        """
        keywords = self.config.get_search_keywords(dataset_type)

        if not keywords:
            logger.warning(f"No keywords configured for dataset type: {dataset_type}")
            return []

        logger.info(f"Searching for {dataset_type} datasets")
        return self.search_all_sources(keywords, max_results_per_source, deduplicate)

    def search_pubmed_only(
        self, keywords: list[str], max_results: int = 100
    ) -> list[DatasetSource]:
        """
        Search PubMed only.

        Args:
            keywords: Search keywords
            max_results: Maximum results

        Returns:
            List of DatasetSource objects
        """
        mesh_terms = self.config.get_mesh_terms()
        return self.pubmed.search_and_fetch(
            keywords=keywords, mesh_terms=mesh_terms, max_results=max_results
        )

    def search_repositories_only(
        self, keywords: list[str], max_results_per_repo: int = 50, deduplicate: bool = True
    ) -> list[DatasetSource]:
        """
        Search repositories only (Dryad, Zenodo, ClinicalTrials.gov).

        Args:
            keywords: Search keywords
            max_results_per_repo: Maximum results per repository
            deduplicate: Whether to deduplicate results

        Returns:
            List of DatasetSource objects
        """
        all_sources = []

        # Search Dryad
        try:
            dryad_sources = self.dryad.search_therapeutic_datasets(
                keywords=keywords, max_results=max_results_per_repo
            )
            all_sources.extend(dryad_sources)
        except Exception as e:
            logger.error(f"Dryad search failed: {e}")

        # Search Zenodo
        try:
            zenodo_sources = self.zenodo.search_therapeutic_datasets(
                keywords=keywords, max_results=max_results_per_repo
            )
            all_sources.extend(zenodo_sources)
        except Exception as e:
            logger.error(f"Zenodo search failed: {e}")

        # Search ClinicalTrials.gov
        try:
            ct_sources = self.clinical_trials.search_mental_health_studies(
                keywords=keywords, max_results=max_results_per_repo
            )
            all_sources.extend(ct_sources)
        except Exception as e:
            logger.error(f"ClinicalTrials.gov search failed: {e}")

        if deduplicate:
            all_sources = self.deduplicator.deduplicate(all_sources)

        return all_sources

    def search_comprehensive(
        self, max_results_per_source: int = 50
    ) -> dict[str, list[DatasetSource]]:
        """
        Perform comprehensive search across all dataset types.

        Args:
            max_results_per_source: Maximum results per source per type

        Returns:
            Dictionary mapping dataset types to source lists
        """
        logger.info("Starting comprehensive search across all dataset types")

        results = {}
        all_keywords = self.config.get_all_search_keywords()

        for dataset_type, keywords in all_keywords.items():
            logger.info(f"Searching for {dataset_type}")
            sources = self.search_all_sources(
                keywords=keywords, max_results_per_source=max_results_per_source, deduplicate=True
            )
            results[dataset_type] = sources
            logger.info(f"Found {len(sources)} unique sources for {dataset_type}")

        return results

    def get_search_statistics(self, results: dict[str, list[DatasetSource]]) -> dict[str, Any]:
        """
        Calculate statistics for search results.

        Args:
            results: Dictionary of search results by dataset type

        Returns:
            Dictionary with statistics
        """
        total_sources = sum(len(sources) for sources in results.values())

        # Count by source type
        source_type_counts = {}
        for sources in results.values():
            for source in sources:
                source_type_counts[source.source_type] = (
                    source_type_counts.get(source.source_type, 0) + 1
                )

        # Count by discovery method
        discovery_method_counts = {}
        for sources in results.values():
            for source in sources:
                discovery_method_counts[source.discovery_method] = (
                    discovery_method_counts.get(source.discovery_method, 0) + 1
                )

        # Count open access
        open_access_count = sum(
            1 for sources in results.values() for source in sources if source.open_access
        )

        # Count with DOI
        doi_count = sum(1 for sources in results.values() for source in sources if source.doi)

        return {
            "total_sources": total_sources,
            "sources_by_dataset_type": {dtype: len(sources) for dtype, sources in results.items()},
            "sources_by_type": source_type_counts,
            "sources_by_discovery_method": discovery_method_counts,
            "open_access_count": open_access_count,
            "open_access_percentage": round(
                (open_access_count / total_sources * 100) if total_sources > 0 else 0, 2
            ),
            "doi_count": doi_count,
            "doi_percentage": round(
                (doi_count / total_sources * 100) if total_sources > 0 else 0, 2
            ),
        }
