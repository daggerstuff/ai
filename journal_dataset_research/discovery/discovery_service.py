"""
Unified Discovery Service

Implements the DiscoveryServiceProtocol to provide a unified interface
for discovering dataset sources from multiple platforms.
"""

import logging
from typing import Dict, List, Optional

from ai.journal_dataset_research.discovery.deduplication import Deduplicator
from ai.journal_dataset_research.discovery.doaj_client import DOAJClient
from ai.journal_dataset_research.discovery.pubmed_client import PubMedClient
from ai.journal_dataset_research.discovery.repository_clients import (
    ClinicalTrialsClient,
    DryadClient,
    ZenodoClient,
)
from ai.journal_dataset_research.models.dataset_models import DatasetSource, ResearchSession
from ai.journal_dataset_research.orchestrator.types import DiscoveryServiceProtocol

logger = logging.getLogger(__name__)


class DiscoveryService(DiscoveryServiceProtocol):
    """Unified discovery service for multiple academic sources."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        pubmed_client: Optional[PubMedClient] = None,
        doaj_client: Optional[DOAJClient] = None,
        dryad_client: Optional[DryadClient] = None,
        zenodo_client: Optional[ZenodoClient] = None,
        clinical_trials_client: Optional[ClinicalTrialsClient] = None,
        deduplicator: Optional[Deduplicator] = None,
    ):
        """Initialize discovery service."""
        self.config = config or {}

        # Initialize clients
        discovery_config = self.config.get("discovery", {})

        # PubMed client
        pubmed_config = discovery_config.get("pubmed", {})
        self.pubmed_client = pubmed_client or PubMedClient(
            api_key=pubmed_config.get("api_key"),
            base_url=pubmed_config.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
            search_limit=pubmed_config.get("search_limit", 100),
        )

        # DOAJ client
        doaj_config = discovery_config.get("doaj", {})
        self.doaj_client = doaj_client or DOAJClient(
            base_url=doaj_config.get("base_url", "https://doaj.org/api/v2"),
        )

        # Repository clients
        repos_config = discovery_config.get("repositories", {})

        dryad_config = repos_config.get("dryad", {})
        self.dryad_client = dryad_client or DryadClient(
            base_url=dryad_config.get("base_url", "https://datadryad.org/api/v2"),
        )

        zenodo_config = repos_config.get("zenodo", {})
        self.zenodo_client = zenodo_client or ZenodoClient(
            base_url=zenodo_config.get("base_url", "https://zenodo.org/api"),
        )

        clinical_trials_config = repos_config.get("clinical_trials", {})
        self.clinical_trials_client = clinical_trials_client or ClinicalTrialsClient(
            base_url=clinical_trials_config.get(
                "base_url", "https://clinicaltrials.gov/api/v2"
            ),
        )

        # Deduplicator
        self.deduplicator = deduplicator or Deduplicator()

    def discover_sources(self, session: ResearchSession) -> List[DatasetSource]:
        """
        Discover dataset sources for the given research session.

        Args:
            session: ResearchSession with target sources and search keywords

        Returns:
            List of discovered DatasetSource objects
        """
        all_sources: List[DatasetSource] = []
        target_sources = session.target_sources or []
        search_keywords = session.search_keywords or {}

        # Extract keywords - handle both dict and list formats
        if isinstance(search_keywords, dict):
            therapeutic_keywords = search_keywords.get("therapeutic", [])
            dataset_keywords = search_keywords.get("dataset", [])
            # Ensure they are lists
            if not isinstance(therapeutic_keywords, list):
                therapeutic_keywords = [therapeutic_keywords] if therapeutic_keywords else []
            if not isinstance(dataset_keywords, list):
                dataset_keywords = [dataset_keywords] if dataset_keywords else []
            all_keywords = therapeutic_keywords + dataset_keywords
        elif isinstance(search_keywords, list):
            # If search_keywords is a list, use it directly
            all_keywords = search_keywords
            therapeutic_keywords = search_keywords
        else:
            # Fallback
            all_keywords = []
            therapeutic_keywords = []

        logger.info(
            f"Starting discovery for sources: {target_sources}, "
            f"keywords: {all_keywords}"
        )

        # Discover from each target source
        if "pubmed" in target_sources or "pubmed_central" in target_sources:
            try:
                sources = self._discover_pubmed(all_keywords)
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources from PubMed")
            except Exception as e:
                logger.error(f"Error discovering from PubMed: {e}", exc_info=True)

        if "doaj" in target_sources:
            try:
                sources = self._discover_doaj(all_keywords, therapeutic_keywords)
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources from DOAJ")
            except Exception as e:
                logger.error(f"Error discovering from DOAJ: {e}", exc_info=True)

        if "dryad" in target_sources:
            try:
                sources = self._discover_dryad(all_keywords)
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources from Dryad")
            except Exception as e:
                logger.error(f"Error discovering from Dryad: {e}", exc_info=True)

        if "zenodo" in target_sources:
            try:
                sources = self._discover_zenodo(all_keywords)
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources from Zenodo")
            except Exception as e:
                logger.error(f"Error discovering from Zenodo: {e}", exc_info=True)

        if "clinical_trials" in target_sources or "clinicaltrials" in target_sources:
            try:
                sources = self._discover_clinical_trials(all_keywords)
                all_sources.extend(sources)
                logger.info(f"Found {len(sources)} sources from ClinicalTrials.gov")
            except Exception as e:
                logger.error(f"Error discovering from ClinicalTrials.gov: {e}", exc_info=True)

        # Deduplicate sources
        if all_sources:
            all_sources = self.deduplicator.deduplicate(all_sources)

        logger.info(f"Total unique sources discovered: {len(all_sources)}")
        return all_sources

    def _discover_pubmed(self, keywords: List[str]) -> List[DatasetSource]:
        """Discover sources from PubMed Central."""
        if not keywords:
            keywords = ["therapy", "counseling", "psychotherapy"]

        return self.pubmed_client.search(
            keywords=keywords,
            max_results=self.config.get("discovery", {}).get("pubmed", {}).get("search_limit", 100),
        )

    def _discover_doaj(
        self, keywords: List[str], therapeutic_keywords: List[str]
    ) -> List[DatasetSource]:
        """Discover sources from DOAJ."""
        if not keywords and not therapeutic_keywords:
            keywords = ["therapy", "counseling", "psychotherapy"]
            therapeutic_keywords = keywords

        return self.doaj_client.search_articles(
            keywords=keywords,
            therapeutic_keywords=therapeutic_keywords,
            max_results=100,
        )

    def _discover_dryad(self, keywords: List[str]) -> List[DatasetSource]:
        """Discover sources from Dryad."""
        if not keywords:
            keywords = ["therapy", "counseling", "psychotherapy"]

        return self.dryad_client.search(keywords=keywords, max_results=100)

    def _discover_zenodo(self, keywords: List[str]) -> List[DatasetSource]:
        """Discover sources from Zenodo."""
        if not keywords:
            keywords = ["therapy", "counseling", "psychotherapy"]

        return self.zenodo_client.search(keywords=keywords, max_results=100)

    def _discover_clinical_trials(self, keywords: List[str]) -> List[DatasetSource]:
        """Discover sources from ClinicalTrials.gov."""
        if not keywords:
            keywords = ["therapy", "counseling", "psychotherapy"]

        return self.clinical_trials_client.search(
            keywords=keywords,
            condition="Mental Health",
            status="COMPLETED",
            max_results=100,
        )

