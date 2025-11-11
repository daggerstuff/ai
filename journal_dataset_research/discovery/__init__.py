"""
Source Discovery Engine

Automated and manual search of academic sources for therapeutic datasets.
"""

from ai.journal_dataset_research.discovery.discovery_service import DiscoveryService
from ai.journal_dataset_research.discovery.deduplication import Deduplicator
from ai.journal_dataset_research.discovery.doaj_client import DOAJClient
from ai.journal_dataset_research.discovery.pubmed_client import PubMedClient
from ai.journal_dataset_research.discovery.repository_clients import (
    ClinicalTrialsClient,
    DryadClient,
    ZenodoClient,
)
from ai.journal_dataset_research.discovery.metadata_parser import MetadataParser

__all__ = [
    "DiscoveryService",
    "Deduplicator",
    "DOAJClient",
    "PubMedClient",
    "DryadClient",
    "ZenodoClient",
    "ClinicalTrialsClient",
    "MetadataParser",
]

