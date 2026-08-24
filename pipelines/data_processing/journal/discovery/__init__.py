"""
Source Discovery Engine

Automated and manual search of academic sources for therapeutic datasets.
"""

from ai.pipelines.data_processing.journal.discovery.deduplication import Deduplicator
from ai.pipelines.data_processing.journal.discovery.discovery_service import DiscoveryService
from ai.pipelines.data_processing.journal.discovery.doaj_client import DOAJClient
from ai.pipelines.data_processing.journal.discovery.metadata_parser import MetadataParser
from ai.pipelines.data_processing.journal.discovery.pubmed_client import PubMedClient
from ai.pipelines.data_processing.journal.discovery.repository_clients import (
    ClinicalTrialsClient,
    DryadClient,
    ZenodoClient,
)

__all__ = [
    "ClinicalTrialsClient",
    "DOAJClient",
    "Deduplicator",
    "DiscoveryService",
    "DryadClient",
    "MetadataParser",
    "PubMedClient",
    "ZenodoClient",
]
