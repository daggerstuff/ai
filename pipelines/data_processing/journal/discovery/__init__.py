"""
Source Discovery Engine

Automated and manual search of academic sources for therapeutic datasets.
"""

from ai.sourcing.journal.discovery.deduplication import Deduplicator
from ai.sourcing.journal.discovery.discovery_service import DiscoveryService
from ai.sourcing.journal.discovery.doaj_client import DOAJClient
from ai.sourcing.journal.discovery.metadata_parser import MetadataParser
from ai.sourcing.journal.discovery.pubmed_client import PubMedClient
from ai.sourcing.journal.discovery.repository_clients import (
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
