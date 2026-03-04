"""
Source Discovery Engine

Automated and manual search of academic sources for therapeutic datasets.
"""

from ai.core.sourcing.journal.discovery.deduplication import Deduplicator
from ai.core.sourcing.journal.discovery.discovery_service import DiscoveryService
from ai.core.sourcing.journal.discovery.doaj_client import DOAJClient
from ai.core.sourcing.journal.discovery.metadata_parser import MetadataParser
from ai.core.sourcing.journal.discovery.pubmed_client import PubMedClient
from ai.core.sourcing.journal.discovery.repository_clients import (
    ClinicalTrialsClient,
    DryadClient,
    ZenodoClient,
)

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

