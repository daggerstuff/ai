"""
Source discovery module for identifying therapeutic datasets from academic sources.

This module provides API clients and search functionality for:
- PubMed Central (NCBI E-utilities)
- DOAJ (Directory of Open Access Journals)
- Dryad Digital Repository
- Zenodo
- ClinicalTrials.gov

Version: 0.1.0
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Lazy imports with error handling
_import_errors: list[str] = []

try:
    from .base_client import APIError, BaseAPIClient, RateLimitError
except ImportError as e:
    logger.warning(f"Failed to import base_client: {e}")
    _import_errors.append("base_client")
    BaseAPIClient = None  # type: ignore
    APIError = None  # type: ignore
    RateLimitError = None  # type: ignore

try:
    from .pubmed_client import PubMedClient
except ImportError as e:
    logger.warning(f"Failed to import PubMedClient: {e}")
    _import_errors.append("PubMedClient")
    PubMedClient = None  # type: ignore

try:
    from .doaj_client import DOAJClient
except ImportError as e:
    logger.warning(f"Failed to import DOAJClient: {e}")
    _import_errors.append("DOAJClient")
    DOAJClient = None  # type: ignore

try:
    from .repository_clients import ClinicalTrialsClient, DryadClient, ZenodoClient
except ImportError as e:
    logger.warning(f"Failed to import repository clients: {e}")
    _import_errors.append("repository_clients")
    DryadClient = None  # type: ignore
    ZenodoClient = None  # type: ignore
    ClinicalTrialsClient = None  # type: ignore

try:
    from .metadata_parser import MetadataParser
except ImportError as e:
    logger.warning(f"Failed to import MetadataParser: {e}")
    _import_errors.append("MetadataParser")
    MetadataParser = None  # type: ignore

try:
    from .deduplication import DatasetDeduplicator
except ImportError as e:
    logger.warning(f"Failed to import DatasetDeduplicator: {e}")
    _import_errors.append("DatasetDeduplicator")
    DatasetDeduplicator = None  # type: ignore

try:
    from .unified_search import UnifiedSearchClient
except ImportError as e:
    logger.warning(f"Failed to import UnifiedSearchClient: {e}")
    _import_errors.append("UnifiedSearchClient")
    UnifiedSearchClient = None  # type: ignore


def create_all_clients(enable_cache: bool = True) -> dict[str, Any]:
    """
    Create instances of all available API clients.
    
    Args:
        enable_cache: Whether to enable request caching for all clients
    
    Returns:
        Dictionary mapping client names to instances
    
    Raises:
        ImportError: If required clients could not be imported
    """
    if _import_errors:
        raise ImportError(
            f"Cannot create clients due to import errors: {', '.join(_import_errors)}"
        )
    
    return {
        "pubmed": PubMedClient(enable_cache=enable_cache),
        "doaj": DOAJClient(enable_cache=enable_cache),
        "dryad": DryadClient(enable_cache=enable_cache),
        "zenodo": ZenodoClient(enable_cache=enable_cache),
        "clinicaltrials": ClinicalTrialsClient(enable_cache=enable_cache),
    }


def create_unified_client(enable_cache: bool = True) -> Any:
    """
    Create a unified search client that coordinates all API clients.
    
    Args:
        enable_cache: Whether to enable request caching for all clients
    
    Returns:
        UnifiedSearchClient instance
    
    Raises:
        ImportError: If UnifiedSearchClient could not be imported
    """
    if UnifiedSearchClient is None:
        raise ImportError("UnifiedSearchClient could not be imported")
    
    return UnifiedSearchClient(enable_cache=enable_cache)


def create_deduplicator(similarity_threshold: float = 0.8) -> Any:
    """
    Create a configured deduplicator instance.
    
    Args:
        similarity_threshold: Minimum similarity for title matching (0-1)
    
    Returns:
        Configured DatasetDeduplicator instance
    
    Raises:
        ImportError: If DatasetDeduplicator could not be imported
    """
    if DatasetDeduplicator is None:
        raise ImportError("DatasetDeduplicator could not be imported")
    
    return DatasetDeduplicator(title_similarity_threshold=similarity_threshold)


def create_metadata_parser() -> Any:
    """
    Create a metadata parser instance.
    
    Returns:
        MetadataParser instance
    
    Raises:
        ImportError: If MetadataParser could not be imported
    """
    if MetadataParser is None:
        raise ImportError("MetadataParser could not be imported")
    
    return MetadataParser()


def get_available_clients() -> list[str]:
    """
    Get list of successfully imported client names.
    
    Returns:
        List of available client names
    """
    available = []
    
    if PubMedClient is not None:
        available.append("PubMedClient")
    if DOAJClient is not None:
        available.append("DOAJClient")
    if DryadClient is not None:
        available.append("DryadClient")
    if ZenodoClient is not None:
        available.append("ZenodoClient")
    if ClinicalTrialsClient is not None:
        available.append("ClinicalTrialsClient")
    if UnifiedSearchClient is not None:
        available.append("UnifiedSearchClient")
    
    return available


def get_import_errors() -> list[str]:
    """
    Get list of components that failed to import.
    
    Returns:
        List of component names with import errors
    """
    return _import_errors.copy()


__all__ = [
    # Version
    "__version__",
    # Base classes and exceptions
    "BaseAPIClient",
    "APIError",
    "RateLimitError",
    # API clients
    "PubMedClient",
    "DOAJClient",
    "DryadClient",
    "ZenodoClient",
    "ClinicalTrialsClient",
    "UnifiedSearchClient",
    # Utilities
    "MetadataParser",
    "DatasetDeduplicator",
    # Factory functions
    "create_all_clients",
    "create_unified_client",
    "create_deduplicator",
    "create_metadata_parser",
    # Introspection
    "get_available_clients",
    "get_import_errors",
]
