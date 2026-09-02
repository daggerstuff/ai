"""
Academic Sourcing Engine for Pixelated Empathy AI Training Data Expansion

Unified module for acquiring psychology and therapy books/papers from:
- API sources (ArXiv, Semantic Scholar, CrossRef)
- Publisher integrations (APA, Springer, Oxford, etc.)
- Web scraping fallback (Google Scholar)

This consolidates:
- ai/pipelines/orchestrator/sourcing/academic_sourcing.py
- ai/scripts/acquire_academic_psychology_books.py
- ai/pipelines/data_processing/academic/ (original OOP design)
"""

from .academic_sourcing import (
    AcademicSourcingEngine,
    BookMetadata,
    SourceType,
    SourcingStrategy,
    create_academic_sourcing_engine,
)

# Note: The following modules are planned but not yet implemented in the directory structure:
from .anonymization.anonymizer import AnonymizationResult, ContentAnonymizer

# API Integration
from .api.main import app as AcademicSourcingAPI

# DOI Resolution
from .doi_resolution.doi_resolver import DOIResolver, DOISearcher
from .metadata_extraction.metadata_extractor import ExtractedMetadata, MetadataExtractor
from .publishers.apa_publisher import APAPublisher
from .publishers.base_publisher import BasePublisher, BookContent, BookFormat
from .publishers.cambridge_publisher import CambridgePublisher
from .publishers.elsevier_publisher import ElsevierPublisher
from .publishers.oxford_publisher import OxfordPublisher
from .publishers.springer_publisher import SpringerPublisher
from .publishers.taylor_francis_publisher import TaylorFrancisPublisher
from .publishers.wiley_publisher import WileyPublisher
from .therapy_dataset_sourcing import (
    ConversationFormat,
    DatasetMetadata,
    DatasetSource,
    TherapyDatasetSourcing,
    find_therapy_datasets,
)

__all__ = [
    "AnonymizationResult",
    "APAPublisher",
    # API
    "AcademicSourcingAPI",
    # Main engine
    "AcademicSourcingEngine",
    # Publisher base classes
    "BasePublisher",
    "BookContent",
    "BookFormat",
    "BookMetadata",
    "CambridgePublisher",
    "ContentAnonymizer",
    "ConversationFormat",
    "DOIResolver",
    # DOI Resolution
    "DOISearcher",
    "DatasetMetadata",
    "DatasetSource",
    "ElsevierPublisher",
    "ExtractedMetadata",
    "MetadataExtractor",
    "OxfordPublisher",
    "SourceType",
    "SourcingStrategy",
    "SpringerPublisher",
    "TaylorFrancisPublisher",
    # Therapy dataset sourcing
    "TherapyDatasetSourcing",
    "WileyPublisher",
    "create_academic_sourcing_engine",
    "find_therapy_datasets",
    "get_all_publishers",
    # Utility functions
    "get_publisher",
]


def get_publisher(publisher_name: str) -> BasePublisher:
    """Get a publisher instance by name"""
    publisher_map = {
        "apa": APAPublisher(),
        "elsevier": ElsevierPublisher(),
        "springer": SpringerPublisher(),
        "wiley": WileyPublisher(),
        "oup": OxfordPublisher(),
        "oxford": OxfordPublisher(),
        "cambridge": CambridgePublisher(),
        "taylor_francis": TaylorFrancisPublisher(),
    }
    if publisher := publisher_map.get(publisher_name.lower()):
        return publisher
    raise ValueError(f"Publisher '{publisher_name}' not implemented yet")


def get_all_publishers() -> list[BasePublisher]:
    """Get all available publisher instances"""
    return [
        APAPublisher(),
        ElsevierPublisher(),
        SpringerPublisher(),
        WileyPublisher(),
        OxfordPublisher(),
        CambridgePublisher(),
        TaylorFrancisPublisher(),
    ]
