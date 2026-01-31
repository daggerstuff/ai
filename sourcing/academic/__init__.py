"""
Academic Sourcing Engine for Pixelated Empathy AI Training Data Expansion

Unified module for acquiring psychology and therapy books/papers from:
- API sources (ArXiv, Semantic Scholar, CrossRef)
- Publisher integrations (APA, Springer, Oxford, etc.)
- Web scraping fallback (Google Scholar)

This consolidates:
- ai/pipelines/orchestrator/sourcing/academic_sourcing.py
- ai/scripts/acquire_academic_psychology_books.py
- ai/sourcing/academic/ (original OOP design)
"""

from .academic_sourcing import (
    AcademicSourcingEngine,
    BookMetadata,
    SourceType,
    SourcingStrategy,
    create_academic_sourcing_engine,
)
from .publishers.apa_publisher import APAPublisher
from .publishers.cambridge_publisher import CambridgePublisher
from .publishers.elsevier_publisher import ElsevierPublisher
from .publishers.oxford_publisher import OxfordPublisher
from .publishers.springer_publisher import SpringerPublisher
from .publishers.taylor_francis_publisher import TaylorFrancisPublisher
from .publishers.wiley_publisher import WileyPublisher
from .publishers.elsevier_publisher import ElsevierPublisher
from .publishers.base_publisher import BasePublisher, BookContent, BookFormat
from .therapy_dataset_sourcing import (
    ConversationFormat,
    DatasetMetadata,
    DatasetSource,
    TherapyDatasetSourcing,
    find_therapy_datasets,
)

# DOI Resolution
from .doi_resolution.doi_resolver import DOISearcher, DOIResolver

# API Integration
from .api.main import app as AcademicSourcingAPI

# Note: The following modules are planned but not yet implemented in the directory structure:
# from .metadata_extraction.metadata_extractor import MetadataExtractor
# from .anonymization.anonymizer import ContentAnonymizer

__all__ = [
    # Main engine
    "AcademicSourcingEngine",
    "BookMetadata",
    "SourceType",
    "SourcingStrategy",
    "create_academic_sourcing_engine",
    # Therapy dataset sourcing
    "TherapyDatasetSourcing",
    "DatasetMetadata",
    "DatasetSource",
    "ConversationFormat",
    "find_therapy_datasets",
    # Publisher base classes
    "BasePublisher",
    "BookContent",
    "BookFormat",
    "APAPublisher",
    "ElsevierPublisher",
    "SpringerPublisher",
    "WileyPublisher",
    "OxfordPublisher",
    "CambridgePublisher",
    "TaylorFrancisPublisher",
    # DOI Resolution
    "DOISearcher",
    "DOIResolver",
    # API
    "AcademicSourcingAPI",
    # Utility functions
    "get_publisher",
    "get_all_publishers",
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
    else:
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
