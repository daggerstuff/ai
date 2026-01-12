"""
Academic Sourcing Engine for Pixelated Empathy AI Training Data Expansion

Unified module for acquiring psychology and therapy books/papers from:
- API sources (ArXiv, Semantic Scholar, CrossRef)
- Publisher integrations (APA, Springer, Oxford, etc.)
- Web scraping fallback (Google Scholar)

This consolidates:
- ai/dataset_pipeline/sourcing/academic_sourcing.py
- ai/scripts/acquire_academic_psychology_books.py
- ai/academic_sourcing/ (original OOP design)
"""

from .academic_sourcing import (
    AcademicSourcingEngine,
    BookMetadata,
    SourceType,
    SourcingStrategy,
    create_academic_sourcing_engine,
)
from .publishers.apa_publisher import APAPublisher
from .publishers.base_publisher import BasePublisher, BookContent, BookFormat
from .therapy_dataset_sourcing import (
    ConversationFormat,
    DatasetMetadata,
    DatasetSource,
    TherapyDatasetSourcing,
    find_therapy_datasets,
)

# TODO: Import these as they are implemented
# from .publishers.elsevier_publisher import ElsevierPublisher
# from .publishers.springer_publisher import SpringerPublisher
# from .publishers.wiley_publisher import WileyPublisher
# from .publishers.oup_publisher import OUPPublisher
# from .publishers.cambridge_publisher import CambridgePublisher
# from .publishers.taylor_francis_publisher import TaylorFrancisPublisher

# TODO: Import these modules as they are implemented
# from .doi_resolution.doi_resolver import DOISearcher, DOIResolver
# from .metadata_extraction.metadata_extractor import MetadataExtractor
# from .anonymization.anonymizer import ContentAnonymizer
# from .api.academic_sourcing_api import AcademicSourcingAPI

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
    # Utility functions
    "get_publisher",
    "get_all_publishers",
]


def get_publisher(publisher_name: str) -> BasePublisher:
    """Get a publisher instance by name"""
    publisher_map = {
        "apa": APAPublisher(),
        # TODO: Add others as implemented
        # 'elsevier': ElsevierPublisher(),
        # 'springer': SpringerPublisher(),
        # 'wiley': WileyPublisher(),
        # 'oup': OUPPublisher(),
        # 'cambridge': CambridgePublisher(),
        # 'taylor_francis': TaylorFrancisPublisher()
    }
    publisher = publisher_map.get(publisher_name.lower())
    if not publisher:
        raise ValueError(f"Publisher '{publisher_name}' not implemented yet")
    return publisher


def get_all_publishers() -> list[BasePublisher]:
    """Get all available publisher instances"""
    return [
        APAPublisher(),
        # TODO: Add others as implemented
    ]
