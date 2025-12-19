"""
Academic Sourcing Engine for Pixelated Empathy AI Training Data Expansion

This module provides the infrastructure for acquiring psychology and therapy books from academic publishers,
including DOI resolution, metadata extraction, content processing, and HIPAA-compliant anonymization.
"""

from .publishers.base_publisher import BasePublisher
from .publishers.apa_publisher import APAPublisher
from .publishers.elsevier_publisher import ElsevierPublisher
from .publishers.springer_publisher import SpringerPublisher
from .publishers.wiley_publisher import WileyPublisher
from .publishers.oup_publisher import OUPPublisher
from .publishers.cambridge_publisher import CambridgePublisher
from .publishers.taylor_francis_publisher import TaylorFrancisPublisher

from .doi_resolution.doi_resolver import DOISearcher, DOIResolver
from .metadata_extraction.metadata_extractor import MetadataExtractor
from .anonymization.anonymizer import ContentAnonymizer
from .api.academic_sourcing_api import AcademicSourcingAPI

__all__ = [
    'BasePublisher',
    'APAPublisher',
    'ElsevierPublisher',
    'SpringerPublisher',
    'WileyPublisher',
    'OUPPublisher',
    'CambridgePublisher',
    'TaylorFrancisPublisher',
    'DOISearcher',
    'DOIResolver',
    'MetadataExtractor',
    'ContentAnonymizer',
    'AcademicSourcingAPI',
    'get_publisher',
    'get_all_publishers'
]

def get_publisher(publisher_name: str) -> BasePublisher:
    """Get a publisher instance by name"""
    publisher_map = {
        'apa': APAPublisher(),
        'elsevier': ElsevierPublisher(),
        'springer': SpringerPublisher(),
        'wiley': WileyPublisher(),
        'oup': OUPPublisher(),
        'cambridge': CambridgePublisher(),
        'taylor_francis': TaylorFrancisPublisher()
    }
    return publisher_map.get(publisher_name.lower())

def get_all_publishers() -> list[BasePublisher]:
    """Get all available publisher instances"""
    return [
        APAPublisher(),
        ElsevierPublisher(),
        SpringerPublisher(),
        WileyPublisher(),
        OUPPublisher(),
        CambridgePublisher(),
        TaylorFrancisPublisher()
    ]