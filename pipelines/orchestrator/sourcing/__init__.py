"""
Data sourcing components for the AI training pipeline.
Handles acquisition of academic resources, research instruments, and therapeutic conversations.
"""

from .ngc_ingestor import NGCIngestor, ingest_ngc_datasets

__all__ = ["NGCIngestor", "ingest_ngc_datasets"]
