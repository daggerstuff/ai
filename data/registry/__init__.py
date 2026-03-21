"""
Dataset Registry - Lazy-loading registry for training datasets.

This module provides lazy-loading access to dataset metadata without loading
the entire registry into memory. Supports queries by stage, quality profile,
and status.
"""

from .lazy_registry import DatasetRegistry, DatasetRef
from .gap_tracker import DatasetGapTracker
from .sources import DatasetSourceManager, JournalSource, EdgeCaseSource, VoiceSource

__all__ = [
    'DatasetRegistry',
    'DatasetRef',
    'DatasetGapTracker',
    'DatasetSourceManager',
    'JournalSource',
    'EdgeCaseSource',
    'VoiceSource',
]
