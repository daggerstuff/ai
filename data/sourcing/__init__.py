"""
Active Dataset Sourcing.

This module provides pipelines for discovering and generating NEW datasets:
- HuggingFace discovery and download
- Journal research ingestion (DOAJ, ClinicalTrials)
- Edge case generation
- Voice pipeline integration
- Prompt corpus extraction from knowledge base
"""

from .huggingface_source import HuggingFaceSource
from .journal_source import JournalSource
from .edge_case_source import EdgeCaseSource
from .voice_source import VoiceSource
from .prompt_corpus_source import PromptCorpusSource

__all__ = [
    'HuggingFaceSource',
    'JournalSource',
    'EdgeCaseSource',
    'VoiceSource',
    'PromptCorpusSource',
]
