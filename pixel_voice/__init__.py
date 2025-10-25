"""
Pixel Voice Pipeline - AI-powered voice analysis and processing pipeline.

This package provides tools for voice data processing, quality control,
feature extraction, and therapeutic conversation analysis.
"""

__version__ = "1.0.0"

# Import main modules for easier access
from . import feature_extraction
from . import audio_quality_control

__all__ = [
    "feature_extraction",
    "audio_quality_control",
]