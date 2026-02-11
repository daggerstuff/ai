"""TechDeck integration package."""

__all__ = []
"""
TechDeck-Python Pipeline Integration Flask API Service.

This package provides a comprehensive Flask API service for integrating TechDeck React frontend
with Python dataset pipeline processing, featuring HIPAA++ compliance, bias detection,
real-time progress tracking, and comprehensive error handling.
"""

__version__ = "1.0.0"
__author__ = "Pixelated Empathy Team"
__description__ = "Flask API service for TechDeck-Python pipeline integration"

from .app import create_app
from .config import Config

__all__ = ['create_app', 'Config']
