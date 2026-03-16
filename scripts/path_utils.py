#!/usr/bin/env python3
"""
Path Utilities for AI Scripts
Provides dynamic workspace root detection to avoid hardcoded paths.
"""

from pathlib import Path
import os


def get_workspace_root() -> Path:
    """
    Dynamically find the workspace root directory.
    
    Searches upward from the current file location until finding
    the workspace root. Handles git submodules by continuing to search
    for the parent repository.
    
    Returns:
        Path: Absolute path to workspace root
    """
    # Start from current file's directory
    current = Path(__file__).resolve().parent
    
    # Search upward for .git directory
    git_roots = []
    search_path = current
    while search_path != search_path.parent:
        if (search_path / ".git").exists():
            git_roots.append(search_path)
        search_path = search_path.parent
    
    # If we found multiple .git directories, use the topmost one (parent repo)
    # This handles the case where ai/ is a git submodule
    if git_roots:
        return git_roots[-1]
    
    # Fallback: assume we're in ai/scripts and go up two levels
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_workspace_root() / "ai" / "data"


def get_ai_dir() -> Path:
    """Get the ai directory path."""
    return get_workspace_root() / "ai"


def get_unified_training_dir() -> Path:
    """Get the unified training data directory."""
    return get_data_dir() / "unified_training"


def get_lightning_dir() -> Path:
    """Get the Lightning.ai directory."""
    return get_ai_dir() / "lightning"


def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return get_ai_dir() / "scripts"


# For backward compatibility and convenience
WORKSPACE_ROOT = get_workspace_root()
DATA_DIR = get_data_dir()
AI_DIR = get_ai_dir()
UNIFIED_TRAINING_DIR = get_unified_training_dir()
LIGHTNING_DIR = get_lightning_dir()
SCRIPTS_DIR = get_scripts_dir()
