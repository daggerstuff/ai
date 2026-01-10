#!/usr/bin/env python3
"""
Create Training Ready Folder Structure

Creates the complete ai/training_ready/ directory structure with all subdirectories.
"""

from pathlib import Path
from typing import List


def create_folder_structure(base_path: Path) -> None:
    """Create complete folder structure for training_ready."""

    directories = [
        # Main structure
        "configs/stage_configs",
        "configs/model_configs",
        "configs/infrastructure",
        "configs/hyperparameters",

        # Datasets by stage
        "datasets/stage1_foundation",
        "datasets/stage2_reasoning",
        "datasets/stage3_edge",
        "datasets/stage4_voice",

        # Models
        "models/moe",
        "models/base",
        "models/experimental",

        # Pipelines
        "pipelines/integrated",
        "pipelines/edge",
        "pipelines/voice",

        # Infrastructure
        "infrastructure/kubernetes",
        "infrastructure/helm",
        "infrastructure/docker",

        # Tools
        "tools/data_preparation",
        "tools/validation",
        "tools/monitoring",

        # Experimental
        "experimental/research_models",
        "experimental/novel_pipelines",
        "experimental/future_features",

        # Scripts output
        "scripts/output",
    ]

    print(f"📁 Creating folder structure in: {base_path}")

    created = []
    existing = []

    for directory in directories:
        full_path = base_path / directory
        if full_path.exists():
            existing.append(directory)
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            created.append(directory)
            # Create .gitkeep to ensure empty directories are tracked
            (full_path / ".gitkeep").touch()

    print(f"\n✅ Created {len(created)} directories")
    if existing:
        print(f"   {len(existing)} directories already existed")

    print("\n📂 Folder structure:")
    print_structure(base_path, base_path, 0)

    return created, existing


def print_structure(base: Path, current: Path, indent: int) -> None:
    """Print folder structure tree."""
    # Get subdirectories (excluding hidden and __pycache__)
    subdirs = sorted([
        d for d in current.iterdir()
        if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__' and d.name != 'scripts'
    ])

    for i, subdir in enumerate(subdirs):
        is_last = i == len(subdirs) - 1
        prefix = "  " * indent
        prefix_char = "└──" if is_last else "├──"
        print(f"{prefix}{prefix_char} {subdir.name}/")
        if subdir.is_dir():
            print_structure(base, subdir, indent + 1)


def main():
    """Main function."""
    base_path = Path.cwd() / "ai" / "training_ready"

    print("🚀 Creating training_ready folder structure...\n")

    created, existing = create_folder_structure(base_path)

    print(f"\n✅ Folder structure created successfully!")
    print(f"   Base path: {base_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

