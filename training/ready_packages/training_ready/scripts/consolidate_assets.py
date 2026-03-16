#!/usr/bin/env python3
"""
Asset Consolidation Script

Consolidates all training assets into ai/training_ready/ structure.
Uses symlinks for large files to avoid duplication.
"""

import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Size threshold for using symlinks (100 MB)
SYMLINK_THRESHOLD = 100 * 1024 * 1024


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load training manifest."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def consolidate_configs(manifest: Dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate training configs."""
    configs = manifest.get("training_configurations", [])
    consolidated = 0

    print(f"📋 Consolidating {len(configs)} config files...")

    for config in configs:
        source_path = Path(config.get("path", ""))
        if not source_path.exists():
            continue

        config_type = config.get("type", "other")
        stage = config.get("stage", "all")

        # Determine target subdirectory
        if config_type == "stage":
            target_subdir = target_dir / "stage_configs"
        elif config_type == "model":
            target_subdir = target_dir / "model_configs"
        elif config_type == "hyperparameter":
            target_subdir = target_dir / "hyperparameters"
        elif config_type == "infrastructure":
            target_subdir = target_dir / "infrastructure"
        else:
            target_subdir = target_dir / "stage_configs"  # Default

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        # Handle duplicates
        if target_file.exists():
            # Add directory prefix to avoid conflicts
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} config files")
    return consolidated


def consolidate_datasets(manifest: Dict, target_dir: Path, base_path: Path, use_symlinks: bool = True) -> int:
    """Consolidate datasets with symlink strategy for large files."""
    datasets = manifest.get("datasets", [])
    consolidated = 0
    symlinked = 0

    print(f"📊 Consolidating {len(datasets)} datasets...")

    for dataset in datasets:
        source_path = Path(dataset.get("path", ""))
        if not source_path.exists():
            continue

        stage = dataset.get("stage", "unassigned")
        if stage == "unassigned":
            stage = "stage1_foundation"  # Default

        target_subdir = target_dir / stage
        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        # Handle duplicates
        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        file_size = dataset.get("size", 0)
        use_symlink = use_symlinks and file_size > SYMLINK_THRESHOLD

        try:
            if use_symlink:
                # Create symlink
                if target_file.exists():
                    target_file.unlink()
                target_file.symlink_to(source_path.resolve())
                symlinked += 1
            else:
                # Copy file
                shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to {'symlink' if use_symlink else 'copy'} {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} datasets ({symlinked} symlinked)")
    return consolidated


def consolidate_models(manifest: Dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate model architectures."""
    models = manifest.get("model_architectures", [])
    consolidated = 0

    print(f"🤖 Consolidating {len(models)} model architectures...")

    for model in models:
        source_path = Path(model.get("path", ""))
        if not source_path.exists():
            continue

        model_type = model.get("type", "other")
        if model_type == "moe":
            target_subdir = target_dir / "moe"
        elif model_type == "experimental":
            target_subdir = target_dir / "experimental"
        elif model_type == "base":
            target_subdir = target_dir / "base"
        else:
            target_subdir = target_dir / "base"  # Default

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        # Handle duplicates
        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} model files")
    return consolidated


def consolidate_pipelines(manifest: Dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate pipeline components."""
    pipelines = manifest.get("pipelines", [])
    consolidated = 0

    print(f"🔧 Consolidating {len(pipelines)} pipeline components...")

    for pipeline in pipelines:
        source_path = Path(pipeline.get("path", ""))
        if not source_path.exists():
            continue

        pipeline_type = pipeline.get("type", "integrated")
        if pipeline_type == "edge":
            target_subdir = target_dir / "edge"
        elif pipeline_type == "voice":
            target_subdir = target_dir / "voice"
        else:
            target_subdir = target_dir / "integrated"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        # Handle duplicates
        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} pipeline files")
    return consolidated


def consolidate_infrastructure(manifest: Dict, target_dir: Path, base_path: Path) -> int:
    """Consolidate infrastructure configs."""
    infrastructure = manifest.get("infrastructure", [])
    consolidated = 0

    print(f"🏗️  Consolidating {len(infrastructure)} infrastructure configs...")

    for infra in infrastructure:
        source_path = Path(infra.get("path", ""))
        if not source_path.exists():
            continue

        infra_type = infra.get("type", "kubernetes")
        if infra_type == "helm":
            target_subdir = target_dir / "helm"
        elif infra_type == "docker":
            target_subdir = target_dir / "docker"
        else:
            target_subdir = target_dir / "kubernetes"

        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / source_path.name

        # Handle duplicates
        if target_file.exists():
            dir_prefix = source_path.parent.name
            target_file = target_subdir / f"{dir_prefix}_{source_path.name}"

        try:
            shutil.copy2(source_path, target_file)
            consolidated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {source_path.name}: {e}")

    print(f"  ✅ Consolidated {consolidated} infrastructure files")
    return consolidated


def main():
    """Main consolidation function."""
    base_path = Path.cwd()
    training_ready = base_path / "ai" / "training_ready"
    manifest_path = training_ready / "TRAINING_MANIFEST.json"

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        print("Please run generate_manifest.py first")
        return 1

    print("🚀 Starting asset consolidation...\n")

    manifest = load_manifest(manifest_path)

    # Consolidate configs
    configs_dir = training_ready / "configs"
    consolidate_configs(manifest, configs_dir, base_path)

    # Consolidate datasets
    datasets_dir = training_ready / "datasets"
    consolidate_datasets(manifest, datasets_dir, base_path, use_symlinks=True)

    # Consolidate models
    models_dir = training_ready / "models"
    consolidate_models(manifest, models_dir, base_path)

    # Consolidate pipelines
    pipelines_dir = training_ready / "pipelines"
    consolidate_pipelines(manifest, pipelines_dir, base_path)

    # Consolidate infrastructure
    infrastructure_dir = training_ready / "infrastructure"
    consolidate_infrastructure(manifest, infrastructure_dir, base_path)

    print("\n✅ Asset consolidation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

