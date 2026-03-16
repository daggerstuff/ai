#!/usr/bin/env python3
"""
Directory Exploration Script for Training Consolidation

Systematically explores all AI directories and catalogs training assets:
- Model architectures
- Training configurations
- Datasets
- Pipeline components
- Infrastructure configs
- Experimental/unused features
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib


# Directories to explore
AI_DIRECTORIES = [
    "ai/integration",
    "ai/helm",
    "ai/infrastructure",
    "ai/distributed_processing",
    "ai/data_designer",
    "ai/dataset_pipeline",
    "ai/datasets",
    "ai/demos",
    "ai/integration_pipeline",
    "ai/journal_dataset_research",
    "ai/lightning_training_package",
    "ai/lightning",
    "ai/models",
    "ai/notebooks",
    "ai/pipelines",
    "ai/pixel",
    "ai/pixel_voice",
    "ai/platform",
    "ai/research_system",
    "ai/tools",
    "ai/training_data_consolidated",
]

# File type patterns
CONFIG_PATTERNS = [".json", ".yaml", ".yml", ".toml", "config", "requirements.txt"]
DATASET_PATTERNS = [".jsonl", ".json", ".parquet", ".csv", ".txt", "dataset", "data"]
MODEL_PATTERNS = ["model", "architecture", "moe", "cnn", "resnet", "quantum", "neuroplasticity"]
PIPELINE_PATTERNS = ["pipeline", "orchestrator", "train", "integration"]
INFRASTRUCTURE_PATTERNS = [".yaml", ".yml", "kubernetes", "k8s", "helm", "docker", "deployment"]
TOOL_PATTERNS = ["tool", "utility", "script", "generator", "validator"]
EXPERIMENTAL_KEYWORDS = ["experimental", "research", "novel", "future", "prototype", "test", "demo"]

# Experimental model architectures to identify
EXPERIMENTAL_MODELS = [
    "moe_architecture",
    "emotional_cnn_layer",
    "emotional_resnet_memory",
    "quantum_emotional_states",
    "neuroplasticity_layer",
    "causal_emotional_reasoning",
]


def get_file_type(file_path: Path, file_name: str) -> str:
    """Determine file type based on path and name patterns."""
    file_name_lower = file_name.lower()
    path_str_lower = str(file_path).lower()

    # Check patterns
    if any(pattern in path_str_lower or pattern in file_name_lower for pattern in CONFIG_PATTERNS):
        if any(pattern in path_str_lower for pattern in INFRASTRUCTURE_PATTERNS):
            return "infrastructure"
        return "config"

    if any(pattern in path_str_lower or pattern in file_name_lower for pattern in DATASET_PATTERNS):
        return "dataset"

    if any(pattern in path_str_lower or pattern in file_name_lower for pattern in MODEL_PATTERNS):
        return "model"

    if any(pattern in path_str_lower or pattern in file_name_lower for pattern in PIPELINE_PATTERNS):
        return "pipeline"

    if any(pattern in path_str_lower or pattern in file_name_lower for pattern in TOOL_PATTERNS):
        return "tool"

    if file_path.suffix in [".py", ".ts", ".tsx", ".js", ".jsx"]:
        return "code"

    if file_path.suffix in [".md", ".txt", ".rst"]:
        return "documentation"

    return "other"


def is_experimental(file_path: Path, file_name: str) -> bool:
    """Check if file/directory is experimental based on keywords."""
    path_str_lower = str(file_path).lower()
    file_name_lower = file_name.lower()

    # Check for experimental keywords
    if any(keyword in path_str_lower or keyword in file_name_lower for keyword in EXPERIMENTAL_KEYWORDS):
        return True

    # Check for experimental model names
    if any(model in path_str_lower or model in file_name_lower for model in EXPERIMENTAL_MODELS):
        return True

    # Check if in research directories
    if "research" in path_str_lower or "experimental" in path_str_lower:
        return True

    return False


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information for cataloging."""
    try:
        stat = file_path.stat()
        file_name = file_path.name
        file_type = get_file_type(file_path, file_name)
        experimental = is_experimental(file_path, file_name)

        return {
            "name": file_name,
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(Path.cwd())) if file_path.is_relative_to(Path.cwd()) else str(file_path),
            "type": file_type,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_experimental": experimental,
            "extension": file_path.suffix,
        }
    except (OSError, ValueError) as e:
        return {
            "name": file_path.name,
            "path": str(file_path),
            "error": str(e),
            "type": "error",
        }


def explore_directory(directory_path: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Explore a directory and catalog all files."""
    if base_path is None:
        base_path = Path.cwd()

    full_path = base_path / directory_path

    if not full_path.exists():
        return {
            "directory_path": directory_path,
            "status": "not_found",
            "error": f"Directory does not exist: {full_path}",
        }

    if not full_path.is_dir():
        return {
            "directory_path": directory_path,
            "status": "not_directory",
            "error": f"Path is not a directory: {full_path}",
        }

    catalog = {
        "directory_path": directory_path,
        "absolute_path": str(full_path.resolve()),
        "status": "success",
        "files": [],
        "subdirectories": [],
        "summary": {
            "total_files": 0,
            "total_size": 0,
            "categories": {
                "config": 0,
                "dataset": 0,
                "model": 0,
                "pipeline": 0,
                "infrastructure": 0,
                "tool": 0,
                "code": 0,
                "documentation": 0,
                "other": 0,
            },
            "experimental_count": 0,
        },
    }

    try:
        for root, dirs, files in os.walk(full_path):
            root_path = Path(root)

            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]

            # Catalog subdirectories
            for dir_name in dirs:
                dir_path = root_path / dir_name
                if dir_path.is_dir():
                    catalog["subdirectories"].append(str(dir_path.relative_to(full_path)))

            # Catalog files
            for file_name in files:
                if file_name.startswith('.'):
                    continue

                file_path = root_path / file_name
                file_info = get_file_info(file_path)

                if "error" not in file_info:
                    catalog["files"].append(file_info)
                    catalog["summary"]["total_files"] += 1
                    catalog["summary"]["total_size"] += file_info.get("size", 0)

                    file_type = file_info.get("type", "other")
                    if file_type in catalog["summary"]["categories"]:
                        catalog["summary"]["categories"][file_type] += 1

                    if file_info.get("is_experimental", False):
                        catalog["summary"]["experimental_count"] += 1

        # Sort files by path
        catalog["files"].sort(key=lambda x: x.get("path", ""))
        catalog["subdirectories"].sort()

    except Exception as e:
        catalog["status"] = "error"
        catalog["error"] = str(e)

    return catalog


def identify_experimental_features(catalogs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify experimental/unused features from directory catalogs."""
    experimental_features = []

    for catalog in catalogs:
        if catalog.get("status") != "success":
            continue

        for file_info in catalog.get("files", []):
            if file_info.get("is_experimental", False):
                feature = {
                    "name": file_info.get("name", "unknown"),
                    "path": file_info.get("path", ""),
                    "relative_path": file_info.get("relative_path", ""),
                    "type": file_info.get("type", "unknown"),
                    "directory": catalog.get("directory_path", ""),
                    "size": file_info.get("size", 0),
                    "modified": file_info.get("modified", ""),
                }
                experimental_features.append(feature)

    return experimental_features


def main():
    """Main function to explore all directories and generate catalogs."""
    base_path = Path.cwd()
    output_dir = base_path / "ai" / "training_ready" / "scripts" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Exploring AI directories...")
    print(f"Base path: {base_path}")
    print(f"Output directory: {output_dir}\n")

    all_catalogs = []
    experimental_features = []

    for directory in AI_DIRECTORIES:
        print(f"Exploring: {directory}")
        catalog = explore_directory(directory, base_path)
        all_catalogs.append(catalog)

        if catalog.get("status") == "success":
            print(f"  ✅ Found {catalog['summary']['total_files']} files, "
                  f"{catalog['summary']['experimental_count']} experimental")
        else:
            print(f"  ⚠️  {catalog.get('status', 'unknown')}: {catalog.get('error', '')}")

    # Identify experimental features
    print("\n🔬 Identifying experimental features...")
    experimental_features = identify_experimental_features(all_catalogs)
    print(f"Found {len(experimental_features)} experimental features")

    # Save catalogs
    catalogs_file = output_dir / "directory_catalogs.json"
    with open(catalogs_file, "w") as f:
        json.dump(all_catalogs, f, indent=2)
    print(f"\n💾 Saved directory catalogs to: {catalogs_file}")

    # Save experimental features
    experimental_file = output_dir / "experimental_features.json"
    with open(experimental_file, "w") as f:
        json.dump(experimental_features, f, indent=2)
    print(f"💾 Saved experimental features to: {experimental_file}")

    # Print summary
    total_files = sum(c.get("summary", {}).get("total_files", 0) for c in all_catalogs)
    total_size = sum(c.get("summary", {}).get("total_size", 0) for c in all_catalogs)
    total_experimental = len(experimental_features)

    print("\n📊 Summary:")
    print(f"  Total directories explored: {len(AI_DIRECTORIES)}")
    print(f"  Total files cataloged: {total_files}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Experimental features: {total_experimental}")

    return all_catalogs, experimental_features


if __name__ == "__main__":
    catalogs, features = main()

