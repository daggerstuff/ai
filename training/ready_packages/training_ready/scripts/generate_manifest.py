#!/usr/bin/env python3
"""
Training Manifest Generator

Generates comprehensive TRAINING_MANIFEST.json from directory catalogs.
Maps datasets to 4-stage architecture and documents all training assets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib.util

# Stage mapping patterns
STAGE1_PATTERNS = ["foundation", "rapport", "tier1", "priority_1", "therapist_sft", "soulchat", "counsel_chat", "psych8k", "mental_health_counseling"]
STAGE2_PATTERNS = ["reasoning", "cot", "chain_of_thought", "therapeutic_expertise", "tier3", "professional", "psychology_knowledge"]
STAGE3_PATTERNS = ["edge", "crisis", "nightmare", "stress_test", "suicidality", "psychosis", "trauma", "abuse", "violence"]
STAGE4_PATTERNS = ["voice", "persona", "delivery", "tim_fletcher", "wayfarer", "personality", "tone", "style"]

# Model architecture patterns
MOE_PATTERNS = ["moe", "mixture_of_experts"]
BASE_MODEL_PATTERNS = ["base", "foundation", "pretrained", "harbringer", "mistral"]
EXPERIMENTAL_MODEL_PATTERNS = ["cnn", "resnet", "quantum", "neuroplasticity", "causal", "emotional", "research"]


def map_dataset_to_stage(file_path: str, file_name: str) -> Optional[str]:
    """Map a dataset file to a training stage based on path and name patterns."""
    path_lower = file_path.lower()
    name_lower = file_name.lower()

    # Check Stage 3 (edge cases) first as it's most specific
    if any(pattern in path_lower or pattern in name_lower for pattern in STAGE3_PATTERNS):
        return "stage3_edge"

    # Check Stage 2 (reasoning)
    if any(pattern in path_lower or pattern in name_lower for pattern in STAGE2_PATTERNS):
        return "stage2_reasoning"

    # Check Stage 4 (voice/persona)
    if any(pattern in path_lower or pattern in name_lower for pattern in STAGE4_PATTERNS):
        return "stage4_voice"

    # Default to Stage 1 (foundation)
    if any(pattern in path_lower or pattern in name_lower for pattern in STAGE1_PATTERNS):
        return "stage1_foundation"

    # If no pattern matches, return None (will be assigned later)
    return None


def classify_model_architecture(file_path: str, file_name: str) -> str:
    """Classify model architecture type."""
    path_lower = file_path.lower()
    name_lower = file_name.lower()

    if any(pattern in path_lower or pattern in name_lower for pattern in MOE_PATTERNS):
        return "moe"

    if any(pattern in path_lower or pattern in name_lower for pattern in EXPERIMENTAL_MODEL_PATTERNS):
        return "experimental"

    if any(pattern in path_lower or pattern in name_lower for pattern in BASE_MODEL_PATTERNS):
        return "base"

    return "other"


def get_config_type(file_path: str, file_name: str) -> str:
    """Determine config type."""
    path_lower = file_path.lower()
    name_lower = file_name.lower()

    if "stage" in path_lower or "stage" in name_lower:
        return "stage"
    if "model" in path_lower or "model" in name_lower:
        return "model"
    if "hyperparameter" in path_lower or "hyperparameter" in name_lower or "train" in name_lower:
        return "hyperparameter"
    if any(x in path_lower for x in ["kubernetes", "k8s", "helm", "docker", "deployment"]):
        return "infrastructure"

    return "other"


def determine_stage_for_config(file_path: str) -> Optional[str]:
    """Determine stage for config files."""
    path_lower = file_path.lower()

    if "stage1" in path_lower or "foundation" in path_lower:
        return "stage1"
    if "stage2" in path_lower or "reasoning" in path_lower or "therapeutic_expertise" in path_lower:
        return "stage2"
    if "stage3" in path_lower or "edge" in path_lower:
        return "stage3"
    if "stage4" in path_lower or "voice" in path_lower or "persona" in path_lower:
        return "stage4"

    return "all"


def load_stage_configs():
    """Load stage configurations from dataset_pipeline."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        spec = importlib.util.spec_from_file_location(
            "stages",
            Path.cwd() / "ai" / "dataset_pipeline" / "configs" / "stages.py"
        )
        stages_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stages_module)
        return stages_module
    except Exception as e:
        print(f"Warning: Could not load stage configs: {e}")
        return None


def generate_manifest(catalogs_file: str, experimental_features_file: str, output_file: str):
    """Generate comprehensive training manifest from directory catalogs."""

    # Load catalogs
    with open(catalogs_file, "r") as f:
        catalogs = json.load(f)

    # Load experimental features
    with open(experimental_features_file, "r") as f:
        experimental_features = json.load(f)

    # Load stage configs
    stages_module = load_stage_configs()

    manifest = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "model_architectures": [],
        "training_configurations": [],
        "datasets": [],
        "pipelines": [],
        "infrastructure": [],
        "experimental_features": [],
        "summary": {
            "total_assets": 0,
            "total_size_bytes": 0,
            "by_stage": {
                "stage1_foundation": 0,
                "stage2_reasoning": 0,
                "stage3_edge": 0,
                "stage4_voice": 0,
                "unassigned": 0,
            },
            "by_type": {
                "config": 0,
                "dataset": 0,
                "model": 0,
                "pipeline": 0,
                "infrastructure": 0,
            },
        },
    }

    # Process catalogs
    for catalog in catalogs:
        if catalog.get("status") != "success":
            continue

        directory = catalog.get("directory_path", "")

        for file_info in catalog.get("files", []):
            file_path = file_info.get("path", "")
            relative_path = file_info.get("relative_path", "")
            file_name = file_info.get("name", "")
            file_type = file_info.get("type", "")
            file_size = file_info.get("size", 0)
            is_experimental = file_info.get("is_experimental", False)

            manifest["summary"]["total_assets"] += 1
            manifest["summary"]["total_size_bytes"] += file_size

            # Process datasets
            if file_type == "dataset":
                stage = map_dataset_to_stage(file_path, file_name)
                if not stage:
                    stage = "unassigned"
                    manifest["summary"]["by_stage"]["unassigned"] += 1
                else:
                    manifest["summary"]["by_stage"][stage] += 1

                dataset_entry = {
                    "name": file_name,
                    "path": file_path,
                    "relative_path": relative_path,
                    "stage": stage,
                    "category": "unknown",  # Could be enhanced with more analysis
                    "size": file_size,
                    "format": Path(file_path).suffix[1:] if Path(file_path).suffix else "unknown",
                    "source": directory,
                    "description": f"Dataset from {directory}",
                    "modified": file_info.get("modified", ""),
                }
                manifest["datasets"].append(dataset_entry)
                manifest["summary"]["by_type"]["dataset"] += 1

            # Process model architectures
            elif file_type == "model":
                model_type = classify_model_architecture(file_path, file_name)
                status = "experimental" if is_experimental else "production"

                model_entry = {
                    "name": file_name,
                    "path": file_path,
                    "relative_path": relative_path,
                    "type": model_type,
                    "description": f"Model architecture from {directory}",
                    "dependencies": [],  # Could be enhanced
                    "status": status,
                    "size": file_size,
                    "modified": file_info.get("modified", ""),
                }
                manifest["model_architectures"].append(model_entry)
                manifest["summary"]["by_type"]["model"] += 1

            # Process configs
            elif file_type == "config":
                config_type = get_config_type(file_path, file_name)
                stage = determine_stage_for_config(file_path)

                config_entry = {
                    "name": file_name,
                    "path": file_path,
                    "relative_path": relative_path,
                    "type": config_type,
                    "description": f"Training configuration from {directory}",
                    "stage": stage,
                    "use_case": f"Configuration for {config_type}",
                    "size": file_size,
                    "modified": file_info.get("modified", ""),
                }
                manifest["training_configurations"].append(config_entry)
                manifest["summary"]["by_type"]["config"] += 1

            # Process pipelines
            elif file_type == "pipeline":
                pipeline_type = "integrated"
                if "edge" in file_path.lower():
                    pipeline_type = "edge"
                elif "voice" in file_path.lower():
                    pipeline_type = "voice"
                elif is_experimental:
                    pipeline_type = "experimental"

                pipeline_entry = {
                    "name": file_name,
                    "path": file_path,
                    "relative_path": relative_path,
                    "type": pipeline_type,
                    "description": f"Pipeline component from {directory}",
                    "dependencies": [],
                    "entry_point": file_path if file_name.endswith(".py") else None,
                    "size": file_size,
                    "modified": file_info.get("modified", ""),
                }
                manifest["pipelines"].append(pipeline_entry)
                manifest["summary"]["by_type"]["pipeline"] += 1

            # Process infrastructure
            elif file_type == "infrastructure":
                infra_type = "kubernetes"
                if "helm" in file_path.lower():
                    infra_type = "helm"
                elif "docker" in file_path.lower():
                    infra_type = "docker"

                infra_entry = {
                    "name": file_name,
                    "path": file_path,
                    "relative_path": relative_path,
                    "type": infra_type,
                    "description": f"Infrastructure config from {directory}",
                    "deployment_target": "production",  # Could be enhanced
                    "size": file_size,
                    "modified": file_info.get("modified", ""),
                }
                manifest["infrastructure"].append(infra_entry)
                manifest["summary"]["by_type"]["infrastructure"] += 1

    # Add experimental features
    for feature in experimental_features:
        feature_entry = {
            "name": feature.get("name", ""),
            "path": feature.get("path", ""),
            "relative_path": feature.get("relative_path", ""),
            "type": feature.get("type", "unknown"),
            "description": f"Experimental feature: {feature.get('name', 'unknown')}",
            "potential_value": "High - requires evaluation",  # Could be enhanced
            "integration_complexity": "medium",  # Could be enhanced
            "dependencies": [],
            "size": feature.get("size", 0),
            "modified": feature.get("modified", ""),
        }
        manifest["experimental_features"].append(feature_entry)

    # Add stage information if available
    if stages_module:
        manifest["stage_configuration"] = {
            "stage1_foundation": {
                "id": "stage1_foundation",
                "name": "Stage 1 – Foundation & Rapport",
                "target_share": 0.40,
            },
            "stage2_reasoning": {
                "id": "stage2_therapeutic_expertise",
                "name": "Stage 2 – Therapeutic Expertise & Reasoning",
                "target_share": 0.25,
            },
            "stage3_edge": {
                "id": "stage3_edge_stress_test",
                "name": "Stage 3 – Edge Stress Test & Scenario Bank",
                "target_share": 0.20,
            },
            "stage4_voice": {
                "id": "stage4_voice_persona",
                "name": "Stage 4 – Voice, Persona & Delivery",
                "target_share": 0.15,
            },
        }

    # Save manifest
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Generated training manifest: {output_file}")
    print(f"   Total assets: {manifest['summary']['total_assets']}")
    print(f"   Total size: {manifest['summary']['total_size_bytes'] / (1024**3):.2f} GB")
    print(f"   Datasets: {len(manifest['datasets'])}")
    print(f"   Models: {len(manifest['model_architectures'])}")
    print(f"   Configs: {len(manifest['training_configurations'])}")
    print(f"   Pipelines: {len(manifest['pipelines'])}")
    print(f"   Infrastructure: {len(manifest['infrastructure'])}")
    print(f"   Experimental features: {len(manifest['experimental_features'])}")

    return manifest


def main():
    """Main function."""
    base_path = Path.cwd()
    scripts_dir = base_path / "ai" / "training_ready" / "scripts"
    output_dir = scripts_dir / "output"
    manifest_file = base_path / "ai" / "training_ready" / "TRAINING_MANIFEST.json"

    catalogs_file = output_dir / "directory_catalogs.json"
    experimental_file = output_dir / "experimental_features.json"

    if not catalogs_file.exists():
        print(f"Error: Directory catalogs not found: {catalogs_file}")
        print("Please run explore_directories.py first")
        return 1

    if not experimental_file.exists():
        print(f"Error: Experimental features not found: {experimental_file}")
        print("Please run explore_directories.py first")
        return 1

    print("📋 Generating training manifest...")
    manifest = generate_manifest(str(catalogs_file), str(experimental_file), str(manifest_file))

    print("\n📊 Stage Distribution:")
    for stage, count in manifest["summary"]["by_stage"].items():
        print(f"   {stage}: {count} datasets")

    return 0


if __name__ == "__main__":
    sys.exit(main())

