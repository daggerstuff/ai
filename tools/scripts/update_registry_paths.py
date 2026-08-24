#!/usr/bin/env python3
"""
Update dataset registry to match actual backup paths in Hetzner Object Storage.
Maps old stage-based paths to actual backup structure.
"""

import json
from pathlib import Path
from typing import Any

OLD_TO_NEW_PATHS = {
    "training/v1/stage1_foundation": "datasets/training_v3/stage1_foundation",
    "training/v1/stage2_expertise": "datasets/training_v3/stage2_specialist_addiction",
    "training/v1/stage3_edge": "datasets/training_v2/stage3_edge_crisis",
    "training/v1/stage4_persona/voice": "datasets/training_v3/stage4_voice_persona",
    "training/v1/stage5_rl": "datasets/training_v3/stage5_rl_alignment",
}


def update_path(path: str) -> str:
    """Update old S3 path to new backup structure."""
    if not path.startswith("s3://pixel-data/"):
        return path

    relative_path = path.replace("s3://pixel-data/", "")

    for old_prefix, new_prefix in OLD_TO_NEW_PATHS.items():
        if relative_path.startswith(old_prefix):
            return f"s3://pixel-data/{relative_path.replace(old_prefix, new_prefix, 1)}"

    return path


def update_dataset_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Update a single dataset entry."""
    if "path" in entry:
        entry["path"] = update_path(entry["path"])

    if "fallback_paths" in entry:
        for key, path in entry["fallback_paths"].items():
            if isinstance(path, str):
                entry["fallback_paths"][key] = update_path(path)

    return entry


def main():
    registry_path = Path("/home/vivi/pixelated/ai/configs/dataset_registry.json")

    with open(registry_path) as f:
        registry = json.load(f)

    updated_count = 0

    if "datasets" in registry:
        for category_name, category_data in registry["datasets"].items():
            if isinstance(category_data, dict):
                for dataset_name, dataset_entry in category_data.items():
                    if isinstance(dataset_entry, dict):
                        registry["datasets"][category_name][dataset_name] = update_dataset_entry(dataset_entry)
                        updated_count += 1

    for section in [
        "rlhf_alignment",
        "emotion_recognition",
        "advanced_reasoning",
        "embeddings",
        "edge_case_sources",
        "voice_persona",
        "supplementary",
    ]:
        if section in registry:
            for dataset_name, dataset_entry in registry[section].items():
                if isinstance(dataset_entry, dict):
                    registry[section][dataset_name] = update_dataset_entry(dataset_entry)
                    updated_count += 1

    registry["last_updated"] = "2026-04-03T16:15:00.000000Z"

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
