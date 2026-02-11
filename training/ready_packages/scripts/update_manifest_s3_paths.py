#!/usr/bin/env python3
"""
Update Training Manifest with S3 Paths
Maps local/Google Drive paths to S3 canonical paths
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
# Script is at: ai/training_ready/scripts/update_manifest_s3_paths.py
# Project root is: /home/vivi/pixelated/
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # Go up 3 levels: scripts -> training_ready -> ai -> project_root
sys.path.insert(0, str(project_root))

from ai.training.ready_packages.utils.s3_dataset_loader import get_s3_dataset_path, S3DatasetLoader


def update_dataset_registry_with_s3_paths(registry_path: Path) -> Dict[str, Any]:
    """
    Update dataset_registry.json with S3 canonical paths.

    Args:
        registry_path: Path to dataset_registry.json

    Returns:
        Updated registry data
    """
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    loader = S3DatasetLoader()
    updated = False

    # Map categories to S3 structure
    category_mapping = {
        "cot_reasoning": "cot_reasoning",
        "professional_therapeutic": "professional_therapeutic",
        "priority_datasets": "priority",
        "edge_case_sources": "edge_cases"
    }

    def update_paths_in_section(section: Dict[str, Any], category: str):
        """Recursively update paths in a section"""
        nonlocal updated
        for key, dataset_info in section.items():
            if isinstance(dataset_info, dict):
                if 'path' in dataset_info:
                    old_path = dataset_info['path']

                    # Skip if already S3 path
                    if old_path.startswith('s3://'):
                        continue

                    # Extract dataset name from path
                    dataset_name = Path(old_path).name

                    # Get S3 canonical path
                    s3_path = get_s3_dataset_path(
                        dataset_name,
                        category=category,
                        prefer_processed=True
                    )

                    # Update registry
                    dataset_info['s3_path'] = s3_path
                    dataset_info['legacy_path'] = old_path
                    updated = True

                    print(f"  {key}:")
                    print(f"    Legacy: {old_path}")
                    print(f"    S3:     {s3_path}")

                # Recurse into nested structures
                update_paths_in_section(dataset_info, category)

    # Update each category
    if 'datasets' in registry:
        for category, datasets in registry['datasets'].items():
            s3_category = category_mapping.get(category, category)
            print(f"\nUpdating {category} -> {s3_category}:")
            update_paths_in_section(datasets, s3_category)

    # Update edge case sources
    if 'edge_case_sources' in registry:
        print(f"\nUpdating edge_case_sources -> edge_cases:")
        update_paths_in_section(registry['edge_case_sources'], 'edge_cases')

    # Add S3 metadata
    registry['s3_consolidation'] = {
        "status": "in_progress",
        "canonical_bucket": "pixel-data",
        "canonical_structure": "gdrive/processed/",
        "raw_backup": "gdrive/raw/",
        "last_updated": str(Path(__file__).stat().st_mtime)
    }

    return registry, updated


def main():
    """Update dataset registry with S3 paths"""
    registry_path = project_root / "ai" / "data" / "dataset_registry.json"

    if not registry_path.exists():
        print(f"Error: Registry not found at {registry_path}")
        return 1

    print("Updating dataset registry with S3 canonical paths...")
    print("=" * 60)

    try:
        registry, updated = update_dataset_registry_with_s3_paths(registry_path)

        if updated:
            # Backup original
            backup_path = registry_path.with_suffix('.json.backup')
            registry_path.rename(backup_path)
            print(f"\n✅ Backup created: {backup_path}")

            # Write updated registry
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=4)

            print(f"✅ Registry updated: {registry_path}")
            print("\nS3 paths are now canonical. Legacy paths preserved for reference.")
        else:
            print("\n✅ Registry already has S3 paths or no updates needed.")

        return 0

    except Exception as e:
        print(f"\n❌ Error updating registry: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
