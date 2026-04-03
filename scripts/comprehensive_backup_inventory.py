#!/usr/bin/env python3
"""
Comprehensive inventory of actual datasets in DigitalOcean Spaces backup.
Creates a new registry based on actual backup structure.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List


def run_rclone(command: str) -> str:
    """Run rclone command and return output."""
    result = subprocess.run(
        f"rclone {command}", shell=True, capture_output=True, text=True
    )
    return result.stdout


def get_size_mb(size_bytes: int) -> float:
    """Convert bytes to MB."""
    return round(size_bytes / 1024 / 1024, 2)


def get_size_gb(size_bytes: int) -> float:
    """Convert bytes to GB."""
    return round(size_bytes / 1024 / 1024 / 1024, 3)


def inventory_directory(remote_path: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Inventory a directory recursively.

    Returns:
        Dict with 'files' and 'subdirs' keys
    """
    result = {"files": [], "subdirs": {}, "total_size": 0, "file_count": 0}

    # List files in current directory
    files_output = run_rclone(f"ls {remote_path}").strip()
    if files_output:
        for line in files_output.split("\n"):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                size, filename = parts
                size_int = int(size)
                result["files"].append(
                    {
                        "name": filename,
                        "size_bytes": size_int,
                        "size_mb": get_size_mb(size_int),
                    }
                )
                result["total_size"] += size_int
                result["file_count"] += 1

    # Recursively inventory subdirectories if we haven't reached max depth
    if max_depth > 0:
        dirs_output = run_rclone(f"lsf {remote_path} --dirs-only").strip()
        if dirs_output:
            for dirname in dirs_output.split("\n"):
                if not dirname.strip():
                    continue
                subdir_path = f"{remote_path}/{dirname.rstrip('/')}"
                subdir_inventory = inventory_directory(subdir_path, max_depth - 1)
                result["subdirs"][dirname.rstrip("/")] = subdir_inventory
                result["total_size"] += subdir_inventory["total_size"]
                result["file_count"] += subdir_inventory["file_count"]

    return result


def build_registry_entry(
    name: str, path: str, inventory: Dict[str, Any], category: str = "unknown"
) -> Dict[str, Any]:
    """Build a registry entry for a dataset."""
    return {
        "name": name,
        "path": f"s3://pixel-data/{path}",
        "category": category,
        "size": {
            "bytes": inventory["total_size"],
            "mb": get_size_mb(inventory["total_size"]),
            "gb": get_size_gb(inventory["total_size"]),
        },
        "file_count": inventory["file_count"],
        "files": inventory["files"][:10],  # Keep first 10 files as samples
        "quality_metrics": {
            "completeness_score": 0,
            "consistency_score": 0,
            "annotation_quality_score": 0,
            "overall_quality_score": 0,
            "quality_tier": "not_scored",
            "duplicate_count": 0,
            "deduplication_ratio": 0,
        },
        "validation": {
            "checksum_sha256": None,
            "checksum_md5": None,
            "schema_valid": False,
            "integrity_valid": False,
            "last_validated": None,
        },
        "usage": {
            "access_count": 0,
            "last_accessed": None,
            "training_jobs": [],
            "data_freshness_days": None,
        },
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE BACKUP INVENTORY & REGISTRY REBUILD")
    print("=" * 80)
    print()

    base_remote = "BackupStorageS3:pixel-data"

    # Top-level directories to inventory
    top_dirs = [
        ("acquired", "raw_acquired"),
        ("compiled_dataset", "training_shards"),
        ("processed_ready", "processed"),
        ("final_dataset", "final"),
        ("datasets", "organized"),
        ("archive", "archived"),
    ]

    full_inventory = {}
    all_entries = []

    for dir_name, category in top_dirs:
        print(f"\n{'=' * 80}")
        print(f"INVENTORYING: {dir_name}/")
        print(f"{'=' * 80}")

        remote_path = f"{base_remote}/{dir_name}"

        # Check if directory exists
        check = run_rclone(f"lsd {remote_path} 2>&1")
        if "directory not found" in check.lower() or "error" in check.lower():
            print(f"  ⚠️  Directory not found or empty: {dir_name}")
            continue

        inventory = inventory_directory(remote_path, max_depth=2)
        full_inventory[dir_name] = inventory

        print(f"  Files: {inventory['file_count']}")
        print(f"  Size: {get_size_gb(inventory['total_size'])} GiB")

        # Build registry entries for subdirectories
        for subdir_name, subdir_inventory in inventory.get("subdirs", {}).items():
            if subdir_inventory["file_count"] > 0:
                entry = build_registry_entry(
                    name=f"{dir_name}_{subdir_name}".replace("/", "_"),
                    path=f"{dir_name}/{subdir_name}",
                    inventory=subdir_inventory,
                    category=category,
                )
                all_entries.append(entry)
                print(
                    f"    - {subdir_name}: {subdir_inventory['file_count']} files, {get_size_gb(subdir_inventory['total_size'])} GiB"
                )

        # Also add entry for top-level files in this directory
        if inventory["file_count"] > 0 and not inventory.get("subdirs"):
            entry = build_registry_entry(
                name=f"{dir_name}_root",
                path=dir_name,
                inventory=inventory,
                category=category,
            )
            all_entries.append(entry)

    # Build new registry structure
    new_registry = {
        "version": "2.0.0",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "description": "Rebuilt registry based on actual backup structure",
        "storage": {
            "primary": {
                "type": "s3",
                "bucket": "pixel-data",
                "endpoint": "${AWS_S3_ENDPOINT}",
                "region": "sfo3",
            }
        },
        "datasets": {},
        "statistics": {
            "total_datasets": len(all_entries),
            "total_size_bytes": sum(e["size"]["bytes"] for e in all_entries),
            "total_size_gb": get_size_gb(sum(e["size"]["bytes"] for e in all_entries)),
            "total_files": sum(e["file_count"] for e in all_entries),
        },
    }

    # Organize entries by category
    for entry in all_entries:
        category = entry.pop("category")
        if category not in new_registry["datasets"]:
            new_registry["datasets"][category] = {}
        new_registry["datasets"][category][entry["name"]] = entry

    # Save inventory
    inventory_path = Path(
        "/home/vivi/pixelated/ai/config/comprehensive_backup_inventory.json"
    )
    with open(inventory_path, "w") as f:
        json.dump(full_inventory, f, indent=2)
    print(f"\n✅ Full inventory saved to: {inventory_path}")

    # Save new registry
    registry_path = Path("/home/vivi/pixelated/ai/config/dataset_registry_v2.json")
    with open(registry_path, "w") as f:
        json.dump(new_registry, f, indent=2, ensure_ascii=False)
    print(f"✅ New registry saved to: {registry_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total datasets: {new_registry['statistics']['total_datasets']}")
    print(f"Total files: {new_registry['statistics']['total_files']}")
    print(f"Total size: {new_registry['statistics']['total_size_gb']} GiB")
    print()
    print("By category:")
    for category, datasets in new_registry["datasets"].items():
        cat_size = sum(d["size"]["bytes"] for d in datasets.values())
        cat_count = len(datasets)
        print(f"  {category}: {cat_count} datasets, {get_size_gb(cat_size)} GiB")


if __name__ == "__main__":
    main()
