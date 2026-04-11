#!/usr/bin/env python3
"""
Inventory actual dataset files in Hetzner Object Storage backup
and create a clean registry based on reality.
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path


def run_rclone(command: str) -> str:
    """Run rclone command and return output."""
    result = subprocess.run(
        f"rclone {command}", shell=True, capture_output=True, text=True
    )
    return result.stdout


def main():
    print("=== INVENTORYING ACTUAL DATASETS IN BACKUP ===\n")

    inventory = defaultdict(lambda: defaultdict(list))

    for training_dir in ["training_v2", "training_v3"]:
        base_path = f"BackupStorageS3:pixel-data/datasets/{training_dir}"
        dirs = run_rclone(f"lsf {base_path} --dirs-only").strip().split("\n")

        for stage_dir in dirs:
            stage_path = f"{base_path}/{stage_dir}"
            files = run_rclone(f"ls {stage_path}").strip().split("\n")

            for file_line in files:
                if not file_line.strip():
                    continue
                parts = file_line.split(maxsplit=1)
                if len(parts) == 2:
                    size, filename = parts
                    inventory[training_dir][stage_dir].append(
                        {
                            "filename": filename,
                            "size_bytes": int(size),
                            "size_mb": round(int(size) / 1024 / 1024, 2),
                        }
                    )

    inventory_path = Path("/home/vivi/pixelated/ai/config/backup_inventory.json")
    with open(inventory_path, "w") as f:
        json.dump(dict(inventory), f, indent=2)

    print(f"Inventory saved to {inventory_path}\n")

    total_files = sum(
        len(files) for stages in inventory.values() for files in stages.values()
    )
    total_size = sum(
        f["size_bytes"]
        for stages in inventory.values()
        for files in stages.values()
        for f in files
    )

    print(f"Total datasets found: {total_files}")
    print(f"Total size: {round(total_size / 1024**3, 2)} GiB\n")

    for training_dir, stages in sorted(inventory.items()):
        print(f"\n{training_dir}:")
        for stage, files in sorted(stages.items()):
            print(f"  {stage}: {len(files)} files")
            for f in files[:5]:  # Show first 5
                print(f"    - {f['filename']} ({f['size_mb']} MB)")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()
