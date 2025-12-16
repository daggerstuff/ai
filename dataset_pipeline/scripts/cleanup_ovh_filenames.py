#!/usr/bin/env python3
"""
Cleanup OVH S3 filenames to match local repository changes.

This script:
1. Checks what exists on OVH S3 under old paths
2. Renames/moves files to match new local structure
3. Reports changes made

Based on git status showing these moves:
- dataset_pipeline/production_exports/v1.0.0/* → training_ready/configs/stage_configs/*
- dataset_pipeline/final_output/* → training_ready/configs/stage_configs/*
- dataset_pipeline/test_output/* → training_ready/configs/stage_configs/*
- wendy_curated_sets → priority_datasets (in paths)
"""

import os
from pathlib import Path
from typing import Dict, List

import boto3


# Load .env files
def load_env():
    """Load environment variables from .env files."""
    env_files = [
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent / "storage.env",
        Path(__file__).parent.parent.parent.parent / ".env",
    ]

    for env_file in env_files:
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Strip quotes if present
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value


# Configuration - load after env files
def get_config():
    """Get S3 configuration from environment."""
    load_env()  # Load env files first
    bucket = (
        os.getenv("OVH_S3_BUCKET") or os.getenv("DATASET_S3_BUCKET") or "pixel-data"
    )
    # Debug: check what we got
    if bucket == "OVH_S3_REGION" or not bucket or bucket.startswith("OVH_"):
        # Something went wrong - force to pixel-data
        bucket = "pixel-data"
    endpoint_raw = os.getenv("OVH_S3_ENDPOINT", "s3.gra.io.cloud.ovh.net")
    endpoint = endpoint_raw.replace("https://", "").replace("http://", "")
    access_key = os.getenv("OVH_S3_ACCESS_KEY")
    secret_key = os.getenv("OVH_S3_SECRET_KEY")
    return bucket, endpoint, access_key, secret_key


# File rename mappings based on git status
FILE_RENAMES = {
    # production_exports → training_ready/configs/stage_configs
    "dataset_pipeline/production_exports/v1.0.0/config_lock.json": (
        "training_ready/configs/stage_configs/config_lock.json"
    ),
    "dataset_pipeline/production_exports/v1.0.0/dataset.json": (
        "training_ready/configs/stage_configs/dataset.json"
    ),
    "dataset_pipeline/production_exports/export_history.json": (
        "training_ready/configs/stage_configs/export_history.json"
    ),
    # final_output → training_ready/configs/stage_configs
    "dataset_pipeline/final_output/dataset_validation_report.json": (
        "training_ready/configs/stage_configs/dataset_validation_report.json"
    ),
    # test_output → training_ready/configs/stage_configs
    "dataset_pipeline/test_output/test_verify.json": (
        "training_ready/configs/stage_configs/test_verify.json"
    ),
}

# Path prefix changes - actual S3 paths
PATH_PREFIX_CHANGES = {
    "datasets/gdrive/processed/wendy_curated": "datasets/gdrive/processed/priority_datasets",
    "wendy_curated": "priority_datasets",
    "wendy_curated_sets": "priority_datasets",
}


def get_s3_client():
    """Create and return S3 client."""
    bucket, endpoint, access_key, secret_key = get_config()
    if not access_key or not secret_key:
        raise ValueError("OVH_S3_ACCESS_KEY and OVH_S3_SECRET_KEY must be set")

    return boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ), bucket


def check_objects_exist(s3_client, bucket: str, keys: List[str]) -> Dict[str, bool]:
    """Check which objects exist in S3."""
    results = {}
    for key in keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            results[key] = True
        except s3_client.exceptions.NoSuchKey:
            results[key] = False
        except Exception as e:
            print(f"  Error checking {key}: {e}")
            results[key] = False
    return results


def find_objects_with_prefix(s3_client, bucket: str, prefix: str) -> List[str]:
    """Find all objects with a given prefix."""
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects.extend(
                obj["Key"] for obj in page["Contents"] if not obj["Key"].endswith("/")
            )
    return objects


def rename_object(
    s3_client,
    bucket: str,
    old_key: str,
    new_key: str,
    dry_run: bool = True,
) -> bool:
    """Rename an object in S3."""
    if dry_run:
        print(f"  [DRY RUN] Would rename: {old_key} → {new_key}")
        return True

    try:
        # Copy object to new location
        copy_source = {"Bucket": bucket, "Key": old_key}
        s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=new_key)

        # Delete old object
        s3_client.delete_object(Bucket=bucket, Key=old_key)
        print(f"  ✓ Renamed: {old_key} → {new_key}")
        return True
    except Exception as e:
        print(f"  ✗ Error renaming {old_key}: {e}")
        return False


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cleanup OVH S3 filenames to match local changes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the renames (overrides --dry-run)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check what exists, do not rename",
    )
    return parser.parse_args()


def confirm_execute():
    """Confirm potentially destructive execute mode."""
    print("⚠️  EXECUTE MODE - Will actually rename files!")
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return False
    return True


def print_header(bucket_name: str, endpoint: str, mode: str) -> None:
    """Print script header and mode information."""
    print("=" * 70)
    print("OVH S3 Filename Cleanup")
    print("=" * 70)
    print(f"Bucket: {bucket_name}")
    print(f"Endpoint: {endpoint}")
    print(f"Mode: {mode}")
    print("=" * 70)


def collect_rename_operations(s3_client, bucket_name: str) -> List[tuple[str, str]]:
    """Determine which specific file keys need renames."""
    print("\n1. Checking file renames...")
    print("-" * 70)

    old_keys = list(FILE_RENAMES.keys())
    new_keys = list(FILE_RENAMES.values())

    old_exist = check_objects_exist(s3_client, bucket_name, old_keys)
    new_exist = check_objects_exist(s3_client, bucket_name, new_keys)

    rename_operations: List[tuple[str, str]] = []
    for old_key, new_key in FILE_RENAMES.items():
        old_exists = old_exist.get(old_key, False)
        new_exists = new_exist.get(new_key, False)

        if old_exists and not new_exists:
            print(f"  ✓ Found: {old_key} (needs rename)")
            rename_operations.append((old_key, new_key))
        elif old_exists:
            print(f"  ⚠ Both exist: {old_key} and {new_key} (manual review needed)")
        elif new_exists:
            print(f"  ✓ Already renamed: {new_key}")
        else:
            print(f"  - Not found: {old_key} (may not exist on S3)")

    return rename_operations


def collect_prefix_changes(
    s3_client, bucket_name: str
) -> List[tuple[str, str, List[str]]]:
    """Determine which prefix-based renames are needed."""
    print("\n2. Checking path prefix changes...")
    print("-" * 70)

    prefix_changes: List[tuple[str, str, List[str]]] = []
    for old_prefix, new_prefix in PATH_PREFIX_CHANGES.items():
        if old_objects := find_objects_with_prefix(
            s3_client, bucket_name, old_prefix
        ):
            print(f"  Found {len(old_objects)} objects with prefix '{old_prefix}':")
            for obj_key in old_objects[:5]:
                new_key = obj_key.replace(old_prefix, new_prefix)
                print(f"    {obj_key} → {new_key}")
            if len(old_objects) > 5:
                print(f"    ... and {len(old_objects) - 5} more")
            prefix_changes.append((old_prefix, new_prefix, old_objects))
        else:
            print(f"  No objects found with prefix '{old_prefix}'")

    return prefix_changes


def perform_renames(
    s3_client,
    bucket_name: str,
    rename_operations: List[tuple[str, str]],
    prefix_changes: List[tuple[str, str, List[str]]],
    dry_run: bool,
) -> None:
    """Execute all rename operations and print a summary."""
    print("\n3. Performing renames...")
    print("-" * 70)

    success_count = 0
    fail_count = 0

    for old_key, new_key in rename_operations:
        if rename_object(s3_client, bucket_name, old_key, new_key, dry_run):
            success_count += 1
        else:
            fail_count += 1

    for old_prefix, new_prefix, objects in prefix_changes:
        print(
            f"\n  Renaming prefix '{old_prefix}' → "
            f"'{new_prefix}' ({len(objects)} objects)..."
        )
        for obj_key in objects:
            new_key = obj_key.replace(old_prefix, new_prefix)
            if rename_object(s3_client, bucket_name, obj_key, new_key, dry_run):
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful operations: {success_count}")
    print(f"Failed operations: {fail_count}")
    if dry_run:
        print("\nThis was a DRY RUN. Use --execute to actually perform renames.")


def main():
    """Main cleanup function."""
    args = parse_args()

    dry_run = not args.execute and not args.check_only

    if args.execute and not confirm_execute():
        return

    bucket_name, endpoint, _, _ = get_config()
    mode = "DRY RUN" if dry_run else "EXECUTE" if args.execute else "CHECK ONLY"
    print_header(bucket_name, endpoint, mode)

    s3_client, _ = get_s3_client()

    rename_operations = collect_rename_operations(s3_client, bucket_name)
    prefix_changes = collect_prefix_changes(s3_client, bucket_name)

    if not args.check_only:
        perform_renames(
            s3_client,
            bucket_name,
            rename_operations,
            prefix_changes,
            dry_run,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
