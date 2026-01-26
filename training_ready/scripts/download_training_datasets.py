#!/usr/bin/env python3
"""
Download Tier 1 Priority datasets for mental health model training.
This script downloads the critical datasets needed for the training pipeline.
"""

import os
import sys
from pathlib import Path

from ai.utils.s3_dataset_loader import S3DatasetLoader


def download_tier1_datasets():
    """Download Tier 1 Priority datasets (1.16GB, 40% training weight) - CRITICAL"""

    # Load manifest to get correct bucket/endpoint
    manifest_path = Path("ai/training_ready/data/s3_manifest.json")
    if manifest_path.exists():
        import json

        with open(manifest_path) as f:
            manifest = json.load(f)
            bucket = manifest.get("bucket", "pixel-data")
            endpoint = manifest.get("endpoint", "https://s3.us-east-va.io.cloud.ovh.us")
            print(f"Loaded config from manifest: Bucket={bucket}, Endpoint={endpoint}")
    else:
        bucket = "pixel-data"
        endpoint = "https://s3.us-east-va.io.cloud.ovh.us"
        print(
            f"Manifest not found, using defaults: Bucket={bucket}, Endpoint={endpoint}"
        )

    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    # Tier 1 Priority datasets based on actual S3 structure discovered
    tier1_datasets = [
        # Priority Wendy Datasets
        {
            "s3_key": "datasets/consolidated/raw/priority/priority_1.jsonl",
            "local_path": "~/datasets/consolidated/priority_wendy/priority_1.jsonl",
            "description": "Tier 1 Priority 1 (Wendy)",
        },
        {
            "s3_key": "datasets/consolidated/raw/priority/priority_2.jsonl",
            "local_path": "~/datasets/consolidated/priority_wendy/priority_2.jsonl",
            "description": "Tier 1 Priority 2 (Wendy)",
        },
        {
            "s3_key": "datasets/consolidated/raw/priority/priority_3.jsonl",
            "local_path": "~/datasets/consolidated/priority_wendy/priority_3.jsonl",
            "description": "Tier 1 Priority 3 (Wendy)",
        },
        # CoT Datasets
        {
            "s3_key": (
                "datasets/gdrive/raw/CoT_Neurodivergent_vs_Neurotypical_Interactions/"
                "CoT_Neurodivergent vs. Neurotypical Interactions.json"
            ),
            "local_path": (
                "~/datasets/consolidated/cot/"
                "CoT_Neurodivergent_vs_Neurotypical_Interactions.json"
            ),
            "description": "CoT Neurodivergent vs Neurotypical",
        },
        {
            "s3_key": (
                "datasets/gdrive/raw/CoT_Philosophical_Understanding/"
                "CoT_Philosophical_Understanding.json"
            ),
            "local_path": (
                "~/datasets/consolidated/cot/CoT_Philosophical_Understanding.json"
            ),
            "description": "CoT Philosophical Understanding",
        },
        # Reddit Data
        {
            "s3_key": (
                "datasets/gdrive/raw/reddit_mental_health/mental_disorders_reddit.csv"
            ),
            "local_path": "~/datasets/consolidated/reddit/mental_disorders_reddit.csv",
            "description": "Reddit Mental Disorders",
        },
        {
            "s3_key": "datasets/gdrive/raw/reddit_mental_health/Suicide_Detection.csv",
            "local_path": "~/datasets/consolidated/reddit/Suicide_Detection.csv",
            "description": "Reddit Suicide Detection",
        },
    ]

    total_downloaded = 0
    successful_downloads = []

    print("🎯 Downloading Tier 1 Priority Datasets (CRITICAL for training)...")
    print("=" * 60)

    for dataset in tier1_datasets:
        try:
            # Expand user path
            local_path = Path(os.path.expanduser(dataset["local_path"]))

            # Create local directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"📥 Downloading: {dataset['description']}")
            print(f"   S3 Key: {dataset['s3_key']}")
            print(f"   Local: {local_path}")

            # Check if file already exists
            if local_path.exists():
                file_size = local_path.stat().st_size / (1024 * 1024)
                print(
                    f"   ⚠️  File already exists ({file_size:.2f} MB). "
                    "Skipping download."
                )
                total_downloaded += file_size
                successful_downloads.append(dataset)
                print()
                continue

            # Download the file using download_file (efficient)
            try:
                print("   📥 Downloading to file (stream)...")

                # Check if loader has s3_client
                if hasattr(loader, "s3_client"):
                    loader.s3_client.download_file(
                        bucket, dataset["s3_key"], str(local_path)
                    )
                    file_size_mb = local_path.stat().st_size / (1024 * 1024)
                else:
                    # Fallback if s3_client not exposed (though we know it is)
                    full_s3_path = f"s3://{bucket}/{dataset['s3_key']}"
                    data = loader.load_bytes(full_s3_path)
                    with open(local_path, "wb") as f:
                        f.write(data)
                    file_size_mb = len(data) / (1024 * 1024)

                total_downloaded += file_size_mb
                successful_downloads.append(dataset)
                print(f"   ✅ SUCCESS: {file_size_mb:.2f} MB downloaded")

            except Exception as download_error:
                print(f"   ❌ FAILED: {str(download_error)}")

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")

        print()

    # Summary
    print("=" * 60)
    print("📊 TIER 1 DOWNLOAD SUMMARY:")
    print(
        f"   ✅ Successfully downloaded: {len(successful_downloads)}/"
        f"{len(tier1_datasets)} files"
    )
    print(f"   📦 Total size: {total_downloaded:.2f} MB")

    if len(successful_downloads) == len(tier1_datasets):
        print("   🎉 ALL TIER 1 DATASETS DOWNLOADED SUCCESSFULLY!")
        return True
    else:
        print("   ⚠️  Some downloads failed. Check logs above.")
        return False


if __name__ == "__main__":
    success = download_tier1_datasets()
    sys.exit(0 if success else 1)
