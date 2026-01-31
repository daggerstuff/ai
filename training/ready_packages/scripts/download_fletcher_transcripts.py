#!/usr/bin/env python3
"""
Download Tim Fletcher transcripts from S3 to local directory.
"""

import os
from pathlib import Path

from ai.utils.s3_dataset_loader import S3DatasetLoader


def download_fletcher_transcripts():
    # Load manifest to get correct bucket/endpoint
    manifest_path = Path("ai/training_ready/data/s3_manifest.json")
    if manifest_path.exists():
        import json

        with open(manifest_path) as f:
            manifest = json.load(f)
            bucket = manifest.get("bucket", "pixel-data")
            endpoint = manifest.get("endpoint", "https://s3.us-east-va.io.cloud.ovh.us")
    else:
        bucket = "pixel-data"
        endpoint = "https://s3.us-east-va.io.cloud.ovh.us"

    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    prefix = "datasets/gdrive/tier4_voice_persona/Tim Fletcher/"
    local_base = Path(
        os.path.expanduser("~/datasets/gdrive/tier4_voice_persona/Tim Fletcher/")
    )
    local_base.mkdir(parents=True, exist_ok=True)

    print(f"Listing objects in {prefix}...")

    # We need to list objects. S3DatasetLoader list_datasets only returns .json/.jsonl
    # We need .txt files.
    # So we'll access s3_client directly
    paginator = loader.s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    count = 0
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if (
                    key.endswith("/") or "._" in key
                ):  # Skip directories and Mac hidden files
                    continue

                local_file = local_base / Path(key).name

                if local_file.exists():
                    # Simple size check could be added
                    continue

                print(f"Downloading {key}...")
                try:
                    loader.s3_client.download_file(bucket, key, str(local_file))
                    count += 1
                except Exception as e:
                    print(f"Failed to download {key}: {e}")

    print(f"Downloaded {count} new files to {local_base}")


if __name__ == "__main__":
    download_fletcher_transcripts()
