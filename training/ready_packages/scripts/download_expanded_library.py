#!/usr/bin/env python3
"""
Download Expanded Library (Nightmare Fuel, Transcripts, Books)
"""

import os
from pathlib import Path

from ai.utils.s3_dataset_loader import S3DatasetLoader


def download_expanded_library():
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

    # 1. Nightmare Fuel Cycles
    nightmare_prefix = "datasets/training_v2/stage3_edge_crisis/"
    nightmare_local = Path("ai/training_ready/data/generated/nightmare_scenarios/")
    nightmare_local.mkdir(parents=True, exist_ok=True)

    # 2. Transcripts Library
    transcripts_prefix = "datasets/consolidated/transcripts/"
    transcripts_local = Path(os.path.expanduser("~/datasets/consolidated/transcripts/"))
    transcripts_local.mkdir(parents=True, exist_ok=True)

    # 3. Books
    books_sources = [
        "datasets/gdrive/raw/Diagnostic and Statistical Manual of... (Z-Library).pdf",
        "datasets/consolidated/datasets/gifts_of_imperfection-brene-brown.pdf",
        "datasets/consolidated/datasets/myth_of_normal-gabor-mate.pdf",
    ]
    books_local = Path(os.path.expanduser("~/datasets/consolidated/books/"))
    books_local.mkdir(parents=True, exist_ok=True)

    # 4. Extra Edge/Crisis Datasets (Addressing "VASTLY go beyond 50 samples")
    extra_edge_sources = [
        "datasets/gdrive/tier3_edge_crisis/crisis_detection_conversations.jsonl",
        "datasets/consolidated/edge_cases/edge_case_output/priority_edge_cases_nvidia.jsonl",
        "datasets/consolidated/conversations/edge_case_dialogues.jsonl",
        "datasets/consolidated/edge_cases/existing_edge_cases.jsonl",
        "datasets/gdrive/raw/dataset_pipeline/crisis_intervention_conversations_dataset.json",
    ]
    edge_local = Path("ai/training_ready/data/generated/edge_case_expanded/")
    edge_local.mkdir(parents=True, exist_ok=True)

    print("🎯 Downloading Expanded Library...")

    # Download Extra Edge Cases
    print("\n⚠️ Downloading Extra Crisis/Edge Datasets...")
    for edge_key in extra_edge_sources:
        local_file = edge_local / Path(edge_key).name
        # If it ends in .3f3e67a2 etc, it's a temp file from a previous failed download
        for tmp_file in edge_local.glob(f"{Path(edge_key).name}.*"):
            print(f"   Cleaning up partial: {tmp_file.name}")
            tmp_file.unlink()

        if not local_file.exists():
            print(f"   Downloading {Path(edge_key).name}...")
            try:
                loader.s3_client.download_file(bucket, edge_key, str(local_file))
                print(f"   ✅ Done: {Path(edge_key).name}")
            except Exception as e:
                print(f"   ❌ Failed {edge_key}: {e}")
        else:
            print(f"   ✅ Already exists: {Path(edge_key).name}")

    # Download Nightmare Scenarios
    print(f"\n🔥 Downloading Nightmare Fuel Scenarios from {nightmare_prefix}...")
    paginator = loader.s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=nightmare_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("jsonl"):
                    local_file = nightmare_local / Path(key).name
                    if not local_file.exists():
                        print(f"   Downloading {Path(key).name}...")
                        try:
                            loader.s3_client.download_file(bucket, key, str(local_file))
                        except Exception as e:
                            print(f"   ❌ Failed {key}: {e}")
                    else:
                        print(f"   ✅ Already exists: {Path(key).name}")

    # Download Transcripts
    print(f"\n📝 Downloading Transcripts Library from {transcripts_prefix}...")
    for page in paginator.paginate(Bucket=bucket, Prefix=transcripts_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(".md") or key.endswith(".txt"):
                    local_file = transcripts_local / Path(key).name
                    if not local_file.exists():
                        print(f"   Downloading {Path(key).name[:50]}...")
                        try:
                            loader.s3_client.download_file(bucket, key, str(local_file))
                        except Exception as e:
                            print(f"   ❌ Failed {key}: {e}")
                    # Don't print "exists" for hundreds of files to avoid clutter

    # Download Books
    print("\n📚 Downloading Books...")
    for book_key in books_sources:
        local_file = books_local / Path(book_key).name
        if not local_file.exists():
            print(f"   Downloading {Path(book_key).name}...")
            try:
                loader.s3_client.download_file(bucket, book_key, str(local_file))
                print(f"   ✅ Done: {Path(book_key).name}")
            except Exception as e:
                print(f"   ❌ Failed {book_key}: {e}")
        else:
            print(f"   ✅ Exists: {Path(book_key).name}")


if __name__ == "__main__":
    download_expanded_library()
