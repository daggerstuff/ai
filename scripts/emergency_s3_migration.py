import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

# --- Configuration ---
BUCKET_NAME = "pixel-data"
ENDPOINT_URL = "https://s3.us-east-va.io.cloud.ovh.us"
REGION_NAME = "us-east-va"
UPLOAD_PREFIX = "legacy_local_backup/"
MAX_WORKERS = 1

# Map of local paths (relative to repo root) to S3 keys
# We will auto-generate the S3 keys to maintain structure under the prefix
FILES_TO_UPLOAD = [
    # The Large Datasets
    "ai/training_ready/data/final_corpus/merged_dataset_raw.jsonl",
    "ai/training_ready/data/ULTIMATE_FINAL_DATASET_cleaned.jsonl",
    "ai/training_ready/data/ULTIMATE_FINAL_DATASET.jsonl",
    "ai/training_ready/configs/stage_configs/ULTIMATE_FINAL_DATASET.jsonl",
    # The Model Checkpoints (Iterate these directories)
    "ai/lightning/pixelated-training/wayfarer-balanced",
    "ai/dataset_pipeline/pixelated-training/training_dataset.json",
    "ai/lightning/pixelated-training/training_dataset.json",
    # The large zips
    "ai/data/compress/RealLifeDeceptionDetection.2016.zip",
    "ai/data/compress/glove.840B.300d.zip",
]

PROJECT_ROOT = Path("/home/vivi/pixelated")


def get_s3_client():
    # Attempt to get credentials from env vars
    access_key = os.environ.get("OVH_S3_ACCESS_KEY") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    secret_key = os.environ.get("OVH_S3_SECRET_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )

    if not access_key or not secret_key:
        print(
            "❌ Error: Missing credentials. Please set OVH_S3_ACCESS_KEY and "
            "OVH_S3_SECRET_KEY."
        )
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def collect_files(root, paths):
    """validates files and expands directories"""
    final_list = []
    total_size = 0

    for p_str in paths:
        p = root / p_str
        if not p.exists():
            print(f"⚠️ Warning: File not found: {p}")
            continue

        if p.is_file():
            final_list.append(p)
            total_size += p.stat().st_size

        elif p.is_dir():
            print(f"📂 Scanning directory: {p}")
            for item in p.rglob("*"):
                if item.is_file():
                    final_list.append(item)
                    total_size += item.stat().st_size

    return final_list, total_size


def upload_file(s3_client, file_path, root_path):
    relative_path = file_path.relative_to(root_path)
    s3_key = f"{UPLOAD_PREFIX}{relative_path}"

    file_size = file_path.stat().st_size
    human_size = f"{file_size / (1024 * 1024):.2f} MB"

    try:
        # Check if exists
        try:
            head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
            remote_size = head["ContentLength"]
            if remote_size == file_size:
                print(
                    f"⏭️  Skipping {file_path.name} "
                    f"(Already exists on S3 - {human_size})"
                )
                return True
        except Exception:
            pass  # Object doesn't exist

        print(f"⬆️  Uploading {file_path.name} ({human_size}) -> {s3_key}...")
        s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
        print(f"✅ Uploaded {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {file_path.name}: {e}")
        return False


def main():
    print("🚀 Starting Emergency S3 Migration for Local Cleanup")
    print(f"   Target Bucket: {BUCKET_NAME}")
    print(f"   Prefix: {UPLOAD_PREFIX}")

    files, size_bytes = collect_files(PROJECT_ROOT, FILES_TO_UPLOAD)
    size_gb = size_bytes / (1024**3)

    print(f"\n📦 Found {len(files)} files totaling {size_gb:.2f} GB")

    if len(files) == 0:
        print("Nothing to upload.")
        return

    s3 = get_s3_client()

    print("\nStarting uploads...")

    # Simple threaded uploader
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(upload_file, s3, f, PROJECT_ROOT) for f in files]
        results = [f.result() for f in futures]

    success_count = sum(1 for r in results if r)
    print(f"\n🎉 Completed. {success_count}/{len(files)} files uploaded.")

    if success_count == len(files):
        print("\n✅ All target files are safely on S3.")
        print("   You may now consider deleting the local copies to free up space.")
        print(
            "   Example command: rm -rf "
            "pixelated/ai/lightning/pixelated-training/wayfarer-balanced"
        )
    else:
        print("\n⚠️ Some uploads failed. Please review the logs above.")


if __name__ == "__main__":
    main()
