#!/usr/bin/env python3
"""
Upload Local Datasets to S3 and Remove Local Copies

Uploads all local dataset files to S3 and removes local copies after successful upload.
This ensures S3 is the canonical source and local storage is freed.
"""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

# Load environment variables
load_dotenv("ai/.env")
load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DatasetUploader")


def get_datasets_to_upload() -> list[tuple[Path, str]]:
    """
    Returns list of (local_path, s3_key) tuples for datasets to upload.

    S3 structure: datasets/local/ai/training_ready/{relative_path}
    """
    project_root = Path(__file__).parents[3]
    training_ready = project_root / "ai" / "training_ready"
    datasets = []

    # Final dataset files
    final_dataset_dir = training_ready / "data" / "final_dataset"
    if final_dataset_dir.exists():
        for file_path in final_dataset_dir.rglob("*.jsonl"):
            relative = file_path.relative_to(training_ready)
            s3_key = f"datasets/local/ai/training_ready/{relative}"
            datasets.append((file_path, s3_key))

    # Generated datasets
    generated_dir = training_ready / "data" / "generated"
    if generated_dir.exists():
        for file_path in generated_dir.rglob("*.jsonl"):
            relative = file_path.relative_to(training_ready)
            s3_key = f"datasets/local/ai/training_ready/{relative}"
            datasets.append((file_path, s3_key))

    # Large dataset files in data/
    data_dir = training_ready / "data"
    large_files = [
        "ULTIMATE_FINAL_DATASET.jsonl",
        "unified_6_component_dataset.jsonl",
    ]
    for filename in large_files:
        file_path = data_dir / filename
        if file_path.exists():
            relative = file_path.relative_to(training_ready)
            s3_key = f"datasets/local/ai/training_ready/{relative}"
            datasets.append((file_path, s3_key))

    # Package data files (apex/velocity)
    for package_dir in ["apex", "velocity"]:
        package_data_dir = training_ready / "packages" / package_dir / "data"
        if package_data_dir.exists():
            for file_path in package_data_dir.rglob("*.jsonl"):
                if file_path.stat().st_size > 1024 * 1024:  # Only files > 1MB
                    relative = file_path.relative_to(training_ready)
                    s3_key = f"datasets/local/ai/training_ready/{relative}"
                    datasets.append((file_path, s3_key))

    return datasets


def upload_and_remove(local_path: Path, s3_key: str, s3_client: Any, bucket: str) -> bool:
    """Upload file to S3 and remove local copy if successful"""
    try:
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"Uploading {local_path.name} ({file_size_mb:.1f}MB) to s3://{bucket}/{s3_key}")

        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info("  ✓ Uploaded successfully")

        # Verify upload
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            logger.info("  ✓ Verified in S3")

            # Remove local copy
            local_path.unlink()
            logger.info(f"  ✓ Removed local copy: {local_path}")
            return True
        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False

    except Exception as e:
        logger.error(f"  ✗ Upload failed: {e}")
        return False


def main() -> int:
    """Main entry point"""
    bucket = os.getenv("OVH_S3_BUCKET", "pixel-data")

    # Initialize S3 loader
    try:
        s3_loader = S3DatasetLoader(bucket=bucket)
        s3_client = s3_loader.s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        logger.error("Make sure OVH_S3_ACCESS_KEY, OVH_S3_SECRET_KEY, and OVH_S3_ENDPOINT are set")
        return 1

    # Get datasets to upload
    datasets = get_datasets_to_upload()

    if not datasets:
        logger.info("No datasets found to upload")
        return 0

    logger.info(f"Found {len(datasets)} dataset files to upload")
    logger.info(f"Bucket: {bucket}")
    logger.info("")

    # Upload each dataset
    uploaded = 0
    failed = 0

    for local_path, s3_key in datasets:
        if upload_and_remove(local_path, s3_key, s3_client, bucket):
            uploaded += 1
        else:
            failed += 1
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Upload Summary:")
    logger.info(f"  Total files: {len(datasets)}")
    logger.info(f"  Successfully uploaded and removed: {uploaded}")
    logger.info(f"  Failed: {failed}")
    logger.info("=" * 60)

    if failed > 0:
        logger.warning("Some files failed to upload. Local copies were NOT removed.")
        return 1

    logger.info("✓ All datasets uploaded to S3 and local copies removed")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
