#!/usr/bin/env python3
import boto3
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from ai/.env
load_dotenv("ai/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("S3Uploader")

def get_s3_client():
    """Creates an S3 client configured for OVH Object Storage."""
    endpoint = os.environ.get("OVH_S3_ENDPOINT")
    access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    secret_key = os.environ.get("OVH_S3_SECRET_KEY")
    region = os.environ.get("OVH_S3_REGION", "us-east-va")

    if not all([endpoint, access_key, secret_key]):
        logger.error("Missing OVH S3 credentials in environment. Check ai/.env")
        return None

    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

def upload_final_artifacts(prefix="datasets/training_v2/"):
    """
    Uploads the contents of final dataset directories to OVH S3.
    """
    bucket_name = os.environ.get("OVH_S3_BUCKET", "pixel-data")

    # Directories to upload
    source_dirs = [
        Path("ai/training_ready/datasets/final_instruct"),
        Path("ai/training_ready/datasets/stage1_foundation"),
        Path("ai/training_ready/datasets/stage2_specialist_addiction"),
        Path("ai/training_ready/datasets/stage2_specialist_ptsd"),
        Path("ai/training_ready/datasets/stage3_edge_crisis"),
        Path("ai/training_ready/datasets/stage4_voice"),
    ]

    s3 = get_s3_client()
    if not s3:
        return "S3 client initialization failed."

    uploaded_count = 0
    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Skipping {source_dir} - does not exist.")
            continue

        for file_path in source_dir.rglob("*.json*"):  # .json and .jsonl
            # Create S3 key preserving directory structure
            relative_path = file_path.relative_to("ai/training_ready/datasets")
            s3_key = f"{prefix}{relative_path}"

            logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}...")
            try:
                s3.upload_file(str(file_path), bucket_name, s3_key)
                uploaded_count += 1
                logger.info("  -> Success")
            except Exception as e:
                logger.error(f"  -> Failed: {e}")

    logger.info(f"Upload Complete. {uploaded_count} files uploaded to s3://{bucket_name}/{prefix}")
    return f"Uploaded {uploaded_count} files to s3://{bucket_name}/{prefix}"

if __name__ == "__main__":
    upload_final_artifacts()
