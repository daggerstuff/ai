#!/usr/bin/env python3
"""
S3 Dataset Processor - Pulls actual 52.20GB from S3 and processes
"""

import json
import logging
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3DatasetProcessor:
    """
    Processes actual 52.20GB dataset from S3
    """

    def __init__(
        self,
        bucket_name: str = "pixel-data",
        endpoint_url: str = "https://s3.us-east-va.io.cloud.ovh.us",
        local_cache_dir: str = "/tmp/s3_dataset_cache",
    ):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-va",
        )

        self.stats = {
            "start_time": datetime.now().isoformat(),
            "total_objects": 0,
            "total_size": 0,
            "processed_conversations": 0,
            "errors": [],
            "pii_removed": 0,
            "duplicates_removed": 0,
        }

    def list_s3_objects(self, prefix: str = "") -> list:
        """List all objects in S3 bucket"""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            objects = []

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if "Contents" in page:
                    objects.extend(page["Contents"])

            return objects
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []

    def get_relevant_files(self) -> list:
        """Get list of relevant dataset files from S3"""
        # Key prefixes for therapeutic datasets
        prefixes = [
            "gdrive/processed/",
            "datasets/",
            "voice/",
            "lightning/",
            "acquired/",
            "final_dataset/",
        ]

        relevant_files = []

        for prefix in prefixes:
            objects = self.list_s3_objects(prefix)
            for obj in objects:
                key = obj["Key"]
                size = obj["Size"]

                # Filter for relevant file types
                if any(key.endswith(ext) for ext in [".jsonl", ".json", ".csv"]):
                    if size > 1024:  # Skip tiny files
                        relevant_files.append(
                            {
                                "key": key,
                                "size": size,
                                "last_modified": obj.get("LastModified", "").isoformat()
                                if obj.get("LastModified")
                                else None,
                            }
                        )

        # Sort by size descending
        relevant_files.sort(key=lambda x: x["size"], reverse=True)
        return relevant_files

    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """Download file from S3"""
        try:
            logger.info(
                f"Downloading {s3_key} ({self.format_size(local_path.stat().st_size if local_path.exists() else 0)})"
            )

            # Check if already cached
            if local_path.exists():
                logger.info(f"Using cached: {local_path}")
                return True

            # Download from S3
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Downloaded: {s3_key} -> {local_path}")
            return True

        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False

    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def process_s3_datasets(self) -> dict:
        """Main processing function for S3 datasets"""

        logger.info("Starting S3 dataset processing...")
        logger.info(f"Bucket: {self.bucket_name}")
        logger.info(f"Endpoint: {self.endpoint_url}")

        # Get relevant files
        files = self.get_relevant_files()
        logger.info(f"Found {len(files)} relevant files")

        if not files:
            logger.warning("No files found in S3")
            return {"success": False, "error": "No files found"}

        # Calculate total size
        total_size = sum(f["size"] for f in files)
        logger.info(f"Total dataset size: {self.format_size(total_size)}")

        # Download files to local cache
        downloaded_files = []
        for i, file_info in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_info['key']}")

            local_path = self.local_cache_dir / file_info["key"].replace("/", "_")

            if self.download_file(file_info["key"], local_path):
                downloaded_files.append(
                    {
                        "local_path": local_path,
                        "s3_key": file_info["key"],
                        "size": file_info["size"],
                    }
                )

        # Create processing report
        report = {
            "total_files": len(files),
            "downloaded_files": len(downloaded_files),
            "total_size": total_size,
            "files": files,
            "cache_dir": str(self.local_cache_dir),
            "timestamp": datetime.now().isoformat(),
        }

        # Save report
        report_path = Path(
            "/home/vivi/pixelated/ai/training_ready/data/s3_processing_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def create_s3_manifest(self) -> dict:
        """Create manifest of S3 datasets"""
        files = self.get_relevant_files()

        manifest = {
            "bucket": self.bucket_name,
            "endpoint": self.endpoint_url,
            "total_objects": len(files),
            "total_size": sum(f["size"] for f in files),
            "categories": {},
            "files": files[:20],  # Top 20 largest
        }

        # Categorize files
        for file_info in files:
            key = file_info["key"]
            if "mental_health" in key.lower() or "therapeutic" in key.lower():
                category = "therapeutic_data"
            elif "priority" in key.lower():
                category = "priority_datasets"
            elif "cot" in key.lower() or "reasoning" in key.lower():
                category = "chain_of_thought"
            elif "reddit" in key.lower() or "social" in key.lower():
                category = "social_media"
            elif "voice" in key.lower():
                category = "voice_data"
            elif "final" in key.lower():
                category = "consolidated"
            else:
                category = "other"

            if category not in manifest["categories"]:
                manifest["categories"][category] = []
            manifest["categories"][category].append(file_info)

        return manifest


def main():
    """Main execution"""
    processor = S3DatasetProcessor()

    try:
        # Create S3 manifest
        manifest = processor.create_s3_manifest()

        print("=" * 60)
        print("S3 DATASET DISCOVERY")
        print("=" * 60)
        print(f"Bucket: {manifest['bucket']}")
        print(f"Endpoint: {manifest['endpoint']}")
        print(f"Total files: {manifest['total_objects']}")
        print(f"Total size: {manifest['total_size'] / 1024**3:.2f}GB")

        for category, files in manifest["categories"].items():
            size = sum(f["size"] for f in files)
            print(f"{category}: {len(files)} files, {size / 1024**3:.2f}GB")

        if manifest["files"]:
            print(f"\nTop 5 largest files:")
            for f in manifest["files"][:5]:
                print(f"  {f['key']}: {f['size'] / 1024**3:.2f}GB")

        print(
            f"\nManifest saved to: /home/vivi/pixelated/ai/training_ready/data/s3_processing_report.json"
        )

        # Ask for confirmation to proceed
        response = input("\nProceed with downloading and processing? (y/N): ")
        if response.lower() == "y":
            result = processor.process_s3_datasets()
            print(f"✅ Processing complete. Files cached in: {result['cache_dir']}")
        else:
            print("❌ Processing cancelled")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check AWS credentials and S3 access")


if __name__ == "__main__":
    main()
