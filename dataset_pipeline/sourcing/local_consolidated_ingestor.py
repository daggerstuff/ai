"""
Local Consolidated Data Ingestor.

Streams the finalized training datasets from `ai/training_data_consolidated/`
directly to OVH S3. These are production-ready datasets identified in the specs.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv

load_dotenv("ai/.env")


def get_s3_client():
    """Creates an S3 client configured for OVH Object Storage."""
    import boto3

    endpoint = os.environ.get("OVH_S3_ENDPOINT")
    access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    secret_key = os.environ.get("OVH_S3_SECRET_KEY")
    region = os.environ.get("OVH_S3_REGION", "us-east-va")

    if not all([endpoint, access_key, secret_key]):
        logger.error("Missing OVH S3 credentials")
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


class LocalConsolidatedIngestor:
    """
    Ingests finalized training data from ai/training_data_consolidated/.

    This includes:
    - ULTIMATE_FINAL_DATASET.jsonl (2.5GB)
    - merged_dataset.jsonl (2.5GB)
    - training_dataset_enhanced.json (58MB, 16K conversations)
    - psychology_knowledge_base_optimized.json (4,867 concepts)
    - Test/validation splits
    """

    BASE_PATH = Path("ai/training_data_consolidated")

    # High-priority files to upload (ordered by importance)
    PRIORITY_FILES = {
        # Tier 1: Final datasets (production-ready)
        "final_datasets/training_dataset_enhanced.json": {
            "target": "tier1_priority",
            "description": "16,386 professional therapeutic conversations",
        },
        "final_datasets/unified_training_data.jsonl": {
            "target": "tier1_priority",
            "description": "Unified training data (212MB)",
        },
        # Knowledge base
        "psychology_knowledge/psychology_knowledge_base_optimized.json": {
            "target": "tier6_knowledge",
            "description": "4,867 psychology concepts",
        },
        # Large datasets (upload separately)
        "final_datasets/merged_dataset.jsonl": {
            "target": "tier1_priority_large",
            "description": "Main training dataset (2.5GB)",
        },
        "final_datasets/ULTIMATE_FINAL_DATASET.jsonl": {
            "target": "tier1_priority_large",
            "description": "Ultimate final dataset (2.5GB)",
        },
        # Test/val splits
        "final_datasets/pixelated_empathy_test_20250526_174637.jsonl": {
            "target": "splits",
            "description": "Test split (190MB)",
        },
        "final_datasets/pixelated_empathy_val_20250526_174637.jsonl": {
            "target": "splits",
            "description": "Validation split (191MB)",
        },
    }

    # Additional directories to scan
    ADDITIONAL_DIRS = [
        ("edge_cases", "tier3_edge"),
        ("conversations", "tier2_professional"),
        ("transcripts", "tier4_voice"),
        ("voice_data", "tier4_voice"),
    ]

    def stream_file_to_s3(self, local_path: Path, s3_key: str, bucket: str) -> bool:
        """Streams a single file to S3."""
        s3 = get_s3_client()
        if not s3:
            return False

        if not local_path.exists():
            logger.warning(f"File not found: {local_path}")
            return False

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Uploading {local_path.name} ({file_size_mb:.1f}MB) to s3://{bucket}/{s3_key}..."
        )

        try:
            # Use multipart upload for large files
            s3.upload_file(str(local_path), bucket, s3_key)
            logger.info("  -> Success")
            return True
        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            return False

    def upload_priority_files(
        self, bucket: str = "pixel-data", prefix: str = "datasets/consolidated/"
    ):
        """Uploads high-priority finalized training files."""
        results = {}

        for rel_path, config in self.PRIORITY_FILES.items():
            local_path = self.BASE_PATH / rel_path
            target = config["target"]

            if local_path.exists():
                s3_key = f"{prefix}{target}/{local_path.name}"
                success = self.stream_file_to_s3(local_path, s3_key, bucket)
                results[rel_path] = {"success": success, "s3_key": s3_key if success else None}
            else:
                logger.warning(f"Skipping {rel_path} - not found")
                results[rel_path] = {"success": False, "s3_key": None}

        return results

    def upload_additional_directories(
        self, bucket: str = "pixel-data", prefix: str = "datasets/consolidated/"
    ):
        """Uploads additional data directories."""
        results = {}

        for dir_name, target in self.ADDITIONAL_DIRS:
            dir_path = self.BASE_PATH / dir_name

            if not dir_path.exists():
                continue

            for file_path in dir_path.rglob("*.json*"):
                s3_key = f"{prefix}{target}/{file_path.name}"
                success = self.stream_file_to_s3(file_path, s3_key, bucket)
                results[str(file_path)] = {
                    "success": success,
                    "s3_key": s3_key if success else None,
                }

        return results

    def run_full_upload(
        self,
        bucket: str = "pixel-data",
        prefix: str = "datasets/consolidated/",
        skip_large: bool = False,
    ):
        """
        Runs the full upload process.

        Args:
            skip_large: If True, skips files > 500MB (for faster testing)
        """
        logger.info("=" * 60)
        logger.info("LOCAL CONSOLIDATED DATA INGESTOR")
        logger.info("=" * 60)

        # Priority files
        logger.info("\n--- Uploading Priority Files ---")
        priority_results = {}

        for rel_path, config in self.PRIORITY_FILES.items():
            local_path = self.BASE_PATH / rel_path

            if not local_path.exists():
                continue

            file_size_mb = local_path.stat().st_size / (1024 * 1024)

            # Skip large files if requested
            if skip_large and file_size_mb > 500:
                logger.info(f"Skipping {local_path.name} ({file_size_mb:.0f}MB) - too large")
                continue

            target = config["target"]
            s3_key = f"{prefix}{target}/{local_path.name}"
            success = self.stream_file_to_s3(local_path, s3_key, bucket)
            priority_results[rel_path] = success

        # Additional directories
        logger.info("\n--- Uploading Additional Directories ---")
        additional_results = self.upload_additional_directories(bucket, prefix)

        # Summary
        total_success = sum(priority_results.values()) + sum(
            1 for r in additional_results.values() if r["success"]
        )
        total_files = len(priority_results) + len(additional_results)

        logger.info("\n" + "=" * 60)
        logger.info(f"UPLOAD COMPLETE: {total_success}/{total_files} files")
        logger.info("=" * 60)

        return {
            "priority": priority_results,
            "additional": additional_results,
            "total_success": total_success,
            "total_files": total_files,
        }


def run_consolidated_upload(skip_large: bool = True):
    """Entry point for uploading consolidated training data."""
    ingestor = LocalConsolidatedIngestor()
    return ingestor.run_full_upload(skip_large=skip_large)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consolidated_upload()
