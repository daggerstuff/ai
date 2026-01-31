#!/usr/bin/env python3
"""
Upload Consolidation Artifacts to S3

Uploads findings, audit reports, plans, and documentation from the training package
consolidation work to S3 for archival and reference.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv("ai/.env")
load_dotenv(".env")  # Also check project root

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConsolidationUploader")


def get_s3_client():
    """Creates an S3 client configured for OVH Object Storage."""
    endpoint = os.environ.get("OVH_S3_ENDPOINT")
    access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    secret_key = os.environ.get("OVH_S3_SECRET_KEY")
    region = os.environ.get("OVH_S3_REGION", "us-east-va")

    if not all([endpoint, access_key, secret_key]):
        logger.error("Missing OVH S3 credentials in environment.")
        logger.error("Required: OVH_S3_ENDPOINT, OVH_S3_ACCESS_KEY, OVH_S3_SECRET_KEY")
        logger.error("Check ai/.env or .env files")
        return None

    try:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None


def upload_file_to_s3(s3_client, bucket: str, local_path: Path, s3_key: str) -> bool:
    """Upload a single file to S3."""
    try:
        logger.info(f"Uploading {local_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info("  ✓ Success")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


def get_consolidation_artifacts() -> list[tuple[Path, str]]:
    """
    Returns list of (local_path, s3_key) tuples for consolidation artifacts.

    S3 structure: datasets/metadata/consolidation/{category}/{filename}
    """
    project_root = Path(__file__).parents[3]
    artifacts = []

    # Consolidation findings and documentation
    findings_files = [
        (
            project_root / ".notes" / "markdown" / "two.md",
            "datasets/metadata/consolidation/findings/training_package_consolidation_audit.md",
        ),
    ]

    # Audit reports and completion summaries
    plan_files = [
        (
            project_root / ".cursor" / "plans" / "training_package_consolidation_audit_report.md",
            "datasets/metadata/consolidation/reports/audit_report.md",
        ),
        (
            project_root
            / ".cursor"
            / "plans"
            / "training_package_consolidation_completion_summary.md",
            "datasets/metadata/consolidation/reports/completion_summary.md",
        ),
    ]

    # Related consolidation documentation (if exists)
    related_docs = [
        (
            project_root / ".notes" / "markdown" / "three.md",  # Google Drive consolidation
            "datasets/metadata/consolidation/findings/gdrive_consolidation_audit.md",
        ),
        (
            project_root / ".notes" / "markdown" / "five.md",  # S3 implementation
            "datasets/metadata/consolidation/findings/s3_implementation_summary.md",
        ),
        (
            project_root / ".notes" / "markdown" / "six.md",  # S3 completion
            "datasets/metadata/consolidation/findings/s3_completion_summary.md",
        ),
    ]

    # Combine all artifacts
    all_artifacts = findings_files + plan_files + related_docs

    # Filter to only existing files
    for local_path, s3_key in all_artifacts:
        if local_path.exists():
            artifacts.append((local_path, s3_key))
        else:
            logger.debug(f"Skipping non-existent file: {local_path}")

    return artifacts


def create_manifest_entry(artifacts: list[tuple[Path, str]], uploaded: list[str]) -> dict:
    """Create a manifest entry for the uploaded artifacts."""
    return {
        "category": "consolidation_artifacts",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(artifacts),
        "uploaded_files": len(uploaded),
        "artifacts": [
            {
                "s3_key": s3_key,
                "local_path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "uploaded": s3_key in uploaded,
            }
            for local_path, s3_key in artifacts
        ],
    }


def upload_consolidation_artifacts(prefix: str = "datasets/metadata/consolidation/") -> dict:
    """
    Upload all consolidation artifacts to S3.

    Returns dict with upload results.
    """
    bucket_name = os.environ.get("OVH_S3_BUCKET", "pixel-data")

    s3_client = get_s3_client()
    if not s3_client:
        return {"success": False, "error": "S3 client initialization failed"}

    # Get artifacts to upload
    artifacts = get_consolidation_artifacts()

    if not artifacts:
        logger.warning("No consolidation artifacts found to upload")
        return {"success": False, "error": "No artifacts found"}

    logger.info(f"Found {len(artifacts)} consolidation artifacts to upload")

    # Upload each artifact
    uploaded = []
    failed = []

    for local_path, s3_key in artifacts:
        if upload_file_to_s3(s3_client, bucket_name, local_path, s3_key):
            uploaded.append(s3_key)
        else:
            failed.append(s3_key)

    # Create manifest entry
    manifest_entry = create_manifest_entry(artifacts, uploaded)

    # Upload manifest itself
    manifest_key = f"{prefix}manifest.json"
    manifest_path = Path("/tmp/consolidation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entry, f, indent=2)

    if upload_file_to_s3(s3_client, bucket_name, manifest_path, manifest_key):
        logger.info(f"✓ Uploaded manifest to s3://{bucket_name}/{manifest_key}")
    else:
        logger.warning("Failed to upload manifest")

    # Cleanup temp manifest
    if manifest_path.exists():
        manifest_path.unlink()

    # Summary
    logger.info("=" * 60)
    logger.info("Upload Summary:")
    logger.info(f"  Total artifacts: {len(artifacts)}")
    logger.info(f"  Successfully uploaded: {len(uploaded)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info(f"  Bucket: {bucket_name}")
    logger.info(f"  Prefix: {prefix}")
    logger.info("=" * 60)

    return {
        "success": len(failed) == 0,
        "bucket": bucket_name,
        "prefix": prefix,
        "total": len(artifacts),
        "uploaded": len(uploaded),
        "failed": len(failed),
        "uploaded_files": uploaded,
        "failed_files": failed,
        "manifest": manifest_entry,
    }


if __name__ == "__main__":
    import sys

    result = upload_consolidation_artifacts()

    if result["success"]:
        logger.info("✓ All consolidation artifacts uploaded successfully")
        sys.exit(0)
    else:
        logger.error("✗ Some artifacts failed to upload")
        if result.get("failed_files"):
            logger.error(f"Failed files: {result['failed_files']}")
        sys.exit(1)
