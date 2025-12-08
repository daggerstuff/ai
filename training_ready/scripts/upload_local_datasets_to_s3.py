#!/usr/bin/env python3
"""
Upload Local-Only Datasets to S3

Reads the catalog and uploads all local-only datasets to S3.
Supports resuming interrupted uploads and progress tracking.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.error("boto3 not available. Install with: uv pip install boto3")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LocalDatasetUploader:
    """Uploads local datasets to S3"""

    def __init__(self, s3_bucket: str, s3_prefix: str = "datasets/local/"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/") + "/"

        # Initialize S3 client (OVH or AWS-compatible)
        endpoint_url = os.getenv("OVH_S3_ENDPOINT") or os.getenv("DATASET_S3_ENDPOINT_URL")
        access_key = (
            os.getenv("OVH_S3_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
            or os.getenv("DATASET_S3_ACCESS_KEY_ID")
        )
        secret_key = (
            os.getenv("OVH_S3_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
            or os.getenv("DATASET_S3_SECRET_ACCESS_KEY")
        )
        region_name = (
            os.getenv("OVH_S3_REGION")
            or os.getenv("DATASET_S3_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        session_kwargs: Dict[str, Any] = {}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        if region_name:
            session_kwargs["region_name"] = region_name
        if endpoint_url:
            session_kwargs["endpoint_url"] = endpoint_url

        self.s3_client = boto3.client("s3", **session_kwargs) if BOTO3_AVAILABLE else None

        if not self.s3_client:
            raise RuntimeError("Failed to initialize S3 client. Check credentials.")

    def upload_file(self, local_path: Path, s3_key: Optional[str] = None) -> Dict[str, Any]:
        """Upload a single file to S3"""
        if not local_path.exists():
            return {
                "success": False,
                "error": f"File not found: {local_path}",
            }

        if not local_path.is_file():
            return {
                "success": False,
                "error": f"Not a file: {local_path}",
            }

        try:
            # Determine S3 key
            if not s3_key:
                # Create a relative path from project root
                project_root = Path.cwd()
                try:
                    relative_path = local_path.relative_to(project_root)
                except ValueError:
                    # File is outside project root, use filename
                    relative_path = Path(local_path.name)

                s3_key = f"{self.s3_prefix}{relative_path.as_posix()}"

            logger.info(f"  📤 Uploading: {local_path.name} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")

            # Upload with progress callback for large files
            file_size = local_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # > 100MB
                # Use multipart upload for large files
                self.s3_client.upload_file(
                    str(local_path),
                    self.s3_bucket,
                    s3_key,
                    ExtraArgs={"ContentType": self._guess_content_type(local_path)}
                )
            else:
                self.s3_client.upload_file(
                    str(local_path),
                    self.s3_bucket,
                    s3_key,
                    ExtraArgs={"ContentType": self._guess_content_type(local_path)}
                )

            return {
                "success": True,
                "s3_path": f"s3://{self.s3_bucket}/{s3_key}",
                "local_path": str(local_path),
                "size": file_size,
            }

        except ClientError as e:
            logger.error(f"  ❌ S3 error: {e}")
            return {
                "success": False,
                "error": f"S3 error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"  ❌ Upload failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _guess_content_type(self, file_path: Path) -> str:
        """Guess content type from file extension"""
        ext = file_path.suffix.lower()
        content_types = {
            ".json": "application/json",
            ".jsonl": "application/jsonl",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".md": "text/markdown",
            ".py": "text/x-python",
        }
        return content_types.get(ext, "application/octet-stream")

    def check_file_exists(self, s3_key: str) -> bool:
        """Check if file already exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False


def load_catalog(catalog_path: Path) -> Dict[str, Any]:
    """Load the dataset accessibility catalog"""
    with open(catalog_path, "r") as f:
        return json.load(f)


def load_upload_list(upload_list_path: Path) -> List[str]:
    """Load the list of local-only files to upload"""
    if not upload_list_path.exists():
        return []

    with open(upload_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Upload local-only datasets to S3")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("ai/training_ready/scripts/output/dataset_accessibility_catalog.json"),
        help="Path to catalog file"
    )
    parser.add_argument(
        "--upload-list",
        type=Path,
        default=Path("ai/training_ready/scripts/output/local_only_upload_list.txt"),
        help="Path to upload list file"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="S3 bucket name (defaults to OVH_S3_BUCKET or S3_BUCKET env var)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="datasets/local/",
        help="S3 prefix for uploaded files"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in S3"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to upload (for testing)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from previous run using results JSON"
    )

    args = parser.parse_args()

    base_path = Path.cwd()
    catalog_path = base_path / args.catalog if not args.catalog.is_absolute() else args.catalog
    upload_list_path = base_path / args.upload_list if not args.upload_list.is_absolute() else args.upload_list

    if not catalog_path.exists():
        logger.error(f"❌ Catalog not found: {catalog_path}")
        return 1

    if not upload_list_path.exists():
        logger.error(f"❌ Upload list not found: {upload_list_path}")
        return 1

    # Get bucket name
    bucket = args.bucket or os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")

    logger.info(f"🔌 Initializing S3 uploader (bucket: {bucket}, prefix: {args.prefix})...")
    try:
        uploader = LocalDatasetUploader(s3_bucket=bucket, s3_prefix=args.prefix)
    except Exception as e:
        logger.error(f"❌ Failed to initialize uploader: {e}")
        return 1

    logger.info("✅ S3 client initialized")
    logger.info("")

    # Load file list
    file_paths = load_upload_list(upload_list_path)

    if args.max_files:
        file_paths = file_paths[:args.max_files]
        logger.info(f"⚠️  Limiting to first {args.max_files} files (testing mode)")

    logger.info(f"📋 Found {len(file_paths)} files to upload")
    logger.info("")

    # Resume from previous run if specified
    already_uploaded = set()
    if args.resume and args.resume.exists():
        logger.info(f"📖 Loading previous results from: {args.resume}")
        with open(args.resume, "r") as f:
            prev_results = json.load(f)
            already_uploaded = {
                r["local_path"] for r in prev_results.get("results", [])
                if r.get("result", {}).get("success")
            }
        logger.info(f"  Skipping {len(already_uploaded)} already uploaded files")
        logger.info("")

    # Upload files
    results = []
    successful = 0
    failed = 0
    skipped = 0

    for i, file_path_str in enumerate(file_paths, 1):
        file_path = Path(file_path_str)

        # Skip if already uploaded
        if str(file_path) in already_uploaded:
            skipped += 1
            continue

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"[{i}/{len(file_paths)}] ⚠️  File not found: {file_path}")
            failed += 1
            results.append({
                "local_path": str(file_path),
                "result": {"success": False, "error": "File not found"}
            })
            continue

        logger.info(f"[{i}/{len(file_paths)}] 📤 {file_path.name}")

        # Check if already in S3
        if args.skip_existing:
            relative_path = file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else Path(file_path.name)
            s3_key = f"{args.prefix.rstrip('/')}/{relative_path.as_posix()}"
            if uploader.check_file_exists(s3_key):
                logger.info(f"  ⏭️  Already exists in S3, skipping")
                skipped += 1
                results.append({
                    "local_path": str(file_path),
                    "result": {"success": True, "skipped": True, "s3_path": f"s3://{bucket}/{s3_key}"}
                })
                continue

        # Upload file
        result = uploader.upload_file(file_path)

        if result.get("success"):
            logger.info(f"  ✅ Success: {result.get('s3_path', 'N/A')}")
            successful += 1
        else:
            logger.error(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            failed += 1

        results.append({
            "local_path": str(file_path),
            "result": result
        })

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Upload Summary:")
    logger.info(f"  Total files: {len(file_paths)}")
    logger.info(f"  ✅ Successful: {successful}")
    logger.info(f"  ⏭️  Skipped: {skipped}")
    logger.info(f"  ❌ Failed: {failed}")
    logger.info("")

    # Save results
    results_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "local_upload_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(file_paths),
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "results": results
        }, f, indent=2)

    logger.info(f"💾 Results saved to: {results_path}")
    logger.info("")
    logger.info("💡 Tip: Use --resume flag with this results file to continue from where you left off")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

