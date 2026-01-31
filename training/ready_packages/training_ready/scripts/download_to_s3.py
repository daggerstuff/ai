#!/usr/bin/env python3
"""
Download Datasets Directly to S3

Downloads datasets from HuggingFace, Kaggle, and URLs directly to S3,
bypassing local storage and VPS.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. Install with: uv pip install boto3")

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("datasets not available. Install with: uv pip install datasets")

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("kaggle not available. Install with: uv pip install kaggle")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DatasetDownloader:
    """Downloads datasets directly to S3"""

    def __init__(self, s3_bucket: str, s3_prefix: str = "datasets/", catalog_path: Optional[Path] = None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/") + "/"
        self.catalog_path = catalog_path

        # Initialize S3 client
        if BOTO3_AVAILABLE:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
            logger.error("boto3 not available. Cannot upload to S3.")

        self.results = []

    def upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload file to S3"""
        if not self.s3_client:
            return False

        try:
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info(f"  ✅ Uploaded to s3://{self.s3_bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"  ❌ Failed to upload {local_path}: {e}")
            return False

    def download_hf_to_s3(self, dataset_name: str, config: Optional[str] = None, s3_key: Optional[str] = None) -> Dict[str, Any]:
        """Download HuggingFace dataset directly to S3"""
        if not HF_AVAILABLE:
            return {
                "success": False,
                "error": "HuggingFace datasets library not installed",
            }

        if not self.s3_client:
            return {
                "success": False,
                "error": "S3 client not available",
            }

        try:
            logger.info(f"📥 Downloading HF dataset: {dataset_name}")

            # Load dataset
            dataset = load_dataset(dataset_name, config)

            # Determine S3 key
            if not s3_key:
                s3_key = f"{self.s3_prefix}huggingface/{dataset_name.replace('/', '_')}.jsonl"

            # Convert to JSONL and upload in chunks
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
                tmp_path = Path(tmp.name)

                # Write dataset to temp file
                if isinstance(dataset, dict):
                    # Multiple splits
                    for split_name, split_data in dataset.items():
                        split_data.to_json(tmp_path)
                else:
                    dataset.to_json(tmp_path)

                # Upload to S3
                success = self.upload_to_s3(tmp_path, s3_key)

                # Cleanup
                tmp_path.unlink()

                if success:
                    return {
                        "success": True,
                        "s3_path": f"s3://{self.s3_bucket}/{s3_key}",
                        "dataset_name": dataset_name,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Upload to S3 failed",
                    }
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def download_kaggle_to_s3(self, dataset_name: str, s3_key: Optional[str] = None) -> Dict[str, Any]:
        """Download Kaggle dataset directly to S3"""
        if not KAGGLE_AVAILABLE:
            return {
                "success": False,
                "error": "Kaggle library not available",
            }

        if not self.s3_client:
            return {
                "success": False,
                "error": "S3 client not available",
            }

        try:
            logger.info(f"📥 Downloading Kaggle dataset: {dataset_name}")

            # Download from Kaggle
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                kaggle.api.dataset_download_files(dataset_name, path=tmpdir, unzip=True)

                # Upload all files to S3
                tmp_path = Path(tmpdir)
                uploaded_files = []

                for file_path in tmp_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(tmp_path)
                        s3_file_key = f"{self.s3_prefix}kaggle/{dataset_name.replace('/', '_')}/{relative_path}"

                        if self.upload_to_s3(file_path, s3_file_key):
                            uploaded_files.append(f"s3://{self.s3_bucket}/{s3_file_key}")

                if uploaded_files:
                    return {
                        "success": True,
                        "s3_paths": uploaded_files,
                        "dataset_name": dataset_name,
                    }
                else:
                    return {
                        "success": False,
                        "error": "No files uploaded",
                    }
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def download_url_to_s3(self, url: str, s3_key: Optional[str] = None) -> Dict[str, Any]:
        """Download from URL directly to S3"""
        if not self.s3_client:
            return {
                "success": False,
                "error": "S3 client not available",
            }

        try:
            import requests
            from urllib.parse import urlparse

            logger.info(f"📥 Downloading from URL: {url}")

            # Determine S3 key
            if not s3_key:
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path) or "downloaded_file"
                s3_key = f"{self.s3_prefix}urls/{filename}"

            # Download to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = Path(tmp.name)

                response = requests.get(url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)

                # Upload to S3
                success = self.upload_to_s3(tmp_path, s3_key)

                # Cleanup
                tmp_path.unlink()

                if success:
                    return {
                        "success": True,
                        "s3_path": f"s3://{self.s3_bucket}/{s3_key}",
                        "url": url,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Upload to S3 failed",
                    }
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def process_catalog(self) -> Dict[str, Any]:
        """Process accessibility catalog and download to S3"""
        if not self.catalog_path or not self.catalog_path.exists():
            logger.error(f"Catalog not found: {self.catalog_path}")
            return {"error": "Catalog not found"}

        with open(self.catalog_path, "r") as f:
            catalog = json.load(f)

        results = {
            "huggingface": [],
            "kaggle": [],
            "url": [],
        }

        # Process HuggingFace datasets
        for dataset in catalog.get("catalog", {}).get("huggingface", []):
            result = self.download_hf_to_s3(dataset.get("name", ""))
            results["huggingface"].append({
                "dataset": dataset,
                "result": result,
            })

        # Process Kaggle datasets
        for dataset in catalog.get("catalog", {}).get("kaggle", []):
            result = self.download_kaggle_to_s3(dataset.get("name", ""))
            results["kaggle"].append({
                "dataset": dataset,
                "result": result,
            })

        # Process URL datasets
        for dataset in catalog.get("catalog", {}).get("url", []):
            result = self.download_url_to_s3(dataset.get("path", ""))
            results["url"].append({
                "dataset": dataset,
                "result": result,
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "results": results,
        }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets directly to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="datasets/", help="S3 prefix (default: datasets/)")
    parser.add_argument("--catalog", help="Path to accessibility catalog JSON")
    parser.add_argument("--hf-dataset", help="Download specific HuggingFace dataset")
    parser.add_argument("--kaggle-dataset", help="Download specific Kaggle dataset")
    parser.add_argument("--url", help="Download from specific URL")

    args = parser.parse_args()

    # Check for AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        logger.error("❌ AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return 1

    downloader = S3DatasetDownloader(
        s3_bucket=args.bucket,
        s3_prefix=args.prefix,
        catalog_path=Path(args.catalog) if args.catalog else None,
    )

    if args.hf_dataset:
        result = downloader.download_hf_to_s3(args.hf_dataset)
        logger.info(f"Result: {result}")
    elif args.kaggle_dataset:
        result = downloader.download_kaggle_to_s3(args.kaggle_dataset)
        logger.info(f"Result: {result}")
    elif args.url:
        result = downloader.download_url_to_s3(args.url)
        logger.info(f"Result: {result}")
    elif args.catalog:
        result = downloader.process_catalog()
        output_path = Path("s3_download_results.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"📊 Results saved to: {output_path}")
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


