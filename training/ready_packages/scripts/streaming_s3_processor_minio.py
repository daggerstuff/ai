#!/usr/bin/env python3
"""
Streaming S3 Dataset Processor - MinIO client for OVH S3
"""

import json
import logging
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
import requests
import os
import tempfile
from typing import Iterator, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MinIOS3Processor:
    """
    Stream-processes dataset using HTTP requests for OVH S3
    """

    def __init__(
        self,
        bucket: str = "pixel-data",
        endpoint: str = "https://s3.us-east-va.io.cloud.ovh.us",
        access_key: str = None,
        secret_key: str = None,
    ):
        self.bucket = bucket
        self.endpoint = endpoint.rstrip("/")
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided"
            )

    def list_objects(self, prefix: str = "") -> list:
        """List objects in bucket using S3 REST API"""
        url = f"{self.endpoint}/{self.bucket}"
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}

        try:
            response = requests.get(
                url, params=params, auth=(self.access_key, self.secret_key)
            )
            response.raise_for_status()

            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.content)

            objects = []
            for contents in root.findall(
                ".//{http://s3.amazonaws.com/doc/2006-03-01/}Contents"
            ):
                key = contents.find("{http://s3.amazonaws.com/doc/2006-03-01/}Key").text
                size = int(
                    contents.find("{http://s3.amazonaws.com/doc/2006-03-01/}Size").text
                )

                if any(
                    key.endswith(ext) for ext in [".json", ".jsonl", ".csv", ".txt"]
                ):
                    objects.append({"key": key, "size": size})

            return objects

        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []

    def get_object(self, key: str) -> str:
        """Get object content using HTTP"""
        url = f"{self.endpoint}/{self.bucket}/{key}"

        try:
            response = requests.get(url, auth=(self.access_key, self.secret_key))
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return ""

    def discover_datasets(self) -> dict:
        """Discover all dataset files"""
        prefixes = ["datasets/", "training/", "conversations/", "", "therapeutic/"]
        all_files = []

        for prefix in prefixes:
            files = self.list_objects(prefix)
            all_files.extend(files)

        # Remove duplicates and sort by size
        seen = set()
        unique_files = []
        for f in all_files:
            if f["key"] not in seen:
                seen.add(f["key"])
                unique_files.append(f)

        return {
            "files": sorted(unique_files, key=lambda x: x["size"], reverse=True),
            "total_files": len(unique_files),
            "total_size": sum(f["size"] for f in unique_files),
        }

    def process_datasets(self) -> None:
        """Discover and report on datasets"""
        print("🔍 Discovering datasets in OVH S3...")

        result = self.discover_datasets()

        if not result["files"]:
            print("❌ No datasets found")
            return

        print(f"📊 Found {result['total_files']} files")
        print(f"📏 Total size: {result['total_size'] / 1024**3:.2f}GB")

        print("\n🗂️  Top 10 largest files:")
        for i, file_info in enumerate(result["files"][:10], 1):
            print(f"   {i}. {file_info['key']}: {file_info['size'] / 1024**3:.2f}GB")

        # Save discovery report
        report_path = Path("training_ready/data/s3_discovery_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n📋 Report saved: {report_path}")


def main():
    """Main function"""
    try:
        processor = MinIOS3Processor()
        processor.process_datasets()

    except ValueError as e:
        print(f"❌ {e}")
        print("🔧 Set credentials:")
        print("   export AWS_ACCESS_KEY_ID=your-key")
        print("   export AWS_SECRET_ACCESS_KEY=your-secret")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
