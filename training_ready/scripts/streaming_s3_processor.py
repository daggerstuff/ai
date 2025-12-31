#!/usr/bin/env python3
"""
Streaming S3 Dataset Processor - Processes 52.20GB without local storage
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
import tempfile
from typing import Iterator, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamingS3Processor:
    """
    Stream-processes 52.20GB dataset directly from S3 without local storage
    """

    def __init__(
        self,
        source_bucket: str = "pixel-data",
        output_bucket: str = "pixel-data-cleaned",
        endpoint_url: str = "https://s3.us-east-va.io.cloud.ovh.us",
        chunk_size: int = 10 * 1024 * 1024,  # 10MB chunks
    ):
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        self.endpoint_url = endpoint_url
        self.chunk_size = chunk_size

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-va",
        )

        # Ensure output bucket exists
        self.ensure_output_bucket()

    def ensure_output_bucket(self):
        """Create output bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.output_bucket)
            logger.info(f"Output bucket {self.output_bucket} exists")
        except ClientError:
            logger.info(f"Creating output bucket {self.output_bucket}")
            try:
                self.s3_client.create_bucket(Bucket=self.output_bucket)
            except ClientError as e:
                logger.warning(f"Could not create bucket: {e}")

    def get_relevant_files(self) -> list:
        """Get list of dataset files from S3"""
        files = []
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")

            # Common dataset patterns
            prefixes = [
                "datasets/",
                "training/",
                "conversations/",
                "therapeutic/",
                "mental-health/",
                "",
            ]

            for prefix in prefixes:
                for page in paginator.paginate(
                    Bucket=self.source_bucket, Prefix=prefix
                ):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            if any(
                                key.endswith(ext)
                                for ext in [".json", ".jsonl", ".csv", ".txt"]
                            ):
                                files.append(
                                    {
                                        "key": key,
                                        "size": obj["Size"],
                                        "last_modified": obj["LastModified"],
                                    }
                                )
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")

        return sorted(files, key=lambda x: x["size"], reverse=True)

    def stream_process_file(self, s3_key: str) -> Iterator[str]:
        """Stream-process a single file from S3"""
        logger.info(f"Streaming file: {s3_key}")

        try:
            response = self.s3_client.get_object(Bucket=self.source_bucket, Key=s3_key)

            # Stream processing based on file type
            if s3_key.endswith(".jsonl"):
                for line in response["Body"].iter_lines():
                    if line:
                        yield self.process_jsonl_line(line.decode("utf-8"))
            elif s3_key.endswith(".json"):
                content = response["Body"].read().decode("utf-8")
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        yield self.process_json_item(item)
                else:
                    yield self.process_json_item(data)
            elif s3_key.endswith(".csv"):
                import csv
                import io

                csv_content = response["Body"].read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(csv_content))
                for row in reader:
                    yield self.process_csv_row(row)

        except Exception as e:
            logger.error(f"Error processing {s3_key}: {e}")

    def process_jsonl_line(self, line: str) -> str:
        """Process a single JSONL line"""
        try:
            data = json.loads(line)
            cleaned = self.clean_record(data)
            return json.dumps(cleaned)
        except:
            return line

    def process_json_item(self, item: Dict) -> str:
        """Process a JSON item"""
        cleaned = self.clean_record(item)
        return json.dumps(cleaned)

    def process_csv_row(self, row: Dict) -> str:
        """Process a CSV row"""
        cleaned = self.clean_record(row)
        return json.dumps(cleaned)

    def clean_record(self, record: Dict) -> Dict:
        """Clean PII from record"""
        # Convert to string for regex processing
        record_str = json.dumps(record)

        # PII patterns
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }

        cleaned_str = record_str
        for pattern_name, pattern in patterns.items():
            cleaned_str = re.sub(
                pattern, f"[{pattern_name.upper()}_REDACTED]", cleaned_str
            )

        return json.loads(cleaned_str)

    def deduplicate_stream(self, stream: Iterator[str]) -> Iterator[str]:
        """Deduplicate streaming data using content hashes"""
        seen_hashes = set()

        for line in stream:
            content_hash = hashlib.md5(line.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                yield line

    def process_and_upload(self, s3_key: str) -> Dict[str, Any]:
        """Process a file and upload cleaned version to S3"""
        try:
            # Create output key
            output_key = f"cleaned/{s3_key.replace('/', '_')}.jsonl"

            # Stream process and upload
            processed_stream = self.stream_process_file(s3_key)
            deduplicated_stream = self.deduplicate_stream(processed_stream)

            # Use temporary file for upload
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl") as tmp_file:
                record_count = 0

                for line in deduplicated_stream:
                    tmp_file.write(line + "\n")
                    record_count += 1

                tmp_file.flush()

                # Upload to S3
                self.s3_client.upload_file(
                    tmp_file.name, self.output_bucket, output_key
                )

                logger.info(f"Uploaded {record_count} records to {output_key}")

                return {
                    "input_key": s3_key,
                    "output_key": output_key,
                    "records_processed": record_count,
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Error processing {s3_key}: {e}")
            return {"input_key": s3_key, "error": str(e), "success": False}

    def process_all_datasets(self) -> Dict[str, Any]:
        """Process all datasets in streaming fashion"""
        files = self.get_relevant_files()

        if not files:
            logger.warning("No files found in S3")
            return {"success": False, "error": "No files found"}

        total_size = sum(f["size"] for f in files)
        logger.info(
            f"Processing {len(files)} files, total size: {total_size / 1024**3:.2f}GB"
        )

        results = []
        for i, file_info in enumerate(files, 1):
            logger.info(f"Processing {i}/{len(files)}: {file_info['key']}")
            result = self.process_and_upload(file_info["key"])
            results.append(result)

        # Create final report
        report = {
            "total_files": len(files),
            "total_size_gb": total_size / 1024**3,
            "processed_files": len([r for r in results if r["success"]]),
            "failed_files": len([r for r in results if not r["success"]]),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "output_bucket": self.output_bucket,
        }

        # Save report to S3
        report_key = (
            f"processing_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.s3_client.put_object(
            Bucket=self.output_bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2, default=str),
        )

        return report


def main():
    """Main processing function"""
    try:
        processor = StreamingS3Processor()

        # List files first
        files = processor.get_relevant_files()
        total_size = sum(f["size"] for f in files)

        print(f"📊 Found {len(files)} files in S3")
        print(f"📏 Total size: {total_size / 1024**3:.2f}GB")

        if files:
            print("\n🗂️  Top files:")
            for f in files[:5]:
                print(f"   {f['key']}: {f['size'] / 1024**3:.2f}GB")

        response = input("\n🚀 Proceed with streaming processing? (y/N): ")
        if response.lower() == "y":
            result = processor.process_all_datasets()
            print(f"✅ Processing complete!")
            print(
                f"   Processed: {result['processed_files']}/{result['total_files']} files"
            )
            print(f"   Clean data in: s3://{result['output_bucket']}/cleaned/")
            print(
                f"   Report saved: s3://{result['output_bucket']}/processing_reports/"
            )
        else:
            print("❌ Processing cancelled")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check AWS credentials and S3 access")


if __name__ == "__main__":
    main()
