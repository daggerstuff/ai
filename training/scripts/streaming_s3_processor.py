#!/usr/bin/env python3
"""
Streaming S3 Dataset Processor - Processes 52.20GB without local storage
Integrated with Ollama LLM Judge for Clinical & Bias Gating.
"""

import csv
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import requests
import itertools
import concurrent.futures
from collections.abc import Iterator
from datetime import timezone, datetime
from typing import Any

from botocore.exceptions import ClientError

from scripts.rclone_boto3_shim import get_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StreamingS3Processor:
    """
    Stream-processes 52.20GB dataset directly from S3 without local storage
    """

    def __init__(
        self,
        source_bucket: str = "pixeldata",
        output_bucket: str = "pixeldata-cleaned",
        endpoint_url: str | None = None,
        chunk_size: int = 10 * 1024 * 1024,  # 10MB chunks
    ):
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        # default to Digital Ocean (bucket name should NOT be in endpoint)
        endpoint = endpoint_url or os.environ.get("AWS_S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com")
        # Remove bucket name from endpoint if present
        if "pixel-data" in endpoint:
            endpoint = endpoint.replace("pixel-data.", "")
        self.endpoint_url = endpoint
        self.chunk_size = chunk_size

        # Initialize S3 client
        self.s3_client = get_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("RCLONE_HETZNER_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("RCLONE_HETZNER_SECRET_ACCESS_KEY"),
            region_name="hel1", # Changed to hel1 based on user's hetzner env
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
                "raw_sources/",
                "datasets/",
                "training/",
                "conversations/",
                "therapeutic/",
                "mental-health/",
                "",
            ]

            for prefix in prefixes:
                for page in paginator.paginate(Bucket=self.source_bucket, Prefix=prefix):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            if any(key.endswith(ext) for ext in [".json", ".jsonl", ".csv", ".txt"]):
                                files.append(
                                    {
                                        "key": key,
                                        "size": obj["Size"],
                                        "last_modified": obj["LastModified"],
                                    }
                                )
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")

        # Deduplicate keys
        unique_files = {f["key"]: f for f in files}.values()
        return sorted(list(unique_files), key=lambda x: x["size"], reverse=True)

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
        except Exception:
            return line

    def process_json_item(self, item: dict) -> str:
        """Process a JSON item"""
        cleaned = self.clean_record(item)
        return json.dumps(cleaned)

    def process_csv_row(self, row: dict) -> str:
        """Process a CSV row"""
        cleaned = self.clean_record(row)
        return json.dumps(cleaned)

    def clean_record(self, record: dict) -> dict:
        """Clean PII from record"""
        record_str = json.dumps(record)

        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }

        cleaned_str = record_str
        for pattern_name, pattern in patterns.items():
            cleaned_str = re.sub(pattern, f"[{pattern_name.upper()}_REDACTED]", cleaned_str)

        return json.loads(cleaned_str)

    def deduplicate_stream(self, stream: Iterator[str]) -> Iterator[str]:
        """Deduplicate streaming data using content hashes"""
        seen_hashes = set()

        for line in stream:
            content_hash = hashlib.md5(line.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                yield line

    def judge_record(self, record_str: str) -> bool:
        """Evaluate a record using local Ollama model for Clinical Validity and Bias."""
        prompt = (
            "You are an expert clinical supervisor evaluating a transcript or journal entry. "
            "Analyze the content for clinical safety, coherence, and BIAS (e.g. cultural, gender, or racial stereotyping). "
            "If the content is clinically safe, coherent, and free of harmful bias, score it a 4 or 5. "
            "If there are boundary violations, dangerous advice, severe hallucinations, or biased assumptions, score it 1 to 3.\n\n"
            "Output ONLY the integer score.\n\nContent:\n" + record_str[:1500]
        )
        try:
            payload = {
                "model": "ornith:9b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            resp = requests.post("https://ollama.pixelated.love/v1/chat/completions", json=payload, headers={"Authorization": "Bearer dummy"}, timeout=60)
            resp.raise_for_status()
            score_str = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\d', score_str)
            if match:
                return int(match.group(0)) >= 4
        except requests.exceptions.RequestException as e:
            logger.error(f"FATAL: Ollama server is down or timing out! {e}")
            raise
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        # Default fail if LLM returns invalid format
        return False

    def judge_stream(self, stream: Iterator[str]) -> Iterator[str]:
        """Apply LLM judge to stream concurrently to prevent extreme slowdowns."""
        def batch_iterator(iterable, size):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, size))
                if not chunk:
                    return
                yield chunk

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for batch in batch_iterator(stream, 50):
                # Map judgments concurrently
                results = list(executor.map(self.judge_record, batch))
                for record_str, passed in zip(batch, results):
                    if passed:
                        yield record_str

    def process_and_upload(self, s3_key: str) -> dict[str, Any]:
        """Process a file and stream cleaned version directly to S3"""
        try:
            output_key = f"judged_and_cleaned/{s3_key.replace('/', '_')}.jsonl"

            # 1. Stream from S3
            # 2. PII Regex Scrubber
            processed_stream = self.stream_process_file(s3_key)
            # 3. Memory-efficient MD5 Deduplication
            deduplicated_stream = self.deduplicate_stream(processed_stream)
            # 4. LLM-as-a-judge (Clinical & Bias Gating)
            judged_stream = self.judge_stream(deduplicated_stream)

            try:
                first_line = next(judged_stream)
            except StopIteration:
                logger.info(f"Dropped all records from {s3_key} (failed judge). No file uploaded.")
                return {
                    "input_key": s3_key,
                    "output_key": "DROPPED",
                    "records_processed": 0,
                    "success": True,
                }

            # State object to keep track of counts inside the generator
            state = {"count": 1}

            def final_stream():
                yield first_line + "\n"
                for line in judged_stream:
                    state["count"] += 1
                    yield line + "\n"

            # Stream directly to S3 via rclone rcat (0 bytes written to disk)
            self.s3_client.put_object(Bucket=self.output_bucket, Key=output_key, Body=final_stream())
            logger.info(f"Uploaded {state['count']} pristine records to {output_key}")

            return {
                "input_key": s3_key,
                "output_key": output_key,
                "records_processed": state["count"],
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error processing {s3_key}: {e}")
            return {"input_key": s3_key, "error": str(e), "success": False}

    def get_completed_files(self) -> set:
        """Get set of already processed output keys"""
        completed = set()
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.output_bucket, Prefix="judged_and_cleaned/"):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Size"] > 0: # Ensure it's not an empty/failed partial file
                            completed.add(obj["Key"])
        except Exception as e:
            logger.error(f"Error checking completed files: {e}")
        return completed

    def process_all_datasets(self) -> dict[str, Any]:
        """Process all datasets in streaming fashion"""
        files = self.get_relevant_files()

        if not files:
            logger.warning("No files found in S3")
            return {"success": False, "error": "No files found"}

        completed_files = self.get_completed_files()
        logger.info(f"Found {len(completed_files)} previously completed files to skip")

        total_size = sum(f["size"] for f in files)
        logger.info(f"Processing {len(files)} files, total size: {total_size / 1024**3:.2f}GB")

        results = []
        for i, file_info in enumerate(files, 1):
            s3_key = file_info["key"]
            output_key = f"judged_and_cleaned/{s3_key.replace('/', '_')}.jsonl"

            if output_key in completed_files:
                logger.info(f"Skipping {i}/{len(files)}: {s3_key} (Already processed)")
                results.append({
                    "input_key": s3_key,
                    "output_key": output_key,
                    "records_processed": "SKIPPED",
                    "success": True,
                })
                continue

            logger.info(f"Processing {i}/{len(files)}: {s3_key}")
            result = self.process_and_upload(s3_key)
            results.append(result)

        # Create final report
        report = {
            "total_files": len(files),
            "total_size_gb": total_size / 1024**3,
            "processed_files": len([r for r in results if r["success"]]),
            "failed_files": len([r for r in results if not r["success"]]),
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_bucket": self.output_bucket,
        }

        report_key = f"processing_reports/report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        self.s3_client.put_object(
            Bucket=self.output_bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2, default=str),
        )

        return report

def main():
    try:
        import sys
        # Auto-proceed if --yes flag is passed for non-interactive
        auto_proceed = "--yes" in sys.argv
        
        processor = StreamingS3Processor()
        files = processor.get_relevant_files()
        total_size = sum(f["size"] for f in files)

        logger.info(f"Found {len(files)} files in S3")
        logger.info(f"Total size: {total_size / 1024**3:.2f}GB")

        if not auto_proceed:
            response = input("\n🚀 Proceed with LLM-Gated streaming processing? (y/N): ")
        else:
            response = "y"
            
        if response.lower() == "y":
            result = processor.process_all_datasets()
            logger.info("Processing complete!")
            logger.info(f"   Clean data in: s3://{result['output_bucket']}/judged_and_cleaned/")
        else:
            logger.info("Processing cancelled")

    except Exception as e:
        logger.info(f"Error: {e}")

if __name__ == "__main__":
    main()
