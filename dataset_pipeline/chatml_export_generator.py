#!/usr/bin/env python3
"""
ChatML Export Generator for Release 0

Streaming exporter that reads source JSONL directly from S3 line-by-line and
writes ChatML JSONL shards back to S3 via multipart uploads (no local files).
Security: credentials must come from environment, and logs avoid content/PII.
"""

import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


def load_env_file(path: Path) -> Dict[str, str]:
    """Minimal .env loader that ignores comments/blank lines without logging values."""
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


class S3MultipartWriter:
    """Multipart JSONL writer for S3. Buffers bytes and uploads parts >=5MB."""

    def __init__(
        self,
        s3: BaseClient,
        bucket: str,
        key: str,
        part_size_bytes: int = 8 * 1024 * 1024,
    ):
        part_size_bytes = max(part_size_bytes, 5 * 1024 * 1024)
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        self.part_size = part_size_bytes
        self._buffer = bytearray()
        self._parts = []
        self._upload_id = self.s3.create_multipart_upload(
            Bucket=bucket, Key=key, ContentType="application/x-ndjson"
        )["UploadId"]
        self._part_number = 1

    def write_line(self, line: str) -> None:
        self._buffer.extend(line.encode("utf-8"))
        self._buffer.extend(b"\n")
        if len(self._buffer) >= self.part_size:
            self._flush_part()

    def _flush_part(self) -> None:
        if not self._buffer:
            return
        body = bytes(self._buffer)
        self._buffer.clear()
        resp = self.s3.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            PartNumber=self._part_number,
            UploadId=self._upload_id,
            Body=body,
        )
        self._parts.append({"ETag": resp["ETag"], "PartNumber": self._part_number})
        self._part_number += 1

    def close(self) -> bool:
        if self._buffer:
            self._flush_part()
        if not self._parts:
            # No data written; abort multipart to avoid malformed XML
            self.s3.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key, UploadId=self._upload_id
            )
            return False
        self.s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self._upload_id,
            MultipartUpload={"Parts": self._parts},
        )
        return True


class ChatMLExportGenerator:
    """Generate ChatML exports from Release 0 consolidated datasets"""

    def __init__(
        self,
        s3_endpoint: str,
        access_key: Optional[str],
        secret_key: Optional[str],
        bucket: str,
    ):
        # Use explicit creds if provided, else fall back to default provider chain
        if access_key and secret_key:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",
            )
        else:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                region_name="us-east-1",
            )
        self.bucket = bucket

    def load_manifest(self, manifest_key: str) -> Dict[str, Any]:
        """Load the unified manifest from S3"""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=manifest_key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def download_dataset(self, s3_key: str) -> List[Dict[str, Any]]:
        """Download and parse a JSONL dataset from S3 with streaming for large files"""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)

        # Stream large JSONL files line by line
        if s3_key.endswith(".jsonl"):
            return self._extracted_from_download_dataset_7(response)
        # For small JSON files, download fully
        data = response["Body"].read().decode("utf-8")
        parsed = json.loads(data)

        # Handle different JSON structures
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "conversations" in parsed:
            return parsed["conversations"]
        if isinstance(parsed, dict) and "data" in parsed:
            return parsed["data"]
        return [parsed]

    # TODO Rename this here and in `download_dataset`
    def _extracted_from_download_dataset_7(self, response):
        print("  Streaming JSONL file (large dataset)...")
        conversations = []
        buffer = ""
        chunk_size = 1024 * 1024  # 1MB chunks

        for chunk in response["Body"].iter_chunks(chunk_size=chunk_size):
            buffer += chunk.decode("utf-8")
            lines = buffer.split("\n")

            # Process all complete lines
            for line in lines[:-1]:
                if line.strip():
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipping malformed JSON line: {e}")

            # Keep incomplete line in buffer
            buffer = lines[-1]

            # Show progress
            if len(conversations) % 1000 == 0 and conversations:
                print(f"  Loaded {len(conversations)} conversations...")

            # Process final buffer
        if buffer.strip():
            with contextlib.suppress(json.JSONDecodeError):
                conversations.append(json.loads(buffer))
        return conversations

    def stream_dataset_lines(self, s3_key: str):
        """Yield JSON objects from S3 without full download.

        JSONL preferred.
        """
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        if s3_key.endswith(".jsonl"):
            for raw in obj["Body"].iter_lines(chunk_size=1024 * 1024):
                if not raw:
                    continue
                try:
                    yield json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
            return

        # Fallback for JSON
        data = obj["Body"].read()
        try:
            parsed = json.loads(data)
        except Exception:
            return
        if isinstance(parsed, list):
            yield from parsed
        elif isinstance(parsed, dict) and "conversations" in parsed:
            yield from parsed["conversations"]
        elif isinstance(parsed, dict) and "data" in parsed:
            yield from parsed["data"]
        else:
            yield parsed

    def convert_to_chatml(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a conversation to ChatML format"""
        messages = []

        # Handle different conversation formats
        if "messages" in conversation:
            # Already in messages format
            messages.extend(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in conversation["messages"]
            )
        elif "prompt" in conversation and "response" in conversation:
            # Prompt-response format
            messages = [
                {"role": "user", "content": conversation["prompt"]},
                {"role": "assistant", "content": conversation["response"]},
            ]
        elif "input" in conversation and "output" in conversation:
            # Input-output format
            messages = [
                {"role": "user", "content": conversation["input"]},
                {"role": "assistant", "content": conversation["output"]},
            ]
        elif "question" in conversation and "answer" in conversation:
            # QA format
            messages = [
                {"role": "user", "content": conversation["question"]},
                {"role": "assistant", "content": conversation["answer"]},
            ]
        elif "conversations" in conversation:
            # Nested conversations format
            messages.extend(
                {"role": turn.get("from", "user"), "content": turn.get("value", "")}
                for turn in conversation["conversations"]
            )
        elif "conversation" in conversation and isinstance(
            conversation.get("conversation"), list
        ):
            # Alternate nested key name
            messages.extend(
                {
                    "role": turn.get("role") or turn.get("from", "user"),
                    "content": turn.get("content") or turn.get("value", ""),
                }
                for turn in conversation["conversation"]
            )
        else:
            # Unknown format - log and skip
            print(
                f"Warning: Unknown conversation format: {list(conversation.keys())[:5]}"
            )
            return None

        return {
            "messages": messages,
            "metadata": {
                "source": conversation.get("source", "unknown"),
                "dataset_family": conversation.get("family", "unknown"),
                "quality": conversation.get("quality", "standard"),
                "exported_at": datetime.now(datetime.UTC).isoformat(),
            },
        }

    def export_family_streaming(
        self,
        family_name: str,
        datasets: List[Dict[str, Any]],
        output_prefix: str,
        shard_lines: int = 250_000,
        part_size_bytes: int = 8 * 1024 * 1024,
        max_per_dataset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stream a dataset family to ChatML JSONL directly to S3 with sharding."""
        stats = {
            "datasets_processed": 0,
            "total_conversations": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "shards_written": 0,
            "output_keys": [],
        }

        shard_idx = 1
        lines_in_shard = 0
        writer: Optional[S3MultipartWriter] = None
        current_key: Optional[str] = None

        def finalize_shard() -> None:
            nonlocal writer, lines_in_shard, current_key
            if writer is None:
                return
            if writer.close():
                stats["shards_written"] += 1
                if current_key:
                    stats["output_keys"].append(current_key)
            writer = None
            current_key = None
            lines_in_shard = 0

        def start_new_shard() -> None:
            nonlocal writer, shard_idx, lines_in_shard, current_key
            finalize_shard()
            current_key = (
                f"{output_prefix}/{family_name}_chatml-shard-{shard_idx:05d}.jsonl"
            )
            writer = S3MultipartWriter(
                self.s3_client, self.bucket, current_key, part_size_bytes
            )
            lines_in_shard = 0

        start_new_shard()

        for dataset_info in datasets:
            s3_key = dataset_info.get("key")
            if not s3_key:
                continue
            print(f"Processing: {s3_key}")
            stats["datasets_processed"] += 1
            processed_this_dataset = 0

            try:
                for record in self.stream_dataset_lines(s3_key):
                    stats["total_conversations"] += 1
                    processed_this_dataset += 1
                    chatml_conv = self.convert_to_chatml(record)
                    if chatml_conv is None:
                        stats["failed_conversions"] += 1
                    else:
                        chatml_conv.setdefault("metadata", {})
                        chatml_conv["metadata"]["family"] = family_name
                        chatml_conv["metadata"]["source_file"] = s3_key
                        writer.write_line(json.dumps(chatml_conv, ensure_ascii=False))
                        lines_in_shard += 1
                        stats["successful_conversions"] += 1

                    if max_per_dataset and processed_this_dataset >= max_per_dataset:
                        break

                    if lines_in_shard >= shard_lines:
                        shard_idx += 1
                        start_new_shard()

            except ClientError as e:
                print(
                    f"Warning: Failed to stream {s3_key}: "
                    f"{e.response.get('Error', {}).get('Code', 'Unknown')}"
                )
                continue

        finalize_shard()

        print(
            f"Exported {family_name}: {stats['successful_conversions']} conversations "
            f"across {stats['shards_written']} shard(s)"
        )
        return stats

    def generate_release_0_export(
        self, manifest_key: str, output_prefix: str
    ) -> Dict[str, Any]:
        """Generate complete Release 0 ChatML export directly to S3 (streaming)."""
        print(f"Loading manifest from {manifest_key}...")
        manifest = self.load_manifest(manifest_key)

        families = manifest["dataset_families"]
        total_stats: Dict[str, Any] = {}

        max_per_dataset_env = os.getenv("CHATML_MAX_PER_DATASET")
        max_per_dataset = int(max_per_dataset_env) if max_per_dataset_env else None

        max_workers_env = os.getenv("CHATML_MAX_WORKERS")
        max_workers = int(max_workers_env) if max_workers_env else 1
        if max_workers < 1:
            max_workers = 1

        # Collect eligible families first
        eligible_families: List[tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
        for family_name, family_data in families.items():
            if not family_data["status"].startswith("✅"):
                continue
            all_datasets: List[Dict[str, Any]] = []
            for _, datasets in family_data.get("datasets", {}).items():
                if isinstance(datasets, list):
                    all_datasets.extend(datasets)
            if all_datasets:
                eligible_families.append((family_name, family_data, all_datasets))

        print(f"Concurrency: families max_workers={max_workers}")

        def run_family(family_name: str, datasets: List[Dict[str, Any]]):
            print(f"\nExporting {family_name} (streaming)...")
            return family_name, self.export_family_streaming(
                family_name=family_name,
                datasets=datasets,
                output_prefix=output_prefix,
                shard_lines=250_000,
                part_size_bytes=8 * 1024 * 1024,
                max_per_dataset=max_per_dataset,
            )

        if max_workers == 1:
            for family_name, _, datasets in eligible_families:
                name, stats = run_family(family_name, datasets)
                total_stats[name] = stats
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(run_family, fam_name, datasets)
                    for fam_name, _, datasets in eligible_families
                ]
                for fut in as_completed(futures):
                    name, stats = fut.result()
                    total_stats[name] = stats

        # Write summary JSON to S3
        summary_key = f"{output_prefix}/release_0_export_summary.json"
        summary_doc = json.dumps(
            {
                "generated_at": datetime.now(datetime.UTC).isoformat(),
                "manifest_source": manifest_key,
                "families": total_stats,
                "total_conversations": sum(
                    s.get("successful_conversions", 0) for s in total_stats.values()
                ),
            },
            indent=2,
        ).encode("utf-8")
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=summary_key,
            Body=summary_doc,
            ContentType="application/json",
        )
        print(f"\n✅ Export complete! Summary: s3://{self.bucket}/{summary_key}")
        return total_stats


def main():
    """Main execution"""

    # Load env files if present (root .env and ai/.env) without logging values
    root_env = load_env_file(Path(__file__).resolve().parents[2] / ".env")
    ai_env = load_env_file(Path(__file__).resolve().parents[1] / ".env")

    def env_get(key: str) -> Optional[str]:
        return os.getenv(key) or root_env.get(key) or ai_env.get(key)

    # S3 configuration (env only; no hardcoded secrets). Support OVH vars as fallback.
    S3_ENDPOINT = (
        env_get("AWS_S3_ENDPOINT")
        or env_get("OVH_S3_ENDPOINT")
        or "https://s3.us-east-va.io.cloud.ovh.us"
    )
    ACCESS_KEY = env_get("AWS_ACCESS_KEY_ID") or env_get("OVH_S3_ACCESS_KEY")
    SECRET_KEY = env_get("AWS_SECRET_ACCESS_KEY") or env_get("OVH_S3_SECRET_KEY")
    BUCKET = env_get("AWS_S3_BUCKET") or env_get("OVH_S3_BUCKET") or "pixel-data"
    MANIFEST_KEY = os.getenv(
        "RELEASE_MANIFEST_KEY", "releases/v2026-01-07/RELEASE_0_UNIFIED_MANIFEST.json"
    )

    # If env credentials are missing, boto3 default chain may still provide creds

    # Output S3 prefix for ChatML shards
    OUTPUT_PREFIX = os.getenv("CHATML_OUTPUT_PREFIX", "releases/v2026-01-07/chatml")

    # Optional: limit per dataset for smoke testing
    if os.getenv("CHATML_MAX_PER_DATASET"):
        limit_str = os.getenv("CHATML_MAX_PER_DATASET")
        print(f"Smoke test active: limiting per-dataset to {limit_str} records")

    # Generate export (streaming → direct to S3)
    generator = ChatMLExportGenerator(S3_ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET)
    generator.generate_release_0_export(MANIFEST_KEY, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
