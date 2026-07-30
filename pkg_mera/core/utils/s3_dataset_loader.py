"""S3 dataset loader compatibility layer used across scripts and tests.

The implementation is intentionally conservative:
- It validates expected credentials like the legacy contract.
- It supports direct local file fallback when tests/scripts provide local paths.
- It optionally uses ``boto3`` when available for real S3 interactions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import boto3
    import botocore
except ImportError:
    boto3 = None
    botocore = None


_CREDENTIAL_ENV_KEYS = (
    "HETZNER_S3_ACCESS_KEY",
    "HETZNER_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "HETZNER_S3_SECRET_KEY",
    "HETZNER_SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY",
)


class S3DatasetLoader:
    """Minimal loader that preserves the historical API used in this repo."""

    def __init__(
        self,
        bucket: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        self.bucket = bucket or os.getenv("HETZNER_S3_BUCKET", "pixel-data")
        self.endpoint_url = endpoint_url or os.getenv("HETZNER_S3_ENDPOINT")
        self.region_name = region_name or os.getenv("HETZNER_S3_REGION", "us-east-1")

        self.aws_access_key_id = (
            aws_access_key_id
            or os.getenv("HETZNER_S3_ACCESS_KEY")
            or os.getenv("HETZNER_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        self.aws_secret_access_key = (
            aws_secret_access_key
            or os.getenv("HETZNER_S3_SECRET_KEY")
            or os.getenv("HETZNER_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if not (aws_access_key_id or aws_secret_access_key) and not any(os.getenv(key) for key in _CREDENTIAL_ENV_KEYS):
            raise ValueError("S3 credentials not found")

        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        kwargs = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.region_name,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        self._client = boto3.client("s3", **{k: v for k, v in kwargs.items() if v})
        return self._client

    @staticmethod
    def _split_bucket_key(bucket: str, key: str) -> tuple[str, str]:
        if key.startswith("s3://"):
            path = key.removeprefix("s3://")
            if "/" in path:
                bucket_part, object_key = path.split("/", 1)
                return bucket_part, object_key
            raise ValueError(f"Invalid S3 path: {key}")
        return bucket, key.lstrip("/")

    def _maybe_local_path(self, bucket: str, key: str) -> Path | None:
        candidate = Path(key)
        if candidate.exists():
            return candidate
        if key.startswith("s3://"):
            candidate = Path(key.removeprefix("s3://"))
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _iter_json_lines(file_path: Path) -> Iterator[dict[str, Any]]:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"text": line.rstrip("\n")}

    def load_json(self, bucket: str, key: str) -> Any:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        local = self._maybe_local_path(bucket_name, object_key)
        if local is not None:
            raw = local.read_text(encoding="utf-8")
            return json.loads(raw)

        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")

        client = self._ensure_client()
        response = client.get_object(Bucket=bucket_name, Key=object_key)
        payload = response["Body"].read()
        return json.loads(payload.decode("utf-8"))

    def stream_json_array(self, bucket: str, key: str) -> Iterator[dict[str, Any]]:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        local = self._maybe_local_path(bucket_name, object_key)
        if local is not None:
            payload = local.read_text(encoding="utf-8").strip()
            if payload:
                data = json.loads(payload)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data
            return

        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        client = self._ensure_client()
        response = client.get_object(Bucket=bucket_name, Key=object_key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        if isinstance(payload, list):
            yield from payload
        else:
            yield payload

    def stream_jsonl(self, bucket: str, key: str) -> Iterator[dict[str, Any]]:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        local = self._maybe_local_path(bucket_name, object_key)
        if local is not None:
            yield from self._iter_json_lines(local)
            return

        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        response = self._ensure_client().get_object(Bucket=bucket_name, Key=object_key)
        body = response["Body"]
        with body:
            for line in body.iter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    yield {"text": line.decode("utf-8", errors="replace")}

    def stream_json(self, bucket: str, key: str) -> Iterator[dict[str, Any]]:
        lowered = key.lower()
        if lowered.endswith((".jsonl", ".ndjson")):
            return self.stream_jsonl(bucket, key)
        return self.stream_json_array(bucket, key)

    def upload_file(self, bucket: str, key: str, data: Any) -> bool:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._ensure_client().put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=payload,
        )
        return True

    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        if bucket_name.startswith("s3://") or object_key.startswith("/"):
            raise ValueError("Invalid destination bucket/key")
        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        response = self._ensure_client().get_object(Bucket=bucket_name, Key=object_key)
        data = response["Body"].read()
        Path(local_path).write_bytes(data)
        return True

    def list_datasets(self, bucket: str, prefix: str = "") -> list[str]:
        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        response = self._ensure_client().list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents") or []
        return [obj["Key"] for obj in contents if isinstance(obj, dict) and obj.get("Key")]

    def object_exists(self, bucket: str, key: str) -> bool:
        bucket_name, object_key = self._split_bucket_key(bucket, key)
        if self._maybe_local_path(bucket_name, object_key) is not None:
            return True
        if boto3 is None:
            raise ImportError("boto3 is required for live S3 operations")
        try:
            self._ensure_client().head_object(Bucket=bucket_name, Key=object_key)
            return True
        except Exception as exc:
            if botocore and hasattr(botocore.exceptions, "ClientError") and isinstance(
                exc,
                botocore.exceptions.ClientError,
            ):
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"404", "NotFound", "NoSuchKey"}:
                    return False
            return False


def get_s3_dataset_path(dataset_name: str, bucket: str | None = None) -> str:
    bucket_name = bucket or os.getenv("HETZNER_S3_BUCKET", "pixel-data")
    return f"s3://{bucket_name}/{dataset_name}"


def load_dataset_from_s3(dataset_name: str) -> Any:
    local = Path(dataset_name)
    if local.exists():
        payload = local.read_text(encoding="utf-8")
        if dataset_name.endswith(".jsonl"):
            return [json.loads(line) for line in payload.splitlines() if line.strip()]
        return json.loads(payload)
    return S3DatasetLoader().load_json("", dataset_name)


__all__ = ["S3DatasetLoader", "get_s3_dataset_path", "load_dataset_from_s3"]
