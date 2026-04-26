#!/usr/bin/env python3
"""
S3 Dataset Loader - Hardened Streaming JSON/JSONL loader.
FIXED: Manual chunk reader for 100% byte-offset precision.
FIXED: errors="replace" for clinical data resilience.
"""

import contextlib
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError as _BotocoreClientError
except ImportError:
    boto3 = None
    _BotocoreClientError = None

if TYPE_CHECKING:
    class ClientError(Exception):
        response: dict[str, Any]
else:
    ClientError = (_BotocoreClientError if _BotocoreClientError is not None else Exception)

logger = logging.getLogger(__name__)

class S3DatasetLoader:
    def __init__(
        self,
        bucket: str = "pixel-data",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str | None = None,
    ):
        self.bucket_name = os.getenv("HETZNER_S3_BUCKET") or bucket
        self.endpoint_url = endpoint_url or os.getenv("HETZNER_S3_ENDPOINT") or "https://hel1.your-objectstorage.com"
        self.access_key = aws_access_key_id or os.getenv("HETZNER_S3_ACCESS_KEY") or os.getenv("HETZNER_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = aws_secret_access_key or os.getenv("HETZNER_S3_SECRET_KEY") or os.getenv("HETZNER_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region_name or os.getenv("HETZNER_S3_REGION") or "hel1"
        self._s3_client = None

    @property
    def s3_client(self):
        if self._s3_client is None:
            if boto3 is None: raise ImportError("boto3 is required.")
            if not self.access_key or not self.secret_key:
                raise ValueError("S3 credentials not found.")
            # FIXED: Hardened S3 Config
            s3_config = Config(
                connect_timeout=30,
                read_timeout=60,
                retries={'max_attempts': 10, 'mode': 'standard'},
                tcp_keepalive=True
            )
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                verify=True,
                config=s3_config
            )
        return self._s3_client

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        if ".." in s3_path: raise ValueError("Path traversal detected.")
        if s3_path.startswith("s3://"): s3_path = s3_path.removeprefix("s3://")
        elif s3_path.startswith("s3:/"): s3_path = s3_path.removeprefix("s3:/")
        if "/" in s3_path:
            parts = s3_path.split("/", 1)
            # FIXED: Strict S3 Bucket Regex
            if parts[0] == self.bucket_name or re.match(r'^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$', parts[0]):
                return parts[0], parts[1]
        return self.bucket_name, s3_path

    def stream_jsonl(self, s3_path: str, byte_offset: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
        bucket, key = self._parse_s3_path(s3_path)
        get_kwargs = {"Bucket": bucket, "Key": key}
        if byte_offset > 0:
            get_kwargs["Range"] = f"bytes={byte_offset}-"

        try:
            response = self.s3_client.get_object(**get_kwargs)
        if byte_offset > 0 and response.get('ResponseMetadata', {}).get('HTTPStatusCode') != 206:
            raise RuntimeError(f"S3 backend does not support Range requests (Status: {response.get('ResponseMetadata', {}).get('HTTPStatusCode')}). Resumption impossible.")
            body = response["Body"]
            current_pos = byte_offset
            
            try:
                with contextlib.closing(body):
                    # FIXED: Manual byte-tracking buffer to prevent EOF offset drift
                    buffer = b""
                    while True:
                        chunk = body.read(1024 * 1024) # 1MB chunks
                        if not chunk:
                            if buffer.strip():
                                try:
                                    record = json.loads(buffer.decode("utf-8", errors="replace"))
                                    yield record, current_pos + len(buffer)
                                except: pass
                            break
                        
                        buffer += chunk
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            line_len = len(line) + 1 # Include newline
                            if not line.strip():
                                current_pos += line_len
                                continue
                            try:
                                # FIXED: errors="replace" prevents crashes on smart-quotes/noise
                                record = json.loads(line.decode("utf-8", errors="replace"))
                                yield record, current_pos + line_len
                            except Exception as e:
                                logger.warning(f"Parse error at {current_pos} in {key}: {e}")
                            current_pos += line_len
            finally:
                if hasattr(body, 'close'): body.close()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Dataset not found: s3://{bucket}/{key}")
            raise
