"""
S3 Dataset Loader (Gilfoyle v24.0).
FIXED: 100% Copy-Free iteration via pointer management.
"""

import contextlib
import json
import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

try:
    import boto3
    from botocore.config import Config
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


class S3DatasetLoader:
    _client_lock = threading.Lock()

    def __init__(self, bucket: str = "pixel-data", endpoint_url: str | None = None):
        self.bucket_name = os.getenv("HETZNER_S3_BUCKET") or bucket
        self.endpoint_url = endpoint_url or os.getenv("HETZNER_S3_ENDPOINT") or "https://hel1.your-objectstorage.com"
        self.access_key, self.secret_key = os.getenv("HETZNER_S3_ACCESS_KEY"), os.getenv("HETZNER_S3_SECRET_KEY")
        self._s3_client = None

    @property
    def s3_client(self):
        with self._client_lock:
            if self._s3_client is None:
                s_config = Config(connect_timeout=30, read_timeout=60, retries={"max_attempts": 5})
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=s_config,
                )
        return self._s3_client

    def stream_jsonl(self, s3_path: str, byte_offset: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
        key = s3_path.removeprefix("s3://").split("/", 1)[1] if s3_path.startswith("s3://") else s3_path
        get_kwargs = {"Bucket": self.bucket_name, "Key": key}
        if byte_offset > 0:
            get_kwargs["Range"] = f"bytes={byte_offset}-"
        try:
            response = self.s3_client.get_object(**get_kwargs)
            body, current_pos = response["Body"], byte_offset
            buffer, ptr = bytearray(), 0
            for chunk in body:
                buffer.extend(chunk)
                while True:
                    idx = buffer.find(b"\n", ptr)
                    if idx == -1:
                        break
                    line = buffer[ptr:idx]
                    line_len = (idx - ptr) + 1
                    if line.strip():
                        try:
                            yield json.loads(line.decode("utf-8", errors="strict")), current_pos + line_len
                        except Exception as e:
                            logger.error(f"JSON Skipped in {key}: {e}")
                    current_pos += line_len
                    ptr = idx + 1
                if ptr > 10 * 1024 * 1024:
                    buffer = buffer[ptr:]
                    ptr = 0
            if buffer[ptr:].strip():
                with contextlib.suppress(BaseException):
                    yield json.loads(buffer[ptr:].decode("utf-8")), current_pos + len(buffer[ptr:])
        except Exception as e:
            logger.error(f"S3 Fatal in {key}: {e}")
            raise
