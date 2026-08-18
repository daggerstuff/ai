import contextlib
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Default remote + bucket — override via env vars for different environments
DEFAULT_REMOTE = os.environ.get("S3_STREAMER_REMOTE", "whitebat")
DEFAULT_BUCKET = os.environ.get("S3_STREAMER_BUCKET", "training")
DEFAULT_PREFIX = os.environ.get("S3_STREAMER_PREFIX", "pixelated-empathy")


def _build_path(key: str, remote: str | None = None, bucket: str | None = None) -> str:
    """Build an rclone remote path from key + optional remote/bucket overrides."""
    r = remote or DEFAULT_REMOTE
    b = bucket or DEFAULT_BUCKET
    # If key already contains the bucket+prefix (e.g. "pixelated-empathy/curated/..."),
    # strip it so we don't double-prefix.
    base = f"{b}/{DEFAULT_PREFIX}"
    if key.startswith(base + "/"):
        key = key[len(base) + 1 :]
    elif key.startswith(DEFAULT_PREFIX + "/"):
        key = key[len(DEFAULT_PREFIX) + 1 :]
    return f"{r}:{b}/{DEFAULT_PREFIX}/{key}" if key else f"{r}:{b}/{DEFAULT_PREFIX}"


class S3Streamer:
    """Streams datasets to and from S3-compatible storage using rclone.

    Defaults to whitebat (Civo) training/pixelated-empathy.
    Override via env vars: S3_STREAMER_REMOTE, S3_STREAMER_BUCKET, S3_STREAMER_PREFIX.
    """

    def __init__(self, remote: str | None = None, bucket: str | None = None, prefix: str | None = None):
        self.remote = remote or DEFAULT_REMOTE
        self.bucket = bucket if bucket is not None else DEFAULT_BUCKET
        self.prefix = prefix if prefix is not None else DEFAULT_PREFIX

    def _path(self, key: str) -> str:
        r = self.remote
        p = self.prefix
        # Strip redundant prefix if caller passed full path
        if p and key.startswith(p + "/"):
            key = key[len(p) + 1 :]
        if self.bucket:
            base = f"{self.bucket}/{p}" if p else self.bucket
            if key.startswith(base + "/"):
                key = key[len(base) + 1 :]
            return f"{r}:{base}/{key}" if key else f"{r}:{base}"
        return f"{r}:{p}/{key}" if key and p else (f"{r}:{p}" if p else f"{r}:{key}")

    def list_files(self, prefix: str, recursive: bool = False) -> list[str]:
        """Returns all object keys under a specific prefix."""
        cmd = ["rclone", "lsf"]
        if recursive:
            cmd.append("--recursive")
        cmd.append(self._path(prefix))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        keys: list[str] = []
        for line in result.stdout.strip().split("\n"):
            if line:
                keys.append(prefix + line)
        return keys

    def stream_jsonl(self, key: str):
        """Yields parsed JSON objects line-by-line from a JSONL file in S3 using rclone cat."""
        cmd = ["rclone", "cat", self._path(key)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        assert proc.stdout is not None
        with contextlib.suppress(Exception):
            for raw_line in proc.stdout:
                stripped = raw_line.strip()
                if stripped:
                    yield json.loads(stripped)
        proc.wait()

    def stream_text(self, key: str):
        """Yields decoded text line-by-line from a text file in S3 using rclone cat."""
        cmd = ["rclone", "cat", self._path(key)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            stripped = raw_line.strip()
            if stripped:
                yield stripped
        proc.wait()

    def download_to_file(self, key: str, local_path: str) -> None:
        """Downloads a file directly to disk for processing."""
        cmd = ["rclone", "copyto", self._path(key), local_path]
        subprocess.run(cmd, check=True)

    def write_jsonl(self, key: str, iterator, chunk_size_mb: int = 64) -> int:
        """Streams an iterable of dicts as JSONL directly to S3 using rclone rcat.

        Avoids writing anything to local disk.
        Returns count of records written.
        """
        cmd = ["rclone", "rcat", self._path(key)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, encoding="utf-8")
        assert proc.stdin is not None
        count = 0
        try:
            for item in iterator:
                line = json.dumps(item) + "\n"
                proc.stdin.write(line)
                count += 1
                if count % 100000 == 0:
                    proc.stdin.flush()
            proc.stdin.close()

            return_code = proc.wait()
            if return_code != 0:
                raise Exception(f"rclone rcat failed with return code {return_code}")

            logger.info("Successfully uploaded %d records to %s via rclone rcat.", count, key)
            return count

        except Exception as e:
            proc.stdin.close()
            proc.terminate()
            raise e from None
