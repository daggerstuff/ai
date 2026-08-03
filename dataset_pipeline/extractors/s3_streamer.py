import contextlib
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


class S3Streamer:
    """Streams datasets to and from Hetzner S3 using rclone."""

    def __init__(self):
        pass

    def list_files(self, prefix, recursive=False):
        """Yields all object keys under a specific prefix."""
        cmd = ["rclone", "lsf"]
        if recursive:
            cmd.append("--recursive")
        cmd.append(f"HetznerS3:pixeldata/{prefix}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in result.stdout.strip().split("\n"):
            if line:
                yield prefix + line

    def stream_jsonl(self, key, chunk_bytes=None):
        """
        Yields parsed JSON objects line-by-line from a JSONL file in S3 using rclone cat.
        """
        cmd = ["rclone", "cat", f"HetznerS3:pixeldata/{key}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        assert proc.stdout is not None
        with contextlib.suppress(Exception):
            for raw_line in proc.stdout:
                stripped = raw_line.strip()
                if stripped:
                    yield json.loads(stripped)
        proc.wait()

    def stream_text(self, key):
        """Yields decoded text line-by-line from a text file in S3 using rclone cat."""
        cmd = ["rclone", "cat", f"HetznerS3:pixeldata/{key}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            stripped = raw_line.strip()
            if stripped:
                yield stripped
        proc.wait()

    def download_to_file(self, key, local_path):
        """Downloads a file directly to disk for processing."""
        cmd = ["rclone", "copyto", f"HetznerS3:pixeldata/{key}", local_path]
        subprocess.run(cmd, check=True)

    def write_jsonl(self, key, iterator, chunk_size_mb=64):
        """
        Streams an iterable of dicts as JSONL directly to S3 using rclone rcat.
        This avoids writing anything to local disk.
        """
        cmd = ["rclone", "rcat", f"HetznerS3:pixeldata/{key}"]
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

            logger.info("Successfully uploaded dataset to %s via rclone rcat.", key)

        except Exception as e:
            proc.stdin.close()
            proc.terminate()
            raise e from None
