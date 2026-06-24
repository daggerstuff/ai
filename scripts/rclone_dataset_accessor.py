#!/usr/bin/env python3
"""
Shared utilities for accessing datasets via rclone.
Uses rclone instead of boto3 to work with Hetzner Object Storage.
"""

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_EXPECTED_PARTS = 2


def run_rclone(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run rclone command and return result."""
    cmd = ["rclone", *shlex.split(command)]
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"rclone command failed: {result.stderr}")
    return result


def s3_path_to_rclone(s3_path: str, remote_name: str | None = None) -> str:
    """
    Convert s3://bucket/path to rclone remote:path format.

    Args:
        s3_path: s3://bucket/path style path
        remote_name: Name of rclone remote

    Returns:
        rclone remote:path format
    """
    if remote_name is None:
        remote_name = os.getenv("RCLONE_REMOTE_NAME", "HetznerS3")
    if s3_path.startswith("s3://"):
        import re
        bucket = os.getenv("HETZNER_S3_BUCKET", "pixeldata")
        return re.sub(r"^s3://[^/]+/", f"{remote_name}:{bucket}/", s3_path)
    return s3_path


def list_files_in_directory(s3_path: str) -> list[dict[str, Any]]:
    """
    List all files in a directory in S3/DO Spaces.

    Args:
        s3_path: s3://bucket/path style path

    Returns:
        List of dicts with 'name', 'size_bytes', 'size_mb' keys
    """
    rclone_path = s3_path_to_rclone(s3_path)
    result = run_rclone(f"ls {rclone_path}", check=False)

    files = []
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == _EXPECTED_PARTS:
                size, filename = parts
                size_int = int(size)
                files.append(
                    {
                        "name": filename,
                        "size_bytes": size_int,
                        "size_mb": round(size_int / 1024 / 1024, 2),
                    }
                )

    return files


def download_file(s3_path: str, local_path: Path | None = None) -> Path | None:
    """
    Download a file from S3/DO Spaces to local temp directory.

    Args:
        s3_path: s3://bucket/path/to/file
        local_path: Optional local path (creates temp file if None)

    Returns:
        Path to downloaded file, or None if failed
    """
    rclone_path = s3_path_to_rclone(s3_path)

    if local_path is None:
        # Create temp file
        suffix = Path(s3_path).suffix
        fd, temp_path = tempfile.mkstemp(suffix=suffix)

        os.close(fd)
        local_path = Path(temp_path)

    result = run_rclone(f"copyto {rclone_path} {local_path}", check=False)

    if result.returncode == 0:
        return local_path
    return None


def load_json_file(s3_path: str) -> Any | None:
    """
    Load a JSON file from S3/DO Spaces.

    Args:
        s3_path: s3://bucket/path/to/file.json

    Returns:
        Parsed JSON data, or None if failed
    """
    local_file = download_file(s3_path)
    if local_file and local_file.exists():
        try:
            with open(local_file) as f:
                data = json.load(f)
            # Clean up temp file
            local_file.unlink()
            return data
        except Exception:
            if local_file.exists():
                local_file.unlink()
    return None


def load_jsonl_file(s3_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Load a JSONL file from S3/DO Spaces.

    Args:
        s3_path: s3://bucket/path/to/file.jsonl
        limit: Maximum number of records to load

    Returns:
        List of records
    """
    local_file = download_file(s3_path)
    records = []

    if local_file and local_file.exists():
        try:
            with open(local_file) as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    if line.strip():
                        records.append(json.loads(line))
        except Exception:
            pass
        finally:
            if local_file.exists():
                local_file.unlink()

    return records


def calculate_checksum(s3_path: str, algorithm: str = "sha256") -> str | None:
    """
    Calculate checksum of a file in S3/DO Spaces.

    Args:
        s3_path: s3://bucket/path/to/file
        algorithm: 'sha256' or 'md5'

    Returns:
        Checksum string, or None if failed
    """

    local_file = download_file(s3_path)
    if local_file and local_file.exists():
        try:
            if algorithm == "sha256":
                hasher = hashlib.sha256()
            elif algorithm == "md5":
                hasher = hashlib.md5()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            with open(local_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)

            checksum = hasher.hexdigest()
            local_file.unlink()
            return checksum
        except Exception:
            if local_file.exists():
                local_file.unlink()

    return None


def get_file_info(s3_path: str) -> dict[str, Any] | None:
    """
    Get information about a file in S3/DO Spaces.

    Args:
        s3_path: s3://bucket/path/to/file

    Returns:
        Dict with 'size', 'mod_time', etc. or None if not found
    """
    rclone_path = s3_path_to_rclone(s3_path)
    result = run_rclone(f"lsjson {rclone_path}", check=False)

    if result.returncode == 0 and result.stdout.strip():
        try:
            files = json.loads(result.stdout)
            if files and len(files) > 0:
                return files[0]
        except Exception:
            pass

    return None


class RcloneDatasetAccessor:
    """Context manager for accessing datasets via rclone."""

    def __init__(self, s3_path: str):
        self.s3_path = s3_path
        self.rclone_path = s3_path_to_rclone(s3_path)
        self.files = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def list_files(self) -> list[dict[str, Any]]:
        """List all files in this dataset."""
        self.files = list_files_in_directory(self.s3_path)
        return self.files

    def load_file(self, filename: str, limit: int | None = None) -> Any:
        """
        Load a specific file from this dataset.

        Args:
            filename: Name of file within the dataset
            limit: For JSONL files, max records to load

        Returns:
            Loaded data
        """
        file_path = f"{self.s3_path}/{filename}"

        if filename.endswith(".jsonl"):
            return load_jsonl_file(file_path, limit=limit)
        if filename.endswith(".json"):
            return load_json_file(file_path)
        return download_file(file_path)

    def get_sample_records(self, sample_size: int = 100) -> list[dict[str, Any]]:
        """
        Get sample records from the dataset.
        Loads from first JSONL file found.

        Args:
            sample_size: Number of records to sample

        Returns:
            List of sample records
        """
        if not self.files:
            self.list_files()

        # Find first JSONL file
        for file_info in self.files:
            if file_info["name"].endswith(".jsonl"):
                return self.load_file(file_info["name"], limit=sample_size)

        return []
