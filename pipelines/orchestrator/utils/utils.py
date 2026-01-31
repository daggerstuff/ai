"""
Utility Functions for Dataset Pipeline

Common operations for file I/O, JSON handling, data processing, and other utilities.
"""

import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


# File I/O Utilities
def load_json(file_path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Load JSON file with error handling (alias for read_json)."""
    return read_json(file_path, encoding)


def read_json(file_path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Read JSON file with error handling."""
    try:
        with open(file_path, encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise


def write_json(
    data: Any,
    file_path: str | Path,
    indent: int = 2,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """Write data to JSON file with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.debug(f"JSON written to: {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        raise


def read_jsonl(
    file_path: str | Path, encoding: str = "utf-8"
) -> list[dict[str, Any]]:
    """Read JSONL (JSON Lines) file."""
    data = []
    try:
        with open(file_path, encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {file_path}: {e}"
                        )

        logger.debug(f"Read {len(data)} records from JSONL: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading JSONL file {file_path}: {e}")
        raise


def write_jsonl(
    data: list[dict[str, Any]],
    file_path: str | Path,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """Write data to JSONL (JSON Lines) file."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")

        logger.debug(f"Written {len(data)} records to JSONL: {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSONL to {file_path}: {e}")
        raise


def read_csv(
    file_path: str | Path, encoding: str = "utf-8", delimiter: str = ","
) -> list[dict[str, Any]]:
    """Read CSV file as list of dictionaries."""
    try:
        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = list(reader)

        logger.debug(f"Read {len(data)} records from CSV: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise


def write_csv(
    data: list[dict[str, Any]],
    file_path: str | Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> None:
    """Write data to CSV file."""
    if not data:
        logger.warning("No data to write to CSV")
        return

    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = data[0].keys()
        with open(file_path, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)

        logger.debug(f"Written {len(data)} records to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Error writing CSV to {file_path}: {e}")
        raise


def read_text(file_path: str | Path, encoding: str = "utf-8") -> str:
    """Read text file."""
    try:
        with open(file_path, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise


def write_text(text: str, file_path: str | Path, encoding: str = "utf-8") -> None:
    """Write text to file."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(text)

        logger.debug(f"Text written to: {file_path}")
    except Exception as e:
        logger.error(f"Error writing text to {file_path}: {e}")
        raise


# File System Utilities
def ensure_directory(dir_path: str | Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(
    src: str | Path, dst: str | Path, create_dirs: bool = True
) -> None:
    """Copy file with directory creation."""
    src_path = Path(src)
    dst_path = Path(dst)

    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(src_path, dst_path)
        logger.debug(f"File copied: {src} -> {dst}")
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        raise


def move_file(
    src: str | Path, dst: str | Path, create_dirs: bool = True
) -> None:
    """Move file with directory creation."""
    src_path = Path(src)
    dst_path = Path(dst)

    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(src_path), str(dst_path))
        logger.debug(f"File moved: {src} -> {dst}")
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        raise


def delete_file(file_path: str | Path, ignore_missing: bool = True) -> bool:
    """Delete file safely."""
    try:
        Path(file_path).unlink()
        logger.debug(f"File deleted: {file_path}")
        return True
    except FileNotFoundError:
        if not ignore_missing:
            logger.error(f"File not found for deletion: {file_path}")
            raise
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise


def get_file_size(file_path: str | Path) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        raise


def get_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """Calculate file hash."""
    hash_func = hashlib.new(algorithm)

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating {algorithm} hash for {file_path}: {e}")
        raise


# Data Processing Utilities
def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten dictionary with dot notation keys."""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def filter_dict(
    d: dict[str, Any], keys: list[str], include: bool = True
) -> dict[str, Any]:
    """Filter dictionary by keys."""
    if include:
        return {k: v for k, v in d.items() if k in keys}
    return {k: v for k, v in d.items() if k not in keys}


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split list into chunks."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_list(lst: list[Any], key_func: Callable | None = None) -> list[Any]:
    """Remove duplicates from list while preserving order."""
    if key_func is None:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    seen = set()
    result = []
    for item in lst:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# Compression Utilities
def compress_json(data: Any, file_path: str | Path) -> None:
    """Write compressed JSON file."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Compressed JSON written to: {file_path}")
    except Exception as e:
        logger.error(f"Error writing compressed JSON to {file_path}: {e}")
        raise


def decompress_json(file_path: str | Path) -> Any:
    """Read compressed JSON file."""
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading compressed JSON from {file_path}: {e}")
        raise


# Pickle Utilities
def save_pickle(obj: Any, file_path: str | Path) -> None:
    """Save object to pickle file."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        logger.debug(f"Object pickled to: {file_path}")
    except Exception as e:
        logger.error(f"Error pickling object to {file_path}: {e}")
        raise


def load_pickle(file_path: str | Path) -> Any:
    """Load object from pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        raise


# Validation Utilities
def validate_json_schema(data: Any, required_keys: list[str]) -> bool:
    """Validate that data has required keys."""
    if not isinstance(data, dict):
        return False

    return all(key in data for key in required_keys)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    import re

    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().isoformat()


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


# Temporary File Utilities
def create_temp_file(
    suffix: str = "", prefix: str = "tmp", dir: str | None = None
) -> str:
    """Create temporary file and return path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)  # Close file descriptor
    return path


def create_temp_dir(
    suffix: str = "", prefix: str = "tmp", dir: str | None = None
) -> str:
    """Create temporary directory and return path."""
    return tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)


# Context Managers
class temporary_file:
    """Context manager for temporary files."""

    def __init__(
        self, suffix: str = "", prefix: str = "tmp", dir: str | None = None
    ):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self) -> str:
        self.path = create_temp_file(self.suffix, self.prefix, self.dir)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class temporary_directory:
    """Context manager for temporary directories."""

    def __init__(
        self, suffix: str = "", prefix: str = "tmp", dir: str | None = None
    ):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self) -> str:
        self.path = create_temp_dir(self.suffix, self.prefix, self.dir)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path)


# Example usage and testing
if __name__ == "__main__":
    # Test JSON utilities
    test_data = {"name": "test", "value": 123, "nested": {"key": "value"}}

    with temporary_file(suffix=".json") as temp_json:
        write_json(test_data, temp_json)
        loaded_data = read_json(temp_json)
        assert loaded_data == test_data

    # Test JSONL utilities
    test_list = [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]

    with temporary_file(suffix=".jsonl") as temp_jsonl:
        write_jsonl(test_list, temp_jsonl)
        loaded_list = read_jsonl(temp_jsonl)
        assert loaded_list == test_list

    # Test utility functions
    flattened = flatten_dict({"a": {"b": {"c": 1}}})
    assert flattened == {"a.b.c": 1}

    # Test file utilities
    test_hash = get_file_hash(__file__)
    assert len(test_hash) == 32  # MD5 hash length

