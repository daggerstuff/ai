#!/usr/bin/env python3
"""
Path Resolver for S3 and Local Paths

Resolves dataset paths to either S3 or local files, with automatic fallback.
"""

import os
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """Resolves paths to S3 or local files"""

    def __init__(self):
        self._s3_loader = None
        self._s3_available = False

        # Try to initialize S3 loader
        try:
            from ai.training_ready.tools.data_preparation.s3_dataset_loader import S3DatasetLoader
            self._s3_loader = S3DatasetLoader()
            self._s3_available = True
        except Exception as e:
            logger.debug(f"S3 loader not available: {e}")
            self._s3_available = False

    def is_s3_path(self, path: str) -> bool:
        """Check if path is an S3 path"""
        return path.startswith("s3://") or (
            self._s3_available and
            path.startswith("datasets/") and
            not Path(path).exists()
        )

    def resolve_path(self, path: str, manifest_entry: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        """
        Resolve path to actual location (S3 or local)

        Args:
            path: Original path from manifest
            manifest_entry: Optional manifest entry with s3_path

        Returns:
            Tuple of (resolved_path, source_type) where source_type is 's3' or 'local'
        """
        # Check for explicit S3 path in manifest
        if manifest_entry and "s3_path" in manifest_entry:
            s3_path = manifest_entry["s3_path"]
            if self._s3_available and self._s3_loader.dataset_exists(s3_path):
                return s3_path, "s3"

        # Check if path is already S3
        if self.is_s3_path(path):
            if self._s3_available and self._s3_loader.dataset_exists(path):
                return path, "s3"

        # Check local path
        local_path = Path(path)
        if local_path.exists():
            return str(local_path), "local"

        # Try to construct S3 path from local path
        if self._s3_available and manifest_entry:
            # Try to map local path to S3
            bucket = os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")
            try:
                abs_path = local_path.resolve()
                project_root = Path.cwd()
                if abs_path.is_relative_to(project_root):
                    rel_path = abs_path.relative_to(project_root)
                    s3_key = f"datasets/local/{rel_path.as_posix()}"
                    s3_path = f"s3://{bucket}/{s3_key}"
                    if self._s3_loader.dataset_exists(s3_path):
                        return s3_path, "s3"
            except (ValueError, OSError):
                pass

        # Fallback to original path (will fail later if doesn't exist)
        return path, "local"

    def load_dataset(self, path: str, source_type: str, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Load dataset from resolved path

        Args:
            path: Resolved path (S3 or local)
            source_type: 's3' or 'local'
            max_records: Maximum records to load (None for all)

        Yields:
            Dataset records
        """
        if source_type == "s3":
            if not self._s3_available:
                raise RuntimeError("S3 loader not available")

            # Determine format from extension
            if path.endswith(".jsonl"):
                yield from self._s3_loader.load_jsonl(path, max_records=max_records)
            elif path.endswith(".json"):
                data = self._s3_loader.load_json(path)
                if isinstance(data, list):
                    for item in data:
                        yield item
                        if max_records and len(data) >= max_records:
                            break
                else:
                    yield data
            else:
                # Try JSONL first
                try:
                    yield from self._s3_loader.load_jsonl(path, max_records=max_records)
                except:
                    data = self._s3_loader.load_json(path)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    else:
                        yield data

        else:  # local
            local_path = Path(path)
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")

            if path.endswith(".jsonl"):
                with open(local_path, "r", encoding="utf-8") as f:
                    import json
                    count = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
                            count += 1
                            if max_records and count >= max_records:
                                break
            elif path.endswith(".json"):
                import json
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                            if max_records and len(data) >= max_records:
                                break
                    else:
                        yield data
            else:
                # Try JSONL first
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        import json
                        for line in f:
                            line = line.strip()
                            if line:
                                yield json.loads(line)
                except:
                    import json
                    with open(local_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                yield item
                        else:
                            yield data

    def file_exists(self, path: str, source_type: str) -> bool:
        """Check if file exists at resolved path"""
        if source_type == "s3":
            if not self._s3_available:
                return False
            return self._s3_loader.dataset_exists(path)
        else:
            return Path(path).exists()


# Global resolver instance
_resolver = None

def get_resolver() -> PathResolver:
    """Get global path resolver instance"""
    global _resolver
    if _resolver is None:
        _resolver = PathResolver()
    return _resolver

