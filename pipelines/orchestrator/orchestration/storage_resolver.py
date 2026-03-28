"""
Storage/path resolution for orchestrator dataset sources.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import TextIOWrapper
import os
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows-only import path
    fcntl = None
    import msvcrt
else:  # pragma: no cover - Unix-only branch metadata
    msvcrt = None

from ai.pipelines.orchestrator.storage_manager import StorageManager
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.storage_resolver")


class StorageCacheError(RuntimeError):
    """Raised when a remote dataset artifact cannot be cached locally."""

    def __init__(self, storage_path: str, message: str) -> None:
        super().__init__(f"{message}: {storage_path}")
        self.storage_path = storage_path
        self.message = message


class StorageResolver:
    """Resolve dataset source paths and cache remote artifacts locally."""

    def __init__(self, storage: StorageManager | None) -> None:
        self.storage = storage
        self.cache_dir = Path.home() / ".cache" / "pixelated" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _lock_path(cached_file: Path) -> Path:
        return cached_file.with_name(f"{cached_file.name}.lock")

    @contextmanager
    def _acquire_lock(self, cached_file: Path) -> Iterator[None]:
        lock_path = self._lock_path(cached_file)
        with lock_path.open("a", encoding="utf-8") as lock_handle:
            self._lock_handle(lock_handle)
            try:
                yield
            finally:
                self._unlock_handle(lock_handle)

    @staticmethod
    def _lock_handle(lock_handle: TextIOWrapper) -> None:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            return
        if msvcrt is not None:  # pragma: no cover - Windows-only code path
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        raise RuntimeError("No supported file locking backend is available")

    @staticmethod
    def _unlock_handle(lock_handle: TextIOWrapper) -> None:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            return
        if msvcrt is not None:  # pragma: no cover - Windows-only code path
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        raise RuntimeError("No supported file locking backend is available")

    @staticmethod
    def resolve_path(manifest_path: str) -> str:
        """Resolve legacy local paths to storage URIs."""
        if manifest_path.startswith("s3://") or manifest_path.startswith("drive:"):
            return manifest_path

        if "consolidated" in manifest_path:
            clean_path = manifest_path.replace("~/", "").replace("../", "")
            if clean_path.startswith("datasets/consolidated/"):
                return (
                    f"datasets/consolidated/"
                    f"{clean_path.split('datasets/consolidated/')[1]}"
                )

        return manifest_path

    def cache_data(self, source_path: str | None) -> Path | None:
        """Download data from storage to local cache if needed."""
        if not source_path:
            return None

        storage_path = self.resolve_path(source_path)

        if not storage_path.startswith(("s3://", "drive:", "datasets/")):
            local_path = Path(os.path.expanduser(source_path))
            if local_path.exists():
                return local_path
            return None

        storage = self.storage
        if storage is None:
            raise StorageCacheError(
                storage_path, "StorageManager is not initialized for remote path"
            )

        safe_name = (
            storage_path.replace("/", "_").replace("s3:__", "").replace(":", "_")
        )
        cached_file = self.cache_dir / safe_name

        if cached_file.exists():
            logger.info("Using cached file: %s", cached_file)
            return cached_file

        try:
            with self._acquire_lock(cached_file):
                if cached_file.exists():
                    logger.info("Using cached file after lock acquisition: %s", cached_file)
                    return cached_file

                logger.info("Downloading %s to cache...", storage_path)
                if storage_path.startswith("drive:"):
                    storage.download_file(storage_path, cached_file)
                elif storage_path.startswith("s3://"):
                    download_key = storage_path
                    if storage.config.s3_bucket and storage_path.startswith(
                        f"s3://{storage.config.s3_bucket}/"
                    ):
                        download_key = storage_path.replace(
                            f"s3://{storage.config.s3_bucket}/", ""
                        )
                    storage.download_file(download_key, cached_file)
                elif storage_path.startswith("datasets/"):
                    storage.download_file(storage_path, cached_file)
                return cached_file
        except Exception as exc:
            logger.error("Failed to download %s: %s", storage_path, exc)
            if cached_file.exists():
                cached_file.unlink(missing_ok=True)
            if isinstance(exc, StorageCacheError):
                raise
            raise StorageCacheError(storage_path, "Failed to cache remote dataset") from exc


__all__ = ["StorageCacheError", "StorageResolver"]
