#!/usr/bin/env python3
"""
Storage Configuration for Dataset Pipeline
Manages S3/GCS bucket configuration for raw data, processed data, exports, and checkpoints
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class StorageBackend(Enum):
    """Supported storage backends"""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


@dataclass
class StorageConfig:
    """Configuration for dataset pipeline storage"""

    # Backend selection
    backend: StorageBackend = StorageBackend.LOCAL

    # S3 Configuration
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint_url: str | None = None  # For S3-compatible services

    # GCS Configuration
    gcs_bucket: str | None = None
    gcs_project_id: str | None = None
    gcs_credentials_path: str | None = None

    # Path prefixes for different data types
    raw_data_prefix: str = "raw"
    processed_data_prefix: str = "processed"
    exports_prefix: str = "exports"
    checkpoints_prefix: str = "checkpoints"
    logs_prefix: str = "logs"

    # Local fallback paths (used when backend is LOCAL or as cache)
    local_base_path: Path = field(default_factory=lambda: Path("ai/dataset_pipeline/data"))

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create config from environment variables"""
        backend_str = os.getenv("DATASET_STORAGE_BACKEND", "local").lower()

        if backend_str == "s3":
            backend = StorageBackend.S3
        elif backend_str == "gcs":
            backend = StorageBackend.GCS
        else:
            backend = StorageBackend.LOCAL

        # Local path
        local_base = Path(os.getenv("DATASET_STORAGE_LOCAL_PATH", "ai/dataset_pipeline/data"))

        return cls(
            backend=backend,
            local_base_path=local_base,
            # S3
            s3_bucket=os.getenv("DATASET_S3_BUCKET"),
            s3_region=os.getenv("DATASET_S3_REGION", "us-east-1"),
            s3_access_key_id=(
                os.getenv("HETZNER_S3_ACCESS_KEY")
                or os.getenv("HETZNER_ACCESS_KEY")
                or os.getenv("AWS_ACCESS_KEY_ID")
                or os.getenv("DATASET_S3_ACCESS_KEY_ID")
            ),
            s3_secret_access_key=(
                os.getenv("HETZNER_S3_SECRET_KEY")
                or os.getenv("HETZNER_SECRET_KEY")
                or os.getenv("AWS_SECRET_ACCESS_KEY")
                or os.getenv("DATASET_S3_SECRET_ACCESS_KEY")
            ),
            s3_endpoint_url=os.getenv("DATASET_S3_ENDPOINT_URL"),
            # GCS
            gcs_bucket=os.getenv("DATASET_GCS_BUCKET"),
            gcs_project_id=os.getenv("DATASET_GCS_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
            gcs_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or os.getenv("DATASET_GCS_CREDENTIALS_PATH"),
            # Prefixes
            raw_data_prefix=os.getenv("DATASET_RAW_PREFIX", "raw"),
            processed_data_prefix=os.getenv("DATASET_PROCESSED_PREFIX", "processed"),
            exports_prefix=os.getenv("DATASET_EXPORTS_PREFIX", "exports"),
            checkpoints_prefix=os.getenv("DATASET_CHECKPOINTS_PREFIX", "checkpoints"),
            logs_prefix=os.getenv("DATASET_LOGS_PREFIX", "logs"),
        )


    def get_export_path(self, version: str, filename: str) -> str:
        """Get storage path for dataset export"""
        if self.backend == StorageBackend.LOCAL:
            export_dir = self.local_base_path / self.exports_prefix / version
            export_dir.mkdir(parents=True, exist_ok=True)
            return str(export_dir / filename)
        if self.backend in (StorageBackend.S3, StorageBackend.GCS):
            return f"{self.exports_prefix}/{version}/{filename}"
        raise ValueError(f"Unknown backend: {self.backend}")

    def get_checkpoint_path(self, run_id: str, checkpoint_name: str) -> str:
        """Get storage path for training checkpoint"""
        if self.backend == StorageBackend.LOCAL:
            checkpoint_dir = self.local_base_path / self.checkpoints_prefix / run_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return str(checkpoint_dir / checkpoint_name)
        if self.backend in (StorageBackend.S3, StorageBackend.GCS):
            return f"{self.checkpoints_prefix}/{run_id}/{checkpoint_name}"
        raise ValueError(f"Unknown backend: {self.backend}")

    def get_log_path(self, run_id: str, log_filename: str) -> str:
        """Get storage path for training logs"""
        if self.backend == StorageBackend.LOCAL:
            log_dir = self.local_base_path / self.logs_prefix / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            return str(log_dir / log_filename)
        if self.backend in (StorageBackend.S3, StorageBackend.GCS):
            return f"{self.logs_prefix}/{run_id}/{log_filename}"
        raise ValueError(f"Unknown backend: {self.backend}")

    def validate(self) -> tuple[bool, str | None]:
        """Validate storage configuration"""
        if self.backend == StorageBackend.S3:
            if not self.s3_bucket:
                return False, "S3 bucket name is required"
            if not self.s3_access_key_id or not self.s3_secret_access_key:
                has_creds = (
                    os.getenv("HETZNER_S3_ACCESS_KEY")
                    or os.getenv("AWS_ACCESS_KEY_ID")
                ) and (
                    os.getenv("HETZNER_S3_SECRET_KEY")
                    or os.getenv("AWS_SECRET_ACCESS_KEY")
                )
                if not has_creds:
                    return False, "S3 credentials are required (HETZNER_S3_ACCESS_KEY / AWS_ACCESS_KEY_ID)"
        elif self.backend == StorageBackend.GCS:
            if not self.gcs_bucket:
                return False, "GCS bucket name is required"
            if not self.gcs_credentials_path and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                return False, "GCS credentials are required (GOOGLE_APPLICATION_CREDENTIALS or credentials_path)"

        return True, None


# Default configuration instance
_default_config: StorageConfig | None = None


def get_storage_config() -> StorageConfig:
    """Get the default storage configuration"""
    global _default_config
    if _default_config is None:
        _default_config = StorageConfig.from_env()
    return _default_config


def set_storage_config(config: StorageConfig) -> None:
    """Set the default storage configuration"""
    global _default_config
    _default_config = config
