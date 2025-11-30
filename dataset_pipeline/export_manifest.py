#!/usr/bin/env python3
"""
Dataset Export Manifest
Defines the manifest structure for dataset exports with checksums, metadata, and provenance
"""

import json
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .config_lock import LockedConfig


@dataclass
class FileManifest:
    """Manifest entry for a single file"""
    filename: str
    format: str  # jsonl, parquet, json, etc.
    size_bytes: int
    sha256: str
    row_count: Optional[int] = None
    source_distribution: Optional[Dict[str, int]] = None

    @classmethod
    def from_file(cls, file_path: Path, format: str,
                  row_count: Optional[int] = None,
                  source_distribution: Optional[Dict[str, int]] = None) -> "FileManifest":
        """Create manifest from file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Calculate checksum
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        sha256 = sha256_hash.hexdigest()

        # Get file size
        size_bytes = file_path.stat().st_size

        return cls(
            filename=file_path.name,
            format=format,
            size_bytes=size_bytes,
            sha256=sha256,
            row_count=row_count,
            source_distribution=source_distribution
        )


@dataclass
class QualitySummary:
    """Summary of quality metrics"""
    total_samples: int
    avg_semantic_coherence: Optional[float] = None
    min_semantic_coherence: Optional[float] = None
    crisis_flags_count: int = 0
    crisis_flags_percentage: float = 0.0
    pii_detected_count: int = 0
    pii_detected_percentage: float = 0.0
    bias_score: Optional[float] = None
    therapeutic_appropriateness_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    """Complete manifest for a dataset export"""
    # Version and identification
    version: str
    created_at: str
    created_by: Optional[str] = None

    # Files in this export
    files: List[FileManifest] = field(default_factory=list)

    # Configuration and reproducibility
    config_lock: Optional[Dict[str, Any]] = None

    # Dataset statistics
    total_samples: int = 0
    samples_by_source: Dict[str, int] = field(default_factory=dict)

    # Quality summary
    quality_summary: Optional[QualitySummary] = None

    # Storage information
    storage_urls: Dict[str, str] = field(default_factory=dict)  # format -> storage_url

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_file(self, file_manifest: FileManifest) -> None:
        """Add a file to the manifest"""
        self.files.append(file_manifest)

    def set_config_lock(self, locked_config: LockedConfig) -> None:
        """Set the configuration lock"""
        self.config_lock = locked_config.to_dict()

    def set_quality_summary(self, quality_summary: QualitySummary) -> None:
        """Set quality summary"""
        self.quality_summary = quality_summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'version': self.version,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'files': [asdict(f) for f in self.files],
            'total_samples': self.total_samples,
            'samples_by_source': self.samples_by_source,
            'storage_urls': self.storage_urls,
            'metadata': self.metadata
        }

        if self.config_lock:
            result['config_lock'] = self.config_lock

        if self.quality_summary:
            result['quality_summary'] = self.quality_summary.to_dict()

        return result

    def save(self, path: Path) -> None:
        """Save manifest to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        manifest = cls(
            version=data['version'],
            created_at=data['created_at'],
            created_by=data.get('created_by'),
            total_samples=data.get('total_samples', 0),
            samples_by_source=data.get('samples_by_source', {}),
            storage_urls=data.get('storage_urls', {}),
            metadata=data.get('metadata', {})
        )

        # Load files
        for file_data in data.get('files', []):
            manifest.files.append(FileManifest(**file_data))

        # Load config lock
        if 'config_lock' in data:
            manifest.config_lock = data['config_lock']

        # Load quality summary
        if 'quality_summary' in data:
            manifest.quality_summary = QualitySummary(**data['quality_summary'])

        return manifest

    def verify_files(self, base_path: Path) -> tuple[bool, List[str]]:
        """Verify all files in manifest exist and checksums match"""
        errors = []

        for file_manifest in self.files:
            file_path = base_path / file_manifest.filename

            # Check file exists
            if not file_path.exists():
                errors.append(f"File not found: {file_manifest.filename}")
                continue

            # Check file size
            actual_size = file_path.stat().st_size
            if actual_size != file_manifest.size_bytes:
                errors.append(
                    f"Size mismatch for {file_manifest.filename}: "
                    f"expected {file_manifest.size_bytes}, got {actual_size}"
                )

            # Check checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            actual_sha256 = sha256_hash.hexdigest()

            if actual_sha256 != file_manifest.sha256:
                errors.append(
                    f"Checksum mismatch for {file_manifest.filename}: "
                    f"expected {file_manifest.sha256}, got {actual_sha256}"
                )

        return len(errors) == 0, errors

