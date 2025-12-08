#!/usr/bin/env python3
"""
Data Sourcing and Download Pipeline

Sources and downloads all datasets from TRAINING_MANIFEST.json.
Supports HuggingFace, local paths, Google Drive, and URLs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

# Add dataset_pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dataset_pipeline"))

# Add training_ready to path for S3 support
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace datasets not available. Install with: uv pip install datasets")

try:
    from ingestion.tier_loaders.huggingface_mental_health_loader import HuggingFaceMentalHealthLoader
    from ingestion.tier_loaders.tier1_priority_loader import Tier1PriorityLoader
    from ingestion.tier_loaders.tier2_professional_loader import Tier2ProfessionalLoader
    from ingestion.tier_loaders.tier3_cot_loader import Tier3CoTLoader
    from ingestion.local_loader import LocalLoader
except ImportError as e:
    logging.warning(f"Some loaders not available: {e}")

try:
    from ai.training_ready.tools.data_preparation.path_resolver import get_resolver
    PATH_RESOLVER_AVAILABLE = True
except ImportError:
    PATH_RESOLVER_AVAILABLE = False
    logging.warning("Path resolver not available - S3 support disabled")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SourceResult:
    """Result of sourcing a dataset"""
    name: str
    path: str
    source_type: str
    success: bool
    error: Optional[str] = None
    size_bytes: Optional[int] = None
    record_count: Optional[int] = None
    cached: bool = False


class DatasetSourcer:
    """Sources and downloads datasets from various sources"""

    def __init__(self, manifest_path: Path, cache_dir: Optional[Path] = None):
        self.manifest_path = manifest_path
        self.cache_dir = cache_dir or Path.cwd() / "ai" / "training_ready" / "datasets" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()
        self.results: List[SourceResult] = []

    def _load_manifest(self) -> Dict[str, Any]:
        """Load training manifest"""
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def source_huggingface_dataset(self, dataset_name: str, config: Optional[str] = None) -> SourceResult:
        """Source dataset from HuggingFace"""
        if not HF_AVAILABLE:
            return SourceResult(
                name=dataset_name,
                path="",
                source_type="huggingface",
                success=False,
                error="HuggingFace datasets library not installed"
            )

        try:
            cache_path = self.cache_dir / "hf" / dataset_name
            cache_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, config, cache_dir=str(cache_path))

            # Save to cache
            output_path = cache_path / f"{dataset_name}.jsonl"
            if isinstance(dataset, dict):
                # Multiple splits
                for split_name, split_data in dataset.items():
                    split_path = cache_path / f"{dataset_name}_{split_name}.jsonl"
                    split_data.to_json(str(split_path))
            else:
                dataset.to_json(str(output_path))

            record_count = len(dataset) if hasattr(dataset, '__len__') else None
            size_bytes = output_path.stat().st_size if output_path.exists() else None

            return SourceResult(
                name=dataset_name,
                path=str(output_path),
                source_type="huggingface",
                success=True,
                size_bytes=size_bytes,
                record_count=record_count,
                cached=True
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
            return SourceResult(
                name=dataset_name,
                path="",
                source_type="huggingface",
                success=False,
                error=str(e)
            )

    def source_s3_dataset(self, s3_path: str, dataset_name: str) -> SourceResult:
        """Source dataset from S3"""
        if not PATH_RESOLVER_AVAILABLE:
            return SourceResult(
                name=dataset_name,
                path=s3_path,
                source_type="s3",
                success=False,
                error="S3 support not available"
            )

        try:
            resolver = get_resolver()
            if resolver.file_exists(s3_path, "s3"):
                # Get file size from S3
                import boto3
                import os
                bucket, key = resolver._s3_loader.parse_s3_path(s3_path)
                s3_client = resolver._s3_loader.s3_client
                response = s3_client.head_object(Bucket=bucket, Key=key)
                size_bytes = response.get('ContentLength', 0)

                return SourceResult(
                    name=dataset_name,
                    path=s3_path,
                    source_type="s3",
                    success=True,
                    size_bytes=size_bytes,
                    cached=True  # S3 is our "cache"
                )
            else:
                return SourceResult(
                    name=dataset_name,
                    path=s3_path,
                    source_type="s3",
                    success=False,
                    error=f"File not found in S3: {s3_path}"
                )
        except Exception as e:
            logger.error(f"Failed to check S3 dataset {s3_path}: {e}")
            return SourceResult(
                name=dataset_name,
                path=s3_path,
                source_type="s3",
                success=False,
                error=str(e)
            )

    def source_local_dataset(self, file_path: str) -> SourceResult:
        """Source dataset from local file path"""
        path = Path(file_path)

        if not path.exists():
            return SourceResult(
                name=path.name,
                path=str(path),
                source_type="local",
                success=False,
                error=f"File not found: {file_path}"
            )

        # Check if already in cache
        cache_path = self.cache_dir / "local" / path.name
        if cache_path.exists():
            return SourceResult(
                name=path.name,
                path=str(cache_path),
                source_type="local",
                success=True,
                size_bytes=cache_path.stat().st_size,
                cached=True
            )

        # Copy to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(path, cache_path)

            return SourceResult(
                name=path.name,
                path=str(cache_path),
                source_type="local",
                success=True,
                size_bytes=cache_path.stat().st_size,
                cached=False
            )
        except Exception as e:
            logger.error(f"Failed to cache local dataset {file_path}: {e}")
            return SourceResult(
                name=path.name,
                path=str(path),
                source_type="local",
                success=True,  # Original file exists
                size_bytes=path.stat().st_size if path.exists() else None,
                cached=False
            )

    def source_all_datasets(self) -> List[SourceResult]:
        """Source all datasets from manifest"""
        datasets = self.manifest.get("datasets", [])

        logger.info(f"Sourcing {len(datasets)} datasets from manifest...")

        for dataset in datasets:
            source = dataset.get("source", "")
            path = dataset.get("path", "")
            name = dataset.get("name", "")

            # Check S3 first if available
            s3_path = dataset.get("s3_path")
            if s3_path and PATH_RESOLVER_AVAILABLE:
                resolver = get_resolver()
                resolved_path, source_type = resolver.resolve_path(path, dataset)
                if source_type == "s3":
                    result = self.source_s3_dataset(resolved_path, name or path)
                    self.results.append(result)
                    continue

            # Determine source type
            if "huggingface" in source.lower() or "hf" in source.lower() or path.startswith("huggingface:"):
                result = self.source_huggingface_dataset(name or path)
            elif path.startswith("http://") or path.startswith("https://"):
                # URL - would need download logic
                result = SourceResult(
                    name=name,
                    path=path,
                    source_type="url",
                    success=False,
                    error="URL downloading not yet implemented"
                )
            elif "gdrive" in source.lower() or "google" in source.lower():
                # Google Drive - would need gdrive API
                result = SourceResult(
                    name=name,
                    path=path,
                    source_type="gdrive",
                    success=False,
                    error="Google Drive downloading not yet implemented"
                )
            else:
                # Assume local path
                result = self.source_local_dataset(path)

            self.results.append(result)

            if result.success:
                logger.info(f"  ✅ {name}: {result.source_type} ({'cached' if result.cached else 'downloaded'})")
            else:
                logger.warning(f"  ⚠️  {name}: {result.error}")

        return self.results

    def generate_sourcing_report(self) -> Dict[str, Any]:
        """Generate report of sourcing results"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        cached = sum(1 for r in self.results if r.cached)
        failed = total - successful

        total_size = sum(r.size_bytes or 0 for r in self.results if r.success)
        total_records = sum(r.record_count or 0 for r in self.results if r.record_count)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": total,
            "successful": successful,
            "failed": failed,
            "cached": cached,
            "total_size_bytes": total_size,
            "total_records": total_records,
            "results": [
                {
                    "name": r.name,
                    "path": r.path,
                    "source_type": r.source_type,
                    "success": r.success,
                    "error": r.error,
                    "size_bytes": r.size_bytes,
                    "record_count": r.record_count,
                    "cached": r.cached,
                }
                for r in self.results
            ]
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    manifest_path = base_path / "ai" / "training_ready" / "TRAINING_MANIFEST.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 1

    logger.info("🚀 Starting dataset sourcing pipeline...")

    sourcer = DatasetSourcer(manifest_path)
    results = sourcer.source_all_datasets()

    # Generate report
    report = sourcer.generate_sourcing_report()
    report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "sourcing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Sourcing Summary:")
    logger.info(f"  Total: {report['total_datasets']}")
    logger.info(f"  Successful: {report['successful']}")
    logger.info(f"  Failed: {report['failed']}")
    logger.info(f"  Cached: {report['cached']}")
    logger.info(f"  Total size: {report['total_size_bytes'] / (1024**3):.2f} GB")
    logger.info(f"  Total records: {report['total_records']:,}")
    logger.info(f"\n💾 Report saved to: {report_path}")

    return 0 if report['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

