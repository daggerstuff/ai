#!/usr/bin/env python3
"""
Update Manifest with S3 Paths

Maps local dataset paths to their S3 equivalents and updates the manifest.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManifestS3Mapper:
    """Maps local paths to S3 paths in the manifest"""

    def __init__(self, manifest_path: Path, upload_results_path: Path, catalog_path: Path):
        self.manifest_path = manifest_path
        self.upload_results_path = upload_results_path
        self.catalog_path = catalog_path

        self.manifest = self._load_json(manifest_path)
        self.upload_results = self._load_json(upload_results_path) if upload_results_path.exists() else {}
        self.catalog = self._load_json(catalog_path) if catalog_path.exists() else {}

        # S3 bucket and prefixes
        self.s3_bucket = os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")
        self.s3_prefixes = {
            "huggingface": "datasets/huggingface/",
            "local": "datasets/local/",
        }

        # Build mapping from upload results
        self.local_to_s3 = self._build_local_to_s3_mapping()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _build_local_to_s3_mapping(self) -> Dict[str, str]:
        """Build mapping from local paths to S3 paths"""
        mapping = {}

        # From upload results (local files)
        for result in self.upload_results.get("results", []):
            local_path = result.get("local_path", "")
            s3_result = result.get("result", {})
            if s3_result.get("success") and "s3_path" in s3_result:
                mapping[local_path] = s3_result["s3_path"]

        # From HuggingFace download results
        hf_results_path = self.manifest_path.parent / "output" / "hf_download_results.json"
        if hf_results_path.exists():
            hf_results = self._load_json(hf_results_path)
            for result in hf_results.get("results", []):
                dataset_id = result.get("dataset_id", "")
                s3_result = result.get("result", {})
                if s3_result.get("success") and "s3_path" in s3_result:
                    # Map dataset ID to S3 path
                    # Note: This is approximate - actual mapping depends on how datasets are referenced
                    mapping[f"huggingface:{dataset_id}"] = s3_result["s3_path"]

        return mapping

    def map_path_to_s3(self, local_path: str) -> Optional[str]:
        """
        Map local path to S3 path

        Args:
            local_path: Local file path

        Returns:
            S3 path if found, None otherwise
        """
        # Direct mapping
        if local_path in self.local_to_s3:
            return self.local_to_s3[local_path]

        # Try relative path
        try:
            abs_path = Path(local_path).resolve()
            project_root = Path.cwd()
            if abs_path.is_relative_to(project_root):
                rel_path = abs_path.relative_to(project_root)
                s3_key = f"{self.s3_prefixes['local']}{rel_path.as_posix()}"
                return f"s3://{self.s3_bucket}/{s3_key}"
        except (ValueError, OSError):
            pass

        return None

    def update_manifest(self) -> Dict[str, Any]:
        """Update manifest with S3 paths"""
        updated_count = 0
        s3_paths_added = 0

        # Update datasets
        for dataset in self.manifest.get("datasets", []):
            local_path = dataset.get("path", "")

            if local_path:
                s3_path = self.map_path_to_s3(local_path)
                if s3_path:
                    if "s3_path" not in dataset or dataset["s3_path"] != s3_path:
                        dataset["s3_path"] = s3_path
                        s3_paths_added += 1
                    updated_count += 1

        logger.info(f"✅ Updated {updated_count} datasets")
        logger.info(f"   Added {s3_paths_added} S3 paths")

        return self.manifest

    def generate_s3_mapping_file(self) -> Dict[str, Any]:
        """Generate a separate S3 mapping file"""
        mapping = {
            "bucket": self.s3_bucket,
            "mappings": {},
            "statistics": {
                "total_local_paths": 0,
                "mapped_to_s3": 0,
                "unmapped": 0,
            }
        }

        for dataset in self.manifest.get("datasets", []):
            local_path = dataset.get("path", "")
            if local_path:
                mapping["statistics"]["total_local_paths"] += 1
                s3_path = self.map_path_to_s3(local_path)
                if s3_path:
                    mapping["mappings"][local_path] = s3_path
                    mapping["statistics"]["mapped_to_s3"] += 1
                else:
                    mapping["statistics"]["unmapped"] += 1

        return mapping


def main():
    """Main function"""
    base_path = Path.cwd()

    manifest_path = base_path / "ai" / "training_ready" / "TRAINING_MANIFEST.json"
    upload_results_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "local_upload_results.json"
    catalog_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "dataset_accessibility_catalog.json"

    if not manifest_path.exists():
        logger.error(f"❌ Manifest not found: {manifest_path}")
        return 1

    logger.info("🔍 Building S3 path mappings...")

    mapper = ManifestS3Mapper(manifest_path, upload_results_path, catalog_path)

    logger.info(f"📋 Found {len(mapper.local_to_s3)} S3 path mappings")
    logger.info("")

    # Update manifest
    logger.info("📝 Updating manifest with S3 paths...")
    updated_manifest = mapper.update_manifest()

    # Save updated manifest
    backup_path = manifest_path.with_suffix(".json.backup")
    if not backup_path.exists():
        logger.info(f"💾 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(manifest_path, backup_path)

    with open(manifest_path, "w") as f:
        json.dump(updated_manifest, f, indent=2)

    logger.info(f"✅ Updated manifest saved: {manifest_path}")
    logger.info("")

    # Generate mapping file
    logger.info("📋 Generating S3 mapping file...")
    mapping = mapper.generate_s3_mapping_file()

    mapping_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "s3_path_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"✅ S3 mapping saved: {mapping_path}")
    logger.info("")
    logger.info("📊 Mapping Statistics:")
    logger.info(f"  Total local paths: {mapping['statistics']['total_local_paths']}")
    logger.info(f"  Mapped to S3: {mapping['statistics']['mapped_to_s3']}")
    logger.info(f"  Unmapped: {mapping['statistics']['unmapped']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

