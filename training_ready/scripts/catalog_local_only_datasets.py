#!/usr/bin/env python3
"""
Catalog Local-Only vs Remote-Accessible Datasets

Identifies which datasets are:
1. Local-only (must be uploaded from local machine)
2. HuggingFace-accessible (can download directly to S3)
3. Kaggle-accessible (can download directly to S3)
4. URL-accessible (can download directly to S3)
"""

import json
import sys
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = None
try:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
except:
    pass


def log(msg):
    if logger:
        logger.info(msg)
    else:
        print(msg)


class DatasetCataloger:
    """Catalogs datasets by accessibility type"""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self.catalog = {
            "local_only": [],
            "huggingface": [],
            "kaggle": [],
            "url": [],
            "unknown": [],
        }
        # Cache for extracted dataset IDs from loader scripts
        self._loader_dataset_ids: Dict[str, List[str]] = {}

    def _load_manifest(self) -> Dict[str, Any]:
        """Load training manifest"""
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _extract_dataset_ids_from_python(self, file_path: Path) -> List[str]:
        """Extract dataset_id values from Python loader scripts"""
        if file_path in self._loader_dataset_ids:
            return self._loader_dataset_ids[file_path]

        dataset_ids = []

        if not file_path.exists() or not file_path.suffix == ".py":
            self._loader_dataset_ids[file_path] = dataset_ids
            return dataset_ids

        try:
            content = file_path.read_text(encoding="utf-8")

            # Method 1: Regex pattern for dataset_id="..." or dataset_id='...'
            # Matches patterns like: dataset_id="user/dataset-name"
            pattern = r'dataset_id\s*=\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, content)
            dataset_ids.extend(matches)

            # Method 2: Look for load_dataset("...") calls
            # Matches patterns like: load_dataset("user/dataset-name")
            load_pattern = r'load_dataset\s*\(\s*["\']([^"\']+)["\']'
            load_matches = re.findall(load_pattern, content)
            dataset_ids.extend(load_matches)

            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for ds_id in dataset_ids:
                if ds_id not in seen and "/" in ds_id:  # Valid HF IDs have format user/dataset
                    seen.add(ds_id)
                    unique_ids.append(ds_id)

            dataset_ids = unique_ids

        except Exception as e:
            log(f"⚠️  Error extracting dataset IDs from {file_path}: {e}")
            dataset_ids = []

        self._loader_dataset_ids[file_path] = dataset_ids
        return dataset_ids

    def classify_dataset(self, dataset: Dict[str, Any]) -> str:
        """Classify dataset by accessibility"""
        path = dataset.get("path", "")
        source = dataset.get("source", "").lower()
        name = dataset.get("name", "").lower()

        # Check for HuggingFace - only Python loader scripts
        # (not markdown files or other docs that mention huggingface)
        if path.endswith(".py") and any(x in path.lower() or x in source or x in name for x in [
            "huggingface", "hf://", "hf:", "huggingface_loader", "dpo_dataset_loader"
        ]):
            return "huggingface"

        # Check for Kaggle
        if any(x in path.lower() or x in source or x in name for x in [
            "kaggle", "kaggle.com", "kaggle/datasets"
        ]):
            return "kaggle"

        # Check for URLs
        if path.startswith("http://") or path.startswith("https://"):
            return "url"

        # Check if local file exists
        local_path = Path(path)
        if local_path.exists() and local_path.is_file():
            return "local_only"

        # Check for Google Drive
        if "gdrive" in path.lower() or "google" in source or "drive.google.com" in path:
            return "url"  # Can be downloaded via URL

        return "unknown"

    def catalog_all(self) -> Dict[str, Any]:
        """Catalog all datasets"""
        datasets = self.manifest.get("datasets", [])

        log(f"📊 Cataloging {len(datasets)} datasets...")

        for dataset in datasets:
            classification = self.classify_dataset(dataset)

            # Extract dataset IDs if this is a HuggingFace loader script
            dataset_ids = []
            if classification == "huggingface":
                path = dataset.get("path", "")
                if path and path.endswith(".py"):
                    loader_path = Path(path)
                    if loader_path.exists():
                        dataset_ids = self._extract_dataset_ids_from_python(loader_path)

            entry = {
                "name": dataset.get("name", ""),
                "path": dataset.get("path", ""),
                "source": dataset.get("source", ""),
                "stage": dataset.get("stage", "unassigned"),
                "size": dataset.get("size", 0),
                "format": dataset.get("format", "unknown"),
            }

            # Add extracted dataset IDs for HuggingFace entries
            if dataset_ids:
                entry["huggingface_ids"] = dataset_ids
                entry["primary_dataset_id"] = dataset_ids[0] if dataset_ids else None

            self.catalog[classification].append(entry)

        # Collect unique HuggingFace dataset IDs
        unique_hf_ids = set()
        for entry in self.catalog["huggingface"]:
            if "huggingface_ids" in entry:
                unique_hf_ids.update(entry["huggingface_ids"])

        # Generate summary
        summary = {
            "total": len(datasets),
            "local_only": len(self.catalog["local_only"]),
            "huggingface": len(self.catalog["huggingface"]),
            "huggingface_unique_ids": len(unique_hf_ids),
            "kaggle": len(self.catalog["kaggle"]),
            "url": len(self.catalog["url"]),
            "unknown": len(self.catalog["unknown"]),
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "catalog": self.catalog,
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    manifest_path = base_path / "ai" / "training_ready" / "TRAINING_MANIFEST.json"

    if not manifest_path.exists():
        log(f"❌ Manifest not found: {manifest_path}")
        return 1

    log("🔍 Cataloging datasets by accessibility...")

    cataloger = DatasetCataloger(manifest_path)
    result = cataloger.catalog_all()

    # Save catalog
    output_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "dataset_accessibility_catalog.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"\n📊 Dataset Accessibility Summary:")
    log(f"  Total datasets: {result['summary']['total']}")
    log(f"  Local-only (need upload): {result['summary']['local_only']}")
    log(f"  HuggingFace loaders: {result['summary']['huggingface']}")
    log(f"  HuggingFace unique IDs: {result['summary'].get('huggingface_unique_ids', 0)}")
    log(f"  Kaggle (direct to S3): {result['summary']['kaggle']}")
    log(f"  URL (direct to S3): {result['summary']['url']}")
    log(f"  Unknown: {result['summary']['unknown']}")
    log(f"\n💾 Catalog saved to: {output_path}")

    # Generate upload list for local-only
    if result['summary']['local_only'] > 0:
        upload_list_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "local_only_upload_list.txt"
        with open(upload_list_path, "w") as f:
            for dataset in result['catalog']['local_only']:
                f.write(f"{dataset['path']}\n")
        log(f"📋 Local-only upload list: {upload_list_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


