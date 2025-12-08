#!/usr/bin/env python3
"""
Download All HuggingFace Datasets to S3

Reads the catalog and downloads all HuggingFace datasets directly to S3.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.training_ready.scripts.download_to_s3 import S3DatasetDownloader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_catalog(catalog_path: Path) -> Dict[str, Any]:
    """Load the dataset accessibility catalog"""
    with open(catalog_path, "r") as f:
        return json.load(f)


def get_all_hf_dataset_ids(catalog: Dict[str, Any]) -> List[str]:
    """Extract all unique HuggingFace dataset IDs from catalog"""
    unique_ids = set()

    for entry in catalog.get("catalog", {}).get("huggingface", []):
        if "huggingface_ids" in entry:
            unique_ids.update(entry["huggingface_ids"])

    return sorted(list(unique_ids))


def main():
    """Main function"""
    base_path = Path.cwd()
    catalog_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "dataset_accessibility_catalog.json"

    if not catalog_path.exists():
        logger.error(f"❌ Catalog not found: {catalog_path}")
        logger.info("Run catalog_local_only_datasets.py first to generate the catalog")
        return 1

    logger.info("📖 Loading catalog...")
    catalog = load_catalog(catalog_path)

    dataset_ids = get_all_hf_dataset_ids(catalog)

    if not dataset_ids:
        logger.warning("⚠️  No HuggingFace dataset IDs found in catalog")
        return 0

    logger.info(f"📋 Found {len(dataset_ids)} unique HuggingFace dataset IDs")
    logger.info("")

    # Initialize S3 downloader
    import os
    bucket = os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")

    logger.info(f"🔌 Initializing S3 downloader (bucket: {bucket})...")
    downloader = S3DatasetDownloader(
        s3_bucket=bucket,
        s3_prefix="datasets/huggingface/"
    )

    if not downloader.s3_client:
        logger.error("❌ S3 client not available. Check credentials in environment.")
        return 1

    logger.info("✅ S3 client initialized")
    logger.info("")

    # Download each dataset
    results = []
    successful = 0
    failed = 0

    for i, dataset_id in enumerate(dataset_ids, 1):
        logger.info(f"[{i}/{len(dataset_ids)}] 📥 Downloading: {dataset_id}")

        try:
            result = downloader.download_hf_to_s3(dataset_id)

            if result.get("success"):
                logger.info(f"  ✅ Success: {result.get('s3_path', 'N/A')}")
                successful += 1
            else:
                logger.error(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                failed += 1

            results.append({
                "dataset_id": dataset_id,
                "result": result
            })

        except Exception as e:
            logger.error(f"  ❌ Exception: {e}")
            failed += 1
            results.append({
                "dataset_id": dataset_id,
                "result": {"success": False, "error": str(e)}
            })

        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("📊 Download Summary:")
    logger.info(f"  Total datasets: {len(dataset_ids)}")
    logger.info(f"  ✅ Successful: {successful}")
    logger.info(f"  ❌ Failed: {failed}")
    logger.info("")

    # Save results
    results_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "hf_download_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "total": len(dataset_ids),
            "successful": successful,
            "failed": failed,
            "results": results
        }, f, indent=2)

    logger.info(f"💾 Results saved to: {results_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

