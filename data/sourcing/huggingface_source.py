"""
HuggingFace Dataset Discovery and Download.

Discovers and downloads mental health/therapy datasets from HuggingFace Hub.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets library not installed. Run: uv add datasets")


class HuggingFaceSource:
    """
    Discover and download datasets from HuggingFace Hub.

    Usage:
        source = HuggingFaceSource()
        refs = source.discover(query="mental health therapy")
        for ref in refs:
            print(f"Found: {ref.name}")
    """

    # Pre-curated queries for mental health/therapy datasets
    MENTAL_HEALTH_QUERIES = [
        "mental health",
        "therapy session",
        "counseling conversation",
        "psychological counseling",
        "psychotherapy dialogue",
        "CBT cognitive behavioral",
        "DBT dialectical behavior",
        "trauma therapy",
        "PTSD counseling",
        "anxiety depression therapy",
        "suicide prevention hotline",
        "crisis counseling",
        "motivational interviewing",
        "clinical psychology",
        "therapeutic alliance",
    ]

    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Initialize HuggingFace source.

        Args:
            output_dir: Directory to store downloaded datasets
        """
        self.output_dir = Path(output_dir) if output_dir else Path("ai/data/acquired_datasets/huggingface")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover(
        self,
        queries: Optional[List[str]] = None,
        min_downloads: int = 100,
        limit: int = 50
    ) -> Iterator[Dict[str, Any]]:
        """
        Discover datasets on HuggingFace.

        Args:
            queries: Search queries (uses MENTAL_HEALTH_QUERIES if None)
            min_downloads: Minimum download count filter
            limit: Max results per query

        Yields:
            Dataset metadata dicts
        """
        if not HAS_DATASETS:
            logger.error("datasets library required. Run: uv add datasets")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        queries = queries or self.MENTAL_HEALTH_QUERIES[:5]  # Start with first 5

        for query in queries:
            logger.info(f"Searching HuggingFace for: {query}")

            try:
                datasets = api.list_datasets(
                    search=query,
                    limit=limit,
                    sort="downloads",
                    direction=-1
                )

                for ds in datasets:
                    # Filter by downloads
                    if ds.downloads < min_downloads:
                        continue

                    yield {
                        'id': ds.id,
                        'title': getattr(ds, 'title', ds.id),
                        'description': getattr(ds, 'description', ''),
                        'downloads': ds.downloads,
                        'likes': getattr(ds, 'likes', 0),
                        'tags': getattr(ds, 'tags', []),
                        'cardData': getattr(ds, 'cardData', None),
                    }

            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue

    def download(
        self,
        dataset_id: str,
        split: str = "train",
        samples: Optional[int] = None
    ) -> Optional[str]:
        """
        Download a specific dataset.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "name/dataset")
            split: Dataset split to download
            samples: Max samples to download (None = all)

        Returns:
            Path to downloaded file
        """
        if not HAS_DATASETS:
            logger.error("datasets library required")
            return None

        try:
            logger.info(f"Downloading {dataset_id}...")

            # Load dataset
            ds = load_dataset(dataset_id, split=split)

            # Limit samples if specified
            if samples and len(ds) > samples:
                ds = ds.select(range(samples))

            # Save to output dir
            output_path = self.output_dir / f"{dataset_id.replace('/', '_')}_{split}.jsonl"
            ds.to_json(str(output_path), orient="records", lines=True)

            logger.info(f"Downloaded {len(ds)} samples to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def download_batch(
        self,
        dataset_ids: List[str],
        samples_per_dataset: int = 10000
    ) -> Dict[str, str]:
        """
        Download multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to download
            samples_per_dataset: Max samples per dataset

        Returns:
            Dict mapping dataset_id -> output_path
        """
        results = {}

        for ds_id in dataset_ids:
            path = self.download(ds_id, samples=samples_per_dataset)
            if path:
                results[ds_id] = path

        return results

    def fill_gap(self, gap: int, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Discover and download datasets to fill a gap.

        Args:
            gap: Number of samples needed

        Yields:
            Downloaded dataset metadata
        """
        logger.info(f"HuggingFaceSource filling gap of {gap} samples")

        # Search and download top datasets
        downloaded = 0
        for result in self.discover(limit=10):
            if downloaded >= gap:
                break

            ds_id = result['id']
            samples_needed = min(gap - downloaded, 10000)

            logger.info(f"Downloading {ds_id} ({samples_needed} samples)")
            path = self.download(ds_id, samples=samples_needed)

            if path:
                downloaded += samples_needed
                yield {
                    'dataset_id': ds_id,
                    'path': path,
                    'samples': samples_needed,
                    'source': 'huggingface'
                }

        logger.info(f"HuggingFace sourcing complete: {downloaded} samples")


if __name__ == "__main__":
    # Test discovery
    source = HuggingFaceSource()

    print("Discovering mental health datasets on HuggingFace...")
    for result in source.discover(limit=5):
        print(f"  {result['id']}: {result['downloads']:,} downloads")
