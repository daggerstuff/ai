"""
NGC Dataset Ingestor

Ingests datasets and resources from NVIDIA GPU Cloud (NGC) catalog.
Similar to HuggingFace ingestor but for NGC resources.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.utils.ngc_cli import (
    NGCCLIAuthError,
    NGCCLIDownloadError,
    NGCCLINotFoundError,
    ensure_ngc_cli_configured,
)

logger = logging.getLogger(__name__)


class NGCIngestor:
    """
    Ingests datasets and resources from NGC catalog.

    Similar to HuggingFaceIngestor but for NGC resources.
    """

    # NGC resources that can be used as training datasets
    NGC_DATASETS: ClassVar[dict[str, dict[str, Any]]] = {
        # Format: "resource_path": {"target": "folder", "version": "version", "description": "..."}
        # NeMo training datasets (if available)
        "nvidia/nemo/datasets/therapeutic_conversations": {
            "target": "stage1_foundation",
            "version": None,  # Use latest
            "description": "Therapeutic conversation dataset from NeMo",
        },
        # Add more NGC datasets as they become available
    }

    def __init__(self, output_base: Path | None = None, api_key: str | None = None):
        """
        Initialize NGC ingestor.

        Args:
            output_base: Base output directory (defaults to dataset_pipeline/data/ngc/)
            api_key: Optional NGC API key
        """
        if output_base is None:
            output_base = Path(__file__).parent.parent / "data" / "ngc"
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NGC_API_KEY")

        try:
            self.cli = ensure_ngc_cli_configured(api_key=api_key)
            logger.info("NGC CLI initialized successfully")
        except (NGCCLINotFoundError, NGCCLIAuthError) as e:
            logger.warning(f"NGC CLI not available: {e}")
            self.cli = None

    def is_available(self) -> bool:
        """Check if NGC CLI is available"""
        return self.cli is not None

    def download_dataset(
        self, resource_path: str, version: str | None = None, target_folder: str | None = None
    ) -> Path:
        """
        Download a dataset from NGC catalog.

        Args:
            resource_path: Resource path (e.g., "nvidia/nemo/datasets/therapeutic_conversations")
            version: Optional version tag
            target_folder: Optional target folder name (defaults to resource name)

        Returns:
            Path to downloaded dataset
        """
        if not self.cli:
            raise NGCCLINotFoundError("NGC CLI not available")

        if target_folder is None:
            target_folder = resource_path.split("/")[-1]

        output_dir = self.output_base / target_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading NGC dataset: {resource_path}...")
        try:
            downloaded_path = self.cli.download_resource(
                resource_path=resource_path, version=version, output_dir=output_dir, extract=True
            )
            logger.info(f"Successfully downloaded to {downloaded_path}")
            return downloaded_path
        except NGCCLIDownloadError as e:
            logger.error(f"Failed to download {resource_path}: {e}")
            raise

    def process_dataset(self, resource_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Process a dataset from NGC catalog.

        Args:
            resource_path: Resource path
            config: Configuration dict with target, version, etc.

        Returns:
            Processing result dictionary
        """
        target = config.get("target", "stage1_foundation")
        version = config.get("version")

        try:
            downloaded_path = self.download_dataset(
                resource_path=resource_path, version=version, target_folder=target
            )

            return {
                "status": "success",
                "resource_path": resource_path,
                "version": version,
                "target": target,
                "downloaded_path": str(downloaded_path),
                "type": "ngc_external",
            }
        except Exception as e:
            logger.error(f"Failed to process {resource_path}: {e}")
            return {
                "status": "error",
                "resource_path": resource_path,
                "error": str(e),
                "type": "ngc_external",
            }

    def process_all_configured(self) -> dict[str, dict[str, Any]]:
        """
        Process all configured NGC datasets.

        Returns:
            Dictionary mapping resource paths to processing results
        """
        if not self.is_available():
            logger.warning("NGC CLI not available, skipping NGC dataset ingestion")
            return {}

        results = {}
        for resource_path, config in self.NGC_DATASETS.items():
            logger.info(f"Processing NGC dataset: {resource_path}")
            results[resource_path] = self.process_dataset(resource_path, config)

        return results

    def list_available_datasets(
        self, org: str | None = None, team: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List available datasets in NGC catalog.

        Args:
            org: Optional organization filter
            team: Optional team filter

        Returns:
            List of available dataset dictionaries
        """
        return self.cli.list_resources(org=org, team=team) if self.cli else []


def ingest_ngc_datasets(output_base: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Convenience function to ingest all configured NGC datasets.

    Args:
        output_base: Base output directory

    Returns:
        Dictionary of processing results
    """
    ingestor = NGCIngestor(output_base=output_base)
    return ingestor.process_all_configured()
