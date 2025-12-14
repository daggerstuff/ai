"""
NGC Resources Downloader for Training Ready

Downloads NeMo resources and training-related assets from NGC catalog.
Integrates with training_ready pipeline for automated resource acquisition.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.utils.ngc_cli import NGCCLIAuthError, NGCCLINotFoundError, ensure_ngc_cli_configured

logger = logging.getLogger(__name__)


class NGCResourceDownloader:
    """
    Downloads NeMo and training resources from NGC catalog.
    """

    # Common NeMo resources used in training
    NEMO_RESOURCES = {
        "nemo-microservices-quickstart": {
            "path": "nvidia/nemo-microservices/nemo-microservices-quickstart",
            "default_version": "25.10",
            "description": "NeMo Microservices quickstart package",
        },
        "nemo-framework": {
            "path": "nvidia/nemo/nemo",
            "default_version": None,  # Use latest
            "description": "NeMo framework for training",
        },
        "nemo-megatron": {
            "path": "nvidia/nemo/nemo-megatron",
            "default_version": None,
            "description": "NeMo Megatron for large-scale training",
        },
    }

    def __init__(self, output_base: Path | None = None, api_key: str | None = None):
        """
        Initialize NGC resource downloader.

        Args:
            output_base: Base directory for downloads (defaults to training_ready/resources/)
            api_key: Optional NGC API key (if not set, will check environment or prompt)
        """
        if output_base is None:
            output_base = Path(__file__).parent.parent / "resources"
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NGC_API_KEY")

        try:
            self.cli = ensure_ngc_cli_configured(api_key=api_key)
        except (NGCCLINotFoundError, NGCCLIAuthError) as e:
            logger.warning(f"NGC CLI not available: {e}")
            self.cli = None

    def download_nemo_quickstart(
        self, version: str | None = None, output_dir: Path | None = None
    ) -> Path:
        """
        Download NeMo Microservices quickstart package.

        Args:
            version: Version to download (defaults to 25.10)
            output_dir: Output directory (defaults to resources/nemo-microservices/)

        Returns:
            Path to downloaded/extracted quickstart directory
        """
        if not self.cli:
            raise NGCCLINotFoundError("NGC CLI not available")

        if version is None:
            version = self.NEMO_RESOURCES["nemo-microservices-quickstart"]["default_version"]

        if output_dir is None:
            output_dir = self.output_base / "nemo-microservices"

        resource_path = self.NEMO_RESOURCES["nemo-microservices-quickstart"]["path"]

        logger.info(f"Downloading NeMo Microservices quickstart v{version}...")
        return self.cli.download_resource(
            resource_path=resource_path, version=version, output_dir=output_dir, extract=True
        )

    def download_nemo_framework(
        self, version: str | None = None, output_dir: Path | None = None
    ) -> Path:
        """
        Download NeMo framework.

        Args:
            version: Version to download
            output_dir: Output directory

        Returns:
            Path to downloaded framework
        """
        if not self.cli:
            raise NGCCLINotFoundError("NGC CLI not available")

        if output_dir is None:
            output_dir = self.output_base / "nemo-framework"

        resource_path = self.NEMO_RESOURCES["nemo-framework"]["path"]

        logger.info("Downloading NeMo framework...")
        return self.cli.download_resource(
            resource_path=resource_path, version=version, output_dir=output_dir, extract=True
        )

    def download_custom_resource(
        self, resource_path: str, version: str | None = None, output_dir: Path | None = None
    ) -> Path:
        """
        Download a custom resource from NGC catalog.

        Args:
            resource_path: Resource path (e.g., "nvidia/nemo-microservices/nemo-microservices-quickstart")
            version: Optional version tag
            output_dir: Optional output directory

        Returns:
            Path to downloaded resource
        """
        if not self.cli:
            raise NGCCLINotFoundError("NGC CLI not available")

        if output_dir is None:
            # Create directory from resource name
            resource_name = resource_path.split("/")[-1]
            output_dir = self.output_base / resource_name

        logger.info(f"Downloading {resource_path}...")
        return self.cli.download_resource(
            resource_path=resource_path, version=version, output_dir=output_dir, extract=True
        )


def download_nemo_quickstart(version: str | None = None, output_dir: Path | None = None) -> Path:
    """
    Convenience function to download NeMo Microservices quickstart.

    Args:
        version: Version to download (defaults to 25.10)
        output_dir: Output directory

    Returns:
        Path to downloaded quickstart directory
    """
    downloader = NGCResourceDownloader()
    return downloader.download_nemo_quickstart(version=version, output_dir=output_dir)
