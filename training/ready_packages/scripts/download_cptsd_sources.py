#!/usr/bin/env python3
"""
Download and catalog CPTSD sources from S3 bucket.

This script downloads CPTSD-related transcript files from the S3 bucket
and catalogs them with proper metadata for processing.
"""


import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CPTSDSourceDownloader:
    """Download and catalog CPTSD sources from S3."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = os.environ.get(
                "CPTSD_DATA_DIR", "ai/training/ready_packages/data/cptsd_sources"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load S3 manifest
        manifest_path = Path(
            os.environ.get("S3_MANIFEST_PATH", "ai/training/ready_packages/data/s3_manifest.json")
        )
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # CPTSD sources to download
        self.cptsd_sources = {
            "heidi_priebe": [
                "datasets/gdrive/tier4_voice_persona/Heidi Priebe/10 'Survival Lies' You May Tell If You Have CPTSD.txt",
                "datasets/gdrive/tier4_voice_persona/Heidi Priebe/CPTSD： Breaking The Toxic Shame⧸Procrastination Cycle With Self-Compassion.txt",
            ],
            "patrick_teahan": [
                "datasets/gdrive/tier4_voice_persona/Patrick Teahan /9 Random Examples of Shame from PTSD & CPTSD.txt"
            ],
            "tim_fletcher": [
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/'Big T' vs 'Little t' Trauma | Understanding Trauma - Part 28 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Betrayal Trauma | Understanding Trauma - Part 13 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Depletion and Self-Care | Re-Parenting - Part 89 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Healing | Understanding Trauma - Part 10 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Recovery and Letting Go | Re-Parenting - Part 93 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Sex | Re-Parenting - Part 25 | #complextrauma.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/These 12 Needs Are Non-Negotiable for a Healthy & Whole Life | #complextrauma #whatstuck.txt",
                "datasets/gdrive/tier4_voice_persona/Tim Fletcher/Trauma Bonding | Understanding Trauma - Part 31 | #complextrauma.txt",
            ],
        }

    def download_file_from_s3(self, s3_key: str, local_path: Path) -> bool:
        """Download a file from S3 using ovhai CLI."""
        try:
            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Use ovhai CLI to download
            cmd = ["ovhai", "object", "download", "pixel-data", s3_key, str(local_path)]

            logger.info(f"Downloading {s3_key} to {local_path}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=False)

            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {s3_key}")
                return True
            else:
                logger.error(f"✗ Failed to download: {s3_key}")
                logger.error(f"  Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout downloading: {s3_key}")
            return False
        except Exception:
            logger.exception(f"✗ Error downloading {s3_key}")
            return False

    def catalog_file(self, source: str, s3_key: str, local_path: Path) -> Dict[str, Any]:
        """Catalog a downloaded file with metadata."""
        # Get file info from manifest
        file_info = None
        file_info = next(
            (
                obj
                for obj in self.manifest.get("categories", {})
                .get("gdrive", {})
                .get("raw", {})
                .get("objects", [])
                if obj.get("key") == s3_key
            ),
            None,
        )

        if not file_info:
            logger.warning(f"File not found in manifest: {s3_key}")
            file_info = {
                "key": s3_key,
                "size": local_path.stat().st_size if local_path.exists() else 0,
                "size_formatted": f"{local_path.stat().st_size / 1024:.2f} KB"
                if local_path.exists()
                else "0.00 KB",
            }

        # Extract topics from filename
        topics = self._extract_topics(s3_key)

        # Create catalog entry
        # Create catalog entry
        # Create catalog entry
        return {
            "source": source,
            "s3_key": s3_key,
            "local_path": str(local_path),
            "size_bytes": file_info.get("size", 0),
            "size_formatted": file_info.get("size_formatted", "0.00 KB"),
            "topics": topics,
            "downloaded": local_path.exists(),
            "cataloged_at": self._get_timestamp(),
        }

    def _extract_topics(self, filename: str) -> List[str]:
        """Extract CPTSD topics from filename."""
        filename_lower = filename.lower()

        # CPTSD topic keywords
        topic_keywords = {
            "survival lies": ["survival lies", "survival_lies"],
            "toxic shame": ["toxic shame", "toxic_shame"],
            "procrastination": ["procrastination"],
            "self-compassion": ["self-compassion", "self_compassion"],
            "shame": ["shame"],
            "ptsd": ["ptsd"],
            "cptsd": ["cptsd", "complex ptsd", "complex_trauma"],
            "big t trauma": ["big t trauma", "big_t_trauma"],
            "little t trauma": ["little t trauma", "little_t_trauma"],
            "betrayal trauma": ["betrayal trauma", "betrayal_trauma"],
            "depletion": ["depletion"],
            "self-care": ["self-care", "self_care"],
            "healing": ["healing"],
            "recovery": ["recovery"],
            "letting go": ["letting go", "letting_go"],
            "sex": ["sex", "sexuality"],
            "needs": ["needs"],
            "trauma bonding": ["trauma bonding", "trauma_bonding"],
            "re-parenting": ["re-parenting", "re_parenting", "reparenting"],
        }

        return [
            topic
            for topic, keywords in topic_keywords.items()
            if any(keyword in filename_lower for keyword in keywords)
        ]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return f"{datetime.now(timezone.utc).isoformat()}Z"

    def download_source(self, source: str, files: List[str]) -> Dict[str, Any]:
        """Download all files for a source."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Downloading {source.replace('_', ' ').title()}")
        logger.info(f"{'=' * 60}")

        source_dir = self.output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)

        catalog = {
            "source": source,
            "total_files": len(files),
            "downloaded_files": 0,
            "failed_files": 0,
            "total_size_bytes": 0,
            "files": [],
        }

        for s3_key in files:
            # Create local filename from S3 key
            filename = Path(s3_key).name
            local_path = source_dir / filename

            # Download file
            success = self.download_file_from_s3(s3_key, local_path)

            # Catalog file
            catalog_entry = self.catalog_file(source, s3_key, local_path)
            catalog["files"].append(catalog_entry)

            if success:
                catalog["downloaded_files"] += 1
                catalog["total_size_bytes"] += catalog_entry["size_bytes"]
            else:
                catalog["failed_files"] += 1

        catalog["total_size_formatted"] = f"{catalog['total_size_bytes'] / 1024:.2f} KB"

        logger.info("\nSource Summary:")
        logger.info(f"  Total files: {catalog['total_files']}")
        logger.info(f"  Downloaded: {catalog['downloaded_files']}")
        logger.info(f"  Failed: {catalog['failed_files']}")
        logger.info(f"  Total size: {catalog['total_size_formatted']}")

        return catalog

    def download_all_sources(self) -> Dict[str, Any]:
        """Download all CPTSD sources."""
        self._log_section_header("CPTSD Source Downloader")
        logger.info(f"Output directory: {self.output_dir}")

        master_catalog = {
            "download_started_at": self._get_timestamp(),
            "output_directory": str(self.output_dir),
            "sources": {},
        }

        total_files = 0
        total_downloaded = 0
        total_size = 0

        for source, files in self.cptsd_sources.items():
            catalog = self.download_source(source, files)
            master_catalog["sources"][source] = catalog

            total_files += catalog["total_files"]
            total_downloaded += catalog["downloaded_files"]
            total_size += catalog["total_size_bytes"]

        master_catalog["summary"] = {
            "total_sources": len(self.cptsd_sources),
            "total_files": total_files,
            "total_downloaded": total_downloaded,
            "total_failed": total_files - total_downloaded,
            "total_size_bytes": total_size,
            "total_size_formatted": f"{total_size / 1024:.2f} KB",
            "download_completed_at": self._get_timestamp(),
        }

        # Save master catalog
        catalog_path = self.output_dir / "download_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(master_catalog, f, indent=2)

        self._log_section_header("Download Summary")
        logger.info(f"Total sources: {master_catalog['summary']['total_sources']}")
        logger.info(f"Total files: {master_catalog['summary']['total_files']}")
        logger.info(f"Downloaded: {master_catalog['summary']['total_downloaded']}")
        logger.info(f"Failed: {master_catalog['summary']['total_failed']}")
        logger.info(f"Total size: {master_catalog['summary']['total_size_formatted']}")
        logger.info(f"Catalog saved to: {catalog_path}")

        return master_catalog

    def _log_section_header(self, title: str) -> None:
        """Log a formatted section header."""
        logger.info("\n" + "=" * 60)
        logger.info(title)
        logger.info("=" * 60)


def main():
    """Main entry point."""
    downloader = CPTSDSourceDownloader()
    downloader.download_all_sources()

    # Print next steps
    logger.info(f"\n{'=' * 60}")
    logger.info("Next Steps")
    logger.info(f"{'=' * 60}")
    logger.info(f"1. Review downloaded files in: {downloader.output_dir}")
    logger.info("2. Analyze content for CPTSD topics")
    logger.info("3. Create voice profiles for each source")
    logger.info("4. Process files with enhanced chunking strategy")
    logger.info("5. Generate synthetic examples for missing topics")


if __name__ == "__main__":
    main()
