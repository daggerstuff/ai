#!/usr/bin/env python3
"""
Download and catalog CPTSD sources from S3 bucket.

This script downloads CPTSD-related transcript files from the S3 bucket
and catalogs them with proper metadata for processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CPTSDSourceDownloader:
    """Download and catalog CPTSD sources from S3."""

    def __init__(self, output_dir: str = "ai/training/ready_packages/data/cptsd_sources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load S3 manifest
        manifest_path = Path("ai/training/ready_packages/data/s3_manifest.json")
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

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"✓ Downloaded: {s3_key}")
                return True
            else:
                print(f"✗ Failed to download: {s3_key}")
                print(f"  Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout downloading: {s3_key}")
            return False
        except Exception as e:
            print(f"✗ Error downloading {s3_key}: {e}")
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
            print(f"⚠ Warning: File not found in manifest: {s3_key}")
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
        from datetime import datetime

        return f"{datetime.utcnow().isoformat()}Z"

    def download_source(self, source: str, files: List[str]) -> Dict[str, Any]:
        """Download all files for a source."""
        print(f"\n{'=' * 60}")
        print(f"Downloading {source.replace('_', ' ').title()}")
        print(f"{'=' * 60}")

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

        print(f"\nSource Summary:")
        print(f"  Total files: {catalog['total_files']}")
        print(f"  Downloaded: {catalog['downloaded_files']}")
        print(f"  Failed: {catalog['failed_files']}")
        print(f"  Total size: {catalog['total_size_formatted']}")

        return catalog

    def download_all_sources(self) -> Dict[str, Any]:
        """Download all CPTSD sources."""
        print(f"\n{'=' * 60}")
        print("CPTSD Source Downloader")
        print(f"{'=' * 60}")
        print(f"Output directory: {self.output_dir}")

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

        print(f"\n{'=' * 60}")
        print("Download Summary")
        print(f"{'=' * 60}")
        print(f"Total sources: {master_catalog['summary']['total_sources']}")
        print(f"Total files: {master_catalog['summary']['total_files']}")
        print(f"Downloaded: {master_catalog['summary']['total_downloaded']}")
        print(f"Failed: {master_catalog['summary']['total_failed']}")
        print(f"Total size: {master_catalog['summary']['total_size_formatted']}")
        print(f"Catalog saved to: {catalog_path}")

        return master_catalog


def main():
    """Main entry point."""
    downloader = CPTSDSourceDownloader()
    catalog = downloader.download_all_sources()

    # Print next steps
    print(f"\n{'=' * 60}")
    print("Next Steps")
    print(f"{'=' * 60}")
    print(f"1. Review downloaded files in: {downloader.output_dir}")
    print("2. Analyze content for CPTSD topics")
    print("3. Create voice profiles for each source")
    print("4. Process files with enhanced chunking strategy")
    print("5. Generate synthetic examples for missing topics")


if __name__ == "__main__":
    main()
