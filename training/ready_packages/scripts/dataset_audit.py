#!/usr/bin/env python3
"""
Complete Dataset Audit & Discovery
Identifies all source files across 52.20GB dataset
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DatasetAuditor:
    """
    Comprehensive audit of all dataset sources for 52.20GB corpus
    """

    def __init__(self):
        self.audit_results = {
            "audit_timestamp": datetime.now().isoformat(),
            "total_size_bytes": 0,
            "total_files": 0,
            "files_by_category": {},
            "s3_manifest": {},
            "local_files": {},
            "missing_files": [],
            "file_types": {},
            "size_distribution": {},
        }

        # Key directories to audit
        self.search_paths = [
            "/home/vivi/pixelated/ai/training_ready/data",
            "/home/vivi/pixelated/ai/lightning",
            "/home/vivi/pixelated/ai/datasets",
            "/home/vivi/pixelated/ai/training_ready/configs",
            "/home/vivi/pixelated/ai/lightning/ghost",
        ]

    def load_s3_manifest(self) -> Dict[str, Any]:
        """Load S3 manifest to understand canonical structure"""
        manifest_path = Path(
            "/home/vivi/pixelated/ai/training_ready/data/s3_manifest.json"
        )

        if not manifest_path.exists():
            logger.warning("S3 manifest not found")
            return {}

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            self.audit_results["s3_manifest"] = {
                "bucket": manifest.get("bucket", "pixel-data"),
                "endpoint": manifest.get("endpoint"),
                "total_objects": manifest.get("total_objects", 0),
                "total_size_bytes": manifest.get("total_size_bytes", 0),
                "categories": list(manifest.get("categories", {}).keys()),
            }

            return manifest
        except Exception as e:
            logger.error(f"Error loading S3 manifest: {e}")
            return {}

    def scan_directory(
        self, base_path: str, extensions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Scan directory for dataset files"""
        if extensions is None:
            extensions = [".jsonl", ".json", ".csv", ".parquet", ".txt"]

        files_found = []

        try:
            base = Path(base_path)
            if not base.exists():
                logger.warning(f"Directory not found: {base_path}")
                return files_found

            for extension in extensions:
                pattern = f"**/*{extension}"
                for file_path in base.glob(pattern):
                    try:
                        stat = file_path.stat()
                        file_info = {
                            "path": str(file_path),
                            "relative_path": str(file_path.relative_to(base)),
                            "size_bytes": stat.st_size,
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "extension": extension,
                            "category": self._categorize_file(str(file_path)),
                        }
                        files_found.append(file_info)

                    except Exception as e:
                        logger.warning(f"Error accessing {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error scanning {base_path}: {e}")

        return files_found

    def _categorize_file(self, file_path: str) -> str:
        """Categorize file based on path and name"""
        path_lower = file_path.lower()

        if "cot" in path_lower or "reasoning" in path_lower:
            return "chain_of_thought"
        elif (
            "mental_health" in path_lower
            or "therapeutic" in path_lower
            or "counsel" in path_lower
        ):
            return "therapeutic_data"
        elif "reddit" in path_lower or "social" in path_lower:
            return "social_media"
        elif "priority" in path_lower:
            return "priority_datasets"
        elif "edge_case" in path_lower or "crisis" in path_lower:
            return "edge_cases"
        elif "voice" in path_lower or "transcript" in path_lower:
            return "voice_data"
        elif "psych" in path_lower:
            return "psychological_data"
        elif "final" in path_lower or "ultimate" in path_lower:
            return "consolidated"
        elif "stage" in path_lower:
            return "training_stages"
        else:
            return "uncategorized"

    def audit_large_files(self) -> List[Dict[str, Any]]:
        """Find and audit large dataset files (>10MB)"""
        large_files = []

        for search_path in self.search_paths:
            files = self.scan_directory(search_path)

            for file_info in files:
                if file_info["size_bytes"] > 10 * 1024 * 1024:  # >10MB
                    large_files.append(file_info)

                    # Update audit results
                    category = file_info["category"]
                    if category not in self.audit_results["files_by_category"]:
                        self.audit_results["files_by_category"][category] = []

                    self.audit_results["files_by_category"][category].append(file_info)
                    self.audit_results["total_size_bytes"] += file_info["size_bytes"]
                    self.audit_results["total_files"] += 1

                    # Track by extension
                    ext = file_info["extension"]
                    self.audit_results["file_types"][ext] = (
                        self.audit_results["file_types"].get(ext, 0) + 1
                    )

        # Sort large files by size
        large_files.sort(key=lambda x: x["size_bytes"], reverse=True)
        return large_files

    def create_file_manifest(self) -> Dict[str, Any]:
        """Create comprehensive file manifest"""

        # Load S3 structure
        s3_data = self.load_s3_manifest()

        # Audit actual files
        large_files = self.audit_large_files()

        # Create consolidated manifest
        manifest = {
            "total_real_size_bytes": self.audit_results["total_size_bytes"],
            "total_real_files": self.audit_results["total_files"],
            "s3_claimed_size": s3_data.get("total_size_bytes", 0),
            "s3_claimed_objects": s3_data.get("total_objects", 0),
            "size_discrepancy": s3_data.get("total_size_bytes", 0)
            - self.audit_results["total_size_bytes"],
            "files_by_category": self.audit_results["files_by_category"],
            "file_types": self.audit_results["file_types"],
            "top_files": large_files[:50],  # Top 50 largest files
            "all_files": large_files,
            "processing_plan": self._create_processing_plan(large_files),
            "missing_categories": self._identify_missing_data(s3_data),
        }

        return manifest

    def _create_processing_plan(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create processing plan for 52.20GB dataset"""

        # Group files by category for processing
        processing_order = {
            "consolidated": [],  # Start with already merged files
            "priority_datasets": [],  # High-priority therapeutic data
            "therapeutic_data": [],  # Core mental health data
            "chain_of_thought": [],  # Reasoning datasets
            "psychological_data": [],  # Psychological datasets
            "edge_cases": [],  # Crisis scenarios
            "voice_data": [],  # Tim Fletcher transcripts
            "social_media": [],  # Reddit mental health
            "training_stages": [],  # Staged training data
            "uncategorized": [],  # Remaining files
        }

        for file_info in files:
            category = file_info["category"]
            if category in processing_order:
                processing_order[category].append(file_info)

        # Calculate processing batches
        total_size = sum(f["size_bytes"] for f in files)

        plan = {
            "total_size_gb": round(total_size / (1024**3), 2),
            "processing_order": processing_order,
            "estimated_memory_needed": round(
                total_size * 1.5 / (1024**3), 2
            ),  # 1.5x for processing
            "recommended_batch_size": self._calculate_batch_sizes(files),
            "streaming_required": total_size > 8 * 1024**3,  # >8GB needs streaming
        }

        return plan

    def _calculate_batch_sizes(
        self, files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate optimal batch sizes for processing"""
        batches = []
        current_batch = []
        current_size = 0
        max_batch_size = 2 * 1024**3  # 2GB per batch

        for file_info in sorted(files, key=lambda x: x["size_bytes"], reverse=True):
            if (
                current_size + file_info["size_bytes"] > max_batch_size
                and current_batch
            ):
                batches.append(
                    {
                        "files": current_batch,
                        "total_size": current_size,
                        "file_count": len(current_batch),
                    }
                )
                current_batch = [file_info]
                current_size = file_info["size_bytes"]
            else:
                current_batch.append(file_info)
                current_size += file_info["size_bytes"]

        if current_batch:
            batches.append(
                {
                    "files": current_batch,
                    "total_size": current_size,
                    "file_count": len(current_batch),
                }
            )

        return batches

    def _identify_missing_data(self, s3_data: Dict[str, Any]) -> List[str]:
        """Identify what might be missing from local vs S3"""
        missing = []

        # Check if we have files for each S3 category
        s3_categories = s3_data.get("categories", {})
        local_categories = set(self.audit_results["files_by_category"].keys())

        for category in s3_categories:
            if category not in local_categories:
                missing.append(category)

        return missing

    def save_audit_report(self, output_path: str) -> None:
        """Save comprehensive audit report"""
        manifest = self.create_file_manifest()

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        print(f"🎯 Audit complete!")
        print(f"📊 Found {manifest['total_real_files']} real files")
        print(f"📈 Total size: {manifest['total_real_size_gb']} GB")
        print(
            f"📋 Processing plan: {len(manifest['processing_plan']['processing_order'])} categories"
        )
        print(f"📄 Report saved: {output_path}")


def main():
    """Run complete dataset audit"""
    auditor = DatasetAuditor()
    auditor.save_audit_report(
        "/home/vivi/pixelated/ai/training_ready/data/dataset_audit_report.json"
    )


if __name__ == "__main__":
    main()
