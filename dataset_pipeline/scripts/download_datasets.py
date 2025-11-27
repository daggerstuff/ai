#!/usr/bin/env python3
"""
Dataset Download Script for Pixelated Empathy

Downloads and organizes HuggingFace datasets for mental health training.
Stores datasets in ai/datasets/ with proper tier organization.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: uv pip install datasets huggingface-hub")
    sys.exit(1)


@dataclass
class DatasetConfig:
    """Configuration for a dataset to download."""
    name: str
    hf_id: str
    tier: int
    description: str
    expected_size_mb: float = 0
    subset: str | None = None
    split: str | None = None
    trust_remote_code: bool = False


# Tier 2: Professional Therapeutic Datasets
TIER2_DATASETS = [
    DatasetConfig(
        name="Psych8k-Modified",
        hf_id="Hussain033/Psych8k-Modified",
        tier=2,
        description="Alexander Street professional therapy conversations (modified version)",
        expected_size_mb=6.3,
    ),
    DatasetConfig(
        name="mental_health_counseling_conversations",
        hf_id="Amod/mental_health_counseling_conversations",
        tier=2,
        description="Licensed therapist responses (~3.5K conversations)",
        expected_size_mb=4.5,
    ),
    DatasetConfig(
        name="SoulChatCorpus",
        hf_id="scutcyr/SoulChatCorpus",
        tier=2,
        description="Advanced psychological counselor digital twin framework",
        expected_size_mb=76,
    ),
    DatasetConfig(
        name="counsel-chat",
        hf_id="nbertagnolli/counsel-chat",
        tier=2,
        description="Professional counseling conversation archive",
        expected_size_mb=4,
    ),
    DatasetConfig(
        name="mental_health_chatbot_dataset",
        hf_id="heliosbrahma/mental_health_chatbot_dataset",
        tier=2,
        description="Mental health chatbot training dataset",
        expected_size_mb=10,
    ),
    DatasetConfig(
        name="Synthetic_Therapy_Conversations",
        hf_id="Mr-Bhaskar/Synthetic_Therapy_Conversations",
        tier=2,
        description="Synthetic therapy conversation dataset",
        expected_size_mb=50,
    ),
    DatasetConfig(
        name="neuro_qa_SFT_Trainer",
        hf_id="wesley7137/neuro_qa_SFT_Trainer",
        tier=2,
        description="Neurology/psychology Q&A for SFT training",
        expected_size_mb=6,
    ),
    DatasetConfig(
        name="motivational-interviewing-therapy",
        hf_id="to-be/annomi-motivational-interviewing-therapy-conversations",
        tier=2,
        description="Motivational interviewing therapy conversations (ANNOMI)",
        expected_size_mb=15,
    ),
    DatasetConfig(
        name="phr-mental-therapy-dataset",
        hf_id="vibhorag101/phr-mental-therapy-dataset-conversational-format",
        tier=2,
        description="PHR mental therapy dataset in conversational format",
        expected_size_mb=30,
    ),
    DatasetConfig(
        name="therapy-conversations-combined",
        hf_id="IINOVAII/therapy-conversations-combined",
        tier=2,
        description="Combined therapy conversations dataset",
        expected_size_mb=20,
    ),
]

# Tier 6: Knowledge Base Datasets
TIER6_DATASETS = [
    DatasetConfig(
        name="psychology-10k",
        hf_id="samhog/psychology-10k",
        tier=6,
        description="Comprehensive psychology knowledge base",
        expected_size_mb=5.2,
    ),
    DatasetConfig(
        name="Psych-101",
        hf_id="marcelbinz/Psych-101",
        tier=6,
        description="Psychology training prompts and fundamentals",
        expected_size_mb=800,
    ),
    DatasetConfig(
        name="xmu_psych_books",
        hf_id="Genius-Society/xmu_psych_books",
        tier=6,
        description="Psychology textbook data corpus (Xiamen University)",
        expected_size_mb=0.043,
    ),
    DatasetConfig(
        name="customized-mental-health-snli2",
        hf_id="MasonYan/customized-mental-health-snli2",
        tier=6,
        description="Mental health natural language inference",
        expected_size_mb=5,
    ),
]

# Tier 4: Reddit Mental Health Archive
TIER4_DATASETS = [
    DatasetConfig(
        name="reddit_mental_health_posts",
        hf_id="solomonk/reddit_mental_health_posts",
        tier=4,
        description="Reddit mental health posts compilation",
        expected_size_mb=100,
    ),
    DatasetConfig(
        name="mental_health_reddit_posts",
        hf_id="jsfactory/mental_health_reddit_posts",
        tier=4,
        description="Mental health Reddit posts dataset",
        expected_size_mb=50,
    ),
    DatasetConfig(
        name="reddit-mental-health-classification",
        hf_id="kamruzzaman-asif/reddit-mental-health-classification",
        tier=4,
        description="Reddit mental health classification dataset",
        expected_size_mb=30,
    ),
    DatasetConfig(
        name="reddit-depression-cleaned",
        hf_id="hugginglearners/reddit-depression-cleaned",
        tier=4,
        description="Cleaned Reddit depression posts",
        expected_size_mb=20,
    ),
    DatasetConfig(
        name="suicide-detection",
        hf_id="Ram07/Detection-for-Suicide",
        tier=4,
        description="Suicide detection dataset",
        expected_size_mb=50,
    ),
    DatasetConfig(
        name="suicide-depression-detection",
        hf_id="joshyii/suicide_depression_detection",
        tier=4,
        description="Suicide and depression detection dataset",
        expected_size_mb=30,
    ),
]

# Additional Research Datasets (Tier 5)
TIER5_DATASETS = [
    DatasetConfig(
        name="RECCON",
        hf_id="martinagvilas/daily-dialog-emotion-cause",
        tier=5,
        description="Emotion cause extraction in conversations",
        expected_size_mb=2,
    ),
    DatasetConfig(
        name="empathetic_dialogues",
        hf_id="empathetic_dialogues",
        tier=5,
        description="Empathetic dialogue dataset",
        expected_size_mb=40,
    ),
    DatasetConfig(
        name="therapist-sft-format",
        hf_id="wesley7137/therapist-sft-format",
        tier=5,
        description="Therapist SFT format training data",
        expected_size_mb=50,
    ),
    DatasetConfig(
        name="mental-health-snli",
        hf_id="iqrakiran/customized-mental-health-snli3",
        tier=5,
        description="Mental health natural language inference",
        expected_size_mb=5,
    ),
    DatasetConfig(
        name="iemocap-emotion-recognition",
        hf_id="AudioLLMs/iemocap_emotion_recognition",
        tier=5,
        description="IEMOCAP emotion recognition dataset",
        expected_size_mb=20,
    ),
]


class DatasetDownloader:
    """Downloads and manages HuggingFace datasets."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.download_log: list[dict[str, Any]] = []
        self.api = HfApi()

    def get_tier_path(self, tier: int) -> Path:
        """Get the path for a specific tier."""
        tier_names = {
            1: "tier1_priority",
            2: "tier2_professional",
            3: "tier3_cot_reasoning",
            4: "tier4_reddit_archive",
            5: "tier5_research",
            6: "tier6_knowledge",
        }
        return self.base_path / tier_names.get(tier, f"tier{tier}")

    def download_dataset(self, config: DatasetConfig) -> dict[str, Any]:
        """Download a single dataset."""
        tier_path = self.get_tier_path(config.tier)
        tier_path.mkdir(parents=True, exist_ok=True)

        dataset_path = tier_path / config.name
        dataset_path.mkdir(parents=True, exist_ok=True)

        result = {
            "name": config.name,
            "hf_id": config.hf_id,
            "tier": config.tier,
            "path": str(dataset_path),
            "status": "pending",
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "size_bytes": 0,
            "num_rows": 0,
        }

        print(f"\n{'='*60}")
        print(f"Downloading: {config.name}")
        print(f"  HuggingFace ID: {config.hf_id}")
        print(f"  Tier: {config.tier}")
        print(f"  Description: {config.description}")
        print(f"  Target: {dataset_path}")
        print(f"{'='*60}")

        try:
            # Load dataset from HuggingFace
            load_kwargs = {
                "trust_remote_code": config.trust_remote_code,
            }

            if config.subset:
                load_kwargs["name"] = config.subset

            if config.split:
                load_kwargs["split"] = config.split

            print(f"  Loading from HuggingFace...")
            dataset = load_dataset(config.hf_id, **load_kwargs)

            # Handle different return types
            if isinstance(dataset, Dataset):
                # Single split returned
                splits = {"train": dataset}
            else:
                # DatasetDict with multiple splits
                splits = dict(dataset)

            total_rows = 0
            total_size = 0

            for split_name, split_data in splits.items():
                output_file = dataset_path / f"{split_name}.jsonl"
                print(f"  Saving {split_name} split ({len(split_data)} rows)...")

                # Save as JSONL
                split_data.to_json(str(output_file), orient="records", lines=True)

                file_size = output_file.stat().st_size
                total_size += file_size
                total_rows += len(split_data)

                print(f"    Saved: {output_file.name} ({file_size / 1024 / 1024:.2f} MB)")

            # Also save dataset info
            info_file = dataset_path / "dataset_info.json"
            info = {
                "name": config.name,
                "hf_id": config.hf_id,
                "description": config.description,
                "downloaded": datetime.now().isoformat(),
                "splits": list(splits.keys()),
                "total_rows": total_rows,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / 1024 / 1024,
            }

            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)

            result["status"] = "success"
            result["size_bytes"] = total_size
            result["num_rows"] = total_rows
            result["splits"] = list(splits.keys())

            print(f"  ✓ Success: {total_rows} rows, {total_size / 1024 / 1024:.2f} MB")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"  ✗ Error: {e}")

        self.download_log.append(result)
        return result

    def download_all(self, datasets: list[DatasetConfig]) -> list[dict[str, Any]]:
        """Download all datasets in a list."""
        results = []
        for config in datasets:
            result = self.download_dataset(config)
            results.append(result)
        return results

    def verify_datasets(self) -> dict[str, Any]:
        """Verify all downloaded datasets."""
        verification = {
            "timestamp": datetime.now().isoformat(),
            "datasets": [],
            "summary": {
                "total": 0,
                "valid": 0,
                "missing": 0,
                "corrupted": 0,
            }
        }

        all_datasets = TIER2_DATASETS + TIER4_DATASETS + TIER5_DATASETS + TIER6_DATASETS

        for config in all_datasets:
            tier_path = self.get_tier_path(config.tier)
            dataset_path = tier_path / config.name

            status = {
                "name": config.name,
                "tier": config.tier,
                "path": str(dataset_path),
                "exists": dataset_path.exists(),
                "valid": False,
                "files": [],
                "total_size_mb": 0,
            }

            verification["summary"]["total"] += 1

            if dataset_path.exists():
                # Check for JSONL files
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                json_files = list(dataset_path.glob("*.json"))

                for f in jsonl_files + json_files:
                    file_info = {
                        "name": f.name,
                        "size_mb": f.stat().st_size / 1024 / 1024,
                    }
                    status["files"].append(file_info)
                    status["total_size_mb"] += file_info["size_mb"]

                if jsonl_files or json_files:
                    status["valid"] = True
                    verification["summary"]["valid"] += 1
                else:
                    verification["summary"]["corrupted"] += 1
            else:
                verification["summary"]["missing"] += 1

            verification["datasets"].append(status)

        return verification

    def generate_report(self) -> str:
        """Generate a human-readable report."""
        verification = self.verify_datasets()

        lines = [
            "# Dataset Download Verification Report",
            f"\nGenerated: {verification['timestamp']}",
            f"\n## Summary",
            f"- Total datasets: {verification['summary']['total']}",
            f"- Valid: {verification['summary']['valid']}",
            f"- Missing: {verification['summary']['missing']}",
            f"- Corrupted: {verification['summary']['corrupted']}",
            "\n## Dataset Status",
        ]

        for tier in [2, 4, 5, 6]:
            tier_datasets = [d for d in verification["datasets"] if d["tier"] == tier]
            if tier_datasets:
                lines.append(f"\n### Tier {tier}")
                for ds in tier_datasets:
                    status = "✓" if ds["valid"] else "✗"
                    lines.append(f"\n**{ds['name']}** {status}")
                    lines.append(f"  - Path: `{ds['path']}`")
                    lines.append(f"  - Size: {ds['total_size_mb']:.2f} MB")
                    if ds["files"]:
                        lines.append(f"  - Files: {', '.join(f['name'] for f in ds['files'])}")

        return "\n".join(lines)

    def save_download_log(self):
        """Save download log to file."""
        log_file = self.base_path / "download_log.json"
        with open(log_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "downloads": self.download_log,
            }, f, indent=2)
        print(f"\nDownload log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for Pixelated Empathy"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent.parent / "datasets",
        help="Base path for datasets (default: ai/datasets/)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[2, 4, 5, 6],
        help="Download only specific tier",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Download only specific dataset by name",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing datasets without downloading",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate verification report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(args.base_path)

    if args.verify or args.report:
        verification = downloader.verify_datasets()
        if args.report:
            print(downloader.generate_report())
        else:
            print(json.dumps(verification, indent=2))
        return

    # Determine which datasets to download
    datasets_to_download = []

    if args.dataset:
        # Find specific dataset
        all_datasets = TIER2_DATASETS + TIER4_DATASETS + TIER5_DATASETS + TIER6_DATASETS
        for ds in all_datasets:
            if ds.name.lower() == args.dataset.lower():
                datasets_to_download.append(ds)
                break
        if not datasets_to_download:
            print(f"Dataset not found: {args.dataset}")
            print("Available datasets:")
            for ds in all_datasets:
                print(f"  - {ds.name} (Tier {ds.tier})")
            sys.exit(1)
    elif args.tier:
        # Download specific tier
        if args.tier == 2:
            datasets_to_download = TIER2_DATASETS
        elif args.tier == 4:
            datasets_to_download = TIER4_DATASETS
        elif args.tier == 5:
            datasets_to_download = TIER5_DATASETS
        elif args.tier == 6:
            datasets_to_download = TIER6_DATASETS
    else:
        # Download all
        datasets_to_download = TIER2_DATASETS + TIER4_DATASETS + TIER5_DATASETS + TIER6_DATASETS

    if args.dry_run:
        print("Dry run - would download:")
        for ds in datasets_to_download:
            print(f"  - {ds.name} ({ds.hf_id}) -> Tier {ds.tier}")
        return

    print(f"Downloading {len(datasets_to_download)} datasets to {args.base_path}")
    print("=" * 60)

    downloader.download_all(datasets_to_download)
    downloader.save_download_log()

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    success = sum(1 for d in downloader.download_log if d["status"] == "success")
    failed = sum(1 for d in downloader.download_log if d["status"] == "error")

    print(f"Successful: {success}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed downloads:")
        for d in downloader.download_log:
            if d["status"] == "error":
                print(f"  - {d['name']}: {d['error']}")


if __name__ == "__main__":
    main()

