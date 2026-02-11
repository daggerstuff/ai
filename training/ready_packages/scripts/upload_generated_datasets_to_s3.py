#!/usr/bin/env python3
"""
Upload generated datasets to S3 canonical locations.

Uploads:
- edge_case_synthetic.jsonl → s3://pixel-data/gdrive/processed/edge_cases/synthetic.jsonl
- long_running_therapy.jsonl → s3://pixel-data/gdrive/processed/professional_therapeutic/long_running_therapy.jsonl
- cptsd_transcripts.jsonl → s3://pixel-data/gdrive/processed/edge_cases/cptsd/cptsd_transcripts.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def upload_file_to_s3(
    loader: S3DatasetLoader,
    local_path: Path,
    s3_key: str,
    bucket: str = "pixel-data",
) -> bool:
    """Upload a local file to S3"""
    if not local_path.exists():
        logger.error(f"❌ Local file not found: {local_path}")
        return False

    s3_path = f"s3://{bucket}/{s3_key}"
    logger.info(f"📤 Uploading {local_path.name} to {s3_path}...")

    try:
        with open(local_path, "rb") as f:
            loader.s3_client.upload_fileobj(
                Fileobj=f,
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={"ContentType": "application/x-ndjson"},
            )
        file_size = local_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   ✅ Uploaded ({file_size:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"   ❌ Upload failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload generated datasets to S3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--bucket",
        default="pixel-data",
        help="S3 bucket name (default: pixel-data)",
    )

    args = parser.parse_args()

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=args.bucket)

    # Define upload mappings
    project_root = Path(__file__).parents[3]
    generated_dir = project_root / "ai" / "training_ready" / "data" / "generated"

    uploads = [
        {
            "local": generated_dir / "edge_case_synthetic.jsonl",
            "s3_key": "datasets/gdrive/processed/edge_cases/synthetic.jsonl",
            "description": "Edge case synthetic dataset",
        },
        {
            "local": generated_dir / "long_running_therapy.jsonl",
            "s3_key": "datasets/gdrive/processed/professional_therapeutic/long_running_therapy.jsonl",
            "description": "Long-running therapy sessions",
        },
        {
            "local": generated_dir / "cptsd_transcripts.jsonl",
            "s3_key": "datasets/gdrive/processed/edge_cases/cptsd/cptsd_transcripts.jsonl",
            "description": "CPTSD transcripts dataset",
        },
        {
            "local": generated_dir / "roleplay_simulator_preferences.jsonl",
            "s3_key": "datasets/gdrive/processed/preference_pairs/roleplay_simulator_preferences.jsonl",
            "description": "Roleplay/simulator preference pairs",
        },
        {
            "local": generated_dir / "edge_case_adversarial_preferences.jsonl",
            "s3_key": "datasets/gdrive/processed/preference_pairs/edge_case_adversarial_preferences.jsonl",
            "description": "Edge case adversarial preference pairs",
        },
        {
            "local": generated_dir / "dpo_preference_pairs.jsonl",
            "s3_key": "datasets/gdrive/processed/preference_pairs/dpo_preference_pairs.jsonl",
            "description": "General DPO preference pairs",
        },
    ]

    logger.info("=" * 80)
    logger.info("📤 Upload Generated Datasets to S3")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be uploaded")
        logger.info("")

    success_count = 0
    for upload in uploads:
        logger.info(f"\n📋 {upload['description']}")
        logger.info(f"   Local: {upload['local']}")
        logger.info(f"   S3:    s3://{args.bucket}/{upload['s3_key']}")

        if args.dry_run:
            if upload["local"].exists():
                size_mb = upload["local"].stat().st_size / (1024 * 1024)
                logger.info(f"   ✅ Would upload ({size_mb:.2f} MB)")
            else:
                logger.warning("   ⚠️  File does not exist")
            continue

        if upload_file_to_s3(loader, upload["local"], upload["s3_key"], args.bucket):
            success_count += 1

    logger.info("")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info(f"🔍 DRY RUN: Would upload {len(uploads)} files")
    else:
        logger.info(f"✅ Upload complete: {success_count}/{len(uploads)} files uploaded")
    logger.info("=" * 80)

    return 0 if success_count == len(uploads) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
