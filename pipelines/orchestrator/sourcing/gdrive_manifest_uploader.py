"""
GDrive Manifest Uploader.

Uploads training data from /mnt/gdrive to OVH S3 based on manifest.
"""

import argparse
import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv("ai/.env")

logger = logging.getLogger(__name__)


def get_s3_client():
    """Creates an S3 client configured for OVH Object Storage."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("OVH_S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("OVH_S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("OVH_S3_SECRET_KEY"),
        region_name=os.environ.get("OVH_S3_REGION", "us-east-va"),
    )


# Manifest of files to upload, organized by tier
MANIFEST: dict[str, list[dict]] = {
    "processed": [
        # Phase 1 Priority (Tier 1)
        {
            "path": "/mnt/gdrive/processed/phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
            "tier": "tier1_priority",
            "size_mb": 153,
        },
        {
            "path": "/mnt/gdrive/processed/phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl",
            "tier": "tier1_priority",
            "size_mb": 143,
        },
        {
            "path": "/mnt/gdrive/processed/phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl",
            "tier": "tier1_priority",
            "size_mb": 123,
        },
        # Phase 3 CoT Reasoning (Tier 3)
        {
            "path": "/mnt/gdrive/processed/phase_3_cot_reasoning/phase_3_cot_reasoning_consolidated.jsonl",
            "tier": "tier3_cot_reasoning",
            "size_mb": 318,
        },
        {
            "path": "/mnt/gdrive/processed/phase_3_cot_reasoning/task_5_15_cot_reasoning/cot_reasoning_conversations_consolidated.jsonl",
            "tier": "tier3_cot_reasoning",
            "size_mb": 280,
        },
        {
            "path": "/mnt/gdrive/processed/phase_3_cot_reasoning/task_5_26_pattern_recognition/therapeutic_reasoning_pattern_database.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 241,
        },
        # Phase 4 Reddit MH - Crisis (Tier 3 Edge)
        {
            "path": "/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
            "tier": "tier3_edge_crisis",
            "size_mb": 637,
        },
        # Phase 4 Reddit MH - Professional (Tier 2) - LARGE FILES
        {
            "path": "/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
            "tier": "tier2_professional",
            "size_mb": 910,
        },
        {
            "path": "/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_33_tfidf_integration/ml_ready_conversations.jsonl",
            "tier": "tier2_professional",
            "size_mb": 905,
        },
        {
            "path": "/mnt/gdrive/processed/phase_4_reddit_mental_health/task_5_31_additional_specialized/additional_specialized_conversations.jsonl",
            "tier": "tier2_professional",
            "size_mb": 127,
        },
        # SoulChat
        {
            "path": "/mnt/gdrive/processed/soulchat_complete/soulchat_2_0_complete_no_limits.jsonl",
            "tier": "tier2_professional",
            "size_mb": 39,
        },
    ],
    "raw": [
        # Therapist SFT (Tier 1)
        {
            "path": "/mnt/gdrive/datasets/therapist-sft-format/train.csv",
            "tier": "tier1_priority",
            "size_mb": 388,
        },
        {
            "path": "/mnt/gdrive/datasets/merged_dataset.jsonl",
            "tier": "tier1_priority",
            "size_mb": 162,
        },
        {
            "path": "/mnt/gdrive/datasets/merged_mental_health_dataset.jsonl",
            "tier": "tier1_priority",
            "size_mb": 86,
        },
        # CoT Reasoning (Tier 3)
        {
            "path": "/mnt/gdrive/datasets/Reasoning_Problem_Solving_Dataset/RPSD.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 651,
        },
        {
            "path": "/mnt/gdrive/datasets/ToT_Reasoning_Problem_Solving_Dataset_V2/ToT-RPSD-V2.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 230,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT_Rare-Diseases_And_Health-Conditions/CoT_Rare Disseases_Health Conditions_9.8k.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 65,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT_Neurodivergent_vs_Neurotypical_Interactions_downloaded.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 53,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT-Reasoning_Cultural_Nuances/CoT-Reasoning_Cultural_Nuances_Dataset.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 42,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT_Heartbreak_and_Breakups_downloaded.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 38,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health_downloaded.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 20,
        },
        {
            "path": "/mnt/gdrive/datasets/CoT_Reasoning_Mens_Mental_Health_downloaded.json",
            "tier": "tier3_cot_reasoning",
            "size_mb": 18,
        },
        # Reddit MH (Tier 2 & 3)
        {
            "path": "/mnt/gdrive/datasets/reddit_mental_health/mental_disorders_reddit.csv",
            "tier": "tier2_professional",
            "size_mb": 562,
        },
        {
            "path": "/mnt/gdrive/datasets/reddit_mental_health/Suicide_Detection.csv",
            "tier": "tier3_edge_crisis",
            "size_mb": 160,
        },
        # Big Five / Personality
        {
            "path": "/mnt/gdrive/datasets/data-final.csv",
            "tier": "tier2_professional",
            "size_mb": 397,
        },
    ],
    "youtube": [
        # YouTube transcripts - all go to tier4_voice_persona
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Tim Fletcher",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Patrick Teahan ",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Crappy Childhood Fairy",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/DoctorRamani",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Therapy in a Nutshell",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Heidi Priebe",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
        {
            "path": "/mnt/gdrive/youtube_transcriptions/transcripts/Doc Snipes",
            "tier": "tier4_voice_persona",
            "is_dir": True,
        },
    ],
}


def upload_file(s3, local_path: str, bucket: str, s3_key: str) -> bool:
    """Upload a single file to S3."""
    try:
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Uploading {local_path} ({file_size:.1f}MB) -> s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        logger.info("  -> Success")
        return True
    except Exception as e:
        logger.error(f"  -> Failed: {e}")
        return False


def upload_directory(s3, local_dir: str, bucket: str, s3_prefix: str) -> int:
    """Upload all files in a directory to S3."""
    count = 0
    for file_path in Path(local_dir).rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative}"
            if upload_file(s3, str(file_path), bucket, s3_key):
                count += 1
    return count


def run_upload(
    tier: str, bucket: str = "pixel-data", prefix: str = "datasets/gdrive/", max_size_mb: int = 0
):
    """
    Upload files from manifest to S3.

    Args:
        tier: 'processed', 'raw', or 'youtube'
        bucket: S3 bucket name
        prefix: S3 prefix for uploads
        max_size_mb: Maximum size in MB for files to upload. 0 means no limit.
    """
    logging.basicConfig(level=logging.INFO)

    s3 = get_s3_client()
    if not s3:
        logger.error("Failed to create S3 client")
        return

    files = MANIFEST.get(tier, [])
    if not files:
        logger.error(f"Unknown tier: {tier}")
        return

    logger.info(f"=== Uploading {len(files)} items from tier '{tier}' ===")

    success = 0
    skipped = 0
    failed = 0

    for item in files:
        path = item["path"]
        target_tier = item["tier"]
        size_mb = item.get("size_mb", 0)  # size_mb is retrieved for logging and optional skipping
        is_dir = item.get("is_dir", False)

        if max_size_mb > 0 and size_mb > max_size_mb:
            logger.info(
                f"Skipping {path} ({size_mb:.1f}MB) as it exceeds max_size_mb ({max_size_mb}MB)"
            )
            skipped += 1
            continue

        if not os.path.exists(path):
            logger.warning(f"Path not found: {path}")
            failed += 1
            continue

        # Build S3 key
        basename = os.path.basename(path.rstrip("/"))
        s3_key = f"{prefix}{target_tier}/{basename}"

        if is_dir:
            count = upload_directory(s3, path, bucket, s3_key)
            success += count
        elif upload_file(s3, path, bucket, s3_key):
            success += 1
        else:
            failed += 1

    logger.info(f"=== Complete: {success} uploaded, {skipped} skipped, {failed} failed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload gdrive training data to S3")
    parser.add_argument("--tier", choices=["processed", "raw", "youtube"], required=True)
    parser.add_argument("--bucket", default="pixel-data")
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=0,
        help="Maximum size in MB for files to upload. 0 means no limit.",
    )
    args = parser.parse_args()

    run_upload(args.tier, args.bucket, max_size_mb=args.max_size_mb)
