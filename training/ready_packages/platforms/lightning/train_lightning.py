#!/usr/bin/env python3
"""
Lightning.ai Training Entry Point for Stage 1 Foundation Training

This wrapper script handles:
1. Loading datasets from S3 or local storage
2. Running PixelTrainer with optimized config
3. Checkpointing to mounted volume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env for local development
env_file = Path(__file__).resolve()
for _ in range(5):  # Go up to project root
    env_file = env_file.parent
env_file = env_file / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip("'\"")
                os.environ.setdefault(key, value)

sys.path.insert(0, "/app/ai/training")
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent.parent.parent / "ai" / "training"
    ),
)

# Load environment variables from .env for local development
env_file = Path(__file__).resolve()
for _ in range(4):
    env_file = env_file.parent
env_file = env_file / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip("'\"")
                os.environ.setdefault(key, value)

sys.path.insert(0, "/app/ai/training")
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "ai" / "training")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Lightning-Trainer")

# Import main trainer
try:
    from train_pixel import PixelTrainer
except ImportError:
    logger.error(
        "Failed to import PixelTrainer. Ensure ai/training/train_pixel.py is available."
    )
    sys.exit(1)


def download_datasets_from_s3(data_dir: Path) -> bool:
    """Download datasets from S3 at runtime instead of baking into image."""
    import boto3
    from botocore.config import Config

    s3_bucket = os.environ.get("OVH_S3_BUCKET", "pixel-data")
    s3_endpoint = os.environ.get(
        "OVH_S3_ENDPOINT", "https://s3.us-east-va.io.cloud.ovh.us"
    )
    s3_region = os.environ.get("OVH_S3_REGION", "us-east-va")
    s3_access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    s3_secret_key = os.environ.get("OVH_S3_SECRET_KEY")

    if not all([s3_access_key, s3_secret_key]):
        logger.warning("S3 credentials not found, skipping download")
        return False

    logger.info(f"Downloading datasets from S3: s3://{s3_bucket}")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        region_name=s3_region,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version="s3v4"),
    )

    # Key dataset prefixes to download
    prefixes = ["compiled_dataset/", "compiled_stage2_dataset/"]
    downloaded = 0

    for prefix in prefixes:
        try:
            response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
            if "Contents" not in response:
                logger.warning(f"No objects found at {prefix}")
                continue

            for obj in response["Contents"]:
                key = obj["Key"]
                local_path = data_dir / key
                local_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Downloading {key}...")
                s3.download_file(s3_bucket, key, str(local_path))
                downloaded += 1
        except Exception as e:
            logger.error(f"Failed to download {prefix}: {e}")

    logger.info(f"Downloaded {downloaded} files from S3")
    return downloaded > 0


def load_baked_datasets(data_dir: Path) -> Dict[str, Any]:
    """Load datasets from data directory (downloaded from S3 or local)."""
    datasets = {}

    # Check for ultimate final dataset
    ultimate_path = data_dir / "ULTIMATE_FINAL_DATASET.jsonl"
    if ultimate_path.exists():
        datasets["ultimate"] = str(ultimate_path)
        logger.info(f"Found ultimate dataset: {ultimate_path}")

    # Check for compiled dataset shards
    compiled_dir = data_dir / "compiled_dataset"
    if compiled_dir.exists() and list(compiled_dir.glob("*.jsonl")):
        datasets["compiled_shards"] = str(compiled_dir)
        logger.info(f"Found compiled shards: {compiled_dir}")

    # Check for stage2 training data
    stage2_path = data_dir / "compiled_stage2_dataset" / "train.jsonl"
    if stage2_path.exists():
        datasets["stage2_train"] = str(stage2_path)
        logger.info(f"Found stage2 train: {stage2_path}")

    return datasets


def create_lightning_config(
    base_config: Dict[str, Any],
    data_dir: Path,
    checkpoint_dir: Path,
    stage: str = "stage1",
) -> Dict[str, Any]:
    """Create Lightning-specific training config with dataset paths."""
    config = base_config.copy()

    # Determine which dataset to use
    datasets = load_baked_datasets(data_dir)

    # Priority: ultimate > merged > compiled shards > stage2
    if "ultimate" in datasets:
        dataset_path = datasets["ultimate"]
        logger.info(f"Using ULTIMATE_FINAL_DATASET: {dataset_path}")
    elif "compiled_shards" in datasets:
        dataset_path = str(data_dir / "merged_dataset.jsonl")
        logger.info(f"Will use merged compiled shards")
    elif "stage2_train" in datasets:
        dataset_path = datasets["stage2_train"]
        logger.info(f"Using Stage 2 train dataset: {dataset_path}")
    else:
        raise FileNotFoundError(f"No datasets found in {data_dir}")

    # Update config with actual paths
    config["dataset_config"]["ultimate_final_dataset"] = dataset_path

    # Set output directory to checkpoint volume
    config["training_parameters"]["output_dir"] = str(checkpoint_dir)

    # Lightning-optimized settings
    if "h100_optimizations" not in config:
        config["h100_optimizations"] = {}
    config["h100_optimizations"]["bf16"] = True
    config["h100_optimizations"]["gradient_checkpointing"] = True
    config["h100_optimizations"]["flash_attention"] = True

    # Memory settings
    if "training_parameters" not in config:
        config["training_parameters"] = {}

    config["training_parameters"]["per_device_train_batch_size"] = 2
    config["training_parameters"]["gradient_accumulation_steps"] = 8

    # Checkpointing config
    config["training_parameters"]["save_strategy"] = "steps"
    config["training_parameters"]["save_steps"] = 500
    config["training_parameters"]["save_total_limit"] = 3

    # Stage-specific modifications
    if stage == "stage1":
        config["training_parameters"]["num_train_epochs"] = 1
        config["training_parameters"]["max_steps"] = -1
        config["training_parameters"]["learning_rate"] = 2e-4
        config["training_parameters"]["warmup_steps"] = 100
    elif stage == "stage2":
        config["training_parameters"]["num_train_epochs"] = 3
        config["training_parameters"]["learning_rate"] = 5e-5
        config["training_parameters"]["warmup_steps"] = 50
    elif stage == "stage3":
        config["training_parameters"]["num_train_epochs"] = 2
        config["training_parameters"]["learning_rate"] = 1e-5
        config["training_parameters"]["warmup_steps"] = 30

    return config


def main():
    parser = argparse.ArgumentParser(description="Lightning.ai Training Entry Point")
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2", "stage3"],
        help="Training stage to run",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/data",
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/checkpoints",
        help="Directory to save checkpoints (mounted volume)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="/app/ai/training/configs/hyperparameters/enhanced_training_config.json",
        help="Base training configuration file",
    )

    args = parser.parse_args()

    # For local development, try local config if default Docker path doesn't exist
    default_config_path = Path(args.base_config)
    if not default_config_path.exists():
        # Hardcoded for now - fix properly later
        local_config = Path(
            "/home/vivi/pixelated/ai/training/configs/dry_run_config.json"
        )
        if local_config.exists():
            args.base_config = str(local_config)
            logger.info(f"Using local config: {args.base_config}")

    # For local development, try local config if default Docker path doesn't exist
    default_config_path = Path(args.base_config)
    if not default_config_path.exists():
        # project_root = ai/training/ready_packages/platforms/lightning/train_lightning.py
        # Need to go up 4 levels: lightning -> platforms -> ready_packages -> training -> ai
        # Then add the path to configs
        project_root = Path(__file__).resolve().parent
        for _ in range(4):
            project_root = project_root.parent
        local_config = (
            project_root / "ai" / "training" / "configs" / "dry_run_config.json"
        )
        if local_config.exists():
            args.base_config = str(local_config)
            logger.info(f"Using local config: {args.base_config}")

    # For local development, try local config if default Docker path doesn't exist
    default_config_path = Path(args.base_config)
    print(
        f"DEBUG: default_config_path = {default_config_path}, exists = {default_config_path.exists()}"
    )
    if not default_config_path.exists():
        project_root = Path(__file__).parent.parent.parent.parent
        print(f"DEBUG: project_root = {project_root}")
        local_config = (
            project_root / "ai" / "training" / "configs" / "dry_run_config.json"
        )
        print(f"DEBUG: local_config = {local_config}, exists = {local_config.exists()}")
        if local_config.exists():
            args.base_config = str(local_config)
            logger.info(f"Using local config: {args.base_config}")

    # For local development, try local config if default Docker path doesn't exist
    default_config_path = Path(args.base_config)
    if not default_config_path.exists():
        project_root = Path(__file__).parent.parent.parent.parent
        local_config = (
            project_root / "ai" / "training" / "configs" / "dry_run_config.json"
        )
        if local_config.exists():
            args.base_config = str(local_config)
            logger.info(f"Using local config: {args.base_config}")
    default_config_path = Path(args.base_config)
    if not default_config_path.exists() or str(default_config_path).startswith("/app/"):
        project_root = Path(__file__).parent.parent.parent.parent.parent
        local_config = (
            project_root / "ai" / "training" / "configs" / "dry_run_config.json"
        )
        if local_config.exists():
            args.base_config = str(local_config)
            logger.info(f"Using local config: {args.base_config}")

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    logger.info(f"=" * 60)
    logger.info(f"Lightning.ai Training - {args.stage.upper()}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"=" * 60)

    # Ensure checkpoint directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for datasets
    logger.info("Checking for datasets...")
    compiled_dir = data_dir / "compiled_dataset"
    if not compiled_dir.exists() or not list(compiled_dir.glob("*.jsonl")):
        logger.info("Datasets not found locally, downloading from S3...")
        download_datasets_from_s3(data_dir)
    else:
        logger.info("Datasets already present locally")

    # Load base config
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        # Try local path
        project_root = Path(__file__).parent.parent.parent.parent
        local_config = (
            project_root
            / "ai"
            / "training"
            / "configs"
            / "hyperparameters"
            / "enhanced_training_config.json"
        )
        if local_config.exists():
            base_config_path = local_config
            logger.info(f"Using local config: {base_config_path}")
        else:
            logger.error(f"Base config not found: {args.base_config}")
            sys.exit(1)

    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    # Create Lightning-specific config
    lightning_config = create_lightning_config(
        base_config, data_dir, checkpoint_dir, args.stage
    )

    # Save config for reproducibility
    config_save_path = checkpoint_dir / f"lightning_{args.stage}_config.json"
    with open(config_save_path, "w") as f:
        json.dump(lightning_config, f, indent=2)
    logger.info(f"Saved config to: {config_save_path}")

    # Initialize trainer
    logger.info("Initializing PixelTrainer...")
    trainer = PixelTrainer(str(config_save_path))

    # Run training
    logger.info(f"Starting {args.stage} training...")
    try:
        trainer.train()
        logger.info(f"✅ {args.stage.upper()} training completed successfully!")

        # Save completion marker
        (checkpoint_dir / f"{args.stage}_COMPLETE").touch()

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        # Save error marker
        with open(checkpoint_dir / f"{args.stage}_ERROR", "w") as f:
            f.write(str(e))
        raise

    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
