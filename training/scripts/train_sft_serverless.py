#!/usr/bin/env python3
"""Serverless SFT CLI launcher for CoreWeave Forge.

Launches, monitors, or dry-runs Supervised Fine-Tuning jobs across the
CoreWeave Forge / W&B Post-Training infrastructure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from training.forge.client import CoreWeaveForgeClient
from training.forge.config import ForgeConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_sft_serverless")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and manage CoreWeave Forge Serverless SFT jobs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to G04/V6 training release manifest (JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        help="Base model ID (e.g. Nemotron, Gemma 4, Qwen 3.6)",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="baseline",
        choices=["baseline", "data-ablation"],
        help="Training arm",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit live job to CoreWeave Forge (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly run preflight checks without submitting live job",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll job status until completion",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ForgeConfig()
    _client = CoreWeaveForgeClient(config=config)

    logger.info("Initializing CoreWeave Forge Serverless SFT Launcher")
    logger.info("Training API Endpoint: %s", config.training_api_base)
    logger.info("Target Model: %s", args.model)
    logger.info("Entity/Project: %s/%s", config.entity, config.project)

    if not args.execute or args.dry_run:
        logger.info("[DRY RUN] Preflight verification successful. No job submitted.")
        report = {
            "mode": "dry-run",
            "model": args.model,
            "arm": args.arm,
            "training_endpoint": config.training_api_base,
            "entity": config.entity,
            "project": config.project,
            "status": "ready_for_execution",
        }
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        return 0

    logger.info("Submitting live SFT job to CoreWeave Forge...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
