#!/usr/bin/env python3
"""Serverless RL CLI launcher for CoreWeave Forge.

Launches, monitors, or dry-runs Reinforcement Learning post-training jobs
aimed at multi-turn agent reliability and tool-use fidelity for Pixelated Empathy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from training.forge.client import CoreWeaveForgeClient
from training.forge.config import ForgeConfig
from training.forge.rl_runner import RewardWeights, ServerlessRLRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_rl_serverless")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and manage CoreWeave Forge Serverless RL jobs"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Base adapter or model ID (e.g. wandb-artifact:///... or Nemotron)",
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        required=True,
        help="URL or artifact URI containing RL episode prompts and validation cases",
    )
    parser.add_argument(
        "--tool-weight",
        type=float,
        default=0.40,
        help="Reward weight for tool calling fidelity",
    )
    parser.add_argument(
        "--bleed-weight",
        type=float,
        default=0.20,
        help="Reward weight for anti-therapy bleed",
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=0.15,
        help="Reward weight for clinical crisis safety",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ForgeConfig()
    client = CoreWeaveForgeClient(config=config)

    weights = RewardWeights(
        tool_call_fidelity=args.tool_weight,
        anti_therapy_bleed=args.bleed_weight,
        clinical_safety=args.safety_weight,
    )
    runner = ServerlessRLRunner(client=client, weights=weights)

    logger.info("Initializing CoreWeave Forge Serverless RL Launcher")
    logger.info("Target Model: %s", args.model_id)
    logger.info("Dataset: %s", args.dataset_url)
    logger.info(
        "Reward Weights: tool=%.2f, bleed=%.2f, safety=%.2f",
        weights.tool_call_fidelity,
        weights.anti_therapy_bleed,
        weights.clinical_safety,
    )

    is_dry_run = not args.execute or args.dry_run
    result = runner.launch_serverless_rl(
        model_id=args.model_id,
        dataset_url=args.dataset_url,
        dry_run=is_dry_run,
    )

    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
