"""
W&B Serverless SFT Training for Pixelated Empathy clinical AI.

Uses OpenPipe ART framework with ServerlessBackend.
Trains LoRA adapter on curated ChatML dataset.

Prerequisites:
  - WANDB_API_KEY env var (already in .env)
  - openpipe-art installed (pip install "openpipe-art[serverless]")
  - Curated data at ai/data/curated/sft_chatml/{train,val}.jsonl

Usage:
  rtk uv run python -m ai.training.scripts.train_sft_serverless [--base-model MODEL] [--epochs N] [--batch-size N]

Free during W&B Serverless preview — only inference + artifact storage charged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Load .env if present
from pathlib import Path as P

_env_file = P(__file__).resolve().parents[3] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY required. Set in .env or environment.")
    sys.exit(1)

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file


# Available models: https://docs.wandb.ai/serverless-training/available-models
AVAILABLE_MODELS = {
    "glm-5.3-flash": "zai-org/glm-5.3-flash",
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "glm-5.2": "THUDM/glm-5.2",
}

DEFAULT_MODEL = "zai-org/glm-5.3-flash"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.1

# Paths
REPO_ROOT = P(__file__).resolve().parents[3]
CURATED_DIR = REPO_ROOT / "ai" / "data" / "curated" / "sft_chatml"
TRAIN_FILE = CURATED_DIR / "train.jsonl"
VAL_FILE = CURATED_DIR / "val.jsonl"

# W&B project
WANDB_PROJECT = "pixelated-empathy-sft"
MODEL_NAME = "pixelated-empathy-v1"


BANNED_OPENERS = (
    "i hear how",
    "it makes sense that you feel",
    "i understand your frustration",
    "i can hear",
    "that sounds really",
    "i'm so sorry to hear",
    "thank you for sharing",
    "it sounds like you",
    "i want you to know",
    "i can imagine how",
    "i hear your",
    "it sounds like",
)

CAVING_PHRASES = (
    "you're right",
    "i apologize",
    "i stand corrected",
    "sorry for",
    "my mistake",
    "if you don't want to talk about it",
    "we don't have to",
    "we don't have to talk about",
    "i'll stop",
    "fair enough",
)


def contains_sycophancy(text: str) -> bool:
    """Check if text contains banned robotic openers or caving phrases."""
    if not text or not isinstance(text, str):
        return False
    t_lower = text.strip().lower()
    for b in BANNED_OPENERS:
        if t_lower.startswith(b) or f"\n{b}" in t_lower:
            return True
    for c in CAVING_PHRASES:
        if c in t_lower:
            return True
    return False


def filter_assistant_ending(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Filter JSONL to assistant-ending records that pass anti-sycophancy & deslop gating.

    ART SFT requires last message to be from assistant role.
    Rejects any record whose assistant turns contain banned sycophantic openers or caving phrases.
    Returns (kept, total) counts.
    """
    kept = 0
    total = 0
    sycophancy_rejected = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = d.get("messages", [])
            if not msgs or msgs[-1].get("role") != "assistant":
                continue

            # Anti-sycophancy check on all assistant turns
            has_syc = False
            for m in msgs:
                if m.get("role") == "assistant" and contains_sycophancy(m.get("content", "")):
                    has_syc = True
                    break

            if has_syc:
                sycophancy_rejected += 1
                continue

            fout.write(line)
            kept += 1

    if sycophancy_rejected > 0:
        print(f"  [Anti-Sycophancy Gate] Rejected {sycophancy_rejected} sycophantic/caving training records.")
    return kept, total


async def run_sft(
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    max_records: int | None = None,
    model_name: str = MODEL_NAME,
) -> None:
    """Run serverless SFT training."""

    # Filter training data to assistant-ending records only
    print(f"\n{'=' * 60}")
    print(f"  Pixelated Empathy — Serverless SFT Training")
    print(f"{'=' * 60}")
    print(f"  Base model:  {base_model}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {learning_rate}")
    print(f"  Warmup:      {warmup_ratio}")
    print(f"  Project:     {WANDB_PROJECT}")
    print(f"  Model name:  {MODEL_NAME}")
    print(f"{'=' * 60}\n")

    # Prepare filtered training file
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found: {TRAIN_FILE}")
        sys.exit(1)

    print(f"Filtering {TRAIN_FILE.name} (assistant-ending only)...")
    tmp_path = CURATED_DIR / f"train_filtered_{model_name}.jsonl"
    kept, total = filter_assistant_ending(TRAIN_FILE, tmp_path)
    if max_records and kept > max_records:
        with open(tmp_path) as f:
            lines = [f.readline() for _ in range(max_records)]
        with open(tmp_path, "w") as f:
            f.writelines(lines)
        kept = max_records
        print(f"  Truncated to {max_records} records for experiment")
    print(f"  {kept}/{total} records kept ({(kept / total) * 100:.1f}%)")
    print(f"  Filtered file: {tmp_path}")

    # Initialize serverless backend
    print("\nInitializing ServerlessBackend...")
    backend = ServerlessBackend(api_key=os.environ["WANDB_API_KEY"])

    # Create trainable model
    model = art.TrainableModel(
        name=model_name,
        project=WANDB_PROJECT,
        base_model=base_model,
    )
    await model.register(backend)
    print(f"  Model registered: {MODEL_NAME}")
    print(f"  Inference endpoint: {model.get_inference_name()}")

    # Run SFT training
    print(f"\nStarting SFT training ({epochs} epochs, {kept} records)...")
    await train_sft_from_file(
        model=model,
        file_path=str(tmp_path),
        epochs=epochs,
        batch_size=batch_size,
        peak_lr=learning_rate,
        schedule_type="cosine",
        warmup_ratio=warmup_ratio,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  SFT Training Complete!")
    print(f"{'=' * 60}")
    print(f"  LoRA adapter saved as W&B Artifact")
    print(f"  Inference endpoint: {model.get_inference_name()}")
    print(f"  W&B Project: {WANDB_PROJECT}")
    print(f"  Next: Run RL training with train_rl_serverless.py")
    print(f"{'=' * 60}\n")

    # Cleanup temp file
    tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serverless SFT training for Pixelated Empathy")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
        choices=list(AVAILABLE_MODELS.values()),
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Peak learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--train-file",
        default=None,
        help="Override training file path (default: ai/data/curated/sft_chatml/train.jsonl)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of training records (for experiments)",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Model name for W&B (default: pixelated-empathy-v1)",
    )
    args = parser.parse_args()

    global TRAIN_FILE
    if args.train_file:
        TRAIN_FILE = Path(args.train_file)

    asyncio.run(
        run_sft(
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            max_records=args.max_records,
            model_name=args.model_name,
        )
    )


if __name__ == "__main__":
    main()
