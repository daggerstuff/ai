"""
W&B Serverless RL Training for Pixelated Empathy clinical AI.

Uses OpenPipe ART framework with ServerlessBackend + GRPO.
Continues from SFT checkpoint (same model name).

Reward function: clinical therapy quality scoring based on:
  - Length appropriateness (20-300 words preferred)
  - No sycophancy (penalize deslop sycophancy patterns)
  - No AI slop markers (penalize generic-ai patterns)
  - Question quality (therapy questions rewarded)
  - No harmful content (penalize crisis mishandling)
  - Empathy markers (reflective listening, validation)

Usage:
  rtk uv run python -m ai.training.scripts.train_rl_serverless [--model-name NAME] [--steps N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

_env_file = Path(__file__).resolve().parents[3] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY required.")
    sys.exit(1)

import art
from art.serverless.backend import ServerlessBackend

REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED_DIR = REPO_ROOT / "ai" / "data" / "curated" / "sft_chatml"
TRAIN_FILE = CURATED_DIR / "train.jsonl"

WANDB_PROJECT = "pixelated-empathy-sft"
DEFAULT_MODEL_NAME = "pixelated-empathy-v1"
DEFAULT_BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
DEFAULT_STEPS = 100
DEFAULT_GROUPS_PER_STEP = 5
DEFAULT_ROLLOUTS_PER_GROUP = 4
DEFAULT_LR = 5e-5
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMP = 0.8

SYCOPHANCY_PATTERNS = [
    r"you're absolutely right",
    r"i completely agree",
    r"i completely understand",
    r"i can only imagine",
    r"that must be really? (hard|difficult|painful)",
    r"it's completely normal",
    r"no wonder",
    r"thank you for sharing",
    r"you're so brave",
    r"that takes a lot of courage",
    r"that's completely valid",
    r"i hear what you're saying",
    r"i see where you're coming from",
    r"it sounds like you",
    r"i want you to know",
]
SYCOPHANCY_RE = re.compile("|".join(SYCOPHANCY_PATTERNS), re.IGNORECASE)

SLOP_PATTERNS = [
    r"\babsolutely\b",
    r"\bof course\b",
    r"\bcertainly\b",
    r"\bdelve\b",
    r"\bleverage\b",
    r"\bseamless\b",
    r"\brobust\b",
    r"\bgame-changer\b",
    r"\bcircle back\b",
    r"\bmoving forward\b",
    r"\bit's important to note\b",
    r"\bstep-by-step\b",
    r"\btapestry\b",
    r"\bi'm here to help\b",
    r"\bhappy to help\b",
]
SLOP_RE = re.compile("|".join(SLOP_PATTERNS), re.IGNORECASE)

QUESTION_RE = re.compile(
    r"\b(can you|could you|what|how|when|where|why|tell me about|describe|explain)\b", re.IGNORECASE
)

CRISIS_PATTERNS = [
    r"\bkill yourself\b",
    r"\bend it all\b",
    r"\bsuicide\b",
    r"\bself-harm\b",
    r"\boverdose\b",
    r"\bdon't want to live\b",
    r"\bbetter off dead\b",
]
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

REFLECTIVE_RE = re.compile(
    r"\b(it sounds like|i hear|i understand that|what i'm hearing is|"
    r"let me make sure|so what you're saying|reflecting back)\b",
    re.IGNORECASE,
)


def load_examples(max_examples: int = 5000) -> list[list[dict]]:
    examples = []
    with open(TRAIN_FILE) as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if len(msgs) >= 2 and msgs[-1].get("role") == "assistant":
                examples.append(msgs)
            if len(examples) >= max_examples:
                break
    return examples


def compute_reward(response: str, expected: str) -> dict[str, float]:
    response_words = response.split()
    response_len = len(response_words)
    expected_words = set(expected.lower().split())
    response_lower = response.lower()

    if 20 <= response_len <= 300:
        length_score = 1.0
    elif response_len < 10:
        length_score = 0.1
    elif response_len > 500:
        length_score = 0.3
    else:
        length_score = 0.5

    sycophancy_hits = len(SYCOPHANCY_RE.findall(response_lower))
    sycophancy_score = max(0.0, 1.0 - sycophancy_hits * 0.3)

    slop_hits = len(SLOP_RE.findall(response_lower))
    slop_score = max(0.0, 1.0 - slop_hits * 0.2)

    has_question = 1.0 if QUESTION_RE.search(response) else 0.0

    crisis_in_input = bool(CRISIS_RE.search(expected))
    crisis_in_response = bool(CRISIS_RE.search(response))
    if crisis_in_input and not crisis_in_response:
        crisis_score = 0.5
    elif crisis_in_response:
        crisis_score = 0.0
    else:
        crisis_score = 1.0

    reflective_score = 1.0 if REFLECTIVE_RE.search(response) else 0.5

    response_word_set = set(response_lower.split())
    overlap = len(response_word_set & expected_words) / max(len(expected_words), 1)

    reward = (
        length_score * 0.15
        + sycophancy_score * 0.25
        + slop_score * 0.15
        + has_question * 0.10
        + crisis_score * 0.20
        + reflective_score * 0.10
        + overlap * 0.05
    )

    return {
        "reward": reward,
        "length_score": length_score,
        "sycophancy_score": sycophancy_score,
        "slop_score": slop_score,
        "question_score": has_question,
        "crisis_score": crisis_score,
        "reflective_score": reflective_score,
        "overlap": overlap,
        "response_len": response_len,
    }


async def rollout(model: art.TrainableModel, messages: list[dict]) -> art.Trajectory:
    context = messages[:-1]
    expected = messages[-1].get("content", "") if messages else ""

    trajectory = art.Trajectory(
        messages_and_choices=list(context),
        reward=0.0,
        metrics={},
    )

    client = model.openai_client
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMP,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)
    response = choice.message.content or ""

    scores = compute_reward(response, expected)
    trajectory.reward = scores["reward"]
    trajectory.metrics = {k: v for k, v in scores.items() if k != "reward"}

    return trajectory


async def run_rl(
    model_name: str,
    base_model: str,
    max_steps: int,
    groups_per_step: int,
    rollouts_per_group: int,
    learning_rate: float,
    max_examples: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Pixelated Empathy — Serverless RL Training")
    print(f"{'=' * 60}")
    print(f"  Model:       {model_name}")
    print(f"  Base:        {base_model}")
    print(f"  Max steps:   {max_steps}")
    print(f"  Groups/step: {groups_per_step}")
    print(f"  Rollouts:    {rollouts_per_group}")
    print(f"  LR:          {learning_rate}")
    print(f"{'=' * 60}\n")

    examples = load_examples(max_examples)
    print(f"Loaded {len(examples)} RL examples")

    model = art.TrainableModel(
        name=model_name,
        project=WANDB_PROJECT,
        base_model=base_model,
    )
    backend = ServerlessBackend(api_key=os.environ["WANDB_API_KEY"])
    await model.register(backend)
    print(f"Model registered: {model.name}")

    start_step = await model.get_step()
    print(f"Starting RL from step {start_step}")

    import random

    for step in range(max_steps):
        batch = random.sample(examples, min(groups_per_step, len(examples)))

        train_groups = await art.gather_trajectory_groups(
            (art.TrajectoryGroup(rollout(model, messages) for _ in range(rollouts_per_group)) for messages in batch),
            pbar_desc=f"RL step {step + start_step}",
        )

        result = await backend.train(
            model,
            train_groups,
            learning_rate=learning_rate,
            precalculate_logprobs=True,
        )
        await model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split="train",
        )

        total_reward = sum(t.reward for g in train_groups for t in g.trajectories)
        total_trajectories = sum(len(g.trajectories) for g in train_groups)
        avg_reward = total_reward / total_trajectories if total_trajectories > 0 else 0.0
        print(f"Step {result.step}: avg_reward={avg_reward:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  RL Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Inference: {model.get_inference_name()}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serverless RL training for Pixelated Empathy")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--groups-per-step", type=int, default=DEFAULT_GROUPS_PER_STEP)
    parser.add_argument("--rollouts-per-group", type=int, default=DEFAULT_ROLLOUTS_PER_GROUP)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-examples", type=int, default=5000)
    args = parser.parse_args()

    asyncio.run(
        run_rl(
            model_name=args.model_name,
            base_model=args.base_model,
            max_steps=args.steps,
            groups_per_step=args.groups_per_step,
            rollouts_per_group=args.rollouts_per_group,
            learning_rate=args.lr,
            max_examples=args.max_examples,
        )
    )


if __name__ == "__main__":
    main()
