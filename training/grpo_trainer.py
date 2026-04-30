#!/usr/bin/env python3
"""GRPO trainer with pure-Python reward function for therapeutic AI.

Implements a composite reward scoring safety, empathy, and crisis resource
citation. The reward function is fully unit-testable without a model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from .shared_config import (
        add_lora_args,
        build_lora_config,
        log_token_length_distribution,
        shared_qlora_config,
    )
    from .multilingual_safety_checker import MultilingualSafetyChecker
except ModuleNotFoundError:
    try:
        from ai.training.shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )
        from ai.training.multilingual_safety_checker import MultilingualSafetyChecker
    except ModuleNotFoundError:
        from shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )
        from multilingual_safety_checker import MultilingualSafetyChecker

logger = logging.getLogger("grpo_trainer")

CRISIS_RESOURCES: frozenset[str] = frozenset({
    "988", "741741", "findahelpline", "911", "999", "112",
    "suicide prevention lifeline", "crisis text line",
    "national suicide prevention", "emergency services",
    "crisis hotline", "suicide hotline",
})

EMPATHY_KEYWORDS: frozenset[str] = frozenset({
    "understand", "hear you", "feeling", "empathy", "compassion",
    "validate", "valid", "support", "care", "sorry you",
    "that sounds", "difficult", "struggling", "pain", "hurt",
    "not alone", "here for you", "listen", "safe space",
    "your feelings", "matters", "important", "brave",
})


def _safety_score(response: str) -> float:
    """0.0 if the response contains unsafe content, 1.0 otherwise."""
    if MultilingualSafetyChecker.is_unsafe(response):
        return 0.0
    return 1.0


def _empathy_score(response: str) -> float:
    """Keyword-based empathy score in [0.0, 1.0]."""
    if not response:
        return 0.0
    response_lower = response.lower()
    matched = sum(1 for kw in EMPATHY_KEYWORDS if kw in response_lower)
    # Scale: 0 keywords = 0.0, 1 = 0.25, 2 = 0.5, 3 = 0.75, 4+ = 1.0
    return min(matched / 4.0, 1.0)


def _crisis_score(prompt: str, response: str) -> float:
    """1.0 if crisis prompt has resource in response, 1.0 if not crisis prompt."""
    if not MultilingualSafetyChecker.is_unsafe(prompt):
        return 1.0
    response_lower = response.lower()
    if any(resource in response_lower for resource in CRISIS_RESOURCES):
        return 1.0
    return 0.0


def score(
    prompt: str,
    response: str,
    safety_weight: float = 0.5,
    empathy_weight: float = 0.3,
    crisis_weight: float = 0.2,
) -> float:
    """Compute composite reward score in [0.0, 1.0].

    Weights must sum to 1.0. Returns weighted sum of sub-scores.
    """
    s = _safety_score(response)
    e = _empathy_score(response)
    c = _crisis_score(prompt, response)
    return safety_weight * s + empathy_weight * e + crisis_weight * c


def filter_by_threshold(
    prompts: list[str],
    responses: list[str],
    threshold: float,
    safety_weight: float = 0.5,
    empathy_weight: float = 0.3,
    crisis_weight: float = 0.2,
) -> list[dict]:
    """Return only samples with composite score >= threshold."""
    kept: list[dict] = []
    for prompt, response in zip(prompts, responses):
        composite = score(prompt, response, safety_weight, empathy_weight, crisis_weight)
        if composite >= threshold:
            kept.append({
                "prompt": prompt,
                "response": response,
                "composite_score": composite,
                "safety_score": _safety_score(response),
                "empathy_score": _empathy_score(response),
                "crisis_score": _crisis_score(prompt, response),
            })
    return kept


def run_grpo(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        logger.error("GRPOTrainer requires trl >= 0.14. Install with: pip install trl>=0.14")
        return

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    logger.info("Loading model from %s", args.base_model_checkpoint)
    bnb_config = shared_qlora_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = build_lora_config(args)

    logger.info(
        "Reward weights: safety=%.2f, empathy=%.2f, crisis=%.2f, threshold=%.2f",
        args.safety_weight, args.empathy_weight, args.crisis_weight,
        args.min_reward_threshold,
    )

    def reward_fn(prompts: list[str], responses: list[str]) -> list[float]:
        return [
            score(p, r, args.safety_weight, args.empathy_weight, args.crisis_weight)
            for p, r in zip(prompts, responses)
        ]

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_json(str(data_path)),
        processing_class=tokenizer,
        peft_config=lora_config,
        reward_funcs=reward_fn,
    )

    train_result = trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "safety_weight": args.safety_weight,
        "empathy_weight": args.empathy_weight,
        "crisis_weight": args.crisis_weight,
        "min_reward_threshold": args.min_reward_threshold,
    }
    metrics_path = output_dir / "grpo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics}, f, indent=2)
        f.write("\n")

    logger.info("GRPO training complete. Final model at %s", final_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO trainer with reward function for therapeutic AI.",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--base_model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--safety_weight", type=float, default=0.5)
    parser.add_argument("--empathy_weight", type=float, default=0.3)
    parser.add_argument("--crisis_weight", type=float, default=0.2)
    parser.add_argument("--min_reward_threshold", type=float, default=0.3)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    add_lora_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_grpo(args)


if __name__ == "__main__":
    main()
