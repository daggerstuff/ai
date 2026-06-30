#!/usr/bin/env python3
"""
Optimized Therapeutic AI Training with Automatic Time Management
Automatically selects best configuration to fit 12-hour window
"""

import contextlib
import json
import signal
import time
from datetime import UTC, datetime

import torch
import wandb
from datasets import Dataset
from models.moe_architecture import MoEConfig, create_therapeutic_moe_model
from train_moe_h100 import MoETrainingCallback, TimeConstraintCallback, setup_wandb, signal_handler
from training_optimizer import optimize_for_dataset
from transformers import AutoTokenizer, Trainer

# Global state
shutdown_requested = False
training_start_time = None

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def analyze_dataset(dataset_path: str | None = None, s3_path: str | None = None):
    """
    Analyze dataset to determine optimal training parameters.

    Args:
        dataset_path: Local file path (for backward compatibility)
        s3_path: S3 path (s3://bucket/key) - preferred, S3 is canonical
    """

    # Prefer S3 if provided
    if s3_path:
        from ai.training.utils.s3_dataset_loader import S3DatasetLoader
        loader = S3DatasetLoader()
        data = loader.load_json(s3_path)
    elif dataset_path:
        # Fallback to local file
        with open(dataset_path) as f:
            data = json.load(f)
    else:
        # Try to find dataset in S3
        from ai.training.utils.s3_dataset_loader import get_s3_dataset_path, load_dataset_from_s3
        try:
            s3_path = get_s3_dataset_path("training_dataset.json", category="professional_therapeutic")
            data = load_dataset_from_s3("training_dataset.json", category="professional_therapeutic")
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset not found. Provide dataset_path or s3_path, "
                f"or ensure training_dataset.json exists in S3. Error: {e}"
            )

    texts = [conv["text"] for conv in data["conversations"]]

    # Calculate statistics
    num_samples = len(texts)
    token_counts = [len(text.split()) for text in texts]
    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)


    return {
        "num_samples": num_samples,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "texts": texts
    }


def main():
    global shutdown_requested, training_start_time


    training_start_time = datetime.now(UTC)
    wandb_run = None

    try:
        # Load configurations
        with open("training_config.json") as f:
            training_config = json.load(f)

        with open("safety_config.json") as f:
            safety_config = json.load(f)

        # Analyze dataset (try S3 first, fallback to local)
        dataset_info = analyze_dataset()

        # Optimize training parameters

        desired_epochs = training_config.get("num_train_epochs", 3)
        priority = training_config.get("optimization_priority", "balanced")
        max_hours = training_config.get("max_training_hours", 12.0)

        profile, estimate, training_args = optimize_for_dataset(
            num_samples=dataset_info["num_samples"],
            avg_tokens_per_sample=int(dataset_info["avg_tokens"]),
            num_epochs=desired_epochs,
            priority=priority,
            max_hours=max_hours
        )

        if not estimate.fits_in_window and estimate.recommended_adjustments:
            adj = estimate.recommended_adjustments
            if "new_num_epochs" in adj:
                desired_epochs = adj["new_num_epochs"]

                # Re-optimize with adjusted epochs
                profile, estimate, training_args = optimize_for_dataset(
                    num_samples=dataset_info["num_samples"],
                    avg_tokens_per_sample=int(dataset_info["avg_tokens"]),
                    num_epochs=desired_epochs,
                    priority="fast",
                    max_hours=max_hours
                )

        # Setup WandB
        wandb_run = setup_wandb()

        # Log optimization info
        wandb.log({
            "optimization/profile": profile.__class__.__name__,
            "optimization/batch_size": profile.batch_size,
            "optimization/gradient_accumulation": profile.gradient_accumulation_steps,
            "optimization/effective_batch_size": profile.batch_size * profile.gradient_accumulation_steps,
            "optimization/estimated_hours": estimate.estimated_hours,
            "optimization/fits_in_window": estimate.fits_in_window,
            "dataset/num_samples": dataset_info["num_samples"],
            "dataset/avg_tokens": dataset_info["avg_tokens"]
        })

        # Check CUDA
        if not torch.cuda.is_available():
            return

        # Create model
        BASE_MODEL_NAME = training_config.get("base_model", "LatitudeGames/Wayfarer-2-12B")


        moe_config = MoEConfig(
            num_experts=4,
            expert_domains=["psychology", "mental_health", "bias_detection", "general_therapeutic"],
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            max_position_embeddings=8192,
            expert_capacity=2,
            load_balancing_weight=0.01
        )

        model = create_therapeutic_moe_model(
            BASE_MODEL_NAME,
            moe_config=moe_config,
            device="auto"
        )


        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/trainable_percent": (trainable_params / total_params) * 100
        })


        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dataset
        dataset = Dataset.from_dict({"text": dataset_info["texts"]})

        # Tokenize
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=profile.max_length
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )

        # Split train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]


        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[
                TimeConstraintCallback(max_hours=max_hours),
                MoETrainingCallback(safety_config)
            ]
        )

        # Train
        if not shutdown_requested:
            wandb.log({"training/status": "started"})


            # Start time tracking
            start_time = time.time()

            # Train
            trainer.train()

            # Calculate actual duration
            actual_duration = (time.time() - start_time) / 3600

            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)

            wandb.log({
                "training/status": "completed",
                "training/actual_hours": actual_duration,
                "training/estimated_hours": estimate.estimated_hours,
                "training/time_accuracy": (estimate.estimated_hours / actual_duration) * 100
            })


    except KeyboardInterrupt:
        if wandb_run:
            wandb.log({"training/status": "interrupted"})

    except Exception as e:
        if wandb_run:
            with contextlib.suppress(BaseException):
                wandb.log({"training/status": "failed", "training/error": str(e)})
        raise

    finally:
        if wandb_run:
            with contextlib.suppress(BaseException):
                wandb.finish()

        if training_start_time:
            (datetime.now(UTC) - training_start_time).total_seconds() / 3600



if __name__ == "__main__":
    main()
