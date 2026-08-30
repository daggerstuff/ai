#!/usr/bin/env python3
"""
Mental Health Instruction Fine-Tuning (IFT) Trainer

Implements task-specific instruction fine-tuning for mental health prediction,
following Mental-LLM's methodology:
- Curriculum learning (classification -> estimation -> generation)
- LoRA / QLoRA for resource efficiency
- Per-task evaluation after each epoch
- Hyperparameter search support
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)

from training.mental_health_instruction_dataset import (
    MentalHealthInstructionDatasetBuilder,
    MentalHealthTaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class IFTConfig:
    """Configuration for mental health IFT."""

    base_model: str = "zai-org/glm-5.3-flash"
    output_dir: str = "./ai/models/mental_ift"
    dataset_path: str | None = "ai/data/curated/sft_chatml/train_master_gold.jsonl"
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_seq_length: int = 2048
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 400
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    curriculum_learning: bool = True
    task_order: tuple[str, ...] = (
        MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value,
        MentalHealthTaskType.SEVERITY_ESTIMATION.value,
        MentalHealthTaskType.RISK_ASSESSMENT.value,
        MentalHealthTaskType.EMPATHY_SCORING.value,
        MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value,
    )
    seed: int = 42


class TaskEvalCallback(TrainerCallback):
    """Callback to evaluate per-task performance after each epoch."""

    def __init__(self, trainer: MentalHealthIFTTrainer, eval_dataset: Dataset):
        self.trainer = trainer
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} ended. Running per-task evaluation...")
        results = self.trainer.evaluate_per_task(self.eval_dataset)
        for task, metrics in results.items():
            logger.info(f"Task {task}: eval_loss={metrics.get('eval_loss', 'N/A')}")
        state.log_history.append({"epoch": state.epoch, "per_task_eval": results})


class MentalHealthIFTTrainer:
    """Trainer for mental health instruction fine-tuning."""

    def __init__(self, config: IFTConfig | None = None):
        self.config = config or IFTConfig()
        self.tokenizer: AutoTokenizer | None = None
        self.model: Any = None
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self._task_datasets: dict[str, Dataset] = {}

    def setup_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return self.tokenizer

    def setup_model(self) -> Any:
        """Load base model with optional QLoRA quantization."""
        logger.info(f"Loading base model: {self.config.base_model}")

        bnb_config = None
        torch_dtype = torch.bfloat16
        if self.config.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            torch_dtype = torch.bfloat16

        attn_impl = None
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                logger.warning("flash_attn not installed; falling back to default attention")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        if self.config.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

    def load_or_build_dataset(self) -> Dataset:
        """Load dataset from path or build default mental health IFT dataset."""
        if self.config.dataset_path and Path(self.config.dataset_path).exists():
            logger.info(f"Loading dataset from {self.config.dataset_path}")
            records = []
            with open(self.config.dataset_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    record: dict[str, Any] = {}
                    if "messages" in item:
                        record["messages"] = item["messages"]
                    if "instruction" in item:
                        record["instruction"] = item["instruction"]
                    if "input" in item:
                        record["input"] = item.get("input", "") or ""
                    if "output" in item:
                        record["output"] = item.get("output", "") or ""
                    record["task_type"] = str(item.get("task_type", "") or "")
                    record["source"] = str(item.get("source", "") or "")
                    records.append(record)
            return Dataset.from_list(records)

        logger.info("Building default mental health IFT dataset")
        builder = MentalHealthInstructionDatasetBuilder(seed=self.config.seed)
        builder.build_from_seed_vignettes(augment_per_vignette=400)
        train, val = builder.stratified_split(train_ratio=0.9)
        return Dataset.from_list([ex.to_alpaca() for ex in train])

    def _split_by_task(self, examples: list[Any]) -> dict[str, list[dict[str, Any]]]:
        """Group examples by task type."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            task = ex.task_type
            groups.setdefault(task, []).append(ex.to_alpaca())
        return groups

    def prepare_datasets(self) -> tuple[Dataset, Dataset]:
        """Tokenize and optionally curriculum-sort datasets."""
        raw_train = self.load_or_build_dataset()

        # If curriculum learning enabled, sort by task difficulty order
        if self.config.curriculum_learning:
            raw_train = self._apply_curriculum(raw_train)

        # Validation set: stratified holdout from actual training data, not separate seed vignettes
        split = raw_train.train_test_split(test_size=0.1, seed=self.config.seed)
        self.train_dataset = split["train"].map(
            self._format_and_tokenize,
            batched=True,
            remove_columns=split["train"].column_names,
        )
        raw_val = split["test"]
        self.eval_dataset = raw_val.map(
            self._format_and_tokenize,
            batched=True,
            remove_columns=raw_val.column_names,
        )

        return self.train_dataset, self.eval_dataset

    def _apply_curriculum(self, dataset: Dataset) -> Dataset:
        """Sort dataset by curriculum difficulty order."""
        task_order = {task: idx for idx, task in enumerate(self.config.task_order)}

        def sort_key(example: dict[str, Any]) -> int:
            return task_order.get(example.get("task_type", ""), 999)

        sorted_indices = sorted(range(len(dataset)), key=lambda i: sort_key(dataset[i]))
        return dataset.select(sorted_indices)

    def _format_and_tokenize(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Format ChatML or Alpaca examples into prompt-completion strings and tokenize."""
        prompts = []
        if "messages" in examples:
            for msgs in examples["messages"]:
                if not msgs:
                    continue
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    try:
                        formatted = self.tokenizer.apply_chat_template(msgs, tokenize=False)
                    except Exception:
                        formatted = "\n".join(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in msgs)
                else:
                    formatted = "\n".join(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in msgs)
                prompts.append(formatted)
        elif "instruction" in examples and "output" in examples:
            inputs = examples.get("input", [""] * len(examples["instruction"]))
            for instruction, input_text, output in zip(examples["instruction"], inputs, examples["output"]):
                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                prompts.append(prompt)
        else:
            raise ValueError("Dataset must contain either 'messages' (ChatML) or 'instruction'/'output' (Alpaca) keys.")

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
            return_overflowing_tokens=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        if "task_type" in examples:
            tokenized["task_type"] = examples["task_type"]
        return tokenized

    def train(self) -> dict[str, Any]:
        """Run instruction fine-tuning."""
        if self.tokenizer is None:
            self.setup_tokenizer()
        if self.model is None:
            self.setup_model()
        if self.train_dataset is None:
            self.prepare_datasets()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            report_to="wandb" if os.getenv("WANDB_API_KEY") else [],
            run_name="mental-health-ift",
            seed=self.config.seed,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
        )

        from transformers import Trainer

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                TaskEvalCallback(self, self.eval_dataset),
            ],
        )

        logger.info("Starting IFT training")
        train_result = trainer.train()

        # Save final model and tokenizer
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))

        # Save training metrics
        metrics = train_result.metrics
        metrics["eval_loss"] = trainer.evaluate().get("eval_loss")
        with open(final_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Model saved to {final_dir}")
        return metrics

    def evaluate_per_task(self, eval_dataset: Dataset) -> dict[str, dict[str, float]]:
        """Evaluate model separately for each mental health task type."""
        from transformers import Trainer

        results: dict[str, dict[str, float]] = {}
        task_order = self.config.task_order
        for task in task_order:
            task_examples = [ex for ex in eval_dataset if ex.get("task_type") == task]
            if not task_examples:
                continue
            task_dataset = Dataset.from_list(task_examples)
            task_dataset = task_dataset.map(
                self._format_and_tokenize,
                batched=True,
                remove_columns=task_dataset.column_names,
            )
            trainer = Trainer(model=self.model, tokenizer=self.tokenizer)
            metrics = trainer.evaluate(task_dataset)
            results[task] = metrics
        return results

    def hyperparameter_search(
        self,
        learning_rates: list[float] | None = None,
        lora_ranks: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a small grid search over key hyperparameters."""
        learning_rates = learning_rates or [1e-4, 2e-4, 5e-4]
        lora_ranks = lora_ranks or [8, 16, 32]

        results: list[dict[str, Any]] = []
        base_output_dir = self.config.output_dir

        for lr in learning_rates:
            for rank in lora_ranks:
                logger.info(f"Hyperparameter search: lr={lr}, lora_r={rank}")
                self.config.learning_rate = lr
                self.config.lora_r = rank
                self.config.output_dir = f"{base_output_dir}_lr{lr}_r{rank}"
                self.model = None  # Force re-init
                metrics = self.train()
                results.append({"learning_rate": lr, "lora_r": rank, "metrics": metrics})

        # Restore best config by eval_loss
        best = min(results, key=lambda r: r["metrics"].get("eval_loss", float("inf")))
        logger.info(f"Best config: {best}")
        return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = IFTConfig()
    trainer = MentalHealthIFTTrainer(config)
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
