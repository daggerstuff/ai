import argparse
import json
import logging
import os
import sys

import psutil
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

# --- Add local training directory to sys.path for local imports ---
logging.getLogger("pixel_training").info("Appending ../training to sys.path for local imports")
sys.path.append(os.path.join(os.path.dirname(__file__), "../training"))
try:
    from training_checkpoint_manager import (
        CheckpointConfig,
        CheckpointManager,
        create_training_state_from_model,
    )

    logging.getLogger("pixel_training").info("Successfully imported training_checkpoint_manager")
except Exception as e:
    logging.getLogger("pixel_training").error(f"Failed to import training_checkpoint_manager: {e}")
    raise

# Load configs
with open(os.path.join(os.path.dirname(__file__), "../config/training_config.json")) as f:
    config = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "../config/deepspeed_config.json")) as f:
    ds_config = json.load(f)

# Parse CLI args for resume-from-checkpoint
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Path or ID of checkpoint to resume from",
)
args, _ = parser.parse_known_args()

# Setup logging with redaction for HIPAA/GDPR


class RedactingFilter(logging.Filter):
    def filter(self, record):
        sensitive_keywords = ["patient", "name", "dob", "ssn", "address", "phone", "email"]
        for keyword in sensitive_keywords:
            if keyword in str(record.msg).lower():
                record.msg = "[REDACTED SENSITIVE DATA]"
        return True


logger = logging.getLogger("pixel_training")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.addFilter(RedactingFilter())
formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


checkpoint_config = CheckpointConfig(
    save_every_steps=config["training"].get("save_steps", 1000),
    save_every_epochs=1,
    save_every_minutes=0,
    keep_last_n_checkpoints=config["training"].get("save_total_limit", 5),
    checkpoint_dir=config["training"].get("output_dir", "checkpoints"),
    compress_checkpoints=True,
    atomic_saves=True,
    validate_on_save=True,
    best_metric_name=config["training"].get("metric_for_best_model", "eval_loss"),
    best_metric_mode="min" if not config["training"].get("greater_is_better", False) else "max",
)
checkpoint_manager = CheckpointManager(checkpoint_config)

# Initialize W&B
try:
    wandb.init(
        project=ds_config.get("wandb", {}).get("project", "Pixel-LLM"),
        group=ds_config.get("wandb", {}).get("group", "distributed-training"),
        name=config["training"].get("run_name", "pixelated_empathy_training"),
        notes=ds_config.get("wandb", {}).get("notes", ""),
        config=config,
        reinit=True,
    )
    logger.info("Weights & Biases initialized.")
except Exception as e:
    logger.error(f"W&B initialization failed: {e}")
    wandb = None

# Load model and tokenizer
model_name = config["model"]["base_model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare training arguments
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    num_train_epochs=config["training"]["num_train_epochs"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
    warmup_ratio=config["training"]["warmup_ratio"],
    lr_scheduler_type=config["training"]["lr_scheduler_type"],
    logging_steps=config["training"]["logging_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    save_total_limit=config["training"]["save_total_limit"],
    load_best_model_at_end=config["training"]["load_best_model_at_end"],
    metric_for_best_model=config["training"]["metric_for_best_model"],
    greater_is_better=config["training"]["greater_is_better"],
    report_to=["wandb"] if wandb else [],
    deepspeed=os.path.join(os.path.dirname(__file__), "../config/deepspeed_config.json"),
    fp16=config["system"]["fp16"],
    bf16=config["system"]["bf16"],
    gradient_checkpointing=config["system"]["gradient_checkpointing"],
    dataloader_num_workers=config["system"]["dataloader_num_workers"],
    dataloader_pin_memory=config["system"]["dataloader_pin_memory"],
    remove_unused_columns=config["system"]["remove_unused_columns"],
)

# Dummy dataset for demonstration (replace with real data loader)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, 100, (32,)), "labels": torch.randint(0, 100, (32,))}


train_dataset = DummyDataset()
eval_dataset = DummyDataset()

# --- Custom validation callback for robust, compliant evaluation logging ---


class PixelValidationCallback(TrainerCallback):
    def on_evaluate(self, _args, state, _control, **kwargs):
        logger.debug(
            "PixelValidationCallback.on_evaluate called; _args and _control are unused and required by interface."
        )
        metrics = kwargs.get("metrics", {})
        redacted_metrics = {}
        sensitive_keywords = ["patient", "name", "dob", "ssn", "address", "phone", "email"]
        for k, v in metrics.items():
            if any(s in k.lower() for s in sensitive_keywords):
                redacted_metrics[k] = "[REDACTED]"
            else:
                redacted_metrics[k] = v
        if wandb:
            try:
                wandb.log(
                    {f"eval/{k}": v for k, v in redacted_metrics.items()}, step=state.global_step
                )
            except Exception as e:
                logger.error(f"W&B validation log error: {e}")
        logger.info(f"Validation metrics (redacted): {redacted_metrics}")


# Custom Trainer to log extra metrics and redact sensitive data


class PixelTrainer(Trainer):
    def log(self, logs):
        for k, _v in list(logs.items()):
            if any(
                s in k.lower()
                for s in ["patient", "name", "dob", "ssn", "address", "phone", "email"]
            ):
                logs[k] = "[REDACTED]"
        try:
            process = psutil.Process(os.getpid())
            logs["cpu_mem_mb"] = process.memory_info().rss / 1024 / 1024
        except Exception:
            logs["cpu_mem_mb"] = -1
        if torch.cuda.is_available():
            logs["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            logs["gpu_mem_mb"] = -1
        super().log(logs)
        if wandb:
            try:
                wandb.log(logs)
            except Exception as e:
                logger.error(f"W&B log error: {e}")

    def _maybe_save_checkpoint(self):
        create_training_state_from_model(
            self.model,
            self.args,
            self.tokenizer,
            self.train_dataset,
            self.eval_dataset,
        )


# --- Resource monitoring callback for system and GPU utilization ---
class PixelResourceMonitorCallback(TrainerCallback):
    def __init__(self, interval_steps=50):
        self.interval_steps = interval_steps
        self.last_logged_step = 0

    def on_train_begin(self):
        self.last_logged_step = 0
        self.interval_steps = 100

    def on_step_end(self, state):
        if state.global_step - self.last_logged_step >= self.interval_steps:
            self.last_logged_step = state.global_step
            metrics = {}
            # CPU
            metrics["cpu_percent"] = psutil.cpu_percent()
            metrics["cpu_mem_mb"] = psutil.virtual_memory().used / 1024 / 1024
            # Disk
            disk = psutil.disk_usage("/")
            metrics["disk_used_gb"] = disk.used / 1024 / 1024 / 1024
            metrics["disk_free_gb"] = disk.free / 1024 / 1024 / 1024
            # Network
            net = psutil.net_io_counters()
            metrics["net_bytes_sent_mb"] = net.bytes_sent / 1024 / 1024
            metrics["net_bytes_recv_mb"] = net.bytes_recv / 1024 / 1024
            # GPU
            if torch.cuda.is_available():
                metrics["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                metrics["gpu_mem_mb"] = -1
            # Log to W&B and logger
            if wandb:
                try:
                    wandb.log(
                        {f"resource/{k}": v for k, v in metrics.items()}, step=state.global_step
                    )
                except Exception as e:
                    logger.error(f"W&B resource log error: {e}")
            logger.info(f"Resource metrics: {metrics}")


trainer = PixelTrainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
)

# Register the callback
trainer.add_callback(PixelValidationCallback())


# --- Orchestration and robust error handling wrapper ---
def orchestrate_training():
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete.")
        if wandb:
            wandb.finish()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if wandb:
            wandb.alert(title="Training Failure", text=str(e))
        # Optionally: add more recovery/cleanup logic here
        # For distributed/DeepSpeed errors, escalate or retry as needed


if __name__ == "__main__":
    orchestrate_training()
