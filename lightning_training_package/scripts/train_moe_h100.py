#!/usr/bin/env python3
"""
Therapeutic AI Training with MoE Architecture on H100
Optimized for 12-hour training window with LoRA fine-tuning
"""

import json
import torch
import wandb
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
from tqdm import tqdm

from moe_architecture import (
    MoEConfig,
    TherapeuticMoEModel,
    create_therapeutic_moe_model
)

# Global shutdown flag
shutdown_requested = False
training_start_time = None
MAX_TRAINING_HOURS = 12


def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n🛑 Shutdown requested")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class TimeConstraintCallback(TrainerCallback):
    """Callback to enforce 12-hour training window"""
    
    def __init__(self, max_hours: int = 12):
        self.max_hours = max_hours
        self.start_time = None
        self.last_checkpoint_time = None
        self.checkpoint_interval_minutes = 30
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        print(f"⏰ Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Maximum training duration: {self.max_hours} hours")
        
    def on_step_end(self, args, state, control, **kwargs):
        global shutdown_requested
        
        if shutdown_requested:
            control.should_training_stop = True
            control.should_save = True
            return control
        
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        
        # Check if we're approaching time limit
        if elapsed_hours >= self.max_hours - 0.5:  # Stop 30 min before limit
            print(f"\n⏰ Approaching {self.max_hours}-hour limit. Stopping training...")
            control.should_training_stop = True
            control.should_save = True
            
            wandb.log({
                'training/stopped_reason': 'time_limit',
                'training/elapsed_hours': elapsed_hours
            })
        
        # Periodic checkpointing (every 30 minutes)
        elapsed_since_checkpoint = (current_time - self.last_checkpoint_time) / 60
        if elapsed_since_checkpoint >= self.checkpoint_interval_minutes:
            control.should_save = True
            self.last_checkpoint_time = current_time
            print(f"💾 Checkpoint at {elapsed_hours:.2f} hours")
        
        # Log time progress
        if state.global_step % 100 == 0:
            remaining_hours = self.max_hours - elapsed_hours
            wandb.log({
                'training/elapsed_hours': elapsed_hours,
                'training/remaining_hours': remaining_hours,
                'training/progress_percent': (elapsed_hours / self.max_hours) * 100
            })
        
        return control


class MoETrainingCallback(TrainerCallback):
    """Callback for MoE-specific monitoring"""
    
    def __init__(self, safety_config: Dict[str, Any]):
        self.safety_config = safety_config
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 3
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        global shutdown_requested
        
        if shutdown_requested:
            print("🛑 Stopping training")
            control.should_training_stop = True
            return control
        
        if logs:
            self.step_count += 1
            current_loss = logs.get('loss', logs.get('train_loss', 0))
            
            # Early stopping based on validation loss
            if 'eval_loss' in logs:
                eval_loss = logs['eval_loss']
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    print(f"\n⚠️ Early stopping triggered (patience={self.patience})")
                    control.should_training_stop = True
                    control.should_save = True
                    
                    wandb.log({
                        'training/stopped_reason': 'early_stopping',
                        'training/best_eval_loss': self.best_loss
                    })
            
            # Log progress
            if 'epoch' in logs:
                progress = (logs['epoch'] / args.num_train_epochs) * 100
                print(f"📊 Progress: {progress:.1f}% | Loss: {current_loss:.4f} | Step: {self.step_count}")
            
            # Enhanced logging
            enhanced_logs = logs.copy()
            enhanced_logs.update({
                'training/steps_completed': self.step_count,
                'training/best_loss': self.best_loss,
                'training/patience_counter': self.patience_counter,
                'system/shutdown_requested': shutdown_requested
            })
            
            wandb.log(enhanced_logs, step=state.global_step)
        
        return control


def setup_wandb(config_path: str = 'wandb_config.json'):
    """Setup Weights & Biases logging"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if not torch.cuda.is_available():
        print("⚠️ No CUDA - using offline mode")
        os.environ['WANDB_MODE'] = 'offline'
    
    run = wandb.init(
        project=config['project'],
        entity=config.get('entity'),
        name=f"{config['name']}_moe_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=config['tags'] + ['moe', 'h100', 'lora'],
        notes=f"MoE training with LoRA on H100 - {config['notes']}",
        config=config['config']
    )
    
    return run


def load_training_data(dataset_path: str = 'training_dataset.json') -> Dataset:
    """Load and prepare training dataset"""
    print(f"📊 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    texts = [conv['text'] for conv in tqdm(data['conversations'], desc="Loading")]
    dataset = Dataset.from_dict({"text": texts})
    
    print(f"📊 Dataset: {len(dataset)} samples")
    
    return dataset, texts


def create_h100_training_args(
    output_dir: str = "./therapeutic_moe_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    max_steps: int = -1
) -> TrainingArguments:
    """
    Create H100-optimized training arguments
    
    Optimized for:
    - 12-hour training window
    - H100 GPU memory (80GB)
    - LoRA fine-tuning efficiency
    """
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        
        # Batch size optimization for H100
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Learning rate with warmup
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        
        # Regularization
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # H100 optimizations
        bf16=True,  # BFloat16 for H100
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        
        # Logging and checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # WandB reporting
        report_to="wandb",
        
        # Performance
        optim="adamw_torch_fused",  # Fused optimizer for H100
        group_by_length=True,
        
        # Disable unnecessary features
        push_to_hub=False,
        remove_unused_columns=True,
    )


def main():
    global shutdown_requested, training_start_time
    
    print("🚀 Therapeutic AI Training with MoE Architecture")
    print("=" * 60)
    
    training_start_time = datetime.now()
    wandb_run = None
    
    try:
        # Setup WandB
        wandb_run = setup_wandb()
        
        # Load configurations
        print("📋 Loading configurations...")
        with open('training_config.json', 'r') as f:
            training_config = json.load(f)
        
        with open('safety_config.json', 'r') as f:
            safety_config = json.load(f)
        
        # Load dataset
        dataset, texts = load_training_data()
        
        wandb.log({
            'dataset/total_conversations': len(texts),
            'dataset/avg_length': sum(len(text.split()) for text in texts) / len(texts)
        })
        
        # Setup model
        BASE_MODEL_NAME = training_config.get('base_model', "LatitudeGames/Wayfarer-2-12B")
        device_available = torch.cuda.is_available()
        
        if not device_available:
            print("❌ CUDA not available. This script requires GPU.")
            return
        
        print(f"🚀 Creating MoE model from {BASE_MODEL_NAME}...")
        
        # Create MoE configuration
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
        
        # Create therapeutic MoE model
        model = create_therapeutic_moe_model(
            BASE_MODEL_NAME,
            moe_config=moe_config,
            device="auto"
        )
        
        print("✅ MoE model created successfully")
        print(f"   - Experts: {moe_config.num_experts}")
        print(f"   - Domains: {', '.join(moe_config.expert_domains)}")
        print(f"   - LoRA rank: {moe_config.lora_r}")
        print(f"   - Context length: {moe_config.max_position_embeddings}")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/trainable_percent': (trainable_params / total_params) * 100,
            'model/num_experts': moe_config.num_experts,
            'model/lora_rank': moe_config.lora_r,
            'model/context_length': moe_config.max_position_embeddings
        })
        
        print(f"📊 Model parameters:")
        print(f"   - Total: {total_params:,}")
        print(f"   - Trainable: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
        
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize dataset
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048  # Use 2048 for training, model supports 8192
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        print(f"📊 Tokenized: {len(tokenized_dataset)} samples")
        
        if len(tokenized_dataset) == 0:
            raise ValueError("Empty dataset after tokenization!")
        
        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        print(f"📊 Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
        
        # Create H100-optimized training arguments
        training_args = create_h100_training_args(
            output_dir="./therapeutic_moe_model",
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 3e-4),
            warmup_steps=training_config.get('warmup_steps', 1000)
        )
        
        # Create trainer with callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[
                TimeConstraintCallback(max_hours=MAX_TRAINING_HOURS),
                MoETrainingCallback(safety_config)
            ]
        )
        
        # Train
        if not shutdown_requested:
            wandb.log({'training/status': 'started'})
            
            print("\n🎯 Starting training...")
            print(f"⏰ Maximum duration: {MAX_TRAINING_HOURS} hours")
            print(f"📊 Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
            print("=" * 60)
            
            trainer.train()
            
            print("\n💾 Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            
            # Save MoE-specific components
            model.save_pretrained(training_args.output_dir)
            
            wandb.log({'training/status': 'completed'})
            
            training_duration = (datetime.now() - training_start_time).total_seconds() / 3600
            print(f"\n✅ Training completed in {training_duration:.2f} hours!")
            print(f"📁 Model saved to: {training_args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        if wandb_run:
            wandb.log({'training/status': 'interrupted'})
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if wandb_run:
            try:
                wandb.log({'training/status': 'failed', 'training/error': str(e)})
            except:
                pass
        raise
    
    finally:
        if wandb_run:
            try:
                wandb.finish()
            except:
                pass
        
        if training_start_time:
            total_duration = (datetime.now() - training_start_time).total_seconds() / 3600
            print(f"\n⏰ Total runtime: {total_duration:.2f} hours")
        
        print("🧹 Cleanup complete")


if __name__ == "__main__":
    main()
