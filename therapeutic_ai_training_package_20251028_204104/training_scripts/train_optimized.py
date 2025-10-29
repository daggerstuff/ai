#!/usr/bin/env python3
"""
Optimized Therapeutic AI Training with Automatic Time Management
Automatically selects best configuration to fit 12-hour window
"""

import json
import torch
import wandb
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, Trainer
from datasets import Dataset
from tqdm import tqdm

from moe_architecture import MoEConfig, create_therapeutic_moe_model
from training_optimizer import (
    TrainingTimeOptimizer,
    optimize_for_dataset
)
from train_moe_h100 import (
    TimeConstraintCallback,
    MoETrainingCallback,
    setup_wandb,
    signal_handler
)

# Global state
shutdown_requested = False
training_start_time = None

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def analyze_dataset(dataset_path: str = 'training_dataset.json'):
    """Analyze dataset to determine optimal training parameters"""
    print("📊 Analyzing dataset...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    texts = [conv['text'] for conv in data['conversations']]
    
    # Calculate statistics
    num_samples = len(texts)
    token_counts = [len(text.split()) for text in texts]
    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    
    print(f"   Samples: {num_samples:,}")
    print(f"   Avg tokens: {avg_tokens:.0f}")
    print(f"   Min tokens: {min_tokens}")
    print(f"   Max tokens: {max_tokens}")
    
    return {
        'num_samples': num_samples,
        'avg_tokens': avg_tokens,
        'max_tokens': max_tokens,
        'min_tokens': min_tokens,
        'texts': texts
    }


def main():
    global shutdown_requested, training_start_time
    
    print("🚀 Optimized Therapeutic AI Training")
    print("=" * 60)
    
    training_start_time = datetime.now()
    wandb_run = None
    
    try:
        # Load configurations
        print("\n📋 Loading configurations...")
        with open('training_config.json', 'r') as f:
            training_config = json.load(f)
        
        with open('safety_config.json', 'r') as f:
            safety_config = json.load(f)
        
        # Analyze dataset
        dataset_info = analyze_dataset()
        
        # Optimize training parameters
        print("\n🎯 Optimizing training parameters for 12-hour window...")
        
        desired_epochs = training_config.get('num_train_epochs', 3)
        priority = training_config.get('optimization_priority', 'balanced')
        max_hours = training_config.get('max_training_hours', 12.0)
        
        profile, estimate, training_args = optimize_for_dataset(
            num_samples=dataset_info['num_samples'],
            avg_tokens_per_sample=int(dataset_info['avg_tokens']),
            num_epochs=desired_epochs,
            priority=priority,
            max_hours=max_hours
        )
        
        if not estimate.fits_in_window:
            print("\n⚠️ Warning: Training may not fit in time window!")
            print("   Applying recommended adjustments...")
            
            if estimate.recommended_adjustments:
                adj = estimate.recommended_adjustments
                if 'new_num_epochs' in adj:
                    desired_epochs = adj['new_num_epochs']
                    print(f"   Reducing epochs to: {desired_epochs}")
                    
                    # Re-optimize with adjusted epochs
                    profile, estimate, training_args = optimize_for_dataset(
                        num_samples=dataset_info['num_samples'],
                        avg_tokens_per_sample=int(dataset_info['avg_tokens']),
                        num_epochs=desired_epochs,
                        priority='fast',
                        max_hours=max_hours
                    )
        
        # Setup WandB
        wandb_run = setup_wandb()
        
        # Log optimization info
        wandb.log({
            'optimization/profile': profile.__class__.__name__,
            'optimization/batch_size': profile.batch_size,
            'optimization/gradient_accumulation': profile.gradient_accumulation_steps,
            'optimization/effective_batch_size': profile.batch_size * profile.gradient_accumulation_steps,
            'optimization/estimated_hours': estimate.estimated_hours,
            'optimization/fits_in_window': estimate.fits_in_window,
            'dataset/num_samples': dataset_info['num_samples'],
            'dataset/avg_tokens': dataset_info['avg_tokens']
        })
        
        # Check CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA not available. This script requires GPU.")
            return
        
        # Create model
        BASE_MODEL_NAME = training_config.get('base_model', "LatitudeGames/Wayfarer-2-12B")
        
        print(f"\n🚀 Creating MoE model from {BASE_MODEL_NAME}...")
        
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
        
        print("✅ MoE model created")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/trainable_percent': (trainable_params / total_params) * 100
        })
        
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
        
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        dataset = Dataset.from_dict({"text": dataset_info['texts']})
        
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
        
        print(f"\n🔤 Tokenizing with max_length={profile.max_length}...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        # Split train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        print(f"   Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")
        
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
            wandb.log({'training/status': 'started'})
            
            print("\n🎯 Starting optimized training...")
            print(f"⏰ Maximum duration: {max_hours} hours")
            print(f"📊 Effective batch size: {profile.batch_size * profile.gradient_accumulation_steps}")
            print(f"🎓 Epochs: {desired_epochs}")
            print(f"📏 Max length: {profile.max_length}")
            print("=" * 60)
            
            # Start time tracking
            start_time = time.time()
            
            # Train
            trainer.train()
            
            # Calculate actual duration
            actual_duration = (time.time() - start_time) / 3600
            
            print(f"\n💾 Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)
            
            wandb.log({
                'training/status': 'completed',
                'training/actual_hours': actual_duration,
                'training/estimated_hours': estimate.estimated_hours,
                'training/time_accuracy': (estimate.estimated_hours / actual_duration) * 100
            })
            
            print(f"\n✅ Training completed!")
            print(f"   Estimated: {estimate.estimated_hours:.2f} hours")
            print(f"   Actual: {actual_duration:.2f} hours")
            print(f"   Accuracy: {(estimate.estimated_hours / actual_duration) * 100:.1f}%")
            print(f"📁 Model saved to: {training_args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")
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
