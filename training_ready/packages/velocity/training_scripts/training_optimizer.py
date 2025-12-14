#!/usr/bin/env python3
"""
Training Optimizer for H100 12-Hour Window
Ensures optimal training completion within time constraints
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from transformers import TrainingArguments


@dataclass
class TrainingTimeEstimate:
    """Estimates for training completion"""
    total_steps: int
    steps_per_second: float
    estimated_hours: float
    estimated_completion: datetime
    fits_in_window: bool
    recommended_adjustments: Dict[str, Any]


@dataclass
class H100OptimizationProfile:
    """H100-specific optimization profile"""
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    num_workers: int
    use_gradient_checkpointing: bool
    use_bf16: bool
    use_fused_optimizer: bool
    estimated_throughput: float  # tokens/sec
    memory_usage_gb: float


class TrainingTimeOptimizer:
    """
    Optimizes training parameters to fit within 12-hour window
    while maximizing model quality
    """
    
    def __init__(
        self,
        max_hours: float = 12.0,
        safety_margin_hours: float = 0.5,
        target_gpu_memory_gb: float = 70.0,  # H100 has 80GB
        h100_peak_throughput: float = 1000.0  # tokens/sec baseline
    ):
        self.max_hours = max_hours
        self.safety_margin_hours = safety_margin_hours
        self.available_hours = max_hours - safety_margin_hours
        self.target_gpu_memory_gb = target_gpu_memory_gb
        self.h100_peak_throughput = h100_peak_throughput
        
        # Optimization profiles for different scenarios
        self.profiles = self._create_optimization_profiles()
        
    def _create_optimization_profiles(self) -> Dict[str, H100OptimizationProfile]:
        """Create predefined optimization profiles"""
        return {
            'fast': H100OptimizationProfile(
                batch_size=8,
                gradient_accumulation_steps=4,
                max_length=1024,
                num_workers=4,
                use_gradient_checkpointing=False,
                use_bf16=True,
                use_fused_optimizer=True,
                estimated_throughput=1200.0,
                memory_usage_gb=75.0
            ),
            'balanced': H100OptimizationProfile(
                batch_size=4,
                gradient_accumulation_steps=8,
                max_length=2048,
                num_workers=4,
                use_gradient_checkpointing=True,
                use_bf16=True,
                use_fused_optimizer=True,
                estimated_throughput=800.0,
                memory_usage_gb=60.0
            ),
            'quality': H100OptimizationProfile(
                batch_size=2,
                gradient_accumulation_steps=16,
                max_length=4096,
                num_workers=4,
                use_gradient_checkpointing=True,
                use_bf16=True,
                use_fused_optimizer=True,
                estimated_throughput=400.0,
                memory_usage_gb=70.0
            ),
            'memory_efficient': H100OptimizationProfile(
                batch_size=1,
                gradient_accumulation_steps=32,
                max_length=2048,
                num_workers=2,
                use_gradient_checkpointing=True,
                use_bf16=True,
                use_fused_optimizer=True,
                estimated_throughput=300.0,
                memory_usage_gb=45.0
            )
        }
    
    def estimate_training_time(
        self,
        num_samples: int,
        avg_tokens_per_sample: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_epochs: int,
        tokens_per_second: Optional[float] = None
    ) -> TrainingTimeEstimate:
        """
        Estimate training time based on dataset and parameters
        
        Args:
            num_samples: Number of training samples
            avg_tokens_per_sample: Average tokens per sample
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation steps
            num_epochs: Number of training epochs
            tokens_per_second: Measured throughput (uses estimate if None)
            
        Returns:
            TrainingTimeEstimate with completion predictions
        """
        # Calculate total steps
        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = num_samples // effective_batch_size
        total_steps = steps_per_epoch * num_epochs
        
        # Calculate total tokens
        total_tokens = num_samples * avg_tokens_per_sample * num_epochs
        
        # Estimate throughput
        if tokens_per_second is None:
            # Use profile-based estimate
            tokens_per_second = self.h100_peak_throughput * 0.7  # Conservative estimate
        
        # Calculate time
        estimated_seconds = total_tokens / tokens_per_second
        estimated_hours = estimated_seconds / 3600
        
        # Check if fits in window
        fits_in_window = estimated_hours <= self.available_hours
        
        # Calculate completion time
        estimated_completion = datetime.now() + timedelta(hours=estimated_hours)
        
        # Generate recommendations if doesn't fit
        recommended_adjustments = {}
        if not fits_in_window:
            # Calculate required speedup
            required_speedup = estimated_hours / self.available_hours
            
            # Suggest adjustments
            if required_speedup <= 1.5:
                # Minor adjustments
                recommended_adjustments = {
                    'action': 'increase_batch_size',
                    'new_batch_size': batch_size * 2,
                    'new_gradient_accumulation': gradient_accumulation_steps // 2,
                    'reason': 'Increase throughput with larger batches'
                }
            elif required_speedup <= 2.0:
                # Moderate adjustments
                recommended_adjustments = {
                    'action': 'reduce_epochs',
                    'new_num_epochs': max(1, int(num_epochs / required_speedup)),
                    'reason': 'Reduce training epochs to fit time window'
                }
            else:
                # Major adjustments
                recommended_adjustments = {
                    'action': 'use_fast_profile',
                    'profile': 'fast',
                    'reduce_epochs': True,
                    'new_num_epochs': max(1, int(num_epochs / 2)),
                    'reason': 'Use fast profile and reduce epochs significantly'
                }
        
        return TrainingTimeEstimate(
            total_steps=total_steps,
            steps_per_second=tokens_per_second / avg_tokens_per_sample,
            estimated_hours=estimated_hours,
            estimated_completion=estimated_completion,
            fits_in_window=fits_in_window,
            recommended_adjustments=recommended_adjustments
        )
    
    def select_optimal_profile(
        self,
        num_samples: int,
        avg_tokens_per_sample: int,
        num_epochs: int,
        priority: str = 'balanced'
    ) -> Tuple[str, H100OptimizationProfile, TrainingTimeEstimate]:
        """
        Select optimal training profile for dataset
        
        Args:
            num_samples: Number of training samples
            avg_tokens_per_sample: Average tokens per sample
            num_epochs: Desired number of epochs
            priority: 'fast', 'balanced', 'quality', or 'memory_efficient'
            
        Returns:
            Tuple of (profile_name, profile, time_estimate)
        """
        # Try requested profile first
        if priority in self.profiles:
            profile = self.profiles[priority]
            estimate = self.estimate_training_time(
                num_samples=num_samples,
                avg_tokens_per_sample=avg_tokens_per_sample,
                batch_size=profile.batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                num_epochs=num_epochs,
                tokens_per_second=profile.estimated_throughput
            )
            
            if estimate.fits_in_window:
                return priority, profile, estimate
        
        # Try profiles in order of speed
        profile_order = ['fast', 'balanced', 'quality', 'memory_efficient']
        
        for profile_name in profile_order:
            profile = self.profiles[profile_name]
            estimate = self.estimate_training_time(
                num_samples=num_samples,
                avg_tokens_per_sample=avg_tokens_per_sample,
                batch_size=profile.batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                num_epochs=num_epochs,
                tokens_per_second=profile.estimated_throughput
            )
            
            if estimate.fits_in_window:
                return profile_name, profile, estimate
        
        # If nothing fits, return fast profile with reduced epochs
        profile = self.profiles['fast']
        
        # Binary search for max epochs that fit
        min_epochs, max_epochs = 1, num_epochs
        best_epochs = 1
        
        while min_epochs <= max_epochs:
            mid_epochs = (min_epochs + max_epochs) // 2
            estimate = self.estimate_training_time(
                num_samples=num_samples,
                avg_tokens_per_sample=avg_tokens_per_sample,
                batch_size=profile.batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                num_epochs=mid_epochs,
                tokens_per_second=profile.estimated_throughput
            )
            
            if estimate.fits_in_window:
                best_epochs = mid_epochs
                min_epochs = mid_epochs + 1
            else:
                max_epochs = mid_epochs - 1
        
        # Get final estimate with best epochs
        estimate = self.estimate_training_time(
            num_samples=num_samples,
            avg_tokens_per_sample=avg_tokens_per_sample,
            batch_size=profile.batch_size,
            gradient_accumulation_steps=profile.gradient_accumulation_steps,
            num_epochs=best_epochs,
            tokens_per_second=profile.estimated_throughput
        )
        
        return 'fast', profile, estimate
    
    def create_optimized_training_args(
        self,
        profile: H100OptimizationProfile,
        output_dir: str = "./model_output",
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000
    ) -> TrainingArguments:
        """
        Create TrainingArguments optimized for H100 and time constraints
        
        Args:
            profile: H100OptimizationProfile to use
            output_dir: Output directory for model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            
        Returns:
            Optimized TrainingArguments
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            
            # Batch size from profile
            per_device_train_batch_size=profile.batch_size,
            per_device_eval_batch_size=profile.batch_size,
            gradient_accumulation_steps=profile.gradient_accumulation_steps,
            
            # Learning rate
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            
            # Regularization
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # H100 optimizations from profile
            bf16=profile.use_bf16,
            bf16_full_eval=profile.use_bf16,
            gradient_checkpointing=profile.use_gradient_checkpointing,
            dataloader_num_workers=profile.num_workers,
            dataloader_pin_memory=True,
            
            # Optimizer
            optim="adamw_torch_fused" if profile.use_fused_optimizer else "adamw_torch",
            
            # Logging and checkpointing
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=5,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Performance
            group_by_length=True,
            
            # WandB
            report_to="wandb",
            
            # Misc
            push_to_hub=False,
            remove_unused_columns=True,
        )
    
    def monitor_and_adjust(
        self,
        start_time: float,
        current_step: int,
        total_steps: int,
        current_loss: float
    ) -> Dict[str, Any]:
        """
        Monitor training progress and suggest adjustments
        
        Args:
            start_time: Training start timestamp
            current_step: Current training step
            total_steps: Total training steps
            current_loss: Current training loss
            
        Returns:
            Dict with monitoring info and recommendations
        """
        elapsed_hours = (time.time() - start_time) / 3600
        progress = current_step / total_steps
        
        # Estimate remaining time
        if progress > 0:
            estimated_total_hours = elapsed_hours / progress
            remaining_hours = estimated_total_hours - elapsed_hours
        else:
            estimated_total_hours = self.max_hours
            remaining_hours = self.max_hours
        
        # Check if on track
        on_track = remaining_hours <= (self.available_hours - elapsed_hours)
        
        # Calculate actual throughput
        steps_per_hour = current_step / elapsed_hours if elapsed_hours > 0 else 0
        
        # Generate recommendations
        recommendations = []
        
        if not on_track and progress < 0.8:
            # Behind schedule
            recommendations.append({
                'type': 'warning',
                'message': 'Training behind schedule',
                'action': 'Consider increasing batch size or reducing epochs'
            })
        
        if elapsed_hours > (self.max_hours - self.safety_margin_hours):
            # Approaching time limit
            recommendations.append({
                'type': 'critical',
                'message': 'Approaching time limit',
                'action': 'Prepare for graceful shutdown'
            })
        
        if current_loss < 0.5 and progress > 0.5:
            # Good progress, might finish early
            recommendations.append({
                'type': 'info',
                'message': 'Excellent progress',
                'action': 'Consider early stopping if loss plateaus'
            })
        
        return {
            'elapsed_hours': elapsed_hours,
            'remaining_hours': remaining_hours,
            'estimated_total_hours': estimated_total_hours,
            'progress_percent': progress * 100,
            'on_track': on_track,
            'steps_per_hour': steps_per_hour,
            'recommendations': recommendations
        }
    
    def save_optimization_report(
        self,
        filepath: str,
        profile_name: str,
        profile: H100OptimizationProfile,
        estimate: TrainingTimeEstimate,
        dataset_info: Dict[str, Any]
    ):
        """Save optimization report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'max_training_hours': self.max_hours,
            'safety_margin_hours': self.safety_margin_hours,
            'profile_name': profile_name,
            'profile': asdict(profile),
            'estimate': asdict(estimate),
            'dataset_info': dataset_info
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Optimization report saved to: {filepath}")


def optimize_for_dataset(
    num_samples: int,
    avg_tokens_per_sample: int,
    num_epochs: int = 3,
    priority: str = 'balanced',
    max_hours: float = 12.0
) -> Tuple[H100OptimizationProfile, TrainingTimeEstimate, TrainingArguments]:
    """
    Convenience function to optimize training for a dataset
    
    Args:
        num_samples: Number of training samples
        avg_tokens_per_sample: Average tokens per sample
        num_epochs: Desired number of epochs
        priority: Optimization priority ('fast', 'balanced', 'quality')
        max_hours: Maximum training hours
        
    Returns:
        Tuple of (profile, estimate, training_args)
    """
    optimizer = TrainingTimeOptimizer(max_hours=max_hours)
    
    profile_name, profile, estimate = optimizer.select_optimal_profile(
        num_samples=num_samples,
        avg_tokens_per_sample=avg_tokens_per_sample,
        num_epochs=num_epochs,
        priority=priority
    )
    
    print(f"\n🎯 Selected Profile: {profile_name}")
    print(f"   Batch Size: {profile.batch_size}")
    print(f"   Gradient Accumulation: {profile.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {profile.batch_size * profile.gradient_accumulation_steps}")
    print(f"   Max Length: {profile.max_length}")
    print(f"   Estimated Throughput: {profile.estimated_throughput:.0f} tokens/sec")
    print(f"   Memory Usage: {profile.memory_usage_gb:.1f} GB")
    
    print(f"\n⏰ Time Estimate:")
    print(f"   Total Steps: {estimate.total_steps:,}")
    print(f"   Estimated Duration: {estimate.estimated_hours:.2f} hours")
    print(f"   Completion: {estimate.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Fits in {max_hours}h window: {'✅ Yes' if estimate.fits_in_window else '❌ No'}")
    
    if estimate.recommended_adjustments:
        print(f"\n⚠️ Recommendations:")
        for key, value in estimate.recommended_adjustments.items():
            print(f"   {key}: {value}")
    
    # Create training args
    training_args = optimizer.create_optimized_training_args(
        profile=profile,
        num_epochs=num_epochs
    )
    
    # Save report
    optimizer.save_optimization_report(
        filepath='training_optimization_report.json',
        profile_name=profile_name,
        profile=profile,
        estimate=estimate,
        dataset_info={
            'num_samples': num_samples,
            'avg_tokens_per_sample': avg_tokens_per_sample,
            'num_epochs': num_epochs
        }
    )
    
    return profile, estimate, training_args


if __name__ == "__main__":
    # Example usage
    print("🚀 H100 Training Optimizer")
    print("=" * 60)
    
    # Example dataset
    num_samples = 8000
    avg_tokens = 500
    num_epochs = 3
    
    profile, estimate, training_args = optimize_for_dataset(
        num_samples=num_samples,
        avg_tokens_per_sample=avg_tokens,
        num_epochs=num_epochs,
        priority='balanced',
        max_hours=12.0
    )
    
    print("\n✅ Optimization complete!")
    print(f"📁 Training args ready for use")
