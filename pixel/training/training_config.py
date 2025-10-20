"""
Pixel LLM Training Configuration
Handles GPU/compute resource setup and training configuration
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class ComputeConfig:
    """GPU and compute resource configuration"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # For distributed training
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 512
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 32000
    use_cache: bool = False  # Disable for training


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: str = "./model_outputs"
    log_dir: str = "./logs"
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    report_to: List[str] = None  # ["wandb", "tensorboard"]
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]


class TrainingConfigManager:
    """Manages training configuration and compute setup"""
    
    def __init__(self):
        self.compute_config = ComputeConfig()
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        self.output_config = OutputConfig()
        
        self._setup_compute()
    
    def _setup_compute(self):
        """Setup compute resources"""
        logger.info("=" * 80)
        logger.info("TIER 1.2: Compute Resource Configuration")
        logger.info("=" * 80)
        
        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if self.compute_config.num_gpus > 0:
            logger.info(f"Number of GPUs: {self.compute_config.num_gpus}")
            for i in range(self.compute_config.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(f"    - Memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"    - Compute Capability: {props.major}.{props.minor}")
        else:
            logger.warning("No GPUs available. Training will be slow on CPU.")
        
        # Setup distributed training if needed
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.compute_config.distributed = True
            self.compute_config.rank = int(os.environ["RANK"])
            self.compute_config.world_size = int(os.environ["WORLD_SIZE"])
            logger.info(f"Distributed training: rank={self.compute_config.rank}, world_size={self.compute_config.world_size}")
        
        # Log compute config
        logger.info("\nCompute Configuration:")
        for key, value in asdict(self.compute_config).items():
            logger.info(f"  {key}: {value}")
    
    def setup_distributed(self):
        """Initialize distributed training"""
        if not self.compute_config.distributed:
            logger.info("Distributed training not enabled")
            return
        
        logger.info("Initializing distributed training...")
        dist.init_process_group(backend=self.compute_config.backend)
        logger.info(f"Distributed training initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    
    def get_device(self) -> torch.device:
        """Get torch device"""
        return torch.device(self.compute_config.device)
    
    def get_training_args_dict(self) -> Dict:
        """Get training arguments as dictionary"""
        args = {
            **asdict(self.training_config),
            **asdict(self.output_config),
            "device": self.compute_config.device,
            "num_gpus": self.compute_config.num_gpus,
            "distributed": self.compute_config.distributed,
            "mixed_precision": self.compute_config.mixed_precision,
            "gradient_checkpointing": self.compute_config.gradient_checkpointing,
        }
        return args
    
    def save_config(self, path: str):
        """Save configuration to JSON"""
        config_dict = {
            "compute": asdict(self.compute_config),
            "training": asdict(self.training_config),
            "model": asdict(self.model_config),
            "output": asdict(self.output_config),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    def load_config(self, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        self.compute_config = ComputeConfig(**config_dict.get("compute", {}))
        self.training_config = TrainingConfig(**config_dict.get("training", {}))
        self.model_config = ModelConfig(**config_dict.get("model", {}))
        self.output_config = OutputConfig(**config_dict.get("output", {}))
        
        logger.info(f"Configuration loaded from {path}")


def create_training_config(
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    output_dir: str = "./model_outputs",
    **kwargs
) -> TrainingConfigManager:
    """
    Convenience function to create training configuration
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size per device
        output_dir: Output directory for models and logs
        **kwargs: Additional configuration parameters
    
    Returns:
        TrainingConfigManager instance
    """
    manager = TrainingConfigManager()
    manager.training_config.num_epochs = num_epochs
    manager.training_config.learning_rate = learning_rate
    manager.training_config.per_device_train_batch_size = batch_size
    manager.output_config.output_dir = output_dir
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(manager.training_config, key):
            setattr(manager.training_config, key, value)
        elif hasattr(manager.compute_config, key):
            setattr(manager.compute_config, key, value)
    
    return manager

