"""
Reproducible training runner for Pixelated Empathy AI project.
Implements containerized, GPU-capable training with comprehensive logging and checkpointing.
"""

import os
import sys
import json
import torch
import wandb
import signal
import logging
import random
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import subprocess
import time
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
                         Trainer, TrainerCallback, EarlyStoppingCallback)
from datasets import Dataset
from torch.utils.data import DataLoader
import psutil
from tqdm import tqdm

# Import the training manifest from the previous task
from .training_manifest import TrainingManifest, Hyperparameters, DatasetReference


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResourceMonitorCallback(TrainerCallback):
    """Callback to monitor system resources during training"""
    
    def __init__(self):
        self.start_time = time.time()
        self.max_gpu_memory = 0
        self.max_system_memory = 0
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log resource usage during training"""
        if logs is None:
            return
            
        # Monitor system resources
        current_system_memory = psutil.virtual_memory().percent
        self.max_system_memory = max(self.max_system_memory, current_system_memory)
        
        logs['resources/system_memory_pct'] = current_system_memory
        logs['resources/max_system_memory_pct'] = self.max_system_memory
        logs['resources/elapsed_time_hours'] = (time.time() - self.start_time) / 3600
        
        # If CUDA is available, monitor GPU
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            gpu_memory_pct = current_gpu_memory / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
            
            self.max_gpu_memory = max(self.max_gpu_memory, current_gpu_memory)
            
            logs['resources/gpu_memory_gb'] = current_gpu_memory
            logs['resources/max_gpu_memory_gb'] = self.max_gpu_memory
            logs['resources/gpu_memory_pct'] = gpu_memory_pct
            logs['resources/gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            
        # Log to wandb if available
        if 'wandb' in sys.modules:
            try:
                import wandb
                if wandb.run:
                    wandb.log(logs, step=state.global_step)
            except:
                pass


class SafetyCallback(TrainerCallback):
    """Callback to implement safety checks during training"""
    
    def __init__(self, safety_metrics: Optional[Dict] = None):
        self.safety_metrics = safety_metrics or {}
        self.step_count = 0
        self.safety_violations = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Check for safety violations during training"""
        if logs is None:
            return
            
        self.step_count += 1
        
        # Check for unusual metrics that might indicate safety issues
        train_loss = logs.get('train_loss', float('inf'))
        learning_rate = logs.get('learning_rate', 0)
        
        # Check if loss is exploding (could indicate toxic learning)
        if train_loss > 1000:  # This is an arbitrary high value as example
            self.safety_violations += 1
            logger.warning(f"Safety alert: High training loss detected at step {state.global_step}: {train_loss}")
            logs['safety/loss_anomaly'] = True
            logs['safety/violation_count'] = self.safety_violations
        
        # Log safety metrics to wandb
        if 'wandb' in sys.modules:
            try:
                import wandb
                if wandb.run:
                    wandb.log(logs, step=state.global_step)
            except:
                pass


class CheckpointManager:
    """Manages training checkpoints with versioning and rollback capabilities"""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints with metadata"""
        checkpoints = []
        for checkpoint_dir in self.checkpoints_dir.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                metadata_path = checkpoint_dir / "trainer_state.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        checkpoints.append({
                            'path': str(checkpoint_dir),
                            'step': metadata.get('global_step', 0),
                            'timestamp': metadata.get('log_history', [{}])[-1].get('timestamp', 0) if metadata.get('log_history') else 0
                        })
                    except:
                        logger.warning(f"Could not read metadata for checkpoint: {checkpoint_dir}")
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x['step'], reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to manage disk space"""
        checkpoints = self.get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            # Keep only the most recent checkpoints
            for checkpoint in checkpoints[self.max_checkpoints:]:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    import shutil
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint"""
        checkpoints = self.get_checkpoints()
        if checkpoints:
            return checkpoints[0]['path']
        return None


class TrainingRunner:
    """Main training runner class that handles the entire training process"""
    
    def __init__(self, manifest: TrainingManifest):
        self.manifest = manifest
        self.checkpoint_manager = CheckpointManager(manifest.output_dir)
        self.logger = logger
        self.setup_reproducibility()
    
    def setup_reproducibility(self):
        """Set up reproducible training by fixing random seeds"""
        # Set random seeds for reproducibility
        seed = self.manifest.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Reproducibility setup complete with seed: {seed}")
    
    def setup_logging(self):
        """Set up logging and monitoring"""
        # Set up logging directory
        log_dir = Path(self.manifest.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up wandb logging if enabled
        if self.manifest.wandb_logging:
            try:
                import wandb
                # Configure wandb
                wandb.init(
                    project=self.manifest.wandb_project,
                    name=f"{self.manifest.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=self.manifest.wandb_tags,
                    config=self.manifest.hyperparameters.to_transformers_config()
                )
                
                # Log manifest information
                wandb.config.update({
                    'manifest_id': self.manifest.manifest_id,
                    'dataset_name': self.manifest.dataset.name,
                    'dataset_version': self.manifest.dataset.version,
                    'framework': self.manifest.framework.value,
                    'compute_target': self.manifest.compute_target.value,
                })
                
                logger.info("WandB logging initialized")
            except ImportError:
                logger.warning("WandB not installed, skipping wandb logging")
                self.manifest.wandb_logging = False
            except Exception as e:
                logger.warning(f"Could not initialize WandB: {e}")
                self.manifest.wandb_logging = False
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset"""
        logger.info(f"Loading dataset from: {self.manifest.dataset.path}")
        
        if not self.manifest.dataset.path or not os.path.exists(self.manifest.dataset.path):
            raise FileNotFoundError(f"Dataset file not found: {self.manifest.dataset.path}")
        
        # Load the dataset
        with open(self.manifest.dataset.path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract text data for training
        if isinstance(raw_data, dict) and 'conversations' in raw_data:
            conversations = raw_data['conversations']
        elif isinstance(raw_data, list):
            conversations = raw_data
        else:
            raise ValueError("Unexpected dataset format")
        
        # Extract text from conversations
        texts = []
        for conv in tqdm(conversations, desc="Processing conversations"):
            if isinstance(conv, dict):
                # Handle different conversation formats
                if 'text' in conv:
                    texts.append(conv['text'])
                elif 'conversation' in conv:
                    # If it's a conversation object, join all messages
                    if isinstance(conv['conversation'], list):
                        text = " ".join([msg.get('content', '') if isinstance(msg, dict) else str(msg) 
                                       for msg in conv['conversation']])
                        texts.append(text)
                elif 'messages' in conv:
                    # If it has messages, join them
                    if isinstance(conv['messages'], list):
                        text = " ".join([msg.get('content', '') if isinstance(msg, dict) else str(msg) 
                                       for msg in conv['messages']])
                        texts.append(text)
            elif isinstance(conv, str):
                texts.append(conv)
        
        # Create HuggingFace dataset
        dataset = Dataset.from_dict({"text": texts})
        logger.info(f"Loaded {len(dataset)} samples from dataset")
        
        # Log dataset statistics
        if self.manifest.wandb_logging:
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        'dataset/total_samples': len(dataset),
                        'dataset/avg_text_length': np.mean([len(text.split()) for text in texts]),
                        'dataset/max_text_length': max([len(text.split()) for text in texts]),
                        'dataset/min_text_length': min([len(text.split()) for text in texts]) if texts else 0
                    })
            except:
                pass
        
        return dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for training"""
        logger.info("Setting up model and tokenizer")
        
        # Determine model name based on framework
        model_name = self.manifest.model_name or "microsoft/DialoGPT-medium"
        
        device_available = torch.cuda.is_available()
        if device_available:
            logger.info("Using GPU for training")
            # Load model with appropriate precision
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.manifest.hyperparameters.bf16 else torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                load_in_8bit=False  # Adjust based on availability
            )
        else:
            logger.info("Using CPU for training (testing mode)")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded: {self.model.__class__.__name__}")
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training"""
        logger.info("Tokenizing dataset")
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.manifest.hyperparameters.max_seq_length,
                return_tensors=None  # Return as lists for datasets
            )
            # Set labels equal to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"] if "text" in dataset.column_names else []
        )
        
        logger.info(f"Tokenized dataset has {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments from manifest hyperparameters"""
        # Create output directory
        output_dir = Path(self.manifest.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get hyperparameters as dict
        hp_dict = self.manifest.hyperparameters.to_transformers_config()
        
        # Override specific values that need special handling
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=self.manifest.evaluation_enabled,
            save_strategy="steps",
            save_steps=self.manifest.hyperparameters.save_steps,
            logging_dir=str(Path(self.manifest.log_dir)),
            logging_steps=self.manifest.hyperparameters.logging_steps,
            evaluation_strategy="steps" if self.manifest.evaluation_enabled else "no",
            eval_steps=self.manifest.hyperparameters.eval_steps,
            load_best_model_at_end=self.manifest.evaluation_enabled,
            metric_for_best_model="eval_loss" if self.manifest.evaluation_enabled else "train_loss",
            greater_is_better=False,
            save_total_limit=self.manifest.hyperparameters.save_total_limit,
            seed=self.manifest.seed,
            data_seed=self.manifest.seed,
            **hp_dict
        )
        
        logger.info(f"Training arguments configured: {training_args.to_dict()}")
        return training_args
    
    def run_training(self):
        """Execute the full training process"""
        logger.info("Starting training process...")
        
        # Setup logging
        self.setup_logging()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Check if dataset is empty
        if len(tokenized_dataset) == 0:
            raise ValueError("Tokenized dataset is empty!")
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Setup callbacks
        callbacks = [
            ResourceMonitorCallback(),
            SafetyCallback(safety_metrics=self.manifest.safety_metrics.__dict__)
        ]
        
        # Add early stopping if evaluation is enabled
        if self.manifest.evaluation_enabled:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # Log training start
        logger.info("Starting training...")
        
        if self.manifest.wandb_logging:
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        'training/start_time': datetime.utcnow().isoformat(),
                        'training/status': 'started',
                        'training/total_parameters': sum(p.numel() for p in self.model.parameters()),
                        'training/trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    })
            except:
                pass
        
        # Train the model
        train_result = self.trainer.train()
        
        # Log training results
        logger.info("Training completed!")
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Model saved to: {training_args.output_dir}")
        
        # Log final metrics
        if self.manifest.wandb_logging:
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        'training/end_time': datetime.utcnow().isoformat(),
                        'training/status': 'completed',
                        'training/final_train_loss': train_result.training_loss,
                        'training/total_steps': train_result.global_step,
                        'training/runtime_hours': train_result.metrics.get('train_runtime', 0) / 3600
                    })
            except:
                pass
        
        # Finish wandb run
        if self.manifest.wandb_logging:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        return train_result
    
    def resume_training(self, checkpoint_path: Optional[str] = None):
        """Resume training from a checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.warning("No checkpoint found, starting fresh training")
            return self.run_training()
        
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Setup everything as in run_training
        self.setup_logging()
        dataset = self.load_dataset()
        self.setup_model_and_tokenizer()
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        if len(tokenized_dataset) == 0:
            raise ValueError("Tokenized dataset is empty!")
        
        training_args = self.setup_training_arguments()
        
        # Setup callbacks
        callbacks = [
            ResourceMonitorCallback(),
            SafetyCallback(safety_metrics=self.manifest.safety_metrics.__dict__)
        ]
        
        if self.manifest.evaluation_enabled:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Create trainer with checkpoint
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # Resume training
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        
        logger.info("Training resumed and completed!")
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        return train_result


class ContainerizedTrainingRunner(TrainingRunner):
    """Extended training runner with containerization support"""
    
    def __init__(self, manifest: TrainingManifest):
        super().__init__(manifest)
        self.dockerfile_content = self._generate_dockerfile()
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for containerized training"""
        return f"""
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install transformers datasets torch wandb tqdm psutil

# Copy training code
COPY . /workspace

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Create output directories
RUN mkdir -p /workspace/model_output /workspace/logs

# Training command
CMD ["python", "-m", "ai.pipelines.orchestrator.training_runner", "--manifest", "/workspace/config/manifest.json"]
"""
    
    def build_container(self, image_name: str = "pixelated-empathy-trainer:latest"):
        """Build the training container"""
        # Save Dockerfile
        with open("Dockerfile.training", "w") as f:
            f.write(self.dockerfile_content)
        
        # Build the image
        try:
            result = subprocess.run([
                "docker", "build", "-t", image_name, "-f", "Dockerfile.training", "."
            ], check=True, capture_output=True, text=True)
            logger.info(f"Container built successfully: {image_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build container: {e}")
            return False
    
    def run_in_container(self, image_name: str = "pixelated-empathy-trainer:latest", 
                        gpu_enabled: bool = True) -> bool:
        """Run training inside a container"""
        # First build the container
        if not self.build_container(image_name):
            return False
        
        # Prepare container run command
        cmd = ["docker", "run", "--rm"]
        
        # Add GPU support if available
        if gpu_enabled and torch.cuda.is_available():
            cmd.extend(["--gpus", "all"])
        
        # Mount necessary volumes
        cmd.extend([
            "-v", f"{os.getcwd()}:/workspace/data",
            "-v", f"{self.manifest.output_dir}:/workspace/model_output",
            "-v", f"{self.manifest.log_dir}:/workspace/logs",
            "-e", f"WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}",
            image_name
        ])
        
        try:
            logger.info(f"Running training in container: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Containerized training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Containerized training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False


def create_training_runner_from_manifest(manifest_path: str) -> TrainingRunner:
    """Create a training runner from a manifest file"""
    manifest = TrainingManifest.load_from_file(manifest_path)
    return TrainingRunner(manifest)


def run_training_from_manifest(manifest_path: str, use_container: bool = False):
    """Run training directly from a manifest file"""
    runner = create_training_runner_from_manifest(manifest_path)
    
    if use_container:
        if isinstance(runner, ContainerizedTrainingRunner):
            return runner.run_in_container()
        else:
            # Convert to containerized runner
            container_runner = ContainerizedTrainingRunner(runner.manifest)
            return container_runner.run_in_container()
    else:
        return runner.run_training()


# Example usage and testing
def test_training_runner():
    """Test the training runner functionality"""
    logger.info("Testing Training Runner...")
    
    # Create a sample manifest for testing
    from .training_manifest import create_default_manifest
    
    # Create a small test dataset
    test_dataset = {
        "conversations": [
            {"text": "Hello, how are you today?"},
            {"text": "I'm feeling anxious about my therapy session."},
            {"text": "That sounds challenging. What specifically are you worried about?"},
            {"text": "I'm afraid I won't be able to open up."},
            {"text": "It's normal to feel that way. What would help you feel more comfortable?"}
        ]
    }
    
    # Write test dataset
    with open("test_training_dataset.json", "w") as f:
        json.dump(test_dataset, f, indent=2)
    
    # Create manifest
    manifest = create_default_manifest("test_training_dataset.json", "test_v1.0")
    manifest.hyperparameters.num_train_epochs = 1  # Fast test run
    manifest.hyperparameters.per_device_train_batch_size = 1  # Small batch for test
    manifest.hyperparameters.gradient_accumulation_steps = 1  # No accumulation for test
    manifest.hyperparameters.logging_steps = 1  # Log every step for test
    
    # Create and run training
    runner = TrainingRunner(manifest)
    
    # Validate manifest before running
    errors = manifest.validate()
    if errors:
        logger.error(f"Manifest validation failed: {errors}")
        return False
    
    logger.info("Training runner created successfully")
    logger.info(f"Model output directory: {manifest.output_dir}")
    logger.info(f"Log directory: {manifest.log_dir}")
    logger.info(f"Dataset path: {manifest.dataset.path}")
    
    # Get available checkpoints
    checkpoints = runner.checkpoint_manager.get_checkpoints()
    logger.info(f"Available checkpoints: {len(checkpoints)}")
    
    # Cleanup test files
    import os
    if os.path.exists("test_training_dataset.json"):
        os.remove("test_training_dataset.json")
    
    logger.info("Training runner test completed!")
    return True


if __name__ == "__main__":
    test_training_runner()