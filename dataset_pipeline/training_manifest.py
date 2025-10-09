"""
Training manifest system for the Pixelated Empathy AI project.
Defines structured manifests for reproducible training runs with all required metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid
from enum import Enum
import os
import hashlib
from pathlib import Path


class ComputeTarget(Enum):
    """Types of compute targets for training"""
    CPU = "cpu"
    GPU_SINGLE = "gpu_single"
    GPU_MULTI = "gpu_multi"
    TPU = "tpu"
    CLOUD_GPU = "cloud_gpu"
    LOCAL_GPU = "local_gpu"


class TrainingFramework(Enum):
    """Supported training frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    TRANSFORMERS = "transformers"
    LIGHTNING = "lightning"
    HUGGINGFACE = "huggingface"


@dataclass
class DatasetReference:
    """Reference to a specific dataset version"""
    name: str
    version: str
    commit_hash: Optional[str] = None
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    record_count: Optional[int] = None
    created_at: Optional[str] = None
    checksum: Optional[str] = None  # SHA256 hash of dataset
    
    def __post_init__(self):
        if self.checksum is None and self.path:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of dataset file"""
        if self.path and os.path.exists(self.path):
            hash_sha256 = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        return ""


@dataclass
class Hyperparameters:
    """Training hyperparameters configuration"""
    # Core training parameters
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Optimizer parameters
    optimizer: str = "adamw"  # adamw, sgd, etc.
    lr_scheduler_type: str = "linear"  # linear, cosine, constant, etc.
    
    # Model-specific parameters
    max_seq_length: int = 512
    gradient_checkpointing: bool = True
    bf16: bool = True  # Use bfloat16 precision
    fp16: bool = False  # Use float16 precision
    
    # Advanced training parameters
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    def to_transformers_config(self) -> Dict[str, Any]:
        """Convert to transformers TrainingArguments format"""
        return {
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "optim": f"{self.optimizer}_8bit" if "adamw" in self.optimizer else self.optimizer,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_seq_length": self.max_seq_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
        }


@dataclass
class SafetyMetrics:
    """Safety-related metrics configuration"""
    max_crisis_content_ratio: float = 0.15  # Max percentage of crisis content
    demographic_balance_threshold: float = 0.1  # Max imbalance in demographics
    toxicity_threshold: float = 0.05  # Max toxicity score allowed
    therapeutic_response_balance: float = 0.15  # Max imbalance in response types
    privacy_preservation_enabled: bool = True
    bias_detection_enabled: bool = True


@dataclass
class ResourceRequirements:
    """Resource requirements for training"""
    min_gpu_memory_gb: float = 8.0
    min_system_memory_gb: float = 16.0
    expected_runtime_hours: float = 24.0
    max_budget_usd: Optional[float] = None
    cloud_provider: Optional[str] = None  # aws, gcp, azure
    region: Optional[str] = None
    instance_type: Optional[str] = None


@dataclass
class TrainingManifest:
    """Complete training manifest with all required information"""
    # Basic identification
    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default_training"
    description: str = "Default training manifest for Pixelated Empathy AI"
    
    # Dataset reference
    dataset: DatasetReference = field(default_factory=DatasetReference)
    
    # Hyperparameters
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    
    # Framework and compute
    framework: TrainingFramework = TrainingFramework.TRANSFORMERS
    compute_target: ComputeTarget = ComputeTarget.GPU_SINGLE
    seed: int = 42
    
    # Safety metrics
    safety_metrics: SafetyMetrics = field(default_factory=SafetyMetrics)
    
    # Resource requirements
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Versioning and tracking
    manifest_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = field(default_factory=lambda: os.environ.get("USER", "system"))
    
    # Output configuration
    output_dir: str = "./model_output"
    model_name: Optional[str] = None  # Name to save the trained model
    log_dir: str = "./logs"
    
    # Advanced configuration
    evaluation_enabled: bool = True
    checkpointing_enabled: bool = True
    wandb_logging: bool = True
    wandb_project: str = "pixelated-empathy-ai"
    wandb_tags: List[str] = field(default_factory=lambda: ["training", "experiment"])
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert manifest to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary format"""
        return {
            "manifest_id": self.manifest_id,
            "name": self.name,
            "description": self.description,
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
                "commit_hash": self.dataset.commit_hash,
                "path": self.dataset.path,
                "size_bytes": self.dataset.size_bytes,
                "record_count": self.dataset.record_count,
                "created_at": self.dataset.created_at,
                "checksum": self.dataset.checksum
            },
            "hyperparameters": self.hyperparameters.to_transformers_config(),
            "framework": self.framework.value,
            "compute_target": self.compute_target.value,
            "seed": self.seed,
            "safety_metrics": {
                "max_crisis_content_ratio": self.safety_metrics.max_crisis_content_ratio,
                "demographic_balance_threshold": self.safety_metrics.demographic_balance_threshold,
                "toxicity_threshold": self.safety_metrics.toxicity_threshold,
                "therapeutic_response_balance": self.safety_metrics.therapeutic_response_balance,
                "privacy_preservation_enabled": self.safety_metrics.privacy_preservation_enabled,
                "bias_detection_enabled": self.safety_metrics.bias_detection_enabled
            },
            "resources": {
                "min_gpu_memory_gb": self.resources.min_gpu_memory_gb,
                "min_system_memory_gb": self.resources.min_system_memory_gb,
                "expected_runtime_hours": self.resources.expected_runtime_hours,
                "max_budget_usd": self.resources.max_budget_usd,
                "cloud_provider": self.resources.cloud_provider,
                "region": self.resources.region,
                "instance_type": self.resources.instance_type
            },
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "output_dir": self.output_dir,
            "model_name": self.model_name,
            "log_dir": self.log_dir,
            "evaluation_enabled": self.evaluation_enabled,
            "checkpointing_enabled": self.checkpointing_enabled,
            "wandb_logging": self.wandb_logging,
            "wandb_project": self.wandb_project,
            "wandb_tags": self.wandb_tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingManifest':
        """Create manifest from dictionary"""
        # Reconstruct DatasetReference
        dataset_data = data.get('dataset', {})
        dataset = DatasetReference(
            name=dataset_data.get('name', ''),
            version=dataset_data.get('version', ''),
            commit_hash=dataset_data.get('commit_hash'),
            path=dataset_data.get('path'),
            size_bytes=dataset_data.get('size_bytes'),
            record_count=dataset_data.get('record_count'),
            created_at=dataset_data.get('created_at'),
            checksum=dataset_data.get('checksum')
        )
        
        # Reconstruct Hyperparameters
        hp_data = data.get('hyperparameters', {})
        hyperparameters = Hyperparameters(
            num_train_epochs=hp_data.get('num_train_epochs', 3),
            learning_rate=hp_data.get('learning_rate', 2e-5),
            per_device_train_batch_size=hp_data.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=hp_data.get('per_device_eval_batch_size', 8),
            gradient_accumulation_steps=hp_data.get('gradient_accumulation_steps', 1),
            max_grad_norm=hp_data.get('max_grad_norm', 1.0),
            weight_decay=hp_data.get('weight_decay', 0.01),
            warmup_steps=hp_data.get('warmup_steps', 500),
            optimizer=hp_data.get('optim', 'adamw'),
            lr_scheduler_type=hp_data.get('lr_scheduler_type', 'linear'),
            max_seq_length=hp_data.get('max_seq_length', 512),
            gradient_checkpointing=hp_data.get('gradient_checkpointing', True),
            bf16=hp_data.get('bf16', True),
            fp16=hp_data.get('fp16', False),
            save_steps=hp_data.get('save_steps', 500),
            logging_steps=hp_data.get('logging_steps', 10),
            eval_steps=hp_data.get('eval_steps'),
            save_total_limit=hp_data.get('save_total_limit', 2),
            dataloader_num_workers=hp_data.get('dataloader_num_workers', 0),
            dataloader_pin_memory=hp_data.get('dataloader_pin_memory', True)
        )
        
        # Reconstruct SafetyMetrics
        safety_data = data.get('safety_metrics', {})
        safety_metrics = SafetyMetrics(
            max_crisis_content_ratio=safety_data.get('max_crisis_content_ratio', 0.15),
            demographic_balance_threshold=safety_data.get('demographic_balance_threshold', 0.1),
            toxicity_threshold=safety_data.get('toxicity_threshold', 0.05),
            therapeutic_response_balance=safety_data.get('therapeutic_response_balance', 0.15),
            privacy_preservation_enabled=safety_data.get('privacy_preservation_enabled', True),
            bias_detection_enabled=safety_data.get('bias_detection_enabled', True)
        )
        
        # Reconstruct ResourceRequirements
        resources_data = data.get('resources', {})
        resources = ResourceRequirements(
            min_gpu_memory_gb=resources_data.get('min_gpu_memory_gb', 8.0),
            min_system_memory_gb=resources_data.get('min_system_memory_gb', 16.0),
            expected_runtime_hours=resources_data.get('expected_runtime_hours', 24.0),
            max_budget_usd=resources_data.get('max_budget_usd'),
            cloud_provider=resources_data.get('cloud_provider'),
            region=resources_data.get('region'),
            instance_type=resources_data.get('instance_type')
        )
        
        # Create the manifest
        return cls(
            manifest_id=data.get('manifest_id', str(uuid.uuid4())),
            name=data.get('name', 'default_training'),
            description=data.get('description', 'Default training manifest'),
            dataset=dataset,
            hyperparameters=hyperparameters,
            framework=TrainingFramework(data.get('framework', 'transformers')),
            compute_target=ComputeTarget(data.get('compute_target', 'gpu_single')),
            seed=data.get('seed', 42),
            safety_metrics=safety_metrics,
            resources=resources,
            manifest_version=data.get('manifest_version', '1.0'),
            created_at=data.get('created_at', datetime.utcnow().isoformat()),
            created_by=data.get('created_by', os.environ.get('USER', 'system')),
            output_dir=data.get('output_dir', './model_output'),
            model_name=data.get('model_name'),
            log_dir=data.get('log_dir', './logs'),
            evaluation_enabled=data.get('evaluation_enabled', True),
            checkpointing_enabled=data.get('checkpointing_enabled', True),
            wandb_logging=data.get('wandb_logging', True),
            wandb_project=data.get('wandb_project', 'pixelated-empathy-ai'),
            wandb_tags=data.get('wandb_tags', ['training', 'experiment']),
            metadata=data.get('metadata', {})
        )
    
    def save_to_file(self, filepath: str):
        """Save manifest to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingManifest':
        """Load manifest from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate the manifest and return list of validation errors"""
        errors = []
        
        # Validate dataset
        if not self.dataset.name:
            errors.append("Dataset name is required")
        if not self.dataset.version:
            errors.append("Dataset version is required")
        
        # Validate compute target
        if self.compute_target == ComputeTarget.CLOUD_GPU:
            if not self.resources.cloud_provider:
                errors.append("Cloud provider required for cloud GPU target")
            if not self.resources.region:
                errors.append("Region required for cloud GPU target")
            if not self.resources.instance_type:
                errors.append("Instance type required for cloud GPU target")
        
        # Validate hyperparameters
        if self.hyperparameters.num_train_epochs <= 0:
            errors.append("num_train_epochs must be positive")
        if self.hyperparameters.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.hyperparameters.per_device_train_batch_size <= 0:
            errors.append("per_device_train_batch_size must be positive")
        
        # Validate safety metrics
        if self.safety_metrics.max_crisis_content_ratio < 0 or self.safety_metrics.max_crisis_content_ratio > 1:
            errors.append("max_crisis_content_ratio must be between 0 and 1")
        if self.safety_metrics.toxicity_threshold < 0 or self.safety_metrics.toxicity_threshold > 1:
            errors.append("toxicity_threshold must be between 0 and 1")
        
        # Validate resources
        if self.resources.min_gpu_memory_gb <= 0:
            errors.append("min_gpu_memory_gb must be positive")
        if self.resources.min_system_memory_gb <= 0:
            errors.append("min_system_memory_gb must be positive")
        if self.resources.expected_runtime_hours <= 0:
            errors.append("expected_runtime_hours must be positive")
        if self.resources.max_budget_usd is not None and self.resources.max_budget_usd <= 0:
            errors.append("max_budget_usd must be positive")
        
        return errors


def create_default_manifest(dataset_path: str = "./training_dataset.json", 
                          dataset_version: str = "1.0") -> TrainingManifest:
    """Create a default training manifest with sensible defaults"""
    dataset = DatasetReference(
        name="therapeutic_conversations_dataset",
        version=dataset_version,
        path=dataset_path,
        created_at=datetime.utcnow().isoformat()
    )
    
    # Calculate size if file exists
    if os.path.exists(dataset_path):
        dataset.size_bytes = os.path.getsize(dataset_path)
    
    return TrainingManifest(
        name="default_therapy_model_training",
        description="Default training manifest for therapeutic conversation model",
        dataset=dataset,
        hyperparameters=Hyperparameters(
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            weight_decay=0.01,
            warmup_steps=500,
            bf16=True,
            gradient_checkpointing=True,
            save_steps=100,
            logging_steps=10
        ),
        compute_target=ComputeTarget.GPU_SINGLE,
        seed=42,
        output_dir="./therapy_model_output",
        log_dir="./therapy_model_logs"
    )


def create_safety_aware_manifest(dataset_path: str, 
                               dataset_version: str = "1.0") -> TrainingManifest:
    """Create a safety-focused training manifest with enhanced safety metrics"""
    manifest = create_default_manifest(dataset_path, dataset_version)
    
    # Enhance safety metrics
    manifest.safety_metrics.max_crisis_content_ratio = 0.1  # Lower for safety
    manifest.safety_metrics.demographic_balance_threshold = 0.05  # Stricter balance
    manifest.safety_metrics.toxicity_threshold = 0.02  # Stricter toxicity
    manifest.safety_metrics.therapeutic_response_balance = 0.1  # Stricter balance
    manifest.safety_metrics.privacy_preservation_enabled = True
    manifest.safety_metrics.bias_detection_enabled = True
    
    # Additional safety-related training parameters
    manifest.name = "safety_aware_therapy_model_training"
    manifest.description = "Safety-focused training manifest for therapeutic conversation model with enhanced safety checks"
    
    # Modify hyperparameters for better safety
    manifest.hyperparameters.learning_rate = 1e-5  # Lower for more stable training
    manifest.hyperparameters.gradient_accumulation_steps = 16  # Higher for stability
    manifest.hyperparameters.logging_steps = 5  # More frequent logging for monitoring
    
    return manifest


# Example usage and testing
def test_training_manifest():
    """Test the training manifest functionality"""
    print("Testing Training Manifest System...")
    
    # Create a default manifest
    manifest = create_default_manifest("./sample_dataset.json", "v2.1")
    print(f"Created manifest: {manifest.name}")
    print(f"Manifest ID: {manifest.manifest_id}")
    
    # Validate the manifest
    errors = manifest.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Manifest validation passed!")
    
    # Convert to dict and back
    manifest_dict = manifest.to_dict()
    restored_manifest = TrainingManifest.from_dict(manifest_dict)
    
    print(f"Restored manifest name: {restored_manifest.name}")
    print(f"Dataset name: {restored_manifest.dataset.name}")
    print(f"Learning rate: {restored_manifest.hyperparameters.learning_rate}")
    
    # Test safety-aware manifest
    safety_manifest = create_safety_aware_manifest("./safety_dataset.json", "v1.2")
    print(f"\nSafety manifest: {safety_manifest.name}")
    print(f"Max crisis content ratio: {safety_manifest.safety_metrics.max_crisis_content_ratio}")
    print(f"Lower learning rate: {safety_manifest.hyperparameters.learning_rate}")
    
    # Save to file
    manifest.save_to_file("test_training_manifest.json")
    print("\nManifest saved to test_training_manifest.json")
    
    # Load from file
    loaded_manifest = TrainingManifest.load_from_file("test_training_manifest.json")
    print(f"Loaded manifest name: {loaded_manifest.name}")
    
    # Test validation with errors
    bad_manifest = TrainingManifest(dataset=DatasetReference(name="", version=""))
    validation_errors = bad_manifest.validate()
    print(f"Expected validation errors: {len(validation_errors)}")
    for error in validation_errors:
        print(f"  - {error}")


if __name__ == "__main__":
    test_training_manifest()