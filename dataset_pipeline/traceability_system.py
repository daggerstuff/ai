"""
Dataset-to-training traceability system for Pixelated Empathy AI project.
Tracks which dataset versions were used to produce specific model artifacts.
"""

import json
import os
import hashlib
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import git
from .training_manifest import TrainingManifest, DatasetReference


logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """Represents a specific version of a dataset"""
    dataset_id: str
    name: str
    version: str
    commit_hash: Optional[str] = None
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    record_count: Optional[int] = None
    created_at: Optional[str] = None
    checksum: Optional[str] = None  # SHA256 hash of dataset
    features: Optional[Dict[str, Any]] = None  # Dataset features/schema
    statistics: Optional[Dict[str, Any]] = None  # Dataset statistics
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingRun:
    """Represents a single training run"""
    run_id: str
    model_id: str
    model_version: str
    manifest_id: str
    dataset_version_id: str
    dataset_commit_hash: Optional[str] = None  # Git commit hash of dataset used
    dataset_version_tag: Optional[str] = None   # Version tag of dataset used
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, failed, cancelled
    artifacts_dir: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    git_commit: Optional[str] = None
    environment_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelArtifact:
    """Represents a trained model artifact with traceability info"""
    artifact_id: str
    model_id: str
    model_version: str
    training_run_id: str
    dataset_version_id: str
    dataset_commit_hash: Optional[str] = None  # Git commit of dataset used
    dataset_version_tag: Optional[str] = None   # Tag of dataset version used
    training_manifest_hash: Optional[str] = None  # Hash of training manifest
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    path: str = ""
    size_bytes: Optional[int] = None
    framework: Optional[str] = None
    format: Optional[str] = None
    evaluation_score: Optional[float] = None
    promotion_status: Optional[str] = None  # training, staging, production, rejected
    training_duration_seconds: Optional[float] = None  # How long training took
    resource_usage: Optional[Dict[str, Any]] = None  # CPU, GPU, memory usage during training
    git_repository_url: Optional[str] = None  # URL of git repository
    git_commit_hash: Optional[str] = None  # Commit hash of training code
    environment_variables: Optional[Dict[str, str]] = None  # Environment variables used
    dependencies: Optional[Dict[str, str]] = None  # Package dependencies and versions
    hardware_specs: Optional[Dict[str, Any]] = None  # Hardware specs used for training
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TraceabilityRecord:
    """Complete traceability record linking dataset to model"""
    record_id: str
    dataset_version: DatasetVersion
    training_run: TrainingRun
    model_artifact: ModelArtifact
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class DatasetRegistry:
    """Registry for tracking dataset versions"""
    
    def __init__(self, registry_path: str = "./dataset_registry.json"):
        self.registry_path = Path(registry_path)
        self.datasets: Dict[str, DatasetVersion] = self._load_registry()
        self.logger = logger
    
    def _load_registry(self) -> Dict[str, DatasetVersion]:
        """Load dataset registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                return {k: self._dict_to_dataset_version(v) for k, v in data.items()}
            except Exception as e:
                self.logger.warning(f"Could not load dataset registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save dataset registry to file"""
        data = {k: self._dataset_version_to_dict(v) for k, v in self.datasets.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _dict_to_dataset_version(self, data: Dict[str, Any]) -> DatasetVersion:
        """Convert dictionary to DatasetVersion"""
        return DatasetVersion(
            dataset_id=data['dataset_id'],
            name=data['name'],
            version=data['version'],
            commit_hash=data.get('commit_hash'),
            path=data.get('path'),
            size_bytes=data.get('size_bytes'),
            record_count=data.get('record_count'),
            created_at=data.get('created_at'),
            checksum=data.get('checksum'),
            features=data.get('features'),
            statistics=data.get('statistics'),
            metadata=data.get('metadata')
        )
    
    def _dataset_version_to_dict(self, dataset: DatasetVersion) -> Dict[str, Any]:
        """Convert DatasetVersion to dictionary"""
        return {
            'dataset_id': dataset.dataset_id,
            'name': dataset.name,
            'version': dataset.version,
            'commit_hash': dataset.commit_hash,
            'path': dataset.path,
            'size_bytes': dataset.size_bytes,
            'record_count': dataset.record_count,
            'created_at': dataset.created_at,
            'checksum': dataset.checksum,
            'features': dataset.features,
            'statistics': dataset.statistics,
            'metadata': dataset.metadata
        }
    
    def register_dataset(self, dataset_ref: DatasetReference) -> DatasetVersion:
        """Register a new dataset version"""
        # Calculate checksum if not provided
        checksum = dataset_ref.checksum
        if not checksum and dataset_ref.path and os.path.exists(dataset_ref.path):
            checksum = self._calculate_checksum(dataset_ref.path)
        
        # Get size if path provided
        size = dataset_ref.size_bytes
        if not size and dataset_ref.path and os.path.exists(dataset_ref.path):
            size = os.path.getsize(dataset_ref.path)
        
        dataset_version = DatasetVersion(
            dataset_id=f"{dataset_ref.name}_{dataset_ref.version}",
            name=dataset_ref.name,
            version=dataset_ref.version,
            commit_hash=dataset_ref.commit_hash,
            path=dataset_ref.path,
            size_bytes=size,
            record_count=dataset_ref.record_count,
            created_at=dataset_ref.created_at or datetime.utcnow().isoformat(),
            checksum=checksum
        )
        
        self.datasets[dataset_version.dataset_id] = dataset_version
        self._save_registry()
        
        self.logger.info(f"Registered dataset version: {dataset_version.dataset_id}")
        return dataset_version
    
    def get_dataset_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        """Get a specific dataset version"""
        return self.datasets.get(dataset_id)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class TrainingRunRegistry:
    """Registry for tracking training runs"""
    
    def __init__(self, registry_path: str = "./training_runs_registry.json"):
        self.registry_path = Path(registry_path)
        self.training_runs: Dict[str, TrainingRun] = self._load_registry()
        self.logger = logger
    
    def _load_registry(self) -> Dict[str, TrainingRun]:
        """Load training runs registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                return {k: self._dict_to_training_run(v) for k, v in data.items()}
            except Exception as e:
                self.logger.warning(f"Could not load training runs registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save training runs registry to file"""
        data = {k: self._training_run_to_dict(v) for k, v in self.training_runs.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _dict_to_training_run(self, data: Dict[str, Any]) -> TrainingRun:
        """Convert dictionary to TrainingRun"""
        return TrainingRun(
            run_id=data['run_id'],
            model_id=data['model_id'],
            model_version=data['model_version'],
            manifest_id=data['manifest_id'],
            dataset_version_id=data['dataset_version_id'],
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            status=data.get('status', 'running'),
            artifacts_dir=data.get('artifacts_dir'),
            metrics=data.get('metrics'),
            hyperparameters=data.get('hyperparameters'),
            git_commit=data.get('git_commit'),
            environment_info=data.get('environment_info'),
            metadata=data.get('metadata')
        )
    
    def _training_run_to_dict(self, run: TrainingRun) -> Dict[str, Any]:
        """Convert TrainingRun to dictionary"""
        return {
            'run_id': run.run_id,
            'model_id': run.model_id,
            'model_version': run.model_version,
            'manifest_id': run.manifest_id,
            'dataset_version_id': run.dataset_version_id,
            'start_time': run.start_time,
            'end_time': run.end_time,
            'status': run.status,
            'artifacts_dir': run.artifacts_dir,
            'metrics': run.metrics,
            'hyperparameters': run.hyperparameters,
            'git_commit': run.git_commit,
            'environment_info': run.environment_info,
            'metadata': run.metadata
        }
    
    def register_training_run(self, run: TrainingRun) -> str:
        """Register a new training run"""
        self.training_runs[run.run_id] = run
        self._save_registry()
        self.logger.info(f"Registered training run: {run.run_id}")
        return run.run_id
    
    def update_training_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing training run"""
        if run_id in self.training_runs:
            run = self.training_runs[run_id]
            for key, value in updates.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            self._save_registry()
            return True
        return False
    
    def get_training_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a specific training run"""
        return self.training_runs.get(run_id)


class ModelArtifactRegistry:
    """Registry for tracking model artifacts"""
    
    def __init__(self, registry_path: str = "./model_artifacts_registry.json"):
        self.registry_path = Path(registry_path)
        self.model_artifacts: Dict[str, ModelArtifact] = self._load_registry()
        self.logger = logger
    
    def _load_registry(self) -> Dict[str, ModelArtifact]:
        """Load model artifacts registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                return {k: self._dict_to_model_artifact(v) for k, v in data.items()}
            except Exception as e:
                self.logger.warning(f"Could not load model artifacts registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model artifacts registry to file"""
        data = {k: self._model_artifact_to_dict(v) for k, v in self.model_artifacts.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _dict_to_model_artifact(self, data: Dict[str, Any]) -> ModelArtifact:
        """Convert dictionary to ModelArtifact"""
        return ModelArtifact(
            artifact_id=data['artifact_id'],
            model_id=data['model_id'],
            model_version=data['model_version'],
            training_run_id=data['training_run_id'],
            dataset_version_id=data['dataset_version_id'],
            created_at=data['created_at'],
            path=data['path'],
            size_bytes=data.get('size_bytes'),
            framework=data.get('framework'),
            format=data.get('format'),
            evaluation_score=data.get('evaluation_score'),
            promotion_status=data.get('promotion_status'),
            metadata=data.get('metadata')
        )
    
    def _model_artifact_to_dict(self, artifact: ModelArtifact) -> Dict[str, Any]:
        """Convert ModelArtifact to dictionary"""
        return {
            'artifact_id': artifact.artifact_id,
            'model_id': artifact.model_id,
            'model_version': artifact.model_version,
            'training_run_id': artifact.training_run_id,
            'dataset_version_id': artifact.dataset_version_id,
            'created_at': artifact.created_at,
            'path': artifact.path,
            'size_bytes': artifact.size_bytes,
            'framework': artifact.framework,
            'format': artifact.format,
            'evaluation_score': artifact.evaluation_score,
            'promotion_status': artifact.promotion_status,
            'metadata': artifact.metadata
        }
    
    def register_model_artifact(self, artifact: ModelArtifact) -> str:
        """Register a new model artifact"""
        self.model_artifacts[artifact.artifact_id] = artifact
        self._save_registry()
        self.logger.info(f"Registered model artifact: {artifact.artifact_id}")
        return artifact.artifact_id
    
    def get_model_artifact(self, artifact_id: str) -> Optional[ModelArtifact]:
        """Get a specific model artifact"""
        return self.model_artifacts.get(artifact_id)
    
    def update_artifact_promotion_status(self, artifact_id: str, status: str) -> bool:
        """Update promotion status of an artifact"""
        if artifact_id in self.model_artifacts:
            self.model_artifacts[artifact_id].promotion_status = status
            self._save_registry()
            return True
        return False


class TraceabilityManager:
    """Main class for managing dataset-to-training traceability"""
    
    def __init__(self, 
                 dataset_registry: DatasetRegistry,
                 training_run_registry: TrainingRunRegistry,
                 model_artifact_registry: ModelArtifactRegistry):
        self.dataset_registry = dataset_registry
        self.training_run_registry = training_run_registry
        self.model_artifact_registry = model_artifact_registry
        self.logger = logging.getLogger(__name__)
    
    def create_traceability_record(self,
                                 manifest: TrainingManifest,
                                 run_id: str,
                                 model_artifact_path: str,
                                 model_id: str,
                                 model_version: str) -> TraceabilityRecord:
        """Create a complete traceability record for a training run"""
        # Register or get dataset version
        dataset_version = self.dataset_registry.register_dataset(manifest.dataset)
        
        # Create training run
        training_run = TrainingRun(
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            manifest_id=manifest.manifest_id,
            dataset_version_id=dataset_version.dataset_id,
            start_time=datetime.utcnow().isoformat(),
            status="completed",
            hyperparameters=manifest.hyperparameters.to_transformers_config(),
            git_commit=self._get_git_commit(),
            environment_info=self._get_environment_info()
        )
        
        # Register training run
        self.training_run_registry.register_training_run(training_run)
        
        # Create model artifact
        artifact_size = os.path.getsize(model_artifact_path) if os.path.exists(model_artifact_path) else None
        model_artifact = ModelArtifact(
            artifact_id=f"{model_id}_{model_version}",
            model_id=model_id,
            model_version=model_version,
            training_run_id=run_id,
            dataset_version_id=dataset_version.dataset_id,
            created_at=datetime.utcnow().isoformat(),
            path=model_artifact_path,
            size_bytes=artifact_size,
            framework=manifest.framework.value,
            format="pytorch"  # Default format
        )
        
        # Register model artifact
        self.model_artifact_registry.register_model_artifact(model_artifact)
        
        # Create traceability record
        record = TraceabilityRecord(
            record_id=f"trace_{run_id}",
            dataset_version=dataset_version,
            training_run=training_run,
            model_artifact=model_artifact
        )
        
        self.logger.info(f"Created traceability record: {record.record_id}")
        return record
    
    def get_model_lineage(self, model_artifact_id: str) -> Optional[TraceabilityRecord]:
        """Get the complete lineage for a model artifact"""
        model_artifact = self.model_artifact_registry.get_model_artifact(model_artifact_id)
        if not model_artifact:
            return None
        
        training_run = self.training_run_registry.get_training_run(model_artifact.training_run_id)
        if not training_run:
            return None
        
        dataset_version = self.dataset_registry.get_dataset_version(model_artifact.dataset_version_id)
        if not dataset_version:
            return None
        
        return TraceabilityRecord(
            record_id=f"trace_{training_run.run_id}",
            dataset_version=dataset_version,
            training_run=training_run,
            model_artifact=model_artifact
        )
    
    def get_dataset_usage(self, dataset_id: str) -> List[TraceabilityRecord]:
        """Get all models trained with a specific dataset version"""
        records = []
        
        # Find training runs that used this dataset
        for run_id, run in self.training_run_registry.training_runs.items():
            if run.dataset_version_id == dataset_id:
                model_artifact = self.model_artifact_registry.get_model_artifact(run.model_id + "_" + run.model_version)
                if model_artifact:
                    dataset_version = self.dataset_registry.get_dataset_version(dataset_id)
                    if dataset_version:
                        record = TraceabilityRecord(
                            record_id=f"trace_{run_id}",
                            dataset_version=dataset_version,
                            training_run=run,
                            model_artifact=model_artifact
                        )
                        records.append(record)
        
        return records
    
    def validate_lineage_integrity(self, record: TraceabilityRecord) -> Tuple[bool, List[str]]:
        """Validate that all links in a traceability record are valid"""
        errors = []
        
        # Check if dataset exists
        dataset = self.dataset_registry.get_dataset_version(record.dataset_version.dataset_id)
        if not dataset:
            errors.append(f"Dataset {record.dataset_version.dataset_id} not found in registry")
        
        # Check if training run exists
        run = self.training_run_registry.get_training_run(record.training_run.run_id)
        if not run:
            errors.append(f"Training run {record.training_run.run_id} not found in registry")
        
        # Check if model artifact exists
        artifact = self.model_artifact_registry.get_model_artifact(record.model_artifact.artifact_id)
        if not artifact:
            errors.append(f"Model artifact {record.model_artifact.artifact_id} not found in registry")
        
        # Verify links between components
        if dataset and record.dataset_version.dataset_id != dataset.dataset_id:
            errors.append("Dataset ID mismatch")
        if run and record.training_run.run_id != run.run_id:
            errors.append("Training run ID mismatch")
        if artifact and record.model_artifact.artifact_id != artifact.artifact_id:
            errors.append("Model artifact ID mismatch")
        
        # Verify cross-references
        if run and run.dataset_version_id != record.dataset_version.dataset_id:
            errors.append("Training run doesn't reference correct dataset")
        if artifact and artifact.training_run_id != record.training_run.run_id:
            errors.append("Model artifact doesn't reference correct training run")
        if artifact and artifact.dataset_version_id != record.dataset_version.dataset_id:
            errors.append("Model artifact doesn't reference correct dataset")
        
        return len(errors) == 0, errors
    
    def _get_git_commit(self) -> Optional[str]:
        """Get the current git commit hash"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.commit.hexsha
        except:
            return None
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for traceability"""
        import platform
        import torch
        import sys
        
        env_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'device_name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                })
            env_info['gpu_info'] = gpu_info
        
        return env_info
    
    def generate_traceability_report(self, model_artifact_id: str) -> str:
        """Generate a human-readable traceability report"""
        lineage = self.get_model_lineage(model_artifact_id)
        if not lineage:
            return f"No traceability record found for model artifact: {model_artifact_id}"
        
        report = [
            f"=== Traceability Report for Model: {model_artifact_id} ===",
            f"Generated at: {lineage.created_at}",
            "",
            "Dataset Information:",
            f"  ID: {lineage.dataset_version.dataset_id}",
            f"  Name: {lineage.dataset_version.name}",
            f"  Version: {lineage.dataset_version.version}",
            f"  Size: {lineage.dataset_version.size_bytes} bytes" if lineage.dataset_version.size_bytes else "  Size: Unknown",
            f"  Records: {lineage.dataset_version.record_count}" if lineage.dataset_version.record_count else "  Records: Unknown",
            f"  Created: {lineage.dataset_version.created_at}",
            f"  Checksum: {lineage.dataset_version.checksum}" if lineage.dataset_version.checksum else "  Checksum: Unknown",
            f"  Git Commit: {lineage.dataset_version.commit_hash}" if lineage.dataset_version.commit_hash else "  Git Commit: Unknown",
            "",
            "Training Run Information:",
            f"  Run ID: {lineage.training_run.run_id}",
            f"  Model ID: {lineage.training_run.model_id}",
            f"  Model Version: {lineage.training_run.model_version}",
            f"  Manifest ID: {lineage.training_run.manifest_id}",
            f"  Start Time: {lineage.training_run.start_time}",
            f"  End Time: {lineage.training_run.end_time}" if lineage.training_run.end_time else "  End Time: In Progress",
            f"  Status: {lineage.training_run.status}",
            f"  Git Commit: {lineage.training_run.git_commit}" if lineage.training_run.git_commit else "  Git Commit: Unknown",
            "",
            "Hyperparameters Used:",
        ]
        
        # Add hyperparameters
        if lineage.training_run.hyperparameters:
            for key, value in lineage.training_run.hyperparameters.items():
                report.append(f"  {key}: {value}")
        else:
            report.append("  No hyperparameters recorded")
        
        report.extend([
            "",
            "Model Artifact Information:",
            f"  Artifact ID: {lineage.model_artifact.artifact_id}",
            f"  Path: {lineage.model_artifact.path}",
            f"  Size: {lineage.model_artifact.size_bytes} bytes" if lineage.model_artifact.size_bytes else "  Size: Unknown",
            f"  Framework: {lineage.model_artifact.framework}",
            f"  Format: {lineage.model_artifact.format}",
            f"  Eval Score: {lineage.model_artifact.evaluation_score}" if lineage.model_artifact.evaluation_score else "  Eval Score: Not available",
            f"  Promotion Status: {lineage.model_artifact.promotion_status}" if lineage.model_artifact.promotion_status else "  Promotion Status: Not promoted",
        ])
        
        return "\n".join(report)
    
    def export_traceability_data(self, output_path: str):
        """Export all traceability data to a file"""
        # Combine all registries into one structure
        export_data = {
            'datasets': {k: self.dataset_registry._dataset_version_to_dict(v) 
                        for k, v in self.dataset_registry.datasets.items()},
            'training_runs': {k: self.training_run_registry._training_run_to_dict(v) 
                             for k, v in self.training_run_registry.training_runs.items()},
            'model_artifacts': {k: self.model_artifact_registry._model_artifact_to_dict(v) 
                               for k, v in self.model_artifact_registry.model_artifacts.items()},
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported traceability data to {output_path}")


def create_traceability_manager() -> TraceabilityManager:
    """Create a default traceability manager with registries"""
    dataset_registry = DatasetRegistry()
    training_run_registry = TrainingRunRegistry()
    model_artifact_registry = ModelArtifactRegistry()
    
    return TraceabilityManager(dataset_registry, training_run_registry, model_artifact_registry)


# Integration with training runner
def integrate_traceability_with_training(training_runner, manifest: TrainingManifest, 
                                       model_artifact_path: str, run_id: str):
    """Integrate traceability with a training run"""
    # Create traceability manager
    traceability_manager = create_traceability_manager()
    
    # Create traceability record after training
    model_id = manifest.model_name or "default_model"
    model_version = f"v{manifest.manifest_version}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    record = traceability_manager.create_traceability_record(
        manifest=manifest,
        run_id=run_id,
        model_artifact_path=model_artifact_path,
        model_id=model_id,
        model_version=model_version
    )
    
    return record


# Example usage and testing
def test_traceability_system():
    """Test the traceability system functionality"""
    logger.info("Testing Traceability System...")
    
    # Create the traceability manager
    manager = create_traceability_manager()
    
    # Create a simple dataset reference
    from .training_manifest import DatasetReference
    dataset_ref = DatasetReference(
        name="test_therapy_dataset",
        version="v1.0",
        path="./test_dataset.json",
        created_at=datetime.utcnow().isoformat()
    )
    
    # Register dataset
    dataset_version = manager.dataset_registry.register_dataset(dataset_ref)
    print(f"Registered dataset: {dataset_version.dataset_id}")
    
    # Create a training run
    training_run = TrainingRun(
        run_id="run_test_123",
        model_id="test_model",
        model_version="1.0.0",
        manifest_id="manifest_456",
        dataset_version_id=dataset_version.dataset_id,
        start_time=datetime.utcnow().isoformat(),
        status="completed",
        hyperparameters={"lr": 1e-5, "epochs": 3}
    )
    
    manager.training_run_registry.register_training_run(training_run)
    print(f"Registered training run: {training_run.run_id}")
    
    # Create a model artifact
    model_artifact = ModelArtifact(
        artifact_id="artifact_test_789",
        model_id="test_model",
        model_version="1.0.0",
        training_run_id=training_run.run_id,
        dataset_version_id=dataset_version.dataset_id,
        created_at=datetime.utcnow().isoformat(),
        path="./test_model_output",
        framework="transformers"
    )
    
    manager.model_artifact_registry.register_model_artifact(model_artifact)
    print(f"Registered model artifact: {model_artifact.artifact_id}")
    
    # Create a traceability record
    record = TraceabilityRecord(
        record_id="trace_test_101",
        dataset_version=dataset_version,
        training_run=training_run,
        model_artifact=model_artifact
    )
    
    # Validate the record
    is_valid, errors = manager.validate_lineage_integrity(record)
    print(f"Record validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Get model lineage
    lineage = manager.get_model_lineage(model_artifact.artifact_id)
    if lineage:
        print(f"Retrieved lineage for model: {model_artifact.artifact_id}")
        print(f"  Dataset: {lineage.dataset_version.dataset_id}")
        print(f"  Training Run: {lineage.training_run.run_id}")
        print(f"  Model Artifact: {lineage.model_artifact.artifact_id}")
    
    # Get dataset usage
    usage = manager.get_dataset_usage(dataset_version.dataset_id)
    print(f"Dataset used in {len(usage)} training runs")
    
    # Generate traceability report
    report = manager.generate_traceability_report(model_artifact.artifact_id)
    print(f"\nTraceability Report:\n{report}")
    
    # Export data
    manager.export_traceability_data("./test_traceability_export.json")
    print("Exported traceability data to ./test_traceability_export.json")


if __name__ == "__main__":
    test_traceability_system()