"""
Hyperparameter sweeps and tracking system for Pixelated Empathy AI project.
Implements hyperparameter optimization and artifact tracking using WandB or similar tools.
"""

import json
import os
import wandb
import random
import itertools
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
from .training_manifest import Hyperparameters, TrainingManifest
from .training_runner import TrainingRunner
from .evaluation_system import ComprehensiveEvaluator, EvaluationResults


logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter sweep"""
    name: str
    values: List[Any]
    type: str  # 'categorical', 'continuous', 'discrete'
    distribution: Optional[str] = None  # 'uniform', 'log_uniform', 'normal', etc.
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


@dataclass
class SweepConfiguration:
    """Complete configuration for hyperparameter sweep"""
    sweep_name: str
    sweep_description: str
    method: str  # 'grid', 'random', 'bayes'
    metric: str  # Metric to optimize (e.g., 'eval_loss', 'accuracy')
    goal: str  # 'minimize' or 'maximize'
    parameters: List[HyperparameterConfig]
    total_trials: int = 10
    early_terminate: bool = True
    early_terminate_patience: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial"""
    trial_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # 'completed', 'failed', 'cancelled'
    start_time: str
    end_time: Optional[str] = None
    training_time_seconds: Optional[float] = None
    artifacts: Optional[List[str]] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SweepResult:
    """Complete result of a hyperparameter sweep"""
    sweep_id: str
    sweep_name: str
    sweep_config: SweepConfiguration
    trial_results: List[TrialResult]
    best_config: Dict[str, Any]
    best_metric_value: float
    best_trial_id: str
    completed_trials: int
    failed_trials: int
    start_time: str
    end_time: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HyperparameterSweeper:
    """System for conducting hyperparameter sweeps"""
    
    def __init__(self):
        self.logger = logger
        self.trial_results: List[TrialResult] = []
    
    def generate_grid_configs(self, sweep_config: SweepConfiguration) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search"""
        param_lists = []
        param_names = []
        
        for param in sweep_config.parameters:
            param_lists.append(param.values)
            param_names.append(param.name)
        
        configs = []
        for combination in itertools.product(*param_lists):
            config = dict(zip(param_names, combination))
            configs.append(config)
        
        # Limit to total_trials if grid is too large
        if len(configs) > sweep_config.total_trials:
            configs = configs[:sweep_config.total_trials]
        
        return configs
    
    def generate_random_configs(self, sweep_config: SweepConfiguration) -> List[Dict[str, Any]]:
        """Generate random combinations for random search"""
        configs = []
        
        for _ in range(sweep_config.total_trials):
            config = {}
            for param in sweep_config.parameters:
                if param.type == 'categorical':
                    config[param.name] = random.choice(param.values)
                elif param.type == 'continuous':
                    if param.distribution == 'uniform':
                        config[param.name] = random.uniform(param.min_value or 0, param.max_value or 1)
                    elif param.distribution == 'log_uniform':
                        config[param.name] = np.exp(random.uniform(np.log(param.min_value or 1e-6), 
                                                                 np.log(param.max_value or 1e6)))
                    else:
                        config[param.name] = random.uniform(param.min_value or 0, param.max_value or 1)
                elif param.type == 'discrete':
                    if param.distribution == 'uniform':
                        config[param.name] = random.randint(int(param.min_value or 0), int(param.max_value or 1))
                    else:
                        # Default to random choice of values if provided
                        if param.values:
                            config[param.name] = random.choice(param.values)
                        else:
                            config[param.name] = random.randint(int(param.min_value or 0), int(param.max_value or 1))
            configs.append(config)
        
        return configs
    
    def generate_bayesian_configs(self, sweep_config: SweepConfiguration) -> List[Dict[str, Any]]:
        """Generate configurations using Bayesian optimization (simplified version)"""
        # For now, this is a placeholder that does random search
        # In a real implementation, you'd use a library like Optuna or scikit-optimize
        return self.generate_random_configs(sweep_config)
    
    def sweep_hyperparameters(self,
                            base_manifest: TrainingManifest,
                            sweep_config: SweepConfiguration,
                            training_function: Callable[[TrainingManifest], Any],
                            evaluation_function: Optional[Callable[[Any], Dict[str, float]]] = None) -> SweepResult:
        """Conduct hyperparameter sweep"""
        self.logger.info(f"Starting hyperparameter sweep: {sweep_config.sweep_name}")
        
        # Generate configurations based on method
        if sweep_config.method == 'grid':
            configs = self.generate_grid_configs(sweep_config)
        elif sweep_config.method == 'random':
            configs = self.generate_random_configs(sweep_config)
        elif sweep_config.method == 'bayes':
            configs = self.generate_bayesian_configs(sweep_config)
        else:
            raise ValueError(f"Unknown sweep method: {sweep_config.method}")
        
        self.logger.info(f"Generated {len(configs)} configurations for sweep")
        
        start_time = datetime.utcnow().isoformat()
        trial_results = []
        best_metric = float('inf') if sweep_config.goal == 'minimize' else float('-inf')
        best_config = None
        best_trial_id = None
        
        for i, config in enumerate(configs):
            trial_id = f"trial_{i:03d}_{sweep_config.sweep_name}"
            self.logger.info(f"Running trial {i+1}/{len(configs)}: {trial_id}")
            
            # Create a new manifest with updated hyperparameters
            trial_manifest = self._update_manifest_with_config(base_manifest, config)
            
            # Run training
            start_trial = datetime.utcnow()
            try:
                # This would actually run training with the config
                model_or_result = training_function(trial_manifest)
                
                # Evaluate if evaluation function is provided
                metrics = {}
                if evaluation_function:
                    metrics = evaluation_function(model_or_result)
                else:
                    # If no evaluation function, use some defaults
                    metrics = {'dummy_metric': random.random()}
                
                # Add the config values to metrics for tracking
                for param_name, param_value in config.items():
                    metrics[f'hp_{param_name}'] = param_value
                
                # Calculate training time
                end_trial = datetime.utcnow()
                training_time = (end_trial - start_trial).total_seconds()
                
                # Check if this is the best configuration
                metric_value = metrics.get(sweep_config.metric, 0)
                is_better = (
                    (sweep_config.goal == 'minimize' and metric_value < best_metric) or
                    (sweep_config.goal == 'maximize' and metric_value > best_metric)
                )
                
                if is_better:
                    best_metric = metric_value
                    best_config = config.copy()
                    best_trial_id = trial_id
                
                # Create trial result
                trial_result = TrialResult(
                    trial_id=trial_id,
                    config=config,
                    metrics=metrics,
                    status='completed',
                    start_time=start_trial.isoformat(),
                    end_time=end_trial.isoformat(),
                    training_time_seconds=training_time,
                    metadata={'sweep_config_name': sweep_config.sweep_name}
                )
                
                trial_results.append(trial_result)
                self.logger.info(f"Trial {trial_id} completed with {sweep_config.metric}={metric_value:.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {trial_id} failed: {str(e)}")
                
                # Record failed trial
                trial_result = TrialResult(
                    trial_id=trial_id,
                    config=config,
                    metrics={},
                    status='failed',
                    start_time=start_trial.isoformat(),
                    end_time=datetime.utcnow().isoformat(),
                    metadata={'error': str(e), 'sweep_config_name': sweep_config.sweep_name}
                )
                trial_results.append(trial_result)
        
        # Calculate sweep statistics
        completed_trials = len([r for r in trial_results if r.status == 'completed'])
        failed_trials = len([r for r in trial_results if r.status == 'failed'])
        
        end_time = datetime.utcnow().isoformat()
        
        sweep_result = SweepResult(
            sweep_id=f"sweep_{sweep_config.sweep_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            sweep_name=sweep_config.sweep_name,
            sweep_config=sweep_config,
            trial_results=trial_results,
            best_config=best_config or {},
            best_metric_value=best_metric if best_metric != float('inf') and best_metric != float('-inf') else 0,
            best_trial_id=best_trial_id or "",
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            start_time=start_time,
            end_time=end_time
        )
        
        self.logger.info(f"Sweep completed. Best {sweep_config.metric}: {best_metric:.4f}")
        return sweep_result
    
    def _update_manifest_with_config(self, base_manifest: TrainingManifest, config: Dict[str, Any]) -> TrainingManifest:
        """Update a training manifest with hyperparameter configuration"""
        # Create a copy of the base manifest
        updated_manifest = TrainingManifest(
            manifest_id=base_manifest.manifest_id + f"_sweep_{len(config)}",
            name=base_manifest.name + "_swept",
            description=base_manifest.description,
            dataset=base_manifest.dataset,
            hyperparameters=Hyperparameters(**{
                **base_manifest.hyperparameters.__dict__,
                **config
            }),
            framework=base_manifest.framework,
            compute_target=base_manifest.compute_target,
            seed=base_manifest.seed,
            safety_metrics=base_manifest.safety_metrics,
            resources=base_manifest.resources,
            manifest_version=base_manifest.manifest_version,
            created_at=base_manifest.created_at,
            created_by=base_manifest.created_by,
            output_dir=base_manifest.output_dir + f"_sweep_{len(config)}",
            model_name=base_manifest.model_name,
            log_dir=base_manifest.log_dir,
            evaluation_enabled=base_manifest.evaluation_enabled,
            checkpointing_enabled=base_manifest.checkpointing_enabled,
            wandb_logging=base_manifest.wandb_logging,
            wandb_project=base_manifest.wandb_project,
            wandb_tags=base_manifest.wandb_tags,
            metadata=base_manifest.metadata
        )
        
        return updated_manifest


class WandBIntegration:
    """Integration with Weights & Biases for hyperparameter tracking"""
    
    def __init__(self):
        self.logger = logger
        self.is_initialized = False
    
    def initialize_sweep(self, sweep_config: SweepConfiguration) -> str:
        """Initialize a WandB sweep"""
        try:
            # Convert sweep config to WandB format
            wandb_config = {
                'method': sweep_config.method,
                'metric': {
                    'name': sweep_config.metric,
                    'goal': sweep_config.goal
                },
                'parameters': {}
            }
            
            for param in sweep_config.parameters:
                param_config = {}
                if param.type == 'categorical':
                    param_config['values'] = param.values
                elif param.type == 'continuous':
                    param_config['min'] = param.min_value or 0
                    param_config['max'] = param.max_value or 1
                    if param.distribution:
                        param_config['distribution'] = param.distribution
                elif param.type == 'discrete':
                    if param.values:
                        param_config['values'] = param.values
                    else:
                        param_config['min'] = param.min_value or 0
                        param_config['max'] = param.max_value or 1
                
                wandb_config['parameters'][param.name] = param_config
            
            # Create sweep in WandB
            sweep_id = wandb.sweep(wandb_config, project=sweep_config.sweep_name)
            self.is_initialized = True
            self.logger.info(f"Initialized WandB sweep: {sweep_id}")
            return sweep_id
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB sweep: {e}")
            return ""
    
    def run_sweep_trial(self, 
                       sweep_config: SweepConfiguration,
                       training_function: Callable,
                       evaluation_function: Optional[Callable] = None) -> List[TrialResult]:
        """Run WandB-managed sweep trials"""
        trial_results = []
        
        def wandb_train():
            with wandb.init() as run:
                # Get hyperparameters from WandB config
                config = dict(wandb.config)
                
                # Create training manifest with WandB config
                # This is a simplified approach - in a real implementation you would 
                # integrate more deeply with the training process
                self.logger.info(f"Running WandB trial with config: {config}")
                
                # Record the configuration in WandB
                wandb.config.update(config)
                
                # Run training
                try:
                    # In real implementation, you would pass config to training
                    # This is a simplified version
                    result = training_function(config)
                    
                    # Evaluate if function provided
                    if evaluation_function:
                        metrics = evaluation_function(result)
                        wandb.log(metrics)
                    else:
                        # Just log a dummy metric
                        dummy_metric = random.random()
                        wandb.log({sweep_config.metric: dummy_metric})
                    
                    # Create trial result
                    trial_result = TrialResult(
                        trial_id=run.id,
                        config=config,
                        metrics=dict(run.summary),
                        status='completed',
                        start_time=datetime.utcnow().isoformat(),
                        end_time=datetime.utcnow().isoformat(),
                        metadata={'sweep_name': sweep_config.sweep_name}
                    )
                    
                    trial_results.append(trial_result)
                    
                except Exception as e:
                    self.logger.error(f"WandB trial failed: {e}")
                    wandb.log({'error': str(e)})
                    
                    trial_result = TrialResult(
                        trial_id=run.id,
                        config=config,
                        metrics={'error': str(e)},
                        status='failed',
                        start_time=datetime.utcnow().isoformat(),
                        end_time=datetime.utcnow().isoformat(),
                        metadata={'sweep_name': sweep_config.sweep_name, 'error': str(e)}
                    )
                    trial_results.append(trial_result)
        
        # Run the sweep
        try:
            wandb.agent(sweep_config.sweep_name, wandb_train, count=sweep_config.total_trials)
        except Exception as e:
            self.logger.error(f"Error running WandB agent: {e}")
        
        return trial_results
    
    def log_artifacts(self, artifacts_dir: str, name: str = "model-artifacts"):
        """Log artifacts to WandB"""
        try:
            artifact = wandb.Artifact(name, type='model')
            artifact.add_dir(artifacts_dir)
            wandb.log_artifact(artifact)
            self.logger.info(f"Logged artifacts from {artifacts_dir}")
        except Exception as e:
            self.logger.error(f"Failed to log artifacts: {e}")


class ArtifactManager:
    """Manager for handling model artifacts during sweeps"""
    
    def __init__(self, base_artifacts_dir: str = "./artifacts"):
        self.base_dir = Path(base_artifacts_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
    
    def save_artifact(self, content: Any, artifact_name: str, trial_id: str) -> str:
        """Save an artifact for a trial"""
        trial_dir = self.base_dir / trial_id
        trial_dir.mkdir(exist_ok=True)
        
        artifact_path = trial_dir / artifact_name
        if isinstance(content, dict):
            with open(artifact_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif isinstance(content, str):
            with open(artifact_path, 'w') as f:
                f.write(content)
        elif hasattr(content, 'save'):
            # If it's a model with a save method
            content.save(str(artifact_path))
        else:
            # Try to convert to string representation
            with open(artifact_path, 'w') as f:
                f.write(str(content))
        
        self.logger.info(f"Saved artifact: {artifact_path}")
        return str(artifact_path)
    
    def load_artifact(self, artifact_path: str):
        """Load an artifact"""
        path = Path(artifact_path)
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            with open(path, 'r') as f:
                return f.read()
    
    def get_trial_artifacts(self, trial_id: str) -> List[Path]:
        """Get all artifacts for a specific trial"""
        trial_dir = self.base_dir / trial_id
        if trial_dir.exists():
            return list(trial_dir.iterdir())
        return []
    
    def cleanup_old_artifacts(self, keep_last_n: int = 5):
        """Clean up old artifacts while keeping the most recent ones"""
        # Get all trial directories (sorted by creation time)
        trial_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        trial_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        
        # Keep only the most recent ones
        for trial_dir in trial_dirs[keep_last_n:]:
            import shutil
            shutil.rmtree(trial_dir)
            self.logger.info(f"Cleaned up old artifacts: {trial_dir}")


class HyperparameterOptimizationSystem:
    """Main system integrating all hyperparameter optimization components"""
    
    def __init__(self):
        self.sweeper = HyperparameterSweeper()
        self.wandb_integration = WandBIntegration()
        self.artifact_manager = ArtifactManager()
        self.logger = logger
    
    def run_hyperparameter_sweep(self,
                               base_manifest: TrainingManifest,
                               sweep_config: SweepConfiguration,
                               use_wandb: bool = True,
                               training_function: Optional[Callable] = None) -> SweepResult:
        """Run a complete hyperparameter sweep"""
        self.logger.info(f"Starting hyperparameter sweep: {sweep_config.sweep_name}")
        
        # If training function is not provided, create a default one
        if training_function is None:
            training_function = self._default_training_function
        
        # Run sweep based on whether we're using WandB
        if use_wandb:
            return self._run_wandb_sweep(base_manifest, sweep_config, training_function)
        else:
            return self._run_local_sweep(base_manifest, sweep_config, training_function)
    
    def _default_training_function(self, manifest: TrainingManifest):
        """Default training function that creates a training runner"""
        runner = TrainingRunner(manifest)
        
        # Instead of running full training, we'll just return hyperparameters
        # as a demo. In a real scenario, this would run the full training
        return manifest.hyperparameters.to_transformers_config()
    
    def _run_wandb_sweep(self, 
                        base_manifest: TrainingManifest,
                        sweep_config: SweepConfiguration,
                        training_function: Callable) -> SweepResult:
        """Run sweep using WandB"""
        # This is a simplified version - in a real implementation you would
        # use WandB's actual API for managing sweeps
        self.logger.info("Running WandB-managed hyperparameter sweep")
        
        # Create WandB sweep
        sweep_id = self.wandb_integration.initialize_sweep(sweep_config)
        if not sweep_id:
            self.logger.warning("Failed to initialize WandB sweep, falling back to local sweep")
            return self._run_local_sweep(base_manifest, sweep_config, training_function)
        
        # Run the sweep trials
        trial_results = self.wandb_integration.run_sweep_trial(
            sweep_config, training_function
        )
        
        # Process results to create a sweep result
        if trial_results:
            # Find the best result
            best_result = max(trial_results, key=lambda r: r.metrics.get(sweep_config.metric, 0) 
                             if sweep_config.goal == 'maximize' else 
                             -r.metrics.get(sweep_config.metric, 0))
            
            sweep_result = SweepResult(
                sweep_id=f"wandb_sweep_{sweep_id}",
                sweep_name=sweep_config.sweep_name,
                sweep_config=sweep_config,
                trial_results=trial_results,
                best_config=best_result.config,
                best_metric_value=best_result.metrics.get(sweep_config.metric, 0),
                best_trial_id=best_result.trial_id,
                completed_trials=len([r for r in trial_results if r.status == 'completed']),
                failed_trials=len([r for r in trial_results if r.status == 'failed']),
                start_time=datetime.utcnow().isoformat(),
                end_time=datetime.utcnow().isoformat()
            )
            
            return sweep_result
        
        # Fallback if no trials completed
        return SweepResult(
            sweep_id=f"wandb_sweep_{sweep_id}_failed",
            sweep_name=sweep_config.sweep_name,
            sweep_config=sweep_config,
            trial_results=[],
            best_config={},
            best_metric_value=0,
            best_trial_id="",
            completed_trials=0,
            failed_trials=len(trial_results),
            start_time=datetime.utcnow().isoformat(),
            end_time=datetime.utcnow().isoformat()
        )
    
    def _run_local_sweep(self,
                        base_manifest: TrainingManifest,
                        sweep_config: SweepConfiguration,
                        training_function: Callable) -> SweepResult:
        """Run sweep locally without WandB"""
        self.logger.info("Running local hyperparameter sweep")
        
        # Define a default evaluation function if none is provided
        def default_evaluation(result):
            # Just return some metrics for demonstration
            import random
            return {
                sweep_config.metric: random.random(),  # Placeholder for actual metric
                'param_count': getattr(result, 'num_parameters', 0) if hasattr(result, 'num_parameters') else 0
            }
        
        # Run the sweep using the local sweeper
        return self.sweeper.sweep_hyperparameters(
            base_manifest,
            sweep_config,
            training_function,
            default_evaluation
        )
    
    def save_sweep_results(self, sweep_result: SweepResult, output_path: str):
        """Save sweep results to file"""
        result_dict = {
            'sweep_id': sweep_result.sweep_id,
            'sweep_name': sweep_result.sweep_name,
            'sweep_config': sweep_result.sweep_config.__dict__,
            'trial_results': [
                {
                    'trial_id': tr.trial_id,
                    'config': tr.config,
                    'metrics': tr.metrics,
                    'status': tr.status,
                    'start_time': tr.start_time,
                    'end_time': tr.end_time,
                    'training_time_seconds': tr.training_time_seconds,
                    'metadata': tr.metadata
                }
                for tr in sweep_result.trial_results
            ],
            'best_config': sweep_result.best_config,
            'best_metric_value': sweep_result.best_metric_value,
            'best_trial_id': sweep_result.best_trial_id,
            'completed_trials': sweep_result.completed_trials,
            'failed_trials': sweep_result.failed_trials,
            'start_time': sweep_result.start_time,
            'end_time': sweep_result.end_time,
            'metadata': sweep_result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Saved sweep results to {output_path}")
    
    def generate_sweep_report(self, sweep_result: SweepResult) -> str:
        """Generate a human-readable report of sweep results"""
        report = [
            f"=== Hyperparameter Sweep Report: {sweep_result.sweep_name} ===",
            f"Generated at: {datetime.utcnow().isoformat()}",
            f"Total Trials: {len(sweep_result.trial_results)}",
            f"Completed: {sweep_result.completed_trials}",
            f"Failed: {sweep_result.failed_trials}",
            f"Goal: {sweep_result.sweep_config.goal} {sweep_result.sweep_config.metric}",
            "",
            f"Best Configuration (Trial: {sweep_result.best_trial_id}):"
        ]
        
        # Add best configuration
        for param_name, param_value in sweep_result.best_config.items():
            report.append(f"  {param_name}: {param_value}")
        
        report.append(f"Best {sweep_result.sweep_config.metric}: {sweep_result.best_metric_value:.6f}")
        report.append("")
        
        # Add top configurations
        report.append("Top 5 Configurations:")
        sorted_trials = sorted(
            sweep_result.trial_results,
            key=lambda x: x.metrics.get(sweep_result.sweep_config.metric, 0) 
                         if sweep_result.sweep_config.goal == 'maximize' else 
                         -x.metrics.get(sweep_result.sweep_config.metric, 0)
        )[:5]
        
        for i, trial in enumerate(sorted_trials, 1):
            metric_val = trial.metrics.get(sweep_result.sweep_config.metric, 0)
            report.append(f"  {i}. {sweep_result.sweep_config.metric}: {metric_val:.6f}")
            for param_name, param_value in trial.config.items():
                report.append(f"     {param_name}: {param_value}")
            report.append("")
        
        return "\n".join(report)
    
    def create_sweep_from_manifest(self, base_manifest: TrainingManifest) -> SweepConfiguration:
        """Create a default sweep configuration based on a training manifest"""
        return SweepConfiguration(
            sweep_name=f"{base_manifest.name}_hyperparameter_sweep",
            sweep_description=f"Hyperparameter sweep for {base_manifest.name}",
            method='random',
            metric='eval_loss',
            goal='minimize',
            parameters=[
                HyperparameterConfig(
                    name='learning_rate',
                    values=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5],
                    type='categorical'
                ),
                HyperparameterConfig(
                    name='per_device_train_batch_size',
                    values=[1, 2, 4, 8],
                    type='categorical'
                ),
                HyperparameterConfig(
                    name='num_train_epochs',
                    values=[1, 2, 3],
                    type='categorical'
                ),
                HyperparameterConfig(
                    name='warmup_steps',
                    values=[100, 500, 1000],
                    type='categorical'
                )
            ],
            total_trials=10
        )


def create_hyperparameter_system() -> HyperparameterOptimizationSystem:
    """Create a default hyperparameter optimization system"""
    return HyperparameterOptimizationSystem()


# Example usage and testing
def test_hyperparameter_system():
    """Test the hyperparameter system functionality"""
    logger.info("Testing Hyperparameter System...")
    
    # Create system
    hp_system = create_hyperparameter_system()
    
    # Create a sample training manifest
    from .training_manifest import create_default_manifest
    manifest = create_default_manifest("./test_dataset.json", "test_v1.0")
    manifest.hyperparameters.learning_rate = 2e-5
    manifest.hyperparameters.num_train_epochs = 1
    
    # Create sweep configuration
    sweep_config = SweepConfiguration(
        sweep_name="test_learning_rate_sweep",
        sweep_description="Testing learning rate optimization",
        method="grid",
        metric="eval_loss",
        goal="minimize",
        parameters=[
            HyperparameterConfig(
                name="learning_rate",
                values=[1e-5, 2e-5, 5e-5],
                type="categorical"
            ),
            HyperparameterConfig(
                name="per_device_train_batch_size",
                values=[1, 2],
                type="categorical"
            )
        ],
        total_trials=6  # 3 learning rates * 2 batch sizes
    )
    
    # Run a simple test sweep (using local mode to avoid WandB dependency)
    def dummy_training_fn(manifest):
        # Simulate training by returning hyperparameters
        return manifest.hyperparameters.to_transformers_config()
    
    def dummy_evaluation_fn(result):
        # Simulate evaluation with random metrics
        import random
        return {
            'eval_loss': random.uniform(1.0, 3.0),
            'accuracy': random.uniform(0.5, 0.9)
        }
    
    # Test sweep with dummy functions
    print("Testing local hyperparameter sweep...")
    sweep_result = hp_system.sweeper.sweep_hyperparameters(
        manifest,
        sweep_config,
        dummy_training_fn,
        dummy_evaluation_fn
    )
    
    print(f"Sweep completed with {len(sweep_result.trial_results)} trials")
    print(f"Best configuration: {sweep_result.best_config}")
    print(f"Best metric value: {sweep_result.best_metric_value}")
    
    # Generate report
    report = hp_system.generate_sweep_report(sweep_result)
    print(f"\nSweep Report:\n{report}")
    
    # Save results
    hp_system.save_sweep_results(sweep_result, "./test_sweep_results.json")
    print("Sweep results saved to ./test_sweep_results.json")
    
    # Test artifact management
    artifact_path = hp_system.artifact_manager.save_artifact(
        {"test": "artifact", "trial": "test_001"}, 
        "config.json", 
        "test_001"
    )
    print(f"Saved artifact to: {artifact_path}")
    
    # Load artifact back
    loaded_artifact = hp_system.artifact_manager.load_artifact(artifact_path)
    print(f"Loaded artifact: {loaded_artifact}")


if __name__ == "__main__":
    test_hyperparameter_system()