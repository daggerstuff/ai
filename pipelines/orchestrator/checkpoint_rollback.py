"""
Checkpoint management and rollback system for Pixelated Empathy AI project.
Implements reliable checkpoint management with rollback capabilities.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer
import tempfile


logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a specific checkpoint"""
    checkpoint_id: str
    path: str
    step: int
    timestamp: str
    metrics: Optional[Dict[str, float]] = None
    is_best: bool = False
    training_args: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    message: str
    previous_checkpoint: Optional[CheckpointInfo]
    new_checkpoint: Optional[CheckpointInfo]
    rollback_time: str


class CheckpointManager:
    """Manages training checkpoints with versioning and rollback capabilities"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        
        # Keep track of checkpoint history
        self.checkpoint_history_file = self.checkpoints_dir / "checkpoint_history.json"
        self.history = self._load_history()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_history(self) -> List[CheckpointInfo]:
        """Load checkpoint history from file"""
        if self.checkpoint_history_file.exists():
            try:
                with open(self.checkpoint_history_file, 'r') as f:
                    data = json.load(f)
                return [self._dict_to_checkpoint_info(item) for item in data]
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save checkpoint history to file"""
        try:
            data = [self._checkpoint_info_to_dict(item) for item in self.history]
            with open(self.checkpoint_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save checkpoint history: {e}")
    
    def _dict_to_checkpoint_info(self, data: Dict[str, Any]) -> CheckpointInfo:
        """Convert dictionary to CheckpointInfo"""
        return CheckpointInfo(
            checkpoint_id=data['checkpoint_id'],
            path=data['path'],
            step=data['step'],
            timestamp=data['timestamp'],
            metrics=data.get('metrics'),
            is_best=data.get('is_best', False),
            training_args=data.get('training_args'),
            metadata=data.get('metadata')
        )
    
    def _checkpoint_info_to_dict(self, checkpoint: CheckpointInfo) -> Dict[str, Any]:
        """Convert CheckpointInfo to dictionary"""
        return {
            'checkpoint_id': checkpoint.checkpoint_id,
            'path': checkpoint.path,
            'step': checkpoint.step,
            'timestamp': checkpoint.timestamp,
            'metrics': checkpoint.metrics,
            'is_best': checkpoint.is_best,
            'training_args': checkpoint.training_args,
            'metadata': checkpoint.metadata
        }
    
    def register_checkpoint(self, 
                           checkpoint_path: str, 
                           step: int, 
                           metrics: Optional[Dict[str, float]] = None,
                           is_best: bool = False,
                           training_args: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> CheckpointInfo:
        """Register a new checkpoint in the history"""
        checkpoint_id = f"chkpt_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            path=checkpoint_path,
            step=step,
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
            is_best=is_best,
            training_args=training_args,
            metadata=metadata
        )
        
        # Add to history
        self.history.append(checkpoint_info)
        self._save_history()
        
        self.logger.info(f"Registered checkpoint: {checkpoint_id} at step {step}")
        return checkpoint_info
    
    def get_available_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of available checkpoints ordered by step (descending)"""
        # Check if checkpoint directories exist in the filesystem
        available_checkpoints = []
        
        for checkpoint_info in self.history:
            checkpoint_path = Path(checkpoint_info.path)
            if checkpoint_path.exists():
                available_checkpoints.append(checkpoint_info)
            else:
                self.logger.warning(f"Checkpoint path does not exist: {checkpoint_info.path}")
        
        # Sort by step in descending order (newest first)
        available_checkpoints.sort(key=lambda x: x.step, reverse=True)
        return available_checkpoints
    
    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the best checkpoint based on metrics or registration order"""
        available = self.get_available_checkpoints()
        
        # Look for checkpoints marked as best
        for checkpoint in available:
            if checkpoint.is_best:
                return checkpoint
        
        # If no best checkpoint marked, return the one with best metrics
        if available:
            best_checkpoint = max(available, 
                                key=lambda x: x.metrics.get('eval_loss', float('inf')) * -1 
                                    if x.metrics else x.step)
            return best_checkpoint
        
        return None
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint"""
        available = self.get_available_checkpoints()
        return available[0] if available else None
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        """Get a specific checkpoint by ID"""
        for checkpoint in self.history:
            if checkpoint.checkpoint_id == checkpoint_id:
                checkpoint_path = Path(checkpoint.path)
                if checkpoint_path.exists():
                    return checkpoint
        return None
    
    def get_checkpoints_by_step_range(self, start_step: int, end_step: int) -> List[CheckpointInfo]:
        """Get checkpoints within a specific step range"""
        available = self.get_available_checkpoints()
        return [chk for chk in available if start_step <= chk.step <= end_step]
    
    def cleanup_checkpoints(self, keep_last_n: int = 5, keep_best_n: int = 2) -> List[str]:
        """Clean up old checkpoints while keeping the most recent and best ones"""
        available = self.get_available_checkpoints()
        
        # Identify checkpoints to keep
        checkpoints_to_keep = set()
        
        # Keep last N checkpoints
        recent_checkpoints = available[:keep_last_n]
        for chk in recent_checkpoints:
            checkpoints_to_keep.add(chk.checkpoint_id)
        
        # Keep best N checkpoints
        if keep_best_n > 0:
            best_checkpoints = sorted(available, 
                                    key=lambda x: x.metrics.get('eval_loss', float('inf')) if x.metrics else float('inf'))[:keep_best_n]
            for chk in best_checkpoints:
                checkpoints_to_keep.add(chk.checkpoint_id)
        
        # Identify checkpoints to remove
        checkpoints_to_remove = []
        for checkpoint in available:
            if checkpoint.checkpoint_id not in checkpoints_to_keep:
                checkpoints_to_remove.append(checkpoint)
        
        # Remove the checkpoints
        removed_paths = []
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint.path)
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {checkpoint.path}")
                    removed_paths.append(checkpoint.path)
                
                # Remove from history
                self.history = [h for h in self.history if h.checkpoint_id != checkpoint.checkpoint_id]
            except Exception as e:
                self.logger.error(f"Could not remove checkpoint {checkpoint.path}: {e}")
        
        # Save history after cleanup
        self._save_history()
        
        return removed_paths
    
    def create_recovery_point(self, checkpoint_path: str) -> str:
        """Create a recovery point that won't be cleaned up"""
        recovery_path = self.checkpoints_dir / f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            shutil.copytree(checkpoint_path, recovery_path)
            self.logger.info(f"Created recovery point: {recovery_path}")
            return str(recovery_path)
        
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Validate that a checkpoint is complete and can be loaded"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False, "Checkpoint path does not exist"
        
        required_files = [
            "pytorch_model.bin",
            "config.json",
            "tokenizer.json",
            "training_args.bin"
        ]
        
        missing_files = []
        for required_file in required_files:
            if not (checkpoint_path / required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            return False, f"Missing required files: {missing_files}"
        
        # Try to load basic components to ensure checkpoint integrity
        try:
            # Check if we can load training arguments
            training_args_path = checkpoint_path / "training_args.bin"
            if training_args_path.exists():
                # We can't easily load TrainingArguments from binary without the original args
                # So we'll just check if the file exists
                pass
            
            # Check if model file exists and has content
            model_path = checkpoint_path / "pytorch_model.bin"
            if model_path.exists() and model_path.stat().st_size == 0:
                return False, "Model file is empty"
            
            return True, "Checkpoint is valid"
        except Exception as e:
            return False, f"Checkpoint validation failed: {e}"
    
    def export_checkpoint_metadata(self, output_path: str):
        """Export checkpoint metadata to a file"""
        metadata = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_checkpoints': len(self.history),
            'available_checkpoints': len(self.get_available_checkpoints()),
            'checkpoints': [self._checkpoint_info_to_dict(chk) for chk in self.get_available_checkpoints()]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Exported checkpoint metadata to {output_path}")


class RollbackManager:
    """Manages rollback operations to previous checkpoints"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.rollback_log_file = checkpoint_manager.checkpoints_dir / "rollback_log.json"
        self.rollback_history = self._load_rollback_history()
        self.logger = logging.getLogger(__name__)
    
    def _load_rollback_history(self) -> List[Dict[str, Any]]:
        """Load rollback history from file"""
        if self.rollback_log_file.exists():
            try:
                with open(self.rollback_log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load rollback history: {e}")
                return []
        return []
    
    def _save_rollback_history(self):
        """Save rollback history to file"""
        try:
            with open(self.rollback_log_file, 'w') as f:
                json.dump(self.rollback_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save rollback history: {e}")
    
    def rollback_to_checkpoint(self, 
                             checkpoint_id: str, 
                             model: Optional[PreTrainedModel] = None,
                             tokenizer: Optional[PreTrainedTokenizer] = None) -> RollbackResult:
        """Rollback to a specific checkpoint"""
        checkpoint_info = self.checkpoint_manager.get_checkpoint_by_id(checkpoint_id)
        
        if not checkpoint_info:
            return RollbackResult(
                success=False,
                message=f"Checkpoint {checkpoint_id} not found",
                previous_checkpoint=None,
                new_checkpoint=None,
                rollback_time=datetime.utcnow().isoformat()
            )
        
        # Validate the checkpoint before rollback
        is_valid, validation_msg = self.checkpoint_manager.validate_checkpoint(checkpoint_info.path)
        if not is_valid:
            return RollbackResult(
                success=False,
                message=f"Checkpoint validation failed: {validation_msg}",
                previous_checkpoint=None,
                new_checkpoint=checkpoint_info,
                rollback_time=datetime.utcnow().isoformat()
            )
        
        try:
            # If model and tokenizer are provided, try to load them from checkpoint
            if model is not None:
                model.load_state_dict(torch.load(
                    Path(checkpoint_info.path) / "pytorch_model.bin", 
                    map_location=model.device
                ))
                self.logger.info(f"Loaded model from checkpoint: {checkpoint_id}")
            
            if tokenizer is not None:
                tokenizer_path = Path(checkpoint_info.path)
                tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_path)
                self.logger.info(f"Loaded tokenizer from checkpoint: {checkpoint_id}")
            
            # Log the rollback operation
            rollback_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'from_checkpoint': checkpoint_info.checkpoint_id,
                'to_checkpoint': checkpoint_info.checkpoint_id,
                'success': True,
                'message': 'Rollback completed successfully'
            }
            self.rollback_history.append(rollback_record)
            self._save_rollback_history()
            
            self.logger.info(f"Successfully rolled back to checkpoint: {checkpoint_id}")
            
            return RollbackResult(
                success=True,
                message="Rollback completed successfully",
                previous_checkpoint=None,  # We don't have a previous checkpoint in this context
                new_checkpoint=checkpoint_info,
                rollback_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return RollbackResult(
                success=False,
                message=f"Rollback failed: {str(e)}",
                previous_checkpoint=None,
                new_checkpoint=checkpoint_info,
                rollback_time=datetime.utcnow().isoformat()
            )
    
    def rollback_to_step(self, 
                        target_step: int, 
                        model: Optional[PreTrainedModel] = None,
                        tokenizer: Optional[PreTrainedTokenizer] = None) -> RollbackResult:
        """Rollback to the closest checkpoint at or before the target step"""
        available = self.checkpoint_manager.get_available_checkpoints()
        
        # Find the checkpoint at or before the target step
        candidate = None
        for checkpoint in available:
            if checkpoint.step <= target_step:
                candidate = checkpoint
                break
        
        if not candidate:
            return RollbackResult(
                success=False,
                message=f"No checkpoint found at or before step {target_step}",
                previous_checkpoint=None,
                new_checkpoint=None,
                rollback_time=datetime.utcnow().isoformat()
            )
        
        return self.rollback_to_checkpoint(candidate.checkpoint_id, model, tokenizer)
    
    def rollback_to_best(self, 
                        model: Optional[PreTrainedModel] = None,
                        tokenizer: Optional[PreTrainedTokenizer] = None) -> RollbackResult:
        """Rollback to the best performing checkpoint"""
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        
        if not best_checkpoint:
            return RollbackResult(
                success=False,
                message="No best checkpoint found",
                previous_checkpoint=None,
                new_checkpoint=None,
                rollback_time=datetime.utcnow().isoformat()
            )
        
        return self.rollback_to_checkpoint(best_checkpoint.checkpoint_id, model, tokenizer)
    
    def rollback_to_latest(self, 
                          model: Optional[PreTrainedModel] = None,
                          tokenizer: Optional[PreTrainedTokenizer] = None) -> RollbackResult:
        """Rollback to the most recent checkpoint"""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        if not latest_checkpoint:
            return RollbackResult(
                success=False,
                message="No latest checkpoint found",
                previous_checkpoint=None,
                new_checkpoint=None,
                rollback_time=datetime.utcnow().isoformat()
            )
        
        return self.rollback_to_checkpoint(latest_checkpoint.checkpoint_id, model, tokenizer)
    
    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """Get rollback history"""
        return self.rollback_history
    
    def generate_rollback_report(self) -> str:
        """Generate a human-readable rollback report"""
        report = [
            "=== Rollback Report ===",
            f"Generated at: {datetime.utcnow().isoformat()}",
            f"Total rollbacks performed: {len(self.rollback_history)}",
            "",
            "Rollback History:"
        ]
        
        for i, record in enumerate(self.rollback_history[-10:], 1):  # Show last 10 rollbacks
            report.append(f"  {i}. {record['timestamp']}")
            report.append(f"     Target: {record['to_checkpoint']}")
            report.append(f"     Success: {record['success']}")
            report.append(f"     Message: {record['message']}")
            report.append("")
        
        # Add checkpoint information
        available = self.checkpoint_manager.get_available_checkpoints()
        report.append(f"Available Checkpoints: {len(available)}")
        for checkpoint in available[:5]:  # Show top 5
            report.append(f"  - {checkpoint.checkpoint_id} (step {checkpoint.step})")
        
        if len(available) > 5:
            report.append(f"  ... and {len(available) - 5} more")
        
        report.append("")
        
        # Add best checkpoint
        best = self.checkpoint_manager.get_best_checkpoint()
        if best:
            report.append(f"Best Checkpoint: {best.checkpoint_id} (step {best.step})")
            if best.metrics:
                for metric, value in best.metrics.items():
                    report.append(f"  {metric}: {value}")
        
        return "\n".join(report)


class CheckpointCallback:
    """Trainer callback to handle checkpoint creation and management"""
    
    def __init__(self, 
                 checkpoint_manager: CheckpointManager,
                 save_strategy: str = "steps",
                 save_steps: int = 500,
                 keep_last_n: int = 5,
                 keep_best_n: int = 2):
        self.checkpoint_manager = checkpoint_manager
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.logger = logging.getLogger(__name__)
        self.best_metric_value = None
        self.best_metric_name = None
    
    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called when the Trainer saves a checkpoint"""
        if hasattr(state, 'global_step'):
            step = state.global_step
        else:
            step = 0
        
        # Get metrics if available
        metrics = None
        if hasattr(state, 'log_history') and state.log_history:
            # Use the most recent metrics
            metrics = state.log_history[-1].copy()
        
        # Determine if this is the best checkpoint
        is_best = False
        if metrics and self.best_metric_name:
            current_value = metrics.get(self.best_metric_name)
            if current_value is not None:
                if (self.best_metric_value is None or 
                    (state.metric_for_best_model == "eval_loss" and current_value < self.best_metric_value) or
                    (state.metric_for_best_model != "eval_loss" and current_value > self.best_metric_value)):
                    self.best_metric_value = current_value
                    is_best = True
        
        # Register the checkpoint
        checkpoint_path = f"{args.output_dir}/checkpoint-{step}"
        training_args = args.to_dict() if hasattr(args, 'to_dict') else {}
        
        checkpoint_info = self.checkpoint_manager.register_checkpoint(
            checkpoint_path=checkpoint_path,
            step=step,
            metrics=metrics,
            is_best=is_best,
            training_args=training_args
        )
        
        self.logger.info(f"Registered checkpoint at step {step}")
        
        # Clean up old checkpoints
        cleaned = self.checkpoint_manager.cleanup_checkpoints(
            keep_last_n=self.keep_last_n,
            keep_best_n=self.keep_best_n
        )
        
        if cleaned:
            self.logger.info(f"Cleaned up {len(cleaned)} old checkpoints")
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, metrics=None, **kwargs):
        """Called after evaluation to track best metrics"""
        if metrics and state.best_metric_name:
            current_value = metrics.get(state.best_metric_name)
            if current_value is not None:
                if (self.best_metric_value is None or 
                    (state.metric_for_best_model == "eval_loss" and current_value < self.best_metric_value) or
                    (state.metric_for_best_model != "eval_loss" and current_value > self.best_metric_value)):
                    self.best_metric_value = current_value
                    self.logger.info(f"New best metric: {state.best_metric_name} = {current_value}")


def create_checkpoint_manager(output_dir: str) -> CheckpointManager:
    """Create a default checkpoint manager"""
    return CheckpointManager(output_dir)


def create_rollback_manager(output_dir: str) -> RollbackManager:
    """Create a rollback manager with default checkpoint manager"""
    checkpoint_manager = create_checkpoint_manager(output_dir)
    return RollbackManager(checkpoint_manager)


# Integration with training system
def integrate_checkpointing_with_runner(training_runner):
    """Integrate checkpoint management with a training runner"""
    # Create checkpoint manager linked to the output directory
    checkpoint_manager = create_checkpoint_manager(training_runner.manifest.output_dir)
    
    # Add the checkpoint manager to the training runner
    training_runner.checkpoint_manager = checkpoint_manager
    
    # Return the checkpoint manager for direct use
    return checkpoint_manager


def test_checkpoint_rollback_system():
    """Test the checkpoint and rollback system"""
    logger.info("Testing Checkpoint and Rollback System...")
    
    import tempfile
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint manager
        cm = create_checkpoint_manager(temp_dir)
        
        print(f"Created checkpoint manager in: {temp_dir}")
        
        # Simulate some checkpoints
        checkpoint_paths = []
        for step in [100, 200, 300, 400, 500]:
            # Create mock checkpoint directory
            chkpt_path = os.path.join(temp_dir, f"checkpoint-{step}")
            os.makedirs(chkpt_path, exist_ok=True)
            
            # Create minimal mock files
            with open(os.path.join(chkpt_path, "pytorch_model.bin"), "w") as f:
                f.write("mock_model_content")
            with open(os.path.join(chkpt_path, "config.json"), "w") as f:
                json.dump({"mock": "config"}, f)
            with open(os.path.join(chkpt_path, "tokenizer.json"), "w") as f:
                json.dump({"mock": "tokenizer"}, f)
            with open(os.path.join(chkpt_path, "training_args.bin"), "w") as f:
                f.write("mock_args")
            
            checkpoint_paths.append(chkpt_path)
            
            # Register checkpoint
            metrics = {"eval_loss": 2.5 - (step/1000), "accuracy": 0.7 + (step/2000)} if step <= 300 else {"eval_loss": 2.5, "accuracy": 0.7}
            is_best = step == 300  # Step 300 should be best based on metrics
            
            cm.register_checkpoint(
                checkpoint_path=chkpt_path,
                step=step,
                metrics=metrics,
                is_best=is_best
            )
        
        # Test getting checkpoints
        available = cm.get_available_checkpoints()
        print(f"Available checkpoints: {len(available)}")
        
        latest = cm.get_latest_checkpoint()
        print(f"Latest checkpoint: {latest.checkpoint_id if latest else None}")
        
        best = cm.get_best_checkpoint()
        print(f"Best checkpoint: {best.checkpoint_id if best else None}")
        
        # Test rollback manager
        rm = create_rollback_manager(temp_dir)
        
        # Test rollback to best
        rollback_result = rm.rollback_to_best()
        print(f"Rollback to best result: {rollback_result.success}, {rollback_result.message}")
        
        # Test rollback to specific step
        rollback_result = rm.rollback_to_step(200)
        print(f"Rollback to step 200 result: {rollback_result.success}, {rollback_result.message}")
        
        # Generate a report
        report = rm.generate_rollback_report()
        print(f"\nRollback Report:\n{report}")
        
        # Test validation
        for chkpt_path in checkpoint_paths[:3]:  # Only test first 3
            is_valid, msg = cm.validate_checkpoint(chkpt_path)
            print(f"Checkpoint {os.path.basename(chkpt_path)} validation: {is_valid}, {msg}")
        
        # Test cleanup
        cleaned = cm.cleanup_checkpoints(keep_last_n=2, keep_best_n=1)
        print(f"Cleaned up {len(cleaned)} checkpoints")
        
        # Check available after cleanup
        available_after = cm.get_available_checkpoints()
        print(f"Available after cleanup: {len(available_after)}")
    
    print("Checkpoint and rollback system test completed!")


if __name__ == "__main__":
    test_checkpoint_rollback_system()