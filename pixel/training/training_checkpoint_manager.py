"""
Training Checkpoint and Resume Functionality

This module provides comprehensive checkpoint management for training pipelines,
including atomic saves, validation, resumption, and lifecycle management.
"""

import contextlib
import hashlib
import json
import logging
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --- Compliance Redaction for HIPAA/GDPR ---
COMPLIANCE_SENSITIVE_KEYS = ["patient", "name", "dob", "ssn", "address", "phone", "email"]


def redact_sensitive_fields(obj, path=""):
    """
    Recursively redacts sensitive fields in dicts/lists for compliance.
    Returns a tuple: (redacted_obj, found_sensitive: bool)
    Only intended for checkpoint payloads, not for internal validator/result dicts.
    """
    found_sensitive = False
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            if any(s in k.lower() for s in COMPLIANCE_SENSITIVE_KEYS):
                redacted[k] = "[REDACTED]"
                found_sensitive = True
            else:
                v_redacted, v_found = redact_sensitive_fields(v, path + "." + k if path else k)
                redacted[k] = v_redacted
                found_sensitive = found_sensitive or v_found
        return redacted, found_sensitive
    if isinstance(obj, list):
        redacted_list = []
        for idx, item in enumerate(obj):
            item_redacted, item_found = redact_sensitive_fields(item, f"{path}[{idx}]")
            redacted_list.append(item_redacted)
            found_sensitive = found_sensitive or item_found
        return redacted_list, found_sensitive
    return obj, False


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior"""

    # Save intervals
    save_every_steps: int = 1000
    save_every_epochs: int = 1
    save_every_minutes: int = 30

    # Retention policies
    keep_last_n_checkpoints: int = 5
    keep_best_n_checkpoints: int = 3
    keep_every_n_epochs: int = 10

    # Storage settings
    checkpoint_dir: str = "checkpoints"
    compress_checkpoints: bool = True
    atomic_saves: bool = True
    validate_on_save: bool = True

    # Best model tracking
    best_metric_name: str = "validation_loss"
    best_metric_mode: str = "min"  # "min" or "max"

    # Metadata
    save_training_config: bool = True
    save_model_code_hash: bool = True
    save_git_commit: bool = True

    # Performance
    async_saves: bool = True
    max_concurrent_saves: int = 2
    save_timeout: int = 600  # seconds


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint"""

    checkpoint_id: str
    timestamp: datetime
    step: int
    epoch: int

    # Training metrics
    loss: float
    validation_loss: float | None = None
    learning_rate: float = 0.0

    # Multi-objective metrics
    language_loss: float | None = None
    eq_loss: float | None = None
    persona_loss: float | None = None
    clinical_loss: float | None = None
    empathy_loss: float | None = None

    # Performance metrics
    training_time_hours: float = 0.0
    samples_processed: int = 0
    tokens_processed: int = 0

    # Validation metrics
    eq_score: float | None = None
    clinical_accuracy: float | None = None
    persona_accuracy: float | None = None
    empathy_score: float | None = None

    # System info
    model_hash: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    gpu_memory_used: float | None = None

    # File info
    file_size_mb: float = 0.0
    is_best: bool = False
    is_corrupted: bool = False

    # Custom metadata
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """Complete training state for checkpointing"""

    # Core training state
    step: int
    epoch: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None = None

    # Loss tracking
    loss_history: list[float] = field(default_factory=list)
    validation_history: list[float] = field(default_factory=list)

    # Multi-objective loss states
    loss_weights: dict[str, float] | None = None
    gradient_norms: dict[str, float] | None = None

    # Random states for reproducibility
    torch_rng_state: torch.Tensor | None = None
    numpy_rng_state: dict[str, Any] | None = None
    python_rng_state: Any | None = None
    cuda_rng_state: torch.Tensor | None = None

    # Training configuration
    training_config: dict[str, Any] | None = None
    model_config: dict[str, Any] | None = None

    # Performance tracking
    training_start_time: datetime | None = None
    last_checkpoint_time: datetime | None = None

    # Custom state
    custom_state: dict[str, Any] = field(default_factory=dict)


class CheckpointValidator:
    """Validates checkpoint integrity and completeness"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _ensure_list(val) -> list[str]:
        """Ensure the value is a list of strings (for errors/warnings fields)."""
        if isinstance(val, list):
            return [str(e) for e in val]
        if val is None:
            return []
        return [str(val)]

    def _invalidate_with_error(self, results: dict[str, Any], error_msg: str) -> dict[str, Any]:
        results["is_valid"] = False
        results["errors"] = self._ensure_list(results.get("errors"))
        results["errors"] = list(results["errors"])  # type: ignore
        results["errors"].append(error_msg)
        return results

    def validate_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """
        Validate a checkpoint file for integrity and completeness

        Returns:
            Dict with validation results and any errors found
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_size_mb": 0.0,
            "contains_model": False,
            "contains_optimizer": False,
            "contains_metadata": False,
            "hash_valid": False,
        }
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # File existence and size
            file_ok, file_size, file_err = self._check_file_exists_and_size(checkpoint_path)
            results["file_size_mb"] = file_size
            if not file_ok:
                errors.append(file_err)
                results["is_valid"] = False
                results["errors"] = errors
                results["warnings"] = warnings
                self.logger.info(
                    f"validate_checkpoint: is_valid={results.get('is_valid')!r}, errors={results.get('errors')!r}, warnings={results.get('warnings')!r}"
                )
                return results

            # Load checkpoint
            checkpoint, load_err = self._load_checkpoint_file(checkpoint_path)
            if checkpoint is None:
                errors.append(load_err)
                results["is_valid"] = False
                results["errors"] = errors
                results["warnings"] = warnings
                self.logger.info(
                    f"validate_checkpoint: is_valid={results.get('is_valid')!r}, errors={results.get('errors')!r}, warnings={results.get('warnings')!r}"
                )
                return results

            # Validate training state
            self._validate_training_state(checkpoint, results, errors, warnings)

            # Validate metadata
            self._validate_metadata(checkpoint, results, warnings)

            # Validate file hash
            self._validate_file_hash(checkpoint, results, errors)

            # Final validation
            if errors:
                results["is_valid"] = False
            results["errors"] = errors
            results["warnings"] = warnings

        except Exception as e:
            self._invalidate_with_error(results, f"Validation failed with error: {e!s}")
            self.logger.error(f"Checkpoint validation error: {e!s}")

        self.logger.info(
            f"validate_checkpoint: is_valid={results.get('is_valid')!r}, errors={results.get('errors')!r}, warnings={results.get('warnings')!r}"
        )
        return results

    def _check_file_exists_and_size(self, checkpoint_path: Path) -> tuple[bool, float, str]:
        if not checkpoint_path.exists():
            return False, 0.0, "Checkpoint file does not exist"
        file_size = checkpoint_path.stat().st_size
        if file_size == 0:
            return False, 0.0, "Checkpoint file is empty"
        return True, file_size / (1024 * 1024), ""

    def _load_checkpoint_file(self, checkpoint_path: Path) -> tuple[Any, str]:
        try:
            torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
            if torch_version >= (2, 1):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            return checkpoint, ""
        except Exception as e:
            return None, f"Failed to load checkpoint: {e!s}"

    def _validate_training_state(
        self, checkpoint: dict, results: dict, errors: list, warnings: list
    ) -> None:
        if "training_state" in checkpoint:
            training_state = checkpoint["training_state"]

            if "model_state_dict" in training_state:
                results["contains_model"] = True
            else:
                errors.append("Missing model state dict")

            if "optimizer_state_dict" in training_state:
                results["contains_optimizer"] = True
            else:
                warnings.append("Missing optimizer state dict")

            if results["contains_model"]:
                model_state = training_state["model_state_dict"]
                if not isinstance(model_state, dict) or len(model_state) == 0:
                    errors.append("Model state dict is invalid or empty")
        else:
            errors.append("Missing training state")

    def _validate_metadata(self, checkpoint: dict, results: dict, warnings: list) -> None:
        if "metadata" in checkpoint:
            results["contains_metadata"] = True
            metadata = checkpoint["metadata"]
            required_fields = ["checkpoint_id", "timestamp", "step", "epoch"]
            for req_field in required_fields:
                if req_field not in metadata:
                    warnings.append(f"Missing metadata field: {req_field}")
        else:
            warnings.append("Missing metadata")

    def _validate_file_hash(self, checkpoint: dict, results: dict, errors: list) -> None:
        if "file_hash" in checkpoint:
            stored_hash = checkpoint["file_hash"]
            temp_checkpoint = {k: v for k, v in checkpoint.items() if k != "file_hash"}
            current_hash = self._calculate_checkpoint_hash(temp_checkpoint)
            if stored_hash == current_hash:
                results["hash_valid"] = True
            else:
                errors.append("Checkpoint hash mismatch - file may be corrupted")

    def _calculate_checkpoint_hash(self, checkpoint_data: dict[str, Any]) -> str:
        """Calculate hash of checkpoint data"""
        try:
            # Convert to string representation for hashing
            checkpoint_str = json.dumps(checkpoint_data, sort_keys=True, default=str)
            return hashlib.sha256(checkpoint_str.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checkpoint hash: {e!s}")
            return ""


class CheckpointManager:
    """Main checkpoint management system"""

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = CheckpointValidator()

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Threading for async saves
        self.save_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_saves)

        # Tracking
        self.best_metrics: dict[str, float] = {}
        self.checkpoint_history: list[CheckpointMetadata] = []
        self.last_save_time = time.time()

        # Load existing checkpoint history
        self._load_checkpoint_history()

        self.logger.info(f"CheckpointManager initialized with directory: {self.checkpoint_dir!r}")

    def should_save_checkpoint(self, step: int, epoch: int) -> bool:
        """Determine if a checkpoint should be saved"""
        current_time = time.time()

        # Ensure step is greater than 0 before saving (prevents saving at step 0)
        if not step > 0:
            return False

        # Check step interval
        if self.config.save_every_steps > 0 and step % self.config.save_every_steps == 0:
            return True

        # Check epoch interval (only save at the end of epochs, not at step-based intervals within epochs)
        # We check if this is truly an epoch boundary by seeing if epoch changed
        if self.config.save_every_epochs > 0:
            # Initialize _last_epoch if not set
            if not hasattr(self, "_last_epoch"):
                self._last_epoch = 0

            # Check if epoch changed and if it's a multiple of save_every_epochs
            if epoch != self._last_epoch:
                result = epoch % self.config.save_every_epochs == 0
                self._last_epoch = epoch
                if result:
                    return True

        # Check time interval
        if self.config.save_every_minutes > 0:
            minutes_since_last = (current_time - self.last_save_time) / 60
            if minutes_since_last >= self.config.save_every_minutes:
                return True

        return False

    def save_checkpoint(
        self,
        training_state: TrainingState,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        custom_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a training checkpoint

        Returns:
            Checkpoint ID for the saved checkpoint
        """
        checkpoint_id = self._generate_checkpoint_id(training_state.step, training_state.epoch)

        # Create metadata
        metadata = self._create_metadata(
            checkpoint_id=checkpoint_id,
            training_state=training_state,
            metrics=metrics or {},
            is_best=is_best,
            custom_metadata=custom_metadata or {},
        )

        if self.config.async_saves:
            self.executor.submit(
                self._save_checkpoint_sync, checkpoint_id, training_state, metadata
            )
            # Don't wait for async save to complete
        else:
            self._save_checkpoint_sync(checkpoint_id, training_state, metadata)

        self.last_save_time = time.time()
        self.logger.info(f"Checkpoint save initiated: {checkpoint_id!r}")

        return checkpoint_id

    def _generate_checkpoint_id(self, step: int, epoch: int) -> str:
        """Generate a unique checkpoint ID based on step, epoch, and timestamp."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
        return f"ckpt-step{step}-epoch{epoch}-{timestamp}"

    def _create_metadata(
        self,
        checkpoint_id: str,
        training_state: TrainingState,
        metrics: dict[str, float],
        is_best: bool,
        custom_metadata: dict[str, Any],
    ) -> CheckpointMetadata:
        """Create CheckpointMetadata for a checkpoint save."""
        now = datetime.now(timezone.utc)  # noqa: UP017
        loss = metrics.get("loss", 0.0)
        validation_loss = metrics.get("validation_loss")
        learning_rate = metrics.get("learning_rate", 0.0)
        return CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=now,
            step=training_state.step,
            epoch=training_state.epoch,
            loss=loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            language_loss=metrics.get("language_loss"),
            eq_loss=metrics.get("eq_loss"),
            persona_loss=metrics.get("persona_loss"),
            clinical_loss=metrics.get("clinical_loss"),
            empathy_loss=metrics.get("empathy_loss"),
            training_time_hours=metrics.get("training_time_hours", 0.0),
            samples_processed=int(metrics.get("samples_processed", 0)),
            tokens_processed=int(metrics.get("tokens_processed", 0)),
            eq_score=metrics.get("eq_score"),
            clinical_accuracy=metrics.get("clinical_accuracy"),
            persona_accuracy=metrics.get("persona_accuracy"),
            empathy_score=metrics.get("empathy_score"),
            model_hash=(
                str(metrics["model_hash"])
                if "model_hash" in metrics and metrics["model_hash"] is not None
                else None
            ),
            config_hash=(
                str(metrics["config_hash"])
                if "config_hash" in metrics and metrics["config_hash"] is not None
                else None
            ),
            git_commit=(
                str(metrics["git_commit"])
                if "git_commit" in metrics and metrics["git_commit"] is not None
                else None
            ),
            gpu_memory_used=metrics.get("gpu_memory_used"),
            file_size_mb=0.0,
            is_best=is_best,
            is_corrupted=False,
            custom_metrics=custom_metadata,
        )

    def _save_checkpoint_sync(
        self,
        checkpoint_id: str,
        training_state: TrainingState,
        metadata: CheckpointMetadata,
    ) -> None:
        """Synchronous checkpoint save with atomic operations"""
        try:
            with self.save_lock:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
                temp_path = self.checkpoint_dir / f"{checkpoint_id}.pt.tmp"

                # Prepare checkpoint data
                checkpoint_data = {
                    "training_state": asdict(training_state),
                    "metadata": asdict(metadata),
                    "save_timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                    "config": asdict(self.config),
                }

                # --- Compliance Redaction ---
                checkpoint_data_redacted, found_sensitive = redact_sensitive_fields(checkpoint_data)
                if found_sensitive:
                    self.logger.warning(
                        "[COMPLIANCE] Sensitive fields detected and redacted in checkpoint data before save."
                    )
                # Ensure redacted data is a dict before adding file_hash
                if not isinstance(checkpoint_data_redacted, dict):
                    self.logger.error(
                        "[COMPLIANCE] Redacted checkpoint data is not a dict. Aborting checkpoint save for compliance."
                    )
                    raise RuntimeError("Redacted checkpoint data is not a dict.")

                # Add file hash for integrity checking
                checkpoint_data_redacted["file_hash"] = self.validator._calculate_checkpoint_hash(
                    {k: v for k, v in checkpoint_data_redacted.items() if k != "file_hash"}
                )

                # Atomic save: write to temp file first
                if self.config.atomic_saves:
                    torch.save(checkpoint_data_redacted, temp_path)

                    # Validate temp file if enabled
                    if self.config.validate_on_save:
                        validation_results = self.validator.validate_checkpoint(temp_path)
                        if not validation_results["is_valid"]:
                            temp_path.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"Checkpoint validation failed: {validation_results['errors']}"
                            )

                    # Atomic move
                    shutil.move(str(temp_path), str(checkpoint_path))
                else:
                    torch.save(checkpoint_data_redacted, checkpoint_path)

                # Update metadata with file size
                metadata.file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

                # Add to history
                self.checkpoint_history.append(metadata)

                # Update best metrics if this is a best checkpoint
                if metadata.is_best and metadata.validation_loss is not None:
                    self.best_metrics[self.config.best_metric_name] = metadata.validation_loss

                # Clean up old checkpoints
                self._cleanup_old_checkpoints()

                # Save checkpoint history
                self._save_checkpoint_history()

                self.logger.info(
                    f"Checkpoint saved successfully: {checkpoint_id!r} ({metadata.file_size_mb:.1f} MB)"
                )

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e!s}")
            # Clean up temp file if it exists
            temp_path = self.checkpoint_dir / f"{checkpoint_id}.pt.tmp"
            temp_path.unlink(missing_ok=True)
            raise

    def load_checkpoint(self, checkpoint_id: str | None = None) -> TrainingState | None:
        """
        Load a checkpoint by ID or latest if no ID provided

        Returns:
            TrainingState if successful, None if failed
        """
        if checkpoint_id is None:
            checkpoint_id = self.get_latest_checkpoint_id()
        if checkpoint_id is None:
            self.logger.info("No checkpoints found to load")
            return None

        try:
            return self._load_checkpoint_data(checkpoint_id)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e!s}")
            return None

    def _load_checkpoint_data(self, checkpoint_id: str) -> TrainingState | None:
        """Load and validate checkpoint data"""
        checkpoint_path = self._get_checkpoint_path_for_loading(checkpoint_id)

        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None

        if not self._validate_and_log_checkpoint(checkpoint_path):
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        training_state_dict = checkpoint["training_state"]
        training_state = TrainingState(**training_state_dict)

        self.logger.info(f"Checkpoint loaded successfully: {checkpoint_id!r}")
        self.logger.info(
            f"Resuming from step {training_state.step!r}, epoch {training_state.epoch!r}"
        )

        return training_state

    def _validate_and_log_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint and log errors if invalid"""
        validation_results = self.validator.validate_checkpoint(checkpoint_path)
        if not validation_results["is_valid"]:
            self.logger.error(f"Checkpoint validation failed: {validation_results['errors']}")
            return False
        return True

    def _get_checkpoint_path_for_loading(self, checkpoint_id: str) -> Path:
        """Get the full path to a checkpoint file given its ID (extracted for clarity)"""
        return self._get_checkpoint_path(checkpoint_id)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the full path to a checkpoint file given its ID"""
        return self.checkpoint_dir / f"{checkpoint_id}.pt"

    def get_latest_checkpoint_id(self) -> str | None:
        """Get the ID of the most recent checkpoint"""
        if not self.checkpoint_history:
            # Try to discover checkpoints from filesystem
            checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
            if not checkpoint_files:
                return None

            # Get most recent by modification time
            latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            return latest_file.stem

        # Get latest from history
        latest_checkpoint = max(self.checkpoint_history, key=lambda c: c.timestamp)
        return latest_checkpoint.checkpoint_id

    def get_best_checkpoint_id(self, metric_name: str | None = None) -> str | None:
        """Get the ID of the best checkpoint based on specified metric"""
        metric_name = metric_name or self.config.best_metric_name
        mode = self.config.best_metric_mode

        best_checkpoints = [c for c in self.checkpoint_history if c.is_best]
        if not best_checkpoints:
            return None

        if mode == "min":
            best_checkpoint = min(
                best_checkpoints, key=lambda c: getattr(c, metric_name, float("inf"))
            )
        else:
            best_checkpoint = max(
                best_checkpoints, key=lambda c: getattr(c, metric_name, float("-inf"))
            )

        return best_checkpoint.checkpoint_id

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints"""
        return sorted(self.checkpoint_history, key=lambda c: c.timestamp, reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # Remove from history
            self.checkpoint_history = [
                c for c in self.checkpoint_history if c.checkpoint_id != checkpoint_id
            ]
            self._save_checkpoint_history()

            self.logger.info(f"Checkpoint deleted: {checkpoint_id!r}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e!s}")
            return False

    def _get_gpu_memory_usage(self) -> float | None:
        """Get current GPU memory usage in GB"""
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        return None

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policies"""
        try:
            # Sort checkpoints by timestamp
            sorted_checkpoints = sorted(
                self.checkpoint_history, key=lambda c: c.timestamp, reverse=True
            )

            # Identify checkpoints to keep
            keep_checkpoints: set[str] = set()

            # Keep last N checkpoints
            if self.config.keep_last_n_checkpoints > 0:
                keep_checkpoints.update(
                    c.checkpoint_id
                    for c in sorted_checkpoints[: self.config.keep_last_n_checkpoints]
                )

            # Keep best N checkpoints
            if self.config.keep_best_n_checkpoints > 0:
                best_checkpoints = [c for c in sorted_checkpoints if c.is_best]
                keep_checkpoints.update(
                    c.checkpoint_id for c in best_checkpoints[: self.config.keep_best_n_checkpoints]
                )

            # Keep every N epochs
            if self.config.keep_every_n_epochs > 0:
                epoch_checkpoints = [
                    c for c in sorted_checkpoints if c.epoch % self.config.keep_every_n_epochs == 0
                ]
                keep_checkpoints.update(c.checkpoint_id for c in epoch_checkpoints)

            # Delete checkpoints not in keep list
            for checkpoint in sorted_checkpoints:
                if checkpoint.checkpoint_id not in keep_checkpoints:
                    self.delete_checkpoint(checkpoint.checkpoint_id)

        except Exception as e:
            self.logger.error(f"Checkpoint cleanup failed: {e!s}")

    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to disk"""
        try:
            history_path = self.checkpoint_dir / "checkpoint_history.json"
            history_data = [asdict(c) for c in self.checkpoint_history]

            # Convert datetime objects to strings
            for checkpoint_data in history_data:
                if "timestamp" in checkpoint_data and isinstance(
                    checkpoint_data["timestamp"], datetime
                ):
                    checkpoint_data["timestamp"] = checkpoint_data["timestamp"].isoformat()

            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint history: {e!s}")

    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from disk"""
        try:
            history_path = self.checkpoint_dir / "checkpoint_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history_data = json.load(f)

                # Convert back to CheckpointMetadata objects
                for checkpoint_data in history_data:
                    if "timestamp" in checkpoint_data and isinstance(
                        checkpoint_data["timestamp"], str
                    ):
                        checkpoint_data["timestamp"] = datetime.fromisoformat(
                            checkpoint_data["timestamp"]
                        )

                self.checkpoint_history = [CheckpointMetadata(**data) for data in history_data]

                self.logger.info(
                    f"Loaded {len(self.checkpoint_history)!r} checkpoints from history"
                )

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint history: {e!s}")
            self.checkpoint_history = []

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during checkpoint manager cleanup: {e!s}")


# Utility functions for integration with training pipelines


def create_training_state_from_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    **kwargs,
) -> TrainingState:
    """Create TrainingState from training components"""
    return TrainingState(
        step=step,
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler else None,
        torch_rng_state=torch.get_rng_state(),
        numpy_rng_state={
            "state": np.random.get_state(),
        },
        python_rng_state=random.getstate(),
        cuda_rng_state=(torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
        **kwargs,
    )


def restore_training_state_to_model(
    training_state: TrainingState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
) -> None:
    """Restore TrainingState to training components"""

    # Restore model and optimizer
    model.load_state_dict(training_state.model_state_dict)
    optimizer.load_state_dict(training_state.optimizer_state_dict)

    if scheduler and training_state.scheduler_state_dict:
        scheduler.load_state_dict(training_state.scheduler_state_dict)

    # Restore random states for reproducibility
    if training_state.torch_rng_state is not None:
        torch.set_rng_state(training_state.torch_rng_state)

    if training_state.numpy_rng_state is not None:
        np.random.set_state(training_state.numpy_rng_state["state"])

    if training_state.python_rng_state is not None:
        random.setstate(training_state.python_rng_state)

    if training_state.cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(training_state.cuda_rng_state)
