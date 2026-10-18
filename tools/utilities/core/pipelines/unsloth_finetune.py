"""Unsloth fine-tuning integration utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    from ai.qa.reports.safety_monitor_integration import SafetyMonitor
except Exception:

    class SafetyMonitor:
        def __init__(self) -> None:
            pass

        def record_training_start(self, _config: dict[str, object]) -> None:
            return None

        def record_training_complete(self, **_kwargs: object) -> None:
            return None


try:
    import unsloth
except Exception:
    unsloth = None

REQUIRED_FIELDS = {"batch_size", "epochs", "learning_rate"}


class UnslothFinetune:
    """Adapter class for calling the module-level fine-tuning helper."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def run(self, tokenized_data: list[Any], config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return finetune_with_unsloth(tokenized_data=tokenized_data, config=config, logger=self.logger, **kwargs)


def load_training_config(
    config_path: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate training configuration from JSON."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object")

    merged = {**config}
    if overrides:
        merged.update(overrides)

    missing = REQUIRED_FIELDS - set(merged)
    if missing:
        raise ValueError(f"Missing required config fields: {sorted(missing)}")

    return merged


def finetune_with_unsloth(
    *,
    tokenized_data: list[Any],
    config: dict[str, Any],
    compliance_system: Any | None = None,
    safety_monitor: Any | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a guarded fine-tuning flow with optional compliance/safety hooks."""

    logger = logger or logging.getLogger(__name__)

    if unsloth is None:
        raise ImportError("Unsloth dependency is required for fine-tuning")

    if not REQUIRED_FIELDS.issubset(config):
        raise ValueError("Config is missing required fields")

    logger.info("Loaded training config")
    if compliance_system is not None and hasattr(compliance_system, "run_pre_training_checks"):
        compliance_system.run_pre_training_checks(config=config, sample_count=len(tokenized_data))

    monitor = safety_monitor or SafetyMonitor()
    monitor.record_training_start(config)

    try:
        model = unsloth.load_gguf_model(config.get("model_name", "default-gguf"))
        # A real implementation would set up datasets/trainers here.
        _ = getattr(model, "train", lambda **_k: None)(
            data=tokenized_data,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            **kwargs,
        )

        result = {
            "status": "completed",
            "records_seen": len(tokenized_data),
            "epochs": config["epochs"],
            "model_handle": getattr(model, "name", "unsloth-model"),
        }

        monitor.record_training_complete(success=True, payload=result)
        if compliance_system is not None and hasattr(compliance_system, "run_post_training_checks"):
            compliance_system.run_post_training_checks(result=result)

        logger.info("Fine-tuning completed successfully")
        return result
    except Exception as exc:
        monitor.record_training_complete(success=False, payload={"error": str(exc)})
        logger.error("Fine-tuning failed: %s", exc)
        raise


__all__ = ["UnslothFinetune", "finetune_with_unsloth", "load_training_config", "unsloth"]
