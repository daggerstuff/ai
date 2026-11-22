"""Lifecycle registry for the models a process has loaded.

The health check needs one place to ask "what is loaded, is it still usable, and
can you release it before shutdown". This module is that place: model loaders
register a handle, the health check reads it, and shutdown unloads everything
through it. Nothing here loads a model by itself; it tracks what was loaded and
releases the references so the memory can be reclaimed.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    """One registered model and the bookkeeping around it."""

    name: str
    model: Any
    adapter: str = "unknown"
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded: bool = True


@dataclass
class UnloadResult:
    """Outcome of an unload call."""

    success: bool
    models_unloaded: int
    failed_models: list[str] = field(default_factory=list)
    error_message: str | None = None


def _describe(model: Any) -> dict[str, Any]:
    """Describe a loaded model by duck-typing, without importing a framework."""
    info: dict[str, Any] = {"model_type": type(model).__name__}

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first = next(iter(parameters()), None)
        except TypeError, RuntimeError, StopIteration:
            first = None
        if first is not None:
            info["device"] = str(getattr(first, "device", "unknown"))
            info["dtype"] = str(getattr(first, "dtype", "unknown"))
            info["parameter_count"] = int(first.numel())

    for attribute in ("config", "name_or_path"):
        if hasattr(model, attribute):
            info[attribute] = str(getattr(model, attribute))[:200]
    return info


class ModelManager:
    """Track registered models and release them on shutdown."""

    def __init__(self) -> None:
        self._handles: dict[str, ModelHandle] = {}

    def register_model(
        self,
        name: str,
        model: Any,
        *,
        adapter: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> ModelHandle:
        """Register a loaded model under ``name``, replacing any previous handle."""
        handle = ModelHandle(
            name=name,
            model=model,
            adapter=adapter,
            metadata=dict(metadata or {}),
        )
        self._handles[name] = handle
        logger.info("registered model %s via %s", name, adapter)
        return handle

    def list_models(self) -> list[str]:
        """Names of every registered model, in registration order."""
        return list(self._handles)

    def get_model(self, name: str) -> Any:
        """The model object itself, or ``None`` when it is not registered."""
        handle = self._handles.get(name)
        return handle.model if handle is not None and handle.loaded else None

    def get_model_info(self, name: str) -> dict[str, Any]:
        """Describe one registered model, including whether it is still loaded."""
        handle = self._handles.get(name)
        if handle is None:
            return {}
        info: dict[str, Any] = {
            "name": handle.name,
            "adapter": handle.adapter,
            "loaded_at": handle.loaded_at,
            "loaded": handle.loaded,
        }
        info.update(handle.metadata)
        info.update(_describe(handle.model))
        return info

    def unload_model(self, name: str) -> UnloadResult:
        """Release one model and drop its handle."""
        handle = self._handles.pop(name, None)
        if handle is None:
            return UnloadResult(success=False, models_unloaded=0, error_message=f"unknown model: {name}")
        handle.loaded = False
        handle.model = None
        logger.info("unloaded model %s", name)
        return UnloadResult(success=True, models_unloaded=1)

    def unload_all_models(self) -> UnloadResult:
        """Release every model and collect the garbage."""
        names = list(self._handles)
        failed: list[str] = []
        unloaded = 0
        for name in names:
            try:
                handle = self._handles.pop(name)
                handle.loaded = False
                handle.model = None
                unloaded += 1
            except (RuntimeError, AttributeError, TypeError, KeyError) as error:
                failed.append(f"{name}: {error}")

        gc.collect()
        return UnloadResult(
            success=not failed,
            models_unloaded=unloaded,
            failed_models=failed,
            error_message="; ".join(failed) if failed else None,
        )

    def clear(self) -> None:
        """Forget every handle without reporting."""
        self._handles.clear()


model_manager = ModelManager()
