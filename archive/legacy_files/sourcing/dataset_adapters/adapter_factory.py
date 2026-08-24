"""Factory for selecting dataset adapters by name.

Usage:
    from ai.sourcing.dataset_adapters.adapter_factory import get_adapter

    adapter = get_adapter("esconv", "ai/data/raw")
    adapter.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_T = TypeVar("_T", bound=type[BaseDatasetAdapter])

# Registry will be populated as adapters are implemented.
# Each entry: dataset_name -> adapter class
_ADAPTER_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register an adapter class in the factory.

    Usage:
        @register_adapter("esconv")
        class ESConvAdapter(BaseDatasetAdapter):
            ...
    """

    def decorator(cls: _T) -> _T:
        _ADAPTER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_adapter(dataset_name: str, output_dir: str | Path) -> BaseDatasetAdapter:
    """Get an adapter instance by dataset name.

    Args:
        dataset_name: Registered adapter name (case-insensitive).
        output_dir: Root directory for adapter output.

    Returns:
        Adapter instance ready to run().

    Raises:
        ValueError: If no adapter is registered for the given name.
    """
    adapter_class = _ADAPTER_REGISTRY.get(dataset_name.lower())
    if adapter_class is None:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys())) or "(none registered)"
        raise ValueError(f"No adapter for '{dataset_name}'. Available: {available}")
    return adapter_class(dataset_name, output_dir)


def list_available_adapters() -> list[str]:
    """Return sorted list of registered adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())
