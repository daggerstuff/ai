"""Shared local memory service exports only."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BaseMemoryManager": ("ai.memory.base", "BaseMemoryManager"),
    "LocalHindsightMemoryManager": ("ai.memory.local_hindsight_manager", "LocalHindsightMemoryManager"),
    "LocalHindsightRepository": ("ai.memory.local_hindsight_repository", "LocalHindsightRepository"),
    "LocalHindsightSchemaManager": ("ai.memory.local_hindsight_schema", "LocalHindsightSchemaManager"),
    "MemoryManagerFactory": ("ai.memory.manager_factory", "MemoryManagerFactory"),
    "get_required_memory_manager": ("ai.memory.manager_factory", "get_required_memory_manager"),
    # v3 exports (preferred)
    "SubconsciousConfig": ("ai.memory.v3.config", "SubconsciousConfig"),
    "SubconsciousState": ("ai.memory.v3.context", "SubconsciousState"),
    "SubconsciousClient": ("ai.memory.v3.client", "SubconsciousClient"),
    "set_subconscious": ("ai.memory.v3.context", "set_subconscious"),
    "get_subconscious": ("ai.memory.v3.context", "get_subconscious"),
    "reset_subconscious": ("ai.memory.v3.context", "reset_subconscious"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
