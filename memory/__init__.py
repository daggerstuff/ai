from __future__ import annotations

"""Shared local memory service exports only."""


from importlib import import_module
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from ai.memory.base import BaseMemoryManager
    from ai.memory.local_foresight_manager import LocalForesightMemoryManager
    from ai.memory.local_foresight_repository import LocalForesightRepository
    from ai.memory.local_foresight_schema import LocalForesightSchemaManager
    from ai.memory.manager_factory import (
        MemoryManagerFactory,
        get_required_memory_manager,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseMemoryManager": (
        "ai.memory.base",
        "BaseMemoryManager",
    ),
    "LocalForesightMemoryManager": (
        "ai.memory.local_foresight_manager",
        "LocalForesightMemoryManager",
    ),
    "LocalForesightRepository": (
        "ai.memory.local_foresight_repository",
        "LocalForesightRepository",
    ),
    "MemoryManagerFactory": (
        "ai.memory.manager_factory",
        "MemoryManagerFactory",
    ),
    "get_required_memory_manager": (
        "ai.memory.manager_factory",
        "get_required_memory_manager",
    ),
    "LocalForesightSchemaManager": (
        "ai.memory.local_foresight_schema",
        "LocalForesightSchemaManager",
    ),
}

__all__ = [
    "BaseMemoryManager",
    "LocalForesightMemoryManager",
    "LocalForesightRepository",
    "LocalForesightSchemaManager",
    "MemoryManagerFactory",
    "get_required_memory_manager",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
