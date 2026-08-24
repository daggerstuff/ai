from __future__ import annotations

"""Shared local memory service exports only."""


from importlib import import_module
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from ai.research.base import BaseMemoryManager
    from ai.research.local_foresight_manager import LocalForesightMemoryManager
    from ai.research.local_foresight_repository import LocalForesightRepository
    from ai.research.local_foresight_schema import LocalForesightSchemaManager
    from ai.research.manager_factory import (
        MemoryManagerFactory,
        get_required_memory_manager,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseMemoryManager": (
        "ai.research.base",
        "BaseMemoryManager",
    ),
    "LocalForesightMemoryManager": (
        "ai.research.local_foresight_manager",
        "LocalForesightMemoryManager",
    ),
    "LocalForesightRepository": (
        "ai.research.local_foresight_repository",
        "LocalForesightRepository",
    ),
    "MemoryManagerFactory": (
        "ai.research.manager_factory",
        "MemoryManagerFactory",
    ),
    "get_required_memory_manager": (
        "ai.research.manager_factory",
        "get_required_memory_manager",
    ),
    "LocalForesightSchemaManager": (
        "ai.research.local_foresight_schema",
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
