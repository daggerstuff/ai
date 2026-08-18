from __future__ import annotations

from dataclasses import dataclass

from .null_memory_command_service import NullMemoryCommandService
from .null_memory_health_service import NullMemoryHealthService
from .null_memory_legacy_service import NullMemoryLegacyService
from .null_memory_protocol_adapter import NullMemoryProtocolAdapter
from .null_memory_query_service import NullMemoryQueryService


@dataclass(frozen=True)
class NullMemoryServiceBundle:
    commands: NullMemoryCommandService
    queries: NullMemoryQueryService
    protocol: NullMemoryProtocolAdapter
    health: NullMemoryHealthService
    legacy: NullMemoryLegacyService
