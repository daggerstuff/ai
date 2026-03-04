"""
Core AI engine package for Pixelated Empathy.

Primary exports:
- GestaltEngine: Unified emotional intelligence fusion engine (PIX-149)
- GestaltState: Fused output dataclass
- CrisisLevel: Risk classification enum
"""

from core.gestalt_engine import (
    CrisisLevel,
    GestaltEngine,
    GestaltState,
)

__all__ = [
    "GestaltEngine",
    "GestaltState",
    "CrisisLevel",
]
