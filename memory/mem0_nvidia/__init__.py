"""
NVIDIA NIM Memory Integration Package.

Provides memory management with NVIDIA NIM inference capabilities
for the Pixelated Empathy therapeutic AI platform.

Components:
- manager.py: Original NvidiaMem0Manager with Mem0 integration
- enhanced_manager.py: Enhanced manager with tiered model selection
- memory_ingestion_config.py: Therapeutic memory configuration
"""

from .enhanced_manager import (
    CrisisDetector,
    EmbeddingGenerator,
    EnhancedNvidiaConfig,
    EnhancedNvidiaNimContext,
    EnhancedNvidiaNimManager,
    ModelSelectionStrategy,
    ModelTier,
    TaskComplexity,
    TieredModelSelector,
    create_enhanced_manager,
)
from .manager import (
    NvidiaMem0Config,
    NvidiaMem0Manager,
)

__all__ = [
    # Enhanced manager
    "EnhancedNvidiaConfig",
    "EnhancedNvidiaNimManager",
    "EnhancedNvidiaNimContext",
    "TieredModelSelector",
    "CrisisDetector",
    "EmbeddingGenerator",
    "TaskComplexity",
    "ModelTier",
    "ModelSelectionStrategy",
    "create_enhanced_manager",
    # Original manager
    "NvidiaMem0Config",
    "NvidiaMem0Manager",
]
