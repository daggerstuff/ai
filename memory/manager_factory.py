import logging
import os
from typing import Optional

from ai.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager
from ai.memory.mem0_nvidia.manager import NvidiaMem0Config, NvidiaMem0Manager
from ai.memory.mem0_nvidia.enhanced_manager import (
    EnhancedNvidiaConfig,
    EnhancedNvidiaNimManager,
    ModelTier,
    TaskComplexity,
)
from ai.api.memory.base import BaseMemoryManager

logger = logging.getLogger("MemoryManagerFactory")


class MemoryManagerFactory:
    """
    Factory class to create and configure memory managers.
    Supports NVIDIA NIM and Google Gemini providers.
    """

    @staticmethod
    def create_manager() -> BaseMemoryManager:
        """
        Creates a memory manager based on environment configuration.

        Logic:
        1. Check MEM0_PROVIDER env var ('nvidia' or 'gemini').
        2. If not set, check for NVIDIA_API_KEY (defaults to nvidia).
        3. If not set, check for GEMINI_API_KEY (fallback to gemini).
        4. If nothing found, return NullMemoryManager.
        """
        provider = os.environ.get("MEM0_PROVIDER", "").lower()

        # 1. Force Provider if specified
        if provider == "nvidia":
            return MemoryManagerFactory._create_nvidia_manager()
        elif provider == "gemini":
            return MemoryManagerFactory._create_gemini_manager()

        # 2. Autodetect
        if os.environ.get("NVIDIA_API_KEY"):
            return MemoryManagerFactory._create_nvidia_manager()
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return MemoryManagerFactory._create_gemini_manager()

        # 3. Fallback to Null
        logger.warning("No memory provider configured. Using NullMemoryManager.")
        from ai.api.memory.null_memory import NullMemoryManager

        return NullMemoryManager()

    @staticmethod
    def _create_nvidia_manager() -> BaseMemoryManager:
        """
        Helper to create NVIDIA manager.

        Uses EnhancedNvidiaNimManager by default (NVIDIA_USE_ENHANCED=true or unset),
        falling back to basic NvidiaMem0Manager if NVIDIA_USE_ENHANCED=false.

        The enhanced manager provides:
        - Tiered model selection (reasoning/generation/embedding)
        - Crisis detection with safety-first model selection
        - Latency-aware model selection
        - Access to 20 curated models from 187 NVIDIA NIM catalog
        """
        use_enhanced = os.environ.get("NVIDIA_USE_ENHANCED", "true").lower() != "false"

        # Use enhanced manager with tiered model selection
        if use_enhanced:
            try:
                config = EnhancedNvidiaConfig(
                    nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
                    model_tiers={
                        # Reasoning tier: Complex therapeutic conversations
                        "reasoning": os.environ.get(
                            "NVIDIA_REASONING_MODEL", ModelTier.NEMOTRON_SUPER.value
                        ),
                        # Generation tier: Fast responses for real-time chat
                        "generation": os.environ.get(
                            "NVIDIA_GENERATION_MODEL", ModelTier.NEMOTRON_NANO.value
                        ),
                        # Embedding tier: RAG-ready semantic search
                        "embedding": os.environ.get(
                            "NVIDIA_EMBEDDING_MODEL", ModelTier.NEMOTRON_EMBED.value
                        ),
                    },
                    # Map task complexity to model tiers
                    complexity_mapping={
                        TaskComplexity.SIMPLE.value: "generation",
                        TaskComplexity.MODERATE.value: "generation",
                        TaskComplexity.COMPLEX.value: "reasoning",
                        TaskComplexity.CRISIS.value: "reasoning",  # Always use reasoning for crisis
                    },
                    # Therapeutic context settings
                    enable_crisis_detection=True,
                    crisis_detection_threshold=0.7,
                    user_id=os.environ.get("DEFAULT_USER_ID", "default_user"),
                    temperature=float(os.environ.get("NVIDIA_TEMPERATURE", "0.7")),
                    streaming_enabled=os.environ.get("NVIDIA_STREAMING", "true").lower()
                    != "false",
                )
                logger.info(
                    "Using EnhancedNvidiaNimManager with tiered model selection"
                )
                return EnhancedNvidiaNimManager(config)
            except Exception as e:
                logger.error(f"Failed to initialize EnhancedNvidiaNimManager: {e}")
                raise

        # Fallback to basic manager
        try:
            config = NvidiaMem0Config(
                nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
                mem0_api_key=os.environ.get("MEM0_API_KEY"),
                model_name=os.environ.get(
                    "NVIDIA_MODEL_NAME", "meta/llama-3.1-405b-instruct"
                ),
            )
            logger.info("Using basic NvidiaMem0Manager")
            return NvidiaMem0Manager(config)
        except Exception as e:
            logger.error(f"Failed to initialize NvidiaMem0Manager: {e}")
            raise

    @staticmethod
    def _create_gemini_manager() -> BaseMemoryManager:
        """Helper to create Gemini manager."""
        try:
            config = GeminiMem0Config(
                gemini_api_key=os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY"),
                mem0_api_key=os.environ.get("MEM0_API_KEY"),
                model_name=os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro"),
            )
            return GeminiMem0Manager(config)
        except Exception as e:
            logger.error(f"Failed to initialize GeminiMem0Manager: {e}")
            raise


def get_memory_manager() -> BaseMemoryManager:
    """Legacy wrapper for create_manager."""
    return MemoryManagerFactory.create_manager()
