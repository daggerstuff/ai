"""
Enhanced NVIDIA NIM Memory Manager with Tiered Model Selection.

This module provides an enhanced memory manager that leverages NVIDIA's
Nemotron model family with intelligent model selection based on task complexity:

- Nemotron-Super-49B: Complex reasoning, crisis detection, nuanced therapeutic responses
- Nemotron-Nano-12B: Fast responses, simple queries, real-time chat
- Nemotron-Embed-VL: 2048-dimension multimodal embeddings for RAG

Implements Phase 1 requirements from the NVIDIA integration roadmap.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, field_validator

from .rate_limiter import NvidiaRateLimiter, SemanticCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_nvidia_nim")

DEFAULT_ROUTING_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""

    SIMPLE = "simple"  # Quick queries, greetings, simple Q&A
    MODERATE = "moderate"  # Standard therapeutic conversation
    COMPLEX = "complex"  # Nuanced reasoning, multi-turn context
    CRISIS = "crisis"  # Safety-critical, requires highest accuracy


class ModelTier(Enum):
    """Available model tiers from NVIDIA NIM - curated selection from 187 available models."""

    # === REASONING TIER (Complex therapeutic conversations) ===
    NEMOTRON_SUPER = "nvidia/llama-3.3-nemotron-super-49b-v1.5"  # NVIDIA's flagship
    DEEPSEEK_V3 = "deepseek-ai/deepseek-v3.2"  # Advanced reasoning
    LLAMA_33_70B = "meta/llama-3.3-70b-instruct"  # Meta's latest
    LLAMA4_MAVERICK = "meta/llama-4-maverick-17b-128e-instruct"  # Llama 4
    QWEN3_NEXT = "qwen/qwen3-next-80b-a3b-instruct"  # Multilingual reasoning
    MISTRAL_LARGE = "mistralai/mistral-large-3-675b-instruct-2512"  # Massive model

    # === BALANCED TIER (General therapeutic interactions) ===
    NEMOTRON_NANO = "nvidia/llama-3.1-nemotron-nano-8b-v1"  # NVIDIA optimized
    LLAMA_31_70B = "meta/llama-3.1-70b-instruct"  # Proven performer
    QWEN_25_CODER = "qwen/qwen2.5-coder-32b-instruct"  # Good for structured output

    # === FAST TIER (Quick responses, high throughput) ===
    PHI4_MINI = "microsoft/phi-4-mini-instruct"  # Ultra-fast, excellent quality
    LLAMA_32_1B = "meta/llama-3.2-1b-instruct"  # Smallest Llama
    GEMMA_3_1B = "google/gemma-3-1b-it"  # Google's lightweight
    NEMOTRON_NANO_4B = "nvidia/llama-3.1-nemotron-nano-4b-v1.1"  # Smaller NVIDIA model

    # === SAFETY TIER (Crisis detection, content moderation) ===
    NEMOTRON_SAFETY = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"  # Safety-focused
    NEMOGUARD_CONTENT = "nvidia/llama-3.1-nemoguard-8b-content-safety"  # Content safety
    LLAMA_GUARD_4 = "meta/llama-guard-4-12b"  # Meta's safety model

    # === MULTILINGUAL TIER (Diverse populations) ===
    QWEN_35_LARGE = "qwen/qwen3.5-397b-a17b"  # Excellent multilingual
    MISTRAL_SMALL = "mistralai/mistral-small-4-119b-2603"  # Multilingual

    # === EMBEDDING TIER ===
    NEMOTRON_EMBED = "nvidia/llama-nemotron-embed-vl-1b-v2"  # Multimodal embeddings
    BGE_M3 = "baai/bge-m3"  # Multilingual embeddings


class ModelSelectionStrategy(Enum):
    """Strategies for model selection."""

    LATENCY_OPTIMIZED = "latency_optimized"  # Prefer fast models
    QUALITY_OPTIMIZED = "quality_optimized"  # Prefer accurate models
    COST_OPTIMIZED = "cost_optimized"  # Balance cost/performance
    CRISIS_AWARE = "crisis_aware"  # Safety-first for crisis detection


@dataclass
class LatencyRequirements:
    """Latency requirements for model selection."""

    max_response_ms: int = 500  # Target max response time
    p95_target_ms: int = 2000  # P95 latency target
    streaming_first_token_ms: int = 1000  # Time to first token for streaming


@dataclass
class ModelCapabilities:
    """Capabilities of each model."""

    model_id: str
    context_window: int
    typical_latency_ms: int
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_tools: bool = True
    cost_per_1k_tokens: float = 0.0
    best_for: list[str] = field(default_factory=list)


# Model capability registry - Curated from 187 available NVIDIA NIM models
MODEL_REGISTRY: dict[str, ModelCapabilities] = {
    # === REASONING TIER ===
    ModelTier.NEMOTRON_SUPER.value: ModelCapabilities(
        model_id=ModelTier.NEMOTRON_SUPER.value,
        context_window=131072,
        typical_latency_ms=1500,
        supports_streaming=True,
        cost_per_1k_tokens=0.0005,
        best_for=["complex_reasoning", "crisis_detection", "nuanced_responses", "therapeutic_conversations"],
    ),
    ModelTier.DEEPSEEK_V3.value: ModelCapabilities(
        model_id=ModelTier.DEEPSEEK_V3.value,
        context_window=128000,
        typical_latency_ms=1200,
        supports_streaming=True,
        cost_per_1k_tokens=0.0003,
        best_for=["advanced_reasoning", "problem_solving", "therapeutic_insights"],
    ),
    ModelTier.LLAMA_33_70B.value: ModelCapabilities(
        model_id=ModelTier.LLAMA_33_70B.value,
        context_window=131072,
        typical_latency_ms=1000,
        supports_streaming=True,
        cost_per_1k_tokens=0.0004,
        best_for=["general_reasoning", "multilingual_support", "balanced_performance"],
    ),
    ModelTier.LLAMA4_MAVERICK.value: ModelCapabilities(
        model_id=ModelTier.LLAMA4_MAVERICK.value,
        context_window=262144,
        typical_latency_ms=1800,
        supports_streaming=True,
        supports_vision=True,
        cost_per_1k_tokens=0.0006,
        best_for=["vision_reasoning", "multimodal_therapy", "complex_dialogue"],
    ),
    ModelTier.QWEN3_NEXT.value: ModelCapabilities(
        model_id=ModelTier.QWEN3_NEXT.value,
        context_window=131072,
        typical_latency_ms=1400,
        supports_streaming=True,
        cost_per_1k_tokens=0.0004,
        best_for=["multilingual_reasoning", "diverse_populations", "cultural_sensitivity"],
    ),
    ModelTier.MISTRAL_LARGE.value: ModelCapabilities(
        model_id=ModelTier.MISTRAL_LARGE.value,
        context_window=128000,
        typical_latency_ms=2000,
        supports_streaming=True,
        cost_per_1k_tokens=0.0008,
        best_for=["highest_complexity", "research_grade", "nuanced_understanding"],
    ),
    # === BALANCED TIER ===
    ModelTier.NEMOTRON_NANO.value: ModelCapabilities(
        model_id=ModelTier.NEMOTRON_NANO.value,
        context_window=128000,
        typical_latency_ms=300,
        supports_streaming=True,
        cost_per_1k_tokens=0.0001,
        best_for=["quick_responses", "real_time_chat", "simple_queries", "follow_ups"],
    ),
    ModelTier.LLAMA_31_70B.value: ModelCapabilities(
        model_id=ModelTier.LLAMA_31_70B.value,
        context_window=131072,
        typical_latency_ms=800,
        supports_streaming=True,
        cost_per_1k_tokens=0.0003,
        best_for=["balanced_reasoning", "general_therapy", "proven_reliability"],
    ),
    ModelTier.QWEN_25_CODER.value: ModelCapabilities(
        model_id=ModelTier.QWEN_25_CODER.value,
        context_window=32768,
        typical_latency_ms=400,
        supports_streaming=True,
        cost_per_1k_tokens=0.0002,
        best_for=["structured_output", "json_responses", "assessment_scoring"],
    ),
    # === FAST TIER ===
    ModelTier.PHI4_MINI.value: ModelCapabilities(
        model_id=ModelTier.PHI4_MINI.value,
        context_window=128000,
        typical_latency_ms=150,
        supports_streaming=True,
        cost_per_1k_tokens=0.00005,
        best_for=["ultra_fast_responses", "high_throughput", "acknowledgments"],
    ),
    ModelTier.LLAMA_32_1B.value: ModelCapabilities(
        model_id=ModelTier.LLAMA_32_1B.value,
        context_window=131072,
        typical_latency_ms=100,
        supports_streaming=True,
        cost_per_1k_tokens=0.00002,
        best_for=["instant_responses", "simple_tasks", "minimal_latency"],
    ),
    ModelTier.GEMMA_3_1B.value: ModelCapabilities(
        model_id=ModelTier.GEMMA_3_1B.value,
        context_window=8192,
        typical_latency_ms=120,
        supports_streaming=True,
        cost_per_1k_tokens=0.00002,
        best_for=["quick_acknowledgments", "simple_classification", "speed_critical"],
    ),
    ModelTier.NEMOTRON_NANO_4B.value: ModelCapabilities(
        model_id=ModelTier.NEMOTRON_NANO_4B.value,
        context_window=128000,
        typical_latency_ms=200,
        supports_streaming=True,
        cost_per_1k_tokens=0.00003,
        best_for=["fast_nvidia_optimized", "balanced_speed_quality", "quick_turnaround"],
    ),
    # === SAFETY TIER ===
    ModelTier.NEMOTRON_SAFETY.value: ModelCapabilities(
        model_id=ModelTier.NEMOTRON_SAFETY.value,
        context_window=128000,
        typical_latency_ms=400,
        supports_streaming=True,
        cost_per_1k_tokens=0.0001,
        best_for=["crisis_detection", "harm_prevention", "safety_checking", "guardrails"],
    ),
    ModelTier.NEMOGUARD_CONTENT.value: ModelCapabilities(
        model_id=ModelTier.NEMOGUARD_CONTENT.value,
        context_window=128000,
        typical_latency_ms=350,
        supports_streaming=True,
        cost_per_1k_tokens=0.0001,
        best_for=["content_safety", "inappropriate_content", "boundary_enforcement"],
    ),
    ModelTier.LLAMA_GUARD_4.value: ModelCapabilities(
        model_id=ModelTier.LLAMA_GUARD_4.value,
        context_window=131072,
        typical_latency_ms=500,
        supports_streaming=True,
        cost_per_1k_tokens=0.00015,
        best_for=["meta_safety", "comprehensive_guard", "multi_category_safety"],
    ),
    # === MULTILINGUAL TIER ===
    ModelTier.QWEN_35_LARGE.value: ModelCapabilities(
        model_id=ModelTier.QWEN_35_LARGE.value,
        context_window=131072,
        typical_latency_ms=1600,
        supports_streaming=True,
        cost_per_1k_tokens=0.0005,
        best_for=["excellent_multilingual", "diverse_cultures", "global_reach"],
    ),
    ModelTier.MISTRAL_SMALL.value: ModelCapabilities(
        model_id=ModelTier.MISTRAL_SMALL.value,
        context_window=128000,
        typical_latency_ms=600,
        supports_streaming=True,
        cost_per_1k_tokens=0.0002,
        best_for=["balanced_multilingual", "efficient_translation", "cross_cultural"],
    ),
    # === EMBEDDING TIER ===
    ModelTier.NEMOTRON_EMBED.value: ModelCapabilities(
        model_id=ModelTier.NEMOTRON_EMBED.value,
        context_window=8192,
        typical_latency_ms=100,
        supports_streaming=False,
        supports_vision=True,
        cost_per_1k_tokens=0.00002,
        best_for=["multimodal_embeddings", "semantic_search", "rag", "similarity"],
    ),
    ModelTier.BGE_M3.value: ModelCapabilities(
        model_id=ModelTier.BGE_M3.value,
        context_window=8192,
        typical_latency_ms=80,
        supports_streaming=False,
        cost_per_1k_tokens=0.00001,
        best_for=["multilingual_embeddings", "cross_lingual_search", "diverse_languages"],
    ),
}


class EnhancedNvidiaConfig(BaseModel):
    """
    Enhanced configuration for NVIDIA NIM with tiered model selection.

    Supports flexible model selection based on task complexity,
    latency requirements, and safety considerations.
    """

    # API Configuration
    nvidia_api_key: str = Field(..., description="NVIDIA API key")
    base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API endpoint",
    )

    # Model Selection Configuration
    model_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "routing": DEFAULT_ROUTING_MODEL,
            "reasoning": ModelTier.NEMOTRON_SUPER.value,
            "generation": ModelTier.NEMOTRON_NANO.value,
            "embedding": ModelTier.NEMOTRON_EMBED.value,
        },
        description="Model selection for different operation types",
    )

    # Task-to-Tier Mapping
    complexity_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            TaskComplexity.SIMPLE.value: "generation",
            TaskComplexity.MODERATE.value: "generation",
            TaskComplexity.COMPLEX.value: "reasoning",
            TaskComplexity.CRISIS.value: "reasoning",
        },
        description="Mapping of task complexity to model tier",
    )

    # Latency Requirements
    latency_requirements: dict[str, int] = Field(
        default_factory=lambda: {
            "max_response_ms": 500,
            "p95_target_ms": 2000,
            "streaming_first_token_ms": 1000,
        },
        description="Latency requirements in milliseconds",
    )

    # Embedding Configuration
    embedding_dimension: int = Field(2048, description="Embedding vector dimension")
    embedding_batch_size: int = Field(32, description="Batch size for embeddings")

    # Safety Configuration
    enable_crisis_detection: bool = Field(True, description="Enable crisis detection")
    crisis_detection_threshold: float = Field(0.7, description="Threshold for crisis detection")

    # User Configuration
    user_id: str = Field("default_user", description="Default user ID for memory")

    # Performance Tuning
    max_context_tokens: int = Field(128000, description="Maximum context window")
    temperature: float = Field(0.7, description="Default temperature for generation")
    streaming_enabled: bool = Field(True, description="Enable streaming responses")

    @field_validator("nvidia_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate NVIDIA API key format."""
        if not v or not v.strip():
            raise ValueError("NVIDIA API key cannot be empty")
        return v.strip()


class TieredModelSelector:
    """
    Intelligent model selector based on task complexity and requirements.

    Selects the appropriate model from the NVIDIA NIM catalog based on:
    - Task complexity (simple, moderate, complex, crisis)
    - Latency requirements
    - Cost optimization preferences
    - Safety considerations
    """

    def __init__(
        self,
        config: EnhancedNvidiaConfig,
        strategy: ModelSelectionStrategy = ModelSelectionStrategy.QUALITY_OPTIMIZED,
    ):
        self.config = config
        self.strategy = strategy
        self._latency_tracker: dict[str, list[float]] = {}
        self._initialize_latency_tracker()

    def _initialize_latency_tracker(self):
        """Initialize latency tracking for each model."""
        for model_id in self.config.model_tiers.values():
            self._latency_tracker[model_id] = []

    def select_model(
        self,
        task_complexity: TaskComplexity,
        _requires_streaming: bool = False,
        latency_budget_ms: int | None = None,
    ) -> str:
        """
        Select the appropriate model for a given task.

        Args:
            task_complexity: Complexity level of the task
            requires_streaming: Whether streaming is required
            latency_budget_ms: Maximum acceptable latency in milliseconds

        Returns:
            Model identifier to use
        """
        # Get base tier from complexity
        tier_name = self.config.complexity_mapping.get(task_complexity.value, "generation")
        base_model = self.config.model_tiers[tier_name]
        if tier_name == "generation":
            base_model = self.config.model_tiers.get("routing", base_model)

        # Check latency constraints
        if (
            latency_budget_ms
            and (model_caps := MODEL_REGISTRY.get(base_model))
            and model_caps.typical_latency_ms > latency_budget_ms
            and tier_name == "reasoning"
        ):
            # Downgrade to faster model if budget exceeded
            base_model = self.config.model_tiers["generation"]
            logger.info(f"Downgrading to {base_model} due to latency budget ({latency_budget_ms}ms)")

        # Crisis detection always uses reasoning model for safety
        if task_complexity == TaskComplexity.CRISIS:
            base_model = self.config.model_tiers["reasoning"]
            logger.info("Using reasoning model for crisis-level task")

        return base_model

    def select_routing_model(self) -> str:
        """Return the dedicated default routing model."""
        return self.config.model_tiers.get("routing", self.config.model_tiers["generation"])

    def select_embedding_model(self) -> str:
        """Get the embedding model identifier."""
        return self.config.model_tiers["embedding"]

    def record_latency(self, model_id: str, latency_ms: float):
        """Record latency observation for adaptive selection."""
        if model_id not in self._latency_tracker:
            self._latency_tracker[model_id] = []
        self._latency_tracker[model_id].append(latency_ms)
        # Keep last 100 observations
        self._latency_tracker[model_id] = self._latency_tracker[model_id][-100:]

    def get_average_latency(self, model_id: str) -> float:
        """Get average latency for a model."""
        observations = self._latency_tracker.get(model_id, [])
        return sum(observations) / len(observations) if observations else 0.0


class CrisisDetector:
    """
    Crisis detection using NVIDIA NIM for safety-critical analysis.

    Uses the most capable model (Nemotron-Super) for nuanced
    detection of crisis indicators in therapeutic conversations.
    """

    CRISIS_INDICATORS: tuple[str, ...] = (
        "self-harm",
        "suicide",
        "hopelessness",
        "isolation",
        "substance use",
        "trauma",
        "abuse",
        "crisis",
    )

    CRISIS_PROMPTS: tuple[str, ...] = (
        "I don't see the point anymore",
        "I'm having thoughts of hurting myself",
        "Nothing seems to matter",
        "I feel so alone",
        "I can't go on like this",
    )

    def __init__(
        self,
        client: AsyncOpenAI,
        model_id: str,
        rate_limiter: NvidiaRateLimiter | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.rate_limiter = rate_limiter or NvidiaRateLimiter()

    async def analyze(
        self,
        content: str,
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        """
        Analyze content for crisis indicators.

        Args:
            content: Text to analyze
            threshold: Confidence threshold for crisis flagging

        Returns:
            Analysis result with risk level and signals detected
        """
        system_prompt = """Analyze the following text for crisis signals.
Consider: self-harm indicators, hopelessness language, isolation signals,
substance use references, recent trauma indicators.

Respond with a JSON object containing:
- risk_level: one of 'low', 'medium', 'high', 'critical'
- confidence: float between 0 and 1
- signals_detected: list of specific signals found
- recommended_action: appropriate response
- immediate_concerns: boolean if immediate intervention needed"""

        try:
            await self.rate_limiter.wait("crisis_detection")
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,  # Lower temperature for safety-critical
                max_tokens=512,
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = self._default_result()
            except json.JSONDecodeError:
                result = self._default_result()

            # Apply threshold
            if result.get("confidence", 0) >= threshold:
                result["crisis_flagged"] = True
            else:
                result["crisis_flagged"] = False

            return result

        except Exception as e:
            logger.error(f"Error in crisis detection: {e}")
            return self._default_result()

    def _default_result(self) -> dict[str, Any]:
        """Return default safe result."""
        return {
            "risk_level": "low",
            "confidence": 0.0,
            "signals_detected": [],
            "recommended_action": "continue_conversation",
            "immediate_concerns": False,
            "crisis_flagged": False,
        }


class EmbeddingGenerator:
    """
    Generate embeddings using NVIDIA Nemotron-Embed-VL.

    Produces 2048-dimension embeddings for semantic search,
    RAG retrieval, and multimodal content understanding.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_id: str,
        dimension: int = 2048,
        rate_limiter: NvidiaRateLimiter | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.dimension = dimension
        self.rate_limiter = rate_limiter or NvidiaRateLimiter()
        self.cache = cache

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text with caching and rate limiting."""
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug("Embedding cache hit for text (length=%d)", len(text))
                return cached

        await self.rate_limiter.wait("embedding")
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding

        if self.cache is not None:
            self.cache.set(text, embedding)

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts with caching and rate limiting."""
        embeddings: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if self.cache is not None:
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append([])

        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i : i + batch_size]
            await self.rate_limiter.wait("embedding")

            response = await self.client.embeddings.create(
                model=self.model_id,
                input=batch,
                encoding_format="float",
            )

            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]

            for batch_position, embedding in enumerate(batch_embeddings):
                global_idx = uncached_indices[i + batch_position]
                embeddings[global_idx] = embedding
                if self.cache is not None:
                    self.cache.set(uncached_texts[i + batch_position], embedding)

        return embeddings

    async def embed_multimodal(
        self,
        text: str,
        _image_data: bytes | None = None,
    ) -> list[float]:
        """
        Generate embedding for multimodal content.

        Note: Current implementation text-only. Vision embedding
        requires additional configuration for image processing.
        """
        # Text-only embedding for now.
        # Multimodal (vision) embedding support deferred until image
        # processing configuration is available on the embedding endpoint.
        return await self.embed_text(text)


@dataclass(frozen=True)
class GenerationOptions:
    """Generation options for text generation requests."""

    temperature: float | None = None
    max_tokens: int = 2048
    stream: bool = False


class EnhancedNvidiaNimManager:
    """
    Enhanced NVIDIA NIM Manager with tiered model selection.

    Provides intelligent model selection based on task complexity,
    with specialized handling for:
    - Crisis detection (safety-first)
    - Real-time chat (latency-optimized)
    - Embedding generation (RAG-ready)

    Follows Phase 1 requirements from implementation roadmap.
    """

    def __init__(self, config: EnhancedNvidiaConfig):
        self.config = config

        # Initialize OpenAI-compatible client
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.nvidia_api_key,
        )
        self.sync_client = OpenAI(
            base_url=config.base_url,
            api_key=config.nvidia_api_key,
        )

        # Initialize model selector
        self.model_selector = TieredModelSelector(config)

        # Initialize rate limiter and semantic cache
        self.rate_limiter = NvidiaRateLimiter()
        self.embedding_cache = SemanticCache(max_size=1000, ttl_seconds=300)

        # Initialize specialized components
        reasoning_model = config.model_tiers.get("reasoning", ModelTier.NEMOTRON_SUPER.value)
        embedding_model = config.model_tiers.get("embedding", ModelTier.NEMOTRON_EMBED.value)

        self.crisis_detector = CrisisDetector(self.client, reasoning_model, rate_limiter=self.rate_limiter)
        self.embedding_generator = EmbeddingGenerator(
            self.client,
            embedding_model,
            config.embedding_dimension,
            rate_limiter=self.rate_limiter,
            cache=self.embedding_cache,
        )

        logger.info(
            f"Initialized EnhancedNvidiaNimManager with "
            f"reasoning={reasoning_model}, "
            f"generation={config.model_tiers.get('generation')}, "
            f"embedding={embedding_model}"
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        generation_options: GenerationOptions | None = None,
        **legacy_options: Any,
    ) -> str | Any:
        """
        Generate content using appropriate model for task complexity.

        Args:
            prompt: User input
            system_instruction: Optional system prompt
            complexity: Task complexity level
            generation_options: Optional generation options or legacy kwargs:
                temperature, max_tokens, stream.

        Returns:
            Generated text or streaming iterator
        """
        model = self.model_selector.select_model(complexity)

        if legacy_options:
            if generation_options is not None:
                raise TypeError("Pass either `generation_options` or legacy kwargs, not both.")
            temperature = legacy_options.pop("temperature", None)
            max_tokens = legacy_options.pop("max_tokens", 2048)
            stream = legacy_options.pop("stream", False)
            if legacy_options:
                unknown = ", ".join(sorted(legacy_options))
                raise TypeError(f"Unexpected kwargs: {unknown}")
            generation_options = GenerationOptions(
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

        generation_options = generation_options or GenerationOptions()
        temp = generation_options.temperature if generation_options.temperature is not None else self.config.temperature

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        await self.rate_limiter.wait("generation")
        start_time = time.time()

        try:
            if generation_options.stream and self.config.streaming_enabled:
                return await self._generate_stream(
                    model,
                    messages,
                    temp,
                    generation_options.max_tokens,
                )
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=generation_options.max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000
            self.model_selector.record_latency(model, latency_ms)

            logger.debug(f"Generated response with {model} in {latency_ms:.0f}ms")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    async def _generate_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ):
        """Generate streaming response."""
        await self.rate_limiter.wait("generation")
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_with_crisis_check(
        self,
        prompt: str,
        system_instruction: str | None = None,
        _user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate response with automatic crisis detection.

        Returns both the response and crisis analysis.
        """
        # First, check for crisis indicators
        crisis_analysis = await self.crisis_detector.analyze(
            prompt,
            self.config.crisis_detection_threshold,
        )

        # Determine task complexity based on crisis level
        if crisis_analysis.get("risk_level") in ["high", "critical"]:
            complexity = TaskComplexity.CRISIS
        else:
            complexity = TaskComplexity.MODERATE

        # Augment system prompt if crisis detected
        augmented_system = system_instruction or ""
        if crisis_analysis.get("crisis_flagged"):
            augmented_system += f"""

IMPORTANT: Crisis signals detected in user input.
Risk level: {crisis_analysis["risk_level"]}
Recommended action: {crisis_analysis["recommended_action"]}

Provide supportive, non-judgmental response. Prioritize safety.
Offer appropriate resources and encourage professional help."""

        # Generate response
        response = await self.generate(
            prompt=prompt,
            system_instruction=augmented_system,
            complexity=complexity,
        )

        return {
            "response": response,
            "model_used": self.model_selector.select_model(complexity),
            "crisis_analysis": crisis_analysis,
            "complexity": complexity.value,
        }

    async def embed(
        self,
        content: str | list[str],
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings for text content.

        Args:
            content: Single text or list of texts

        Returns:
            Embedding vector(s) with 2048 dimensions
        """
        if isinstance(content, list):
            return await self.embedding_generator.embed_batch(content)
        return await self.embedding_generator.embed_text(content)

    async def analyze_for_crisis(
        self,
        content: str,
    ) -> dict[str, Any]:
        """
        Dedicated crisis analysis endpoint.

        Use for pre-screening content before processing.
        """
        return await self.crisis_detector.analyze(content)

    def get_model_for_task(
        self,
        complexity: TaskComplexity,
    ) -> str:
        """Get the model that would be used for a given complexity."""
        return self.model_selector.select_model(complexity)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all configured models."""
        results = {}

        for tier_name, model_id in self.config.model_tiers.items():
            try:
                # Simple test request
                if tier_name == "embedding":
                    embedding = await self.embedding_generator.embed_text("test")
                    results[tier_name] = {
                        "status": "healthy",
                        "model": model_id,
                        "dimension": len(embedding),
                    }
                else:
                    await self.rate_limiter.wait("generation")
                    response = await self.client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=10,
                    )
                    results[tier_name] = {
                        "status": "healthy",
                        "model": model_id,
                        "response_length": len(response.choices[0].message.content or ""),
                    }
            except Exception as e:
                results[tier_name] = {
                    "status": "error",
                    "model": model_id,
                    "error": str(e),
                }

        return results


# Factory function for easy instantiation
def create_enhanced_manager(
    nvidia_api_key: str | None = None,
    _strategy: ModelSelectionStrategy = ModelSelectionStrategy.QUALITY_OPTIMIZED,
    **kwargs,
) -> EnhancedNvidiaNimManager:
    """
    Create an enhanced NVIDIA NIM manager.

    Args:
        nvidia_api_key: NVIDIA API key (defaults to env var)
        strategy: Model selection strategy
        **kwargs: Additional configuration options

    Returns:
        Configured EnhancedNvidiaNimManager instance
    """
    api_key = nvidia_api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA API key required. Set NVIDIA_API_KEY environment variable or pass nvidia_api_key parameter."
        )

    config = EnhancedNvidiaConfig(nvidia_api_key=api_key, **kwargs)
    return EnhancedNvidiaNimManager(config)


# Async context manager for convenient usage
class EnhancedNvidiaNimContext:
    """Async context manager for EnhancedNvidiaNimManager."""

    def __init__(self, config: EnhancedNvidiaConfig):
        self.config = config
        self.manager: EnhancedNvidiaNimManager | None = None

    async def __aenter__(self) -> EnhancedNvidiaNimManager:
        self.manager = EnhancedNvidiaNimManager(self.config)
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


async def main():
    """Demo/test function."""

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return

    # Create manager
    manager = create_enhanced_manager(api_key)

    # Health check
    health = await manager.health_check()
    for _tier, _status in health.items():
        pass

    # Test generation
    await manager.generate(
        "Hello, I'm feeling a bit stressed today.",
        complexity=TaskComplexity.SIMPLE,
    )

    # Test embedding
    await manager.embed("Test embedding generation")

    # Test crisis detection
    await manager.analyze_for_crisis("I've been feeling really hopeless lately")


if __name__ == "__main__":
    asyncio.run(main())
