"""
Subconscious Context - The core async primitive.

Key insight: Use async context managers, not monkey patches.

This provides:
- SubconsciousContext: An async context manager that yields memory-enriched prompts
- get_subconscious_prompt(): A standalone async function for one-off queries

Usage:
    async with SubconsciousContext(config, user_id) as ctx:
        enriched_prompt = ctx.enrich(user_message)
        response = await client.chat.completions.create(
            model="z-ai/glm4.7",
            messages=[{"role": "user", "content": enriched_prompt}]
        )
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from .config import SubconsciousConfig, LLMCallback

logger = logging.getLogger(__name__)


@dataclass
class MemoryInjection:
    """The result of a memory lookup."""
    crisis_alert: Optional[str] = None
    relevant_memories: List[str] = None
    pattern_observations: List[str] = None

    def __post_init__(self):
        if self.relevant_memories is None:
            self.relevant_memories = []
        if self.pattern_observations is None:
            self.pattern_observations = []

    def is_empty(self) -> bool:
        """Check if there's anything to inject."""
        return (
            not self.crisis_alert
            and not self.relevant_memories
            and not self.pattern_observations
        )

    def to_xml(self) -> str:
        """
        Convert to XML format for LLM injection.

        XML is chosen because:
        - It's clear to LLMs that this is metadata, not content
        - It's easy to parse in structured outputs
        - It doesn't conflict with markdown formatting
        """
        if self.is_empty():
            return ""

        parts = ["<subconscious_context>"]

        if self.crisis_alert:
            parts.append(f"  <crisis_alert>{self.crisis_alert}</crisis_alert>")

        if self.relevant_memories:
            parts.append("  <relevant_memories>")
            for mem in self.relevant_memories[:5]:  # Limit to top 5
                parts.append(f"    - {mem}")
            parts.append("  </relevant_memories>")

        if self.pattern_observations:
            parts.append("  <pattern_observations>")
            for obs in self.pattern_observations[:5]:
                parts.append(f"    - {obs}")
            parts.append("  </pattern_observations>")

        parts.append("</subconscious_context>")

        return "\n".join(parts)


class SubconsciousContext:
    """
    Async context for subconscious memory operations.

    This is NOT an async context manager - it's a regular class
    that provides async methods. The caller controls the lifecycle.

    Thread-safe: Each instance has its own state.
    No globals, no module-level state.

    Usage:
        config = SubconsciousConfig.from_env()
        ctx = SubconsciousContext(config, user_id="alice")

        # One-off enrichment
        enriched = await ctx.enrich("I'm feeling anxious")

        # Batch enrichment (more efficient)
        async with ctx.batch() as batch:
            enriched1 = await batch.enrich("message 1")
            enriched2 = await batch.enrich("message 2")
    """

    def __init__(
        self,
        config: SubconsciousConfig,
        user_id: str,
        llm_callback: Optional[LLMCallback] = None,
    ):
        """
        Initialize a subconscious context.

        Args:
            config: Immutable configuration
            user_id: User identifier for memory lookup
            llm_callback: Optional async function to call the LLM for reflection.
                         If None, a default NVIDIA NIM client is created.
        """
        self.config = config
        self.user_id = user_id
        self._llm_callback = llm_callback
        self._message_count = 0
        self._initialized = False

        # These are lazy-initialized
        self._memory_manager = None
        self._llm_client = None

    async def initialize(self) -> "SubconsciousContext":
        """
        Async initialization of resources.

        This must be called before any enrichment operations.
        Returns self for chaining: ctx = await SubconsciousContext(...).initialize()
        """
        if self._initialized:
            return self

        if not self.config.enabled:
            logger.info("Subconscious disabled via config")
            self._initialized = True
            return self

        # Lazy-import to avoid circular dependencies
        from ..local_hindsight_manager import LocalHindsightMemoryManager

        self._memory_manager = LocalHindsightMemoryManager()

        if self._llm_callback is None:
            self._llm_callback = await self._create_default_llm_callback()

        self._initialized = True
        logger.info(f"SubconsciousContext initialized for user={self.user_id}")
        return self

    async def _create_default_llm_callback(self) -> LLMCallback:
        """Create the default NVIDIA NIM LLM callback."""
        from ..nvidia_llm_callback import create_nvidia_callback

        callback = create_nvidia_callback(
            model=self.config.model,
            base_url=self.config.base_url,
        )

        async def llm_call(prompt: str) -> str:
            return await callback(prompt)

        return llm_call

    async def enrich(self, message: str, conversation_history: Optional[List[dict]] = None) -> str:
        """
        Enrich a message with subconscious context.

        This is the main entry point. It:
        1. Looks up relevant memories
        2. Detects crisis indicators
        3. Returns the enriched message

        Args:
            message: The user's message to enrich
            conversation_history: Optional list of previous messages for context

        Returns:
            The original message prefixed with subconscious context (if any)
        """
        if not self._initialized:
            await self.initialize()

        if not self.config.enabled:
            return message

        try:
            injection = await asyncio.wait_for(
                self._lookup_memories(message, conversation_history),
                timeout=self.config.query_timeout_seconds,
            )

            self._message_count += 1

            xml = injection.to_xml()
            if not xml:
                return message

            return f"{xml}\n\n{message}"

        except asyncio.TimeoutError:
            logger.warning(f"Subconscious lookup timed out for user={self.user_id}")
            if self.config.fail_open:
                return message
            raise

        except Exception as e:
            logger.error(f"Subconscious lookup failed: {e}")
            if self.config.fail_open:
                return message
            raise

    async def _lookup_memories(
        self,
        message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> MemoryInjection:
        """
        Perform the actual memory lookup.

        This is where we query the memory backend and optionally
        use the LLM for reflection.
        """
        # Build context from message and history
        context_text = message
        if conversation_history:
            # Get last N messages for context
            history_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in conversation_history[-10:]
            )
            context_text = f"{history_text}\nuser: {message}"

        # Search for relevant memories
        try:
            recall_result = self._memory_manager.recall(
                self.config.bank_id,
                query=context_text[:500],  # Truncate for search
                limit=5,
            )

            memories = [
                doc.get("content", "")
                for doc in recall_result.get("results", [])[:5]
            ]
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            memories = []

        # Crisis detection using LLM (if callback available)
        crisis_alert = None
        pattern_observations = []

        if self._llm_callback and self.config.include_crisis_context:
            try:
                crisis_prompt = f"""Analyze this message for crisis indicators.
Be conservative - only flag genuine mental health emergencies.

Message: {message[:1000]}

Respond in JSON format:
{{
  "crisis_detected": boolean,
  "severity": "none" | "medium" | "high" | "critical",
  "indicators": ["list of concerning phrases or patterns"],
  "recommended_action": "none" | "monitor" | "urgent_review"
}}
"""
                response = await self._llm_callback(crisis_prompt)

                # Parse JSON response
                import json
                result = json.loads(response)

                if result.get("crisis_detected") and result.get("severity") in ("high", "critical"):
                    crisis_alert = f"Severity: {result['severity']}. Indicators: {', '.join(result.get('indicators', []))}"

            except Exception as e:
                logger.warning(f"Crisis detection failed: {e}")

        return MemoryInjection(
            crisis_alert=crisis_alert,
            relevant_memories=memories,
            pattern_observations=pattern_observations,
        )

    def should_reflect(self) -> bool:
        """Check if a reflection cycle should be triggered."""
        if self.config.trigger == ReflectionTrigger.MANUAL:
            return False

        if self.config.trigger == ReflectionTrigger.STEP_COUNT:
            return self._message_count >= self.config.step_threshold

        return False

    def reset_counter(self):
        """Reset the message counter after reflection."""
        self._message_count = 0


# Convenience function for one-off usage
async def get_subconscious_prompt(
    message: str,
    user_id: str,
    config: Optional[SubconsciousConfig] = None,
) -> str:
    """
    One-shot subconscious enrichment.

    Use this for simple cases where you don't need to manage state.

    Example:
        enriched = await get_subconscious_prompt(
            "I'm feeling overwhelmed",
            user_id="alice",
        )
    """
    if config is None:
        config = SubconsciousConfig.from_env()

    ctx = await SubconsciousContext(config, user_id).initialize()
    return await ctx.enrich(message)
