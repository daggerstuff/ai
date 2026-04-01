"""
Subconscious LLM Wrapper - Injects memory context into LLM calls.

This wrapper intercepts LLM prompts, queries the reflection subagent
for relevant context/memories, and injects them into the prompt
before the LLM sees it - mimicking Claude's subconscious injection.

Usage:
    from ai.memory.subconscious_wrapper import SubconsciousLLM

    llm = SubconsciousLLM(base_llm, user_id="user-123")
    response = await llm.complete(prompt, conversation_context)
"""
import logging
from typing import Optional, List, Dict, Any

from .reflection_bootstrap import ReflectionBootstrap  # noqa: E402

logger = logging.getLogger(__name__)


class SubconsciousContext:
    """Context injected by the subconscious."""

    def __init__(
        self,
        relevant_memories: List[str] = None,
        emotional_state: Optional[str] = None,
        crisis_indicators: List[str] = None,
        therapeutic_goals: List[str] = None,
        pattern_observations: List[str] = None,
    ):
        self.relevant_memories = relevant_memories if relevant_memories else []
        self.emotional_state = emotional_state
        self.crisis_indicators = crisis_indicators if crisis_indicators else []
        self.therapeutic_goals = therapeutic_goals if therapeutic_goals else []
        self.pattern_observations = pattern_observations if pattern_observations else []

    def to_injection_prompt(self) -> str:
        """Convert context to XML injection format (like Claude Subconscious)."""
        parts = []

        if self.crisis_indicators:
            parts.append(
                f"<crisis_alert>Active crisis indicators: {', '.join(self.crisis_indicators)}</crisis_alert>"
            )

        if self.relevant_memories:
            memories_text = "\n".join(f"- {m}" for m in self.relevant_memories[:5])
            parts.append(f"<relevant_memories>\n{memories_text}\n</relevant_memories>")

        if self.emotional_state:
            parts.append(f"<emotional_state>{self.emotional_state}</emotional_state>")

        if self.therapeutic_goals:
            goals_text = "\n".join(f"- {g}" for g in self.therapeutic_goals[:3])
            parts.append(f"<therapeutic_goals>\n{goals_text}\n</therapeutic_goals>")

        if self.pattern_observations:
            pattern_text = "\n".join(f"- {p}" for p in self.pattern_observations[:5])
            parts.append(f"<pattern_observations>\n{pattern_text}\n</pattern_observations>")

        if parts:
            return "<subconscious_context>\n" + "\n".join(parts) + "\n</subconscious_context>"
        return ""


class SubconsciousLLM:
    """
    LLM wrapper that injects subconscious context.

    Wraps any LLM call (OpenAI, Anthropic, etc.) and injects
    relevant memories, patterns, and therapeutic context before
    the LLM sees the prompt.
    """

    def __init__(
        self,
        llm_callback,
        user_id: str,
        bootstrap: Optional[ReflectionBootstrap] = None,
        model: str = "qwen/qwen3.5-397b-a17b",
    ):
        """
        Initialize subconscious LLM wrapper.

        Args:
            llm_callback: Function to call the actual LLM (e.g., openai.ChatCompletion)
            user_id: User identifier for memory lookup
            bootstrap: Optional ReflectionBootstrap instance
            model: Model identifier
        """
        self.llm_callback = llm_callback
        self.user_id = user_id
        self.bootstrap = bootstrap
        self.model = model
        self._conversation_history: List[Dict[str, Any]] = []

    async def _query_subconscious(
        self,
        prompt: str,  # noqa: F841
        conversation_context: str,
    ) -> SubconsciousContext:
        """Query the subconscious for relevant context."""
        if not self.bootstrap or not self.bootstrap._subagent:
            return SubconsciousContext()

        try:
            # Analyze the conversation for relevant memories
            result = await self.bootstrap._subagent.analyze_conversation(
                conversation_text=conversation_context,
                user_id=self.user_id,
            )

            # Build context from memories
            relevant_memories = []
            if result.memories_preserved:
                # These are memories flagged as important
                relevant_memories.extend(result.memories_preserved[:5])

            # Add crisis indicators if present
            crisis_indicators = result.crisis_indicators if result.crisis_detected else []

            # Generate pattern observations from the analysis
            pattern_observations = []
            if result.recommendations:
                pattern_observations.extend(result.recommendations[:5])

            return SubconsciousContext(
                relevant_memories=relevant_memories,
                crisis_indicators=crisis_indicators,
                pattern_observations=pattern_observations,
                therapeutic_goals=[],  # Would come from therapeutic config
            )
        except Exception as e:
            logger.error(f"Subconscious query failed: {e}")
            return SubconsciousContext([])

    def _inject_context_into_prompt(
        self,
        prompt: str,
        subconscious_context: SubconsciousContext,
    ) -> str:
        """Inject subconscious context into the prompt."""
        injection = subconscious_context.to_injection_prompt()
        if not injection:
            return prompt

        # Format: subconscious context appears BEFORE the user's prompt
        # This is how Claude Subconscious works - it prepends to the prompt
        return f"{injection}\n\n{prompt}"

    async def complete(
        self,
        prompt: str,
        conversation_context: str,
        **kwargs,
    ) -> str:
        """
        Complete a prompt with subconscious context injection.

        Args:
            prompt: User's prompt
            conversation_context: Full conversation for subconscious analysis
            **kwargs: Additional args for the LLM callback

        Returns:
            LLM response with subconscious context injected
        """
        # Query subconscious for context
        subconscious_context = await self._query_subconscious(
            prompt=prompt,
            conversation_context=conversation_context,
        )

        # Inject context into prompt
        injected_prompt = self._inject_context_into_prompt(
            prompt=prompt,
            subconscious_context=subconscious_context,
        )

        # Call the actual LLM
        response = await self.llm_callback(injected_prompt, **kwargs)

        # Store in conversation history
        self._conversation_history.append({
            "role": "user",
            "content": prompt,
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": response,
        })

        return response

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()


def create_subconscious_llm(
    llm_callback,
    user_id: str,
    bootstrap: Optional[ReflectionBootstrap] = None,
    model: str = "qwen/qwen3.5-397b-a17b",
) -> SubconsciousLLM:
    """
    Factory function to create SubconsciousLLM wrapper.

    Usage:
        llm = create_subconscious_llm(
            llm_callback=openai_chat,
            user_id="user-123",
        )
        response = await llm.complete(prompt, conversation_context)
    """
    return SubconsciousLLM(
        llm_callback=llm_callback,
        user_id=user_id,
        bootstrap=bootstrap,
        model=model,
    )
