"""
NVIDIA NIM + Mem0 Integration Manager.

Production-ready integration of NVIDIA NIM with Mem0 long-term memory,
implementing Mem0 cookbook best practices for:
- Custom instruction-based memory ingestion
- Confidence thresholds for high-stakes data
- PII filtering for HIPAA compliance
- Memory updates without duplication

Based on:
- https://docs.mem0.ai/cookbooks/essentials/controlling-memory-ingestion
- https://docs.mem0.ai/platform/overview
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from pydantic import BaseModel, Field

# Third-party imports
try:
    from mem0 import MemoryClient
except ImportError:
    try:
        from mem0ai import MemoryClient
    except ImportError:
        MemoryClient = None

try:
    from mem0 import Memory
except ImportError:
    try:
        from mem0ai import Memory
    except ImportError:
        Memory = None

if not MemoryClient and not Memory:
    raise ImportError("Please install mem0ai: uv add mem0ai")

from ai.core.api.memory.null_memory import NullMemoryManager

from .memory_ingestion_config import (
    CrisisDetector,
    MemoryCategory,
    PIIFilter,
    SpeculationFilter,
    TherapeuticMemoryConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0_nim")


@dataclass(frozen=True)
class _QueryContext:
    request_id: str
    user_id: str
    session_id: Optional[str]
    crisis_severity: str
    memory_context: str
    prompt: str


@dataclass(frozen=True)
class _InteractionResult:
    response_text: str
    latency_ms: float
    memories_used: int


@dataclass(frozen=True)
class _StoreReport:
    user_memory_stored: bool
    assistant_memory_stored: bool
    crisis_flagged: bool


class NIMMem0Config(BaseModel):
    """Configuration for NVIDIA NIM and Mem0 integration."""

    nim_api_key: Optional[str] = Field(None, description="NVIDIA NIM API key")
    nim_base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL",
    )
    mem0_api_key: Optional[str] = Field(None, description="Mem0 API key (for cloud)")
    model_name: str = Field(
        "meta/llama-3.1-405b-instruct", description="NIM model to use"
    )
    user_id: str = Field("default_user", description="Default user ID for memory")
    memory_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333},
            }
        },
        description="Mem0 memory configuration",
    )
    therapeutic_config: Optional[TherapeuticMemoryConfig] = Field(
        default=None, description="Therapeutic memory ingestion configuration"
    )


class NIMMem0Manager:
    """
    Manager for NVIDIA NIM with Mem0 long-term memory.

    Implements a production-ready interface for empathetic AI with memory,
    including Mem0 cookbook best practices:
    - Custom instructions for memory ingestion
    - PII filtering for HIPAA compliance
    - Speculation filtering for high-stakes therapeutic data
    - Crisis detection and flagging
    - Memory update/correction capabilities
    """

    def __init__(self, config: NIMMem0Config, memory_provider: Any = None):
        self.config = config
        self.therapeutic_config = config.therapeutic_config or TherapeuticMemoryConfig()
        self.nim_api_key = (
            config.nim_api_key
            or os.getenv("NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        if not self.nim_api_key:
            logger.warning(
                "No NVIDIA NIM API key configured. Set NIM_API_KEY, NVIDIA_API_KEY, or "
                "LLM_API_KEY, or pass nim_api_key."
            )

        # Initialize filters
        self.pii_filter = PIIFilter(self.therapeutic_config.pii_patterns)
        self.crisis_detector = CrisisDetector()

        # Initialize NIM endpoint
        self.client_endpoint = self.config.nim_base_url.rstrip("/")

        # Initialize Memory
        if memory_provider:
            self.memory = memory_provider
            self._is_platform_client = False
            logger.info("Using custom memory provider")
        else:
            self._initialize_mem0()

        # Apply custom instructions to project if using Platform API
        self._apply_custom_instructions()

        logger.info(f"Initialized NIMMem0Manager with model {self.config.model_name}")

    def _log_stage(
        self, request_id: str, stage: str, details: Optional[Dict[str, Any]] = None
    ):
        """Structured stage logging for better workflow visibility."""
        payload = {"request_id": request_id, "stage": stage}
        if details:
            payload.update(details)
        logger.info("memory-stack-stage=%s details=%s", payload["stage"], payload)

    def _extract_memory_results(self, raw_result: Any) -> List[Dict[str, Any]]:
        """Normalize memory client responses to a list of memory records."""
        if raw_result is None:
            return []
        if isinstance(raw_result, list):
            return raw_result
        if isinstance(raw_result, dict):
            return raw_result.get("results", []) or []
        return []

    def _extract_memory_id(self, result: Any) -> Optional[str]:
        """Extract a memory ID from normalized memory add/update responses."""
        results = self._extract_memory_results(result)
        if not results:
            return None
        return results[0].get("id", "stored")

    def _call_memory(self, operation: str, *args, **kwargs) -> Any:
        """Call raw memory client operation with consistent error handling."""
        method = getattr(self.memory, operation)
        return method(*args, **kwargs)

    def _initialize_mem0(self):
        """Initialize Mem0 client with fallback chain."""
        try:
            if self.config.mem0_api_key:
                # Use Platform API (recommended for production)
                if MemoryClient:
                    self.memory = MemoryClient(api_key=self.config.mem0_api_key)
                    self._is_platform_client = True
                    logger.info("Initialized Mem0 Platform Client")
                else:
                    self.memory = Memory.from_config(
                        {"api_key": self.config.mem0_api_key}
                    )
                    self._is_platform_client = False
                    logger.info("Initialized Mem0 with API key")
            elif Memory:
                # Use local/self-hosted Mem0
                self.memory = Memory.from_config(self.config.memory_config)
                self._is_platform_client = False
                logger.info("Initialized local Mem0")
            else:
                raise ImportError("Memory class not available")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Mem0: {e}. Falling back to null memory."
            )
            self.memory = NullMemoryManager()
            self._is_platform_client = False

    def _apply_custom_instructions(self):
        """Apply therapeutic custom instructions to Mem0 project."""
        try:
            if hasattr(self.memory, "project") and hasattr(
                self.memory.project, "update"
            ):
                self.memory.project.update(
                    custom_instructions=self.therapeutic_config.custom_instructions
                )
                logger.info("Applied therapeutic custom instructions to Mem0 project")
        except Exception as e:
            logger.debug(f"Could not apply custom instructions (non-Platform API): {e}")

    async def _call_nim(self, system_instructions: str, query: str) -> str:
        """Call NVIDIA NIM chat completions and return response text."""
        if not self.nim_api_key:
            raise RuntimeError(
                "No NVIDIA NIM API key is configured. "
                "Provide NIM_API_KEY or pass nim_api_key."
            )

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": query},
            ],
            "temperature": 0.6,
        }
        headers = {
            "Authorization": f"Bearer {self.nim_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.client_endpoint}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"NVIDIA NIM request failed: {exc}") from exc

        body = response.json()
        try:
            choices = body.get("choices", [])
            if not choices:
                raise RuntimeError("NVIDIA NIM returned no choices.")
            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                raise RuntimeError("NVIDIA NIM returned empty message content.")
            return str(content)
        except Exception as exc:
            raise RuntimeError(f"Unexpected NVIDIA NIM response: {body}") from exc

    def _filter_for_storage(self, content: str) -> Optional[str]:
        """
        Filter content before memory storage.

        Applies PII filtering and speculation detection based on cookbook patterns.

        Args:
            content: Raw content to filter

        Returns:
            Filtered content safe for storage, or None if should not be stored
        """
        # Check for PII
        filtered = self.pii_filter.filter_for_storage(content)
        if filtered is None:
            logger.debug("Content rejected: too much PII")
            return None

        # Check speculation based on confidence threshold
        if SpeculationFilter.is_speculative(filtered):
            confidence = SpeculationFilter.get_confidence_adjustment(filtered)
            if confidence < self.therapeutic_config.confidence_threshold:
                logger.debug(f"Content rejected: speculation confidence {confidence}")
                return None

        # Truncate if too long
        if len(filtered) > self.therapeutic_config.max_memory_length:
            filtered = f"{filtered[: self.therapeutic_config.max_memory_length]}..."

        return filtered

    async def get_response(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a response from NVIDIA NIM using Mem0 for context.

        Args:
            query: User's input message
            user_id: Unique identifier for the user
            session_id: Optional session identifier
            context: Additional fixed context

        Returns:
            Dictionary with response text and metadata
        """
        uid = user_id or self.config.user_id

        request_id = uuid4().hex
        self._log_stage(
            request_id,
            "start",
            {
                "user_id": uid,
                "has_session": bool(session_id),
                "platform_memory": self._is_platform_client,
            },
        )

        # 1. Check for crisis signals
        crisis_severity = self.crisis_detector.get_crisis_severity(query)
        if crisis_severity != "none":
            logger.warning(f"Crisis signal detected: {crisis_severity} for user {uid}")

        # 2. Retrieve relevant memories
        memories = self._search_memories(query, uid)
        memory_context = self._format_memories(memories)
        self._log_stage(
            request_id,
            "memory_retrieved",
            {
                "memories_found": len(memories),
                "memory_context_empty": memory_context
                == "No previous relevant memories.",
            },
        )

        # 3. Build the system prompt with memory
        system_instructions = self._build_system_prompt(
            memory_context, context, crisis_severity
        )
        query_context = _QueryContext(
            request_id=request_id,
            user_id=uid,
            session_id=session_id,
            crisis_severity=crisis_severity,
            memory_context=memory_context,
            prompt=system_instructions,
        )
        self._log_stage(
            request_id,
            "prompt_built",
            {"prompt_length": len(query_context.prompt)},
        )

        # 4. Generate response using NVIDIA NIM
        start_time = datetime.now()
        response_text = await self._call_nim(
            system_instructions=system_instructions, query=query
        )
        end_time = datetime.now()
        self._log_stage(
            request_id,
            "llm_response",
            {"latency_ms": (end_time - start_time).total_seconds() * 1000},
        )

        # 5. Store the new interaction in Mem0 (with filtering)
        store_report = self._store_interaction(
            query, response_text, uid, session_id, crisis_severity
        )

        latency = (end_time - start_time).total_seconds() * 1000
        interaction_result = _InteractionResult(
            response_text=response_text,
            latency_ms=latency,
            memories_used=len(memories),
        )
        self._log_stage(
            request_id,
            "complete",
            {
                "latency_ms": interaction_result.latency_ms,
                "memories_used": interaction_result.memories_used,
            },
        )

        return {
            "response": interaction_result.response_text,
            "latency_ms": interaction_result.latency_ms,
            "memories_used": interaction_result.memories_used,
            # "memories_content": [
            #     m.get("memory") or m.get("content", "") for m in memories
            # ],  # Removed for privacy
            "user_id": uid,
            "crisis_detected": crisis_severity != "none",
            "crisis_severity": crisis_severity,
            "request_id": request_id,
            "store_report": {
                "user_memory_stored": store_report.user_memory_stored,
                "assistant_memory_stored": store_report.assistant_memory_stored,
                "crisis_flagged": store_report.crisis_flagged,
            },
            "model": self.config.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def search_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Public alias for memory search."""
        return self._search_memories(query, user_id)

    def _search_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        try:
            result = self._call_memory("search", query, user_id=user_id)
            return self._extract_memory_results(result)
        except Exception:
            logger.exception("Error searching memories")
            return []

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for context."""
        if not memories:
            return "No previous relevant memories."

        formatted = []
        for m in memories:
            if content := m.get("memory") or m.get("content", ""):
                formatted.append(f"- {content}")

        return "\n".join(formatted) if formatted else "No previous relevant memories."

    def _build_system_prompt(
        self,
        memory_context: str,
        additional_context: Optional[str],
        crisis_severity: str,
    ) -> str:
        """Build the system prompt with memory and crisis handling."""
        prompt = (
            "You are Pixelated Empathy, an empathetic AI assistant trained"
            " in therapeutic dialogue. Use the following memories about the user"
            " to personalize your response and demonstrate continuity. "
            "If the memories contradict the current query, prioritize the current "
            "query but acknowledge the change if appropriate.\n\n"
            f"USER MEMORIES:\n{memory_context}\n\n"
        )

        if additional_context:
            prompt += f"ADDITIONAL CONTEXT:\n{additional_context}\n\n"

        if crisis_severity != "none":
            prompt += (
                f"⚠️ CRISIS ALERT (Severity: {crisis_severity.upper()}):\n"
                "The user may be expressing thoughts of self-harm or crisis. "
                "Respond with compassion, validate their feelings, and gently "
                "encourage professional support. Provide crisis resources if "
                "appropriate. Do NOT dismiss their concerns or offer toxic "
                "positivity.\n\n"
            )

        return prompt

    def _store_interaction(
        self,
        query: str,
        response: str,
        user_id: str,
        session_id: Optional[str],
        crisis_severity: str,
    ) -> _StoreReport:
        """Store interaction in memory with filtering."""
        user_memory_stored = False
        assistant_memory_stored = False
        crisis_flagged = crisis_severity != "none"
        try:
            # Filter and store user query
            if filtered_query := self._filter_for_storage(f"User shared: {query}"):
                metadata = {"role": "user"}
                if session_id:
                    metadata["session_id"] = session_id
                if crisis_severity != "none":
                    metadata["crisis_flag"] = True
                    metadata["crisis_severity"] = crisis_severity
                    metadata["category"] = MemoryCategory.CRISIS_CONTEXT.value

                self._call_memory(
                    "add",
                    filtered_query,
                    user_id=user_id,
                    metadata=metadata,
                )
                user_memory_stored = True

            # Store key response insights (truncated)
            response_summary = response[:500] if len(response) > 500 else response
            if filtered_response := self._filter_for_storage(
                f"Assistant provided: {response_summary}"
            ):
                self._call_memory(
                    "add",
                    filtered_response,
                    user_id=user_id,
                    metadata={"role": "assistant", "session_id": session_id}
                    if session_id
                    else {"role": "assistant"},
                )
                assistant_memory_stored = True
            return _StoreReport(
                user_memory_stored=user_memory_stored,
                assistant_memory_stored=assistant_memory_stored,
                crisis_flagged=crisis_flagged,
            )
        except Exception:
            logger.exception("Error storing interaction")
            return _StoreReport(
                user_memory_stored=user_memory_stored,
                assistant_memory_stored=assistant_memory_stored,
                crisis_flagged=crisis_flagged,
            )

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory without creating duplicates.

        Based on Mem0 cookbook: https://docs.mem0.ai/cookbooks/essentials/controlling-memory-ingestion

        Args:
            memory_id: ID of the memory to update
            new_content: New content for the memory
            metadata: Optional updated metadata

        Returns:
            True if update succeeded
        """
        try:
            filtered_content = self._filter_for_storage(new_content)
            if not filtered_content:
                logger.warning("Update rejected: content failed filtering")
                return False

            update_args = {"memory_id": memory_id, "text": filtered_content}
            if metadata:
                update_args["metadata"] = metadata

            self._call_memory("update", **update_args)
            logger.info(f"Updated memory {memory_id}")
            return True
        except Exception:
            logger.exception(f"Error updating memory {memory_id}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deletion succeeded
        """
        try:
            self._call_memory("delete", memory_id=memory_id)
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception:
            logger.exception(f"Error deleting memory {memory_id}")
            return False

    def clear_memory(self, user_id: str):
        """Clear all memories for a specific user."""
        try:
            if hasattr(self.memory, "delete_all"):
                self._call_memory("delete_all", user_id=user_id)
            else:
                # Fallback: get all and delete individually
                all_memories = self.get_all_memories(user_id)
                for m in all_memories:
                    if m.get("id"):
                        self.delete_memory(m["id"])
            logger.info(f"Cleared all memories for user: {user_id}")
        except Exception:
            logger.exception(f"Error clearing memory for user {user_id}")

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user."""
        try:
            result = self._call_memory("get_all", user_id=user_id)
            return self._extract_memory_results(result)
        except Exception:
            logger.exception(f"Error retrieving memories for user {user_id}")
            return []

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        try:
            result = self._call_memory("get", memory_id=memory_id)
            if isinstance(result, dict):
                return result
            return None
        except Exception:
            logger.exception(f"Error retrieving memory {memory_id}")
            return None

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add a single memory with PII and speculation filtering.

        Args:
            content: Content to store
            user_id: User identifier
            metadata: Optional metadata
            category: Optional category

        Returns:
            Memory ID if stored, None if filtered out or error
        """
        try:
            if not (filtered_content := self._filter_for_storage(content)):
                logger.warning("Memory addition rejected: content failed filtering")
                return None

            full_metadata = metadata or {}
            if category:
                full_metadata["category"] = category

            # Add timestamp if not present
            if "timestamp" not in full_metadata:
                full_metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

            result = self._call_memory(
                "add",
                filtered_content,
                user_id=user_id,
                metadata=full_metadata,
            )

            return self._extract_memory_id(result)

        except Exception as e:
            logger.exception(f"Error adding memory: {e}")
            return None


async def test_integration():
    """Simple test for the integration."""
    nim_key = (
        os.environ.get("NIM_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    mem0_key = os.environ.get("MEM0_API_KEY")

    if not nim_key:
        logger.error("Error: NIM API key not found in environment.")
        return

    # Create therapeutic config with high confidence threshold
    therapeutic_config = TherapeuticMemoryConfig(
        confidence_threshold=0.8,
        enable_crisis_detection=True,
    )

    config = NIMMem0Config(
        nim_api_key=nim_key,
        mem0_api_key=mem0_key,
        user_id="test_user_001",
        therapeutic_config=therapeutic_config,
    )

    manager = NIMMem0Manager(config)

    # Test queries including speculation filtering
    queries = [
        (
            "Hi, I'm Alex. I've been feeling a bit overwhelmed with "
            "my new job as a developer."
        ),
        # Should be filtered (speculation)
        "I think I might have anxiety, but I'm not sure.",
        # Should be stored
        "Dr. Smith diagnosed me with generalized anxiety disorder last month.",
        "Do you remember my name and what's bothering me?",
    ]

    for q in queries:
        logger.info(f"\nUSER: {q}")
        result = await manager.get_response(q)
        logger.info(f"AI: {result['response']}")
        logger.info(
            f"(Memories used: {result['memories_used']}, "
            f"Crisis detected: {result['crisis_detected']})"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_integration())
