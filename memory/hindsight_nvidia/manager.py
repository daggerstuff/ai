"""
Nvidia NIM + Hindsight Integration Manager.

Production-ready integration of NVIDIA NIM with Hindsight long-term memory,
implementing Hindsight cookbook best practices for:
- Custom instruction-based memory ingestion
- Confidence thresholds for high-stakes data
- PII filtering for HIPAA compliance
- Memory updates without duplication

Transitioned from Google Gemini to NVIDIA NIM for the Pixelated Empathy platform.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

from ai.api.memory.base import BaseMemoryManager
from ai.api.memory.null_memory import NullMemoryManager
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from .memory_ingestion_config import (
    MemoryCategory,
    TherapeuticMemoryConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hindsight_nvidia")


class NvidiaHindsightConfig(BaseModel):
    """Configuration for NVIDIA NIM and Hindsight integration."""

    nvidia_api_key: str = Field(..., description="NVIDIA API key")
    hindsight_api_key: Optional[str] = Field(None, description="Hindsight API key (for cloud)")
    model_name: str = Field(
        "meta/llama-3.1-405b-instruct", description="NVIDIA NIM model to use"
    )
    base_url: str = Field(
        "https://integrate.api.nvidia.com/v1", description="NVIDIA NIM Base URL"
    )
    user_id: str = Field("default_user", description="Default user ID for memory")
    memory_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333},
            }
        },
        description="Hindsight memory configuration",
    )
    therapeutic_config: Optional[TherapeuticMemoryConfig] = Field(
        default=None, description="Therapeutic memory ingestion configuration"
    )


class NvidiaHindsightManager(BaseMemoryManager):
    """
    Manager for NVIDIA NIM with Hindsight long-term memory.

    Implements a production-ready interface for empathetic AI with memory,
    including Hindsight cookbook best practices:
    - Custom instructions for memory ingestion
    - PII filtering for HIPAA compliance
    - Speculation filtering for high-stakes therapeutic data
    - Crisis detection and flagging
    - Memory update/correction capabilities
    """

    def __init__(self, config: NvidiaHindsightConfig, memory_provider: Any = None):
        self.config = config
        self.therapeutic_config = config.therapeutic_config or TherapeuticMemoryConfig()

        from ai.memory.therapeutic_processor import TherapeuticProcessor
        self.processor = TherapeuticProcessor(self.therapeutic_config)

        # Initialize OpenAI clients
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.nvidia_api_key
        )
        self.async_client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.nvidia_api_key
        )

        # Initialize Memory
        if memory_provider:
            self.memory = memory_provider
            self._is_platform_client = False
            logger.info("Using custom memory provider")
        else:
            self._initialize_hindsight()

        # Apply custom instructions to project if using Platform API
        self._apply_custom_instructions()

        logger.info(
            f"Initialized NvidiaHindsightManager with model {self.config.model_name}"
        )

    def _initialize_hindsight(self):
        """Initialize Hindsight client with fallback chain."""
        try:
            if self.config.hindsight_api_key:
                # Use Platform API (recommended for production)
                if MemoryClient:
                    self.memory = MemoryClient(api_key=self.config.hindsight_api_key)
                    self._is_platform_client = True
                    logger.info("Initialized Hindsight Platform Client")
                else:
                    self.memory = Memory.from_config(
                        {"api_key": self.config.hindsight_api_key}
                    )
                    self._is_platform_client = False
                    logger.info("Initialized Hindsight with API key")
            elif Memory:
                # Use local/self-hosted Hindsight
                self.memory = Memory.from_config(self.config.memory_config)
                self._is_platform_client = False
                logger.info("Initialized local Hindsight")
            else:
                raise ImportError("Memory class not available")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Hindsight: {e}. Falling back to null memory."
            )
            self.memory = NullMemoryManager()
            self._is_platform_client = False

    def _apply_custom_instructions(self):
        """Apply therapeutic custom instructions to Hindsight project."""
        try:
            if hasattr(self.memory, "project") and hasattr(
                self.memory.project, "update"
            ):
                self.memory.project.update(
                    custom_instructions=self.therapeutic_config.custom_instructions
                )
                logger.info("Applied therapeutic custom instructions to Hindsight project")
        except Exception as e:
            logger.debug(f"Could not apply custom instructions (non-Platform API): {e}")

    def _filter_for_storage(self, content: str) -> tuple[Optional[str], str]:
        """
        Filter content before memory storage.

        Applies PII filtering and speculation detection based on cookbook patterns.

        Args:
            content: Raw content to filter

        Returns:
            Tuple of (Filtered content safe for storage or None, Hindsight memory type)
        """
        return self.processor.filter_for_storage(content)

    async def generate_content(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> str:
        """Generate content using NVIDIA NIM."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in NVIDIA generate_content: {e}")
            return f"Error: {e}"

    def _get_base_instructions(self) -> str:
        """Provides the base instructions for the therapeutic processor."""
        return """You are Antigravity, a therapeutic companion AI.
Your goal is to provide empathetic, validating, and safe support.
Maintain professional boundaries and safety protocols at all times."""

    async def get_response(
        self, user_id: str, message: str, session_id: Optional[str] = None
    ) -> str:
        """Get response from the memory-augmented agent using NVIDIA NIM."""
        # 1. Search relevant memories (Hindsight is currently sync)
        memories = self.search_memories(message, user_id)
        
        # 2. Extract facts for the prompt
        facts = [m.get("memory", "") for m in memories]
        
        # 3. Build augmented prompt
        base_instructions = self._get_base_instructions()
        system_prompt = self.processor.build_system_prompt(base_instructions, facts)

        # 4. Generate response using NVIDIA NIM
        response_text = await self.generate_content(message, system_prompt)

        # 5. Detect crisis severity
        crisis_severity = self.processor.detect_crisis(message)

        # 6. Store the new interaction in Hindsight (with filtering)
        self._store_interaction(
            message, response_text, user_id, session_id, crisis_severity
        )

        return response_text

    def _store_interaction(
        self,
        query: str,
        response: str,
        user_id: str,
        session_id: Optional[str],
        crisis_severity: str,
    ):
        """Store interaction in memory with filtering."""
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

                self.memory.add(
                    filtered_query,
                    user_id=user_id,
                    metadata=metadata,
                )

            # Store key response insights (truncated)
            response_summary = response[:500] if len(response) > 500 else response
            if filtered_response := self._filter_for_storage(
                f"Assistant provided: {response_summary}"
            ):
                self.memory.add(
                    filtered_response,
                    user_id=user_id,
                    metadata={"role": "assistant", "session_id": session_id}
                    if session_id
                    else {"role": "assistant"},
                )
        except Exception:
            logger.exception("Error storing interaction")

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory without creating duplicates.
        """
        try:
            filtered_content = self._filter_for_storage(new_content)
            if not filtered_content:
                logger.warning("Update rejected: content failed filtering")
                return False

            update_args = {"memory_id": memory_id, "text": filtered_content}
            if metadata:
                update_args["metadata"] = metadata

            self.memory.update(**update_args)
            logger.info(f"Updated memory {memory_id}")
            return True
        except Exception:
            logger.exception(f"Error updating memory {memory_id}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        """
        try:
            self.memory.delete(memory_id=memory_id)
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception:
            logger.exception(f"Error deleting memory {memory_id}")
            return False

    def clear_memory(self, user_id: str):
        """Clear all memories for a specific user."""
        try:
            if hasattr(self.memory, "delete_all"):
                self.memory.delete_all(user_id=user_id)
            else:
                # Fallback: get all and delete individually
                all_memories = self.get_all_memories(user_id)
                for m in all_memories:
                    if m.get("id"):
                        self.delete_memory(m["id"])
            logger.info(f"Cleared all memories for user: {user_id}")
        except Exception:
            logger.exception(f"Error clearing memory for user {user_id}")

    def list_entities(self, limit: int = 20, page: int = 1) -> List[Dict[str, Any]]:
        """List all entities (users/agents) with pagination."""
        try:
            if hasattr(self.memory, "users"):
                entities = self.memory.users()
                return self._paginate(entities, limit, page)
            return []
        except Exception:
            logger.exception("Error listing entities")
            return []

    def get_all_memories(
        self, user_id: str, limit: int = 100, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user with pagination."""
        try:
            result = self.memory.get_all(user_id=user_id)
            if isinstance(result, dict):
                memories = result.get("results", [])
            else:
                memories = result if isinstance(result, list) else []
            return self._paginate(memories, limit, page)
        except Exception:
            logger.exception(f"Error retrieving memories for user {user_id}")
            return []

    def search_memories(
        self, query: str, user_id: str, limit: int = 10, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        try:
            result = self.memory.search(query, user_id=user_id, limit=limit)
            if isinstance(result, dict):
                memories = result.get("results", [])
            else:
                memories = result if isinstance(result, list) else []
            return self._paginate(memories, limit, page)
        except Exception:
            logger.exception(f"Error searching memories for user {user_id}")
            return []

    def _paginate(self, items: List[Any], limit: int, page: int) -> List[Any]:
        """Manually paginate results."""
        if not items:
            return []
        start = (page - 1) * limit
        end = start + limit
        return items[start:end]

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        try:
            return self.memory.get(memory_id=memory_id)
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

            result = self.memory.add(
                filtered_content,
                user_id=user_id,
                metadata=full_metadata,
            )

            # Extract memory ID - Hindsight returns dict with 'results' list
            if isinstance(result, dict):
                results = result.get("results") or []
                return results[0].get("id", "stored") if results else None
            return (
                result[0].get("id", "stored")
                if isinstance(result, list) and result
                else None
            )

        except Exception as e:
            logger.exception(f"Error adding memory: {e}")
            return None


async def test_integration():
    """Simple test for the integration."""
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    hindsight_key = os.environ.get("HINDSIGHT_API_KEY")

    if not nvidia_key:
        logger.error("Error: NVIDIA_API_KEY not found in environment.")
        return

    # Create therapeutic config with high confidence threshold
    therapeutic_config = TherapeuticMemoryConfig(
        confidence_threshold=0.8,
        enable_crisis_detection=True,
    )

    config = NvidiaHindsightConfig(
        nvidia_api_key=nvidia_key,
        hindsight_api_key=hindsight_key,
        user_id="test_user_001",
        therapeutic_config=therapeutic_config,
    )

    manager = NvidiaHindsightManager(config)

    # Test queries
    queries = [
        "Hi, I'm Alex. I've been feeling a bit overwhelmed.",
        "Do you remember my name?",
    ]

    for q in queries:
        logger.info(f"USER: {q}")
        result = await manager.get_response(user_id=config.user_id, message=q)
        logger.info(f"AI: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integration())
