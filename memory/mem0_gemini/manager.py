"""
Gemini-tuned Mem0 Integration Manager.

Implements a production-ready interface for empathetic AI with memory,
tuned for Google Gemini's context handling and Mem0 long-term storage.
"""

import logging
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

from pydantic import BaseModel, Field

from ai.api.memory.base import BaseMemoryManager
from ai.api.memory.null_memory import NullMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0_gemini")


class GeminiMem0Config(BaseModel):
    """Configuration for Gemini and Mem0 integration."""

    gemini_api_key: str = Field(..., description="Gemini/Google API key")
    mem0_api_key: Optional[str] = Field(None, description="Mem0 API key (for cloud)")
    model_name: str = Field("gemini-1.5-pro", description="Gemini model to use")
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


class GeminiMem0Manager(BaseMemoryManager):
    """
    Manager for Google Gemini with Mem0 long-term memory.
    """

    def __init__(self, config: GeminiMem0Config):
        self.config = config

        # Initialize Mem0 client
        # If mem0_api_key is provided, use cloud; otherwise use local
        if config.mem0_api_key:
            self.client = MemoryClient(api_key=config.mem0_api_key)
        else:
            self.client = Memory.from_config(config.memory_config)

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> str:
        """Add a memory and return ID."""
        if metadata is None:
            metadata = {}

        if category:
            metadata["category"] = category

        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        metadata["provider"] = "gemini"

        result = self.client.add(content, user_id=user_id, metadata=metadata)

        # Handle different return formats between cloud and local
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("id") or str(result[0])
        elif isinstance(result, dict) and "results" in result:
            return result["results"][0]["id"]
        return str(result)

    def search_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        return self.client.search(query, user_id=user_id)

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all memories for a user."""
        return self.client.get_all(user_id=user_id)

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        return self.client.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory."""
        try:
            self.client.update(memory_id, new_content, metadata=metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            self.client.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def clear_memory(self, user_id: str) -> bool:
        """Clear all memories for a user."""
        try:
            self.client.delete_all(user_id=user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to clear memory for user {user_id}: {e}")
            return False
