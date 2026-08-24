"""
Agent Memory Tools Module.

Provides async memory tools for integration with agent frameworks.

These tools now target the repository's shared local memory service instead of
creating their own cloud-backed Foresight client.
"""

import logging
from dataclasses import dataclass
from typing import Any

from ai.research.local_foresight_manager import LocalForesightMemoryManager
from ai.research.local_memory_settings import resolve_local_memory_settings

logger = logging.getLogger("agent_memory_tools")


@dataclass
class AgentContext:
    """
    Context for agent memory operations.

    Tracks user identity, session, and agent for memory partitioning.
    """

    user_id: str
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    scope: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Convert context to metadata dict for Foresight."""
        metadata = {"user_id": self.user_id}
        if self.session_id:
            metadata["session_id"] = self.session_id
        if self.agent_id:
            metadata["agent_id"] = self.agent_id
        if self.run_id:
            metadata["run_id"] = self.run_id
        if self.scope:
            metadata["scope"] = self.scope
        return metadata


class AgentMemoryTools:
    """
    Async memory tools for agent frameworks.

    Wraps Foresight operations as callable tools that can be registered with
    OpenAI Agent SDK, LangChain, or other agent frameworks.

    Usage:
        import os
        tools = AgentMemoryTools(api_key=os.environ.get("API_KEY"))
        await tools.add_to_memory(context, "User prefers morning meetings")
        results = await tools.search_memory(context, "meeting preferences")
    """

    def __init__(
        self,
        api_key: str | None = None,
        memory_client: Any | None = None,
        db_path: str | None = None,
        bank_id: str | None = None,
    ):
        """
        Initialize agent memory tools.

        Args:
            api_key: Deprecated compatibility argument. Ignored.
            memory_client: Optional pre-configured shared local memory manager
            db_path: Optional path to the shared local memory database
            bank_id: Optional bank identifier
        """
        del api_key
        if memory_client:
            self.memory = memory_client
        else:
            settings = resolve_local_memory_settings(db_path=db_path, bank_id=bank_id)
            self.memory = LocalForesightMemoryManager(
                db_path=settings.db_path,
                bank_id=settings.bank_id,
            )

    async def add_to_memory(
        self,
        context: AgentContext,
        content: str,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add information to memory.

        This is the primary tool for storing new memories during agent execution.

        Args:
            context: Agent context with user/session info
            content: Content to store in memory
            category: Optional category for the memory
            metadata: Optional additional metadata

        Returns:
            Memory ID of the stored content

        Example:
            memory_id = await tools.add_to_memory(
                context,
                "User mentioned they have a dog named Max"
            )
        """
        try:
            full_metadata = context.to_metadata()
            if category:
                full_metadata["category"] = category
            if metadata:
                full_metadata.update(metadata)

            return self.memory.add_memory(
                content=content,
                user_id=context.user_id,
                metadata=full_metadata,
            )

        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            raise

    async def search_memory(
        self,
        context: AgentContext,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search relevant memories.

        Performs semantic search across the user's memories to find relevant context.

        Args:
            context: Agent context with user info
            query: Search query
            limit: Maximum number of results

        Returns:
            List of relevant memory objects

        Example:
            memories = await tools.search_memory(context, "favorite food")
            for m in memories:
                print(m["memory"])
        """
        try:
            result = self.memory.search_memories(
                query=query,
                user_id=context.user_id,
                limit=limit,
            )
            return result if isinstance(result, list) else []

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []

    async def get_all_memory(
        self,
        context: AgentContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all memories for the user.

        Args:
            context: Agent context with user info
            limit: Maximum number of results

        Returns:
            List of all memory objects for the user
        """
        try:
            memories = self.memory.get_all_memories(user_id=context.user_id, limit=limit)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            return []

    async def get_memory(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: ID of the memory to retrieve

        Returns:
            Memory object or None if not found
        """
        try:
            return self.memory.get_memory(memory_id)
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None

    async def update_memory(
        self,
        context: AgentContext,
        memory_id: str,
        new_content: str,
    ) -> bool:
        """
        Update an existing memory.

        Use this to correct or refine existing information without creating duplicates.

        Args:
            context: Agent context for verification
            memory_id: ID of the memory to update
            new_content: New content for the memory

        Returns:
            True if update succeeded
        """
        try:
            self.memory.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                metadata=context.to_metadata(),
            )
            logger.info(f"Updated memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False

    async def delete_memory(
        self,
        _context: AgentContext,
        memory_id: str,
    ) -> bool:
        """
        Delete a specific memory.

        Args:
            context: Agent context for verification
            memory_id: ID of the memory to delete

        Returns:
            True if deletion succeeded
        """
        try:
            self.memory.delete_memory(memory_id)
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for these memory tools.

        Returns definitions that can be passed to OpenAI Agent SDK or
        similar frameworks.

        Returns:
            List of tool definition dicts
        """
        return [
            {
                "name": "add_to_memory",
                "description": "Store important information about the user in long-term memory. Use this to remember facts, preferences, or context that will be useful in future conversations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to store in memory",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category for the memory (e.g., 'preference', 'fact', 'context')",
                            "enum": ["preference", "fact", "context", "therapeutic", "goal"],
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "search_memory",
                "description": "Search for relevant memories about the user. Use this to recall previously stored information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant memories",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_all_memory",
                "description": "Retrieve all stored memories for the current user. Use sparingly as it returns all data.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "update_memory",
                "description": "Update an existing memory with corrected information. Use when you need to fix or refine previously stored data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to update",
                        },
                        "new_content": {
                            "type": "string",
                            "description": "The updated content for the memory",
                        },
                    },
                    "required": ["memory_id", "new_content"],
                },
            },
            {
                "name": "delete_memory",
                "description": "Delete a specific memory. Use when information is no longer relevant or was stored in error.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete",
                        }
                    },
                    "required": ["memory_id"],
                },
            },
        ]


# Convenience function for creating agent tools with context bound
def create_memory_tool_handler(
    tools: AgentMemoryTools,
    context: AgentContext,
):
    """
    Create a tool handler function for agent frameworks.

    This returns a function that can handle tool calls from an agent,
    routing them to the appropriate memory operation.

    Args:
        tools: AgentMemoryTools instance
        context: Agent context for memory operations

    Returns:
        Async function that handles tool calls
    """

    async def handle_tool_call(name: str, arguments: dict[str, Any]) -> Any:
        """Handle a memory tool call from an agent."""
        if name == "add_to_memory":
            return await tools.add_to_memory(
                context,
                arguments["content"],
                category=arguments.get("category"),
            )
        if name == "search_memory":
            return await tools.search_memory(context, arguments["query"])
        if name == "get_all_memory":
            return await tools.get_all_memory(context)
        if name == "update_memory":
            return await tools.update_memory(
                context,
                arguments["memory_id"],
                arguments["new_content"],
            )
        if name == "delete_memory":
            return await tools.delete_memory(context, arguments["memory_id"])
        raise ValueError(f"Unknown tool: {name}")

    return handle_tool_call
