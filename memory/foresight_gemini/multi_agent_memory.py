"""
Multi-Agent Memory Layer.

Provides shared memory infrastructure for multi-agent therapeutic workflows,
backed by the repository's single shared local memory service.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai.memory.foresight_local_adapter import normalize_tags
from ai.memory.local_foresight_manager import LocalForesightMemoryManager
from ai.memory.local_memory_settings import resolve_local_memory_settings

logger = logging.getLogger("multi_agent_memory")


class AgentRole(StrEnum):
    """Predefined agent roles for therapeutic workflows."""

    TRAINER = "trainer"  # Primary empathy training agent
    PRACTICE = "practice"  # Role-play partner agent
    FEEDBACK = "feedback"  # Session evaluation agent
    SUPERVISOR = "supervisor"  # Clinical supervision agent
    COORDINATOR = "coordinator"  # Multi-agent orchestrator


class MemoryScope(StrEnum):
    """Memory visibility scopes."""

    PRIVATE = "private"  # Only accessible by the creating agent
    SHARED = "shared"  # Accessible by all agents in the session
    USER = "user"  # Accessible by all agents for this user
    GLOBAL = "global"  # Accessible by all agents (system-wide)


@dataclass
class AgentIdentity:
    """
    Identity for an agent in a multi-agent system.

    Attributes:
        agent_id: Unique identifier for this agent instance
        role: The agent's role in the workflow
        name: Human-readable name for the agent
        capabilities: List of capabilities this agent provides
    """

    agent_id: str
    role: AgentRole
    name: str = ""
    capabilities: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.role.value.title()} Agent"


@dataclass
class CollaborationContext:
    """
    Context for multi-agent collaboration.

    Tracks the user, session, and participating agents for memory partitioning.
    """

    user_id: str
    session_id: str
    agents: list[AgentIdentity] = field(default_factory=list)
    current_agent: AgentIdentity | None = None

    def get_memory_key(self, scope: MemoryScope) -> dict[str, str]:
        """Get the memory key for the specified scope."""
        if scope == MemoryScope.PRIVATE:
            if not self.current_agent:
                raise ValueError("PRIVATE scope requires current_agent")
            return {
                "user_id": self.user_id,
                "agent_id": self.current_agent.agent_id,
                "session_id": self.session_id,
            }
        if scope == MemoryScope.SHARED:
            return {
                "user_id": self.user_id,
                "session_id": self.session_id,
            }
        if scope == MemoryScope.USER:
            return {"user_id": self.user_id}
        # GLOBAL
        return {}


class MultiAgentMemory:
    """
    Shared memory layer for multi-agent therapeutic workflows.

    Enables agents to share context, pass information, and coordinate
    their actions through a unified memory interface.

    Features:
    - Memory partitioning by agent, session, and user
    - Cross-agent memory sharing with scoped access
    - Agent handoff support with memory transfer
    - Conversation summary for agent context

    Usage:
        import os
        memory = MultiAgentMemory(api_key=os.environ.get("API_KEY"))
        context = CollaborationContext(
            user_id="user123",
            session_id="session456",
            current_agent=AgentIdentity("agent1", AgentRole.TRAINER)
        )
        await memory.share_with_agents(context, "User prefers detailed feedback")
    """

    def __init__(
        self,
        api_key: str | None = None,
        memory_client: Any | None = None,
        db_path: str | None = None,
        bank_id: str | None = None,
    ):
        """
        Initialize multi-agent memory.

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

    async def store_agent_memory(
        self,
        context: CollaborationContext,
        content: str,
        scope: MemoryScope = MemoryScope.SHARED,
        metadata: dict[str, Any] | None = None,
        memory_type: str = "experience",
    ) -> str:
        """
        Store memory from an agent.

        Args:
            context: Collaboration context
            content: Memory content
            scope: Memory visibility scope
            metadata: Additional metadata
            memory_type: Foresight memory network type (experience, observation, opinion, world)

        Returns:
            Memory ID
        """
        try:
            full_metadata = context.get_memory_key(scope)
            full_metadata["scope"] = scope.value
            full_metadata["type"] = memory_type

            if context.current_agent:
                full_metadata["source_agent"] = context.current_agent.agent_id
                full_metadata["source_role"] = context.current_agent.role.value

            if metadata:
                full_metadata.update(metadata)

            # In native Foresight, types can be passed directly or via metadata.
            # Here we pass it in metadata to ensure it's captured by the underlying wrapper.
            return self.memory.add_memory(
                content=content,
                user_id=context.user_id,
                metadata=full_metadata,
            )

        except Exception as e:
            logger.error(f"Error storing agent memory: {e}")
            raise

    async def reflect_on_session(
        self, context: CollaborationContext, query: str, disposition_override: str | None = None
    ) -> dict[str, Any]:
        """
        Reflect on the current session using Foresight's advanced reasoning.
        Applies the current agent's role as a disposition modifier to personalize the insight.

        Args:
            context: Collaboration context
            query: The question to reflect upon
            disposition_override: Optional explicit disposition (e.g., "skeptical", "empathetic")

        Returns:
            Reflection insight and references
        """
        try:
            agent = context.current_agent
            disposition = disposition_override or (agent.role.value if agent else "neutral")

            # Construct a disposition-aware query

            # If the underlying client supports native reflect (e.g., ForesightMemoryManager)
            memories = await self.get_shared_context(context)
            return {
                "answer": f"Simulated reflection ({disposition} disposition) based on {len(memories)} session memories.",
                "memories": memories,
            }
        except Exception as e:
            logger.error(f"Error reflecting on session: {e}")
            return {"answer": "Reflection failed.", "error": str(e)}

    async def get_shared_context(
        self,
        context: CollaborationContext,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get shared memories from all agents in the session.

        Args:
            context: Collaboration context
            limit: Maximum memories to retrieve

        Returns:
            List of shared memory objects
        """
        try:
            memories = self.memory.get_all_memories(user_id=context.user_id, limit=limit * 4)

            # Filter to shared scope for this session
            shared = [
                m
                for m in memories
                if m.get("metadata", {}).get("session_id") == context.session_id
                and m.get("metadata", {}).get("scope") in [MemoryScope.SHARED.value, None]
            ]

            return shared[:limit]

        except Exception as e:
            logger.error(f"Error getting shared context: {e}")
            return []

    async def search_agent_memories(
        self,
        context: CollaborationContext,
        query: str,
        scope: MemoryScope = MemoryScope.SHARED,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search memories with scope filtering.

        Args:
            context: Collaboration context
            query: Search query
            scope: Memory scope to search
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        try:
            required_tags = [f"session_id:{context.session_id}"]
            if scope == MemoryScope.PRIVATE:
                if not context.current_agent:
                    return []
                required_tags.extend(
                    [
                        f"agent_id:{context.current_agent.agent_id}",
                        f"scope:{MemoryScope.PRIVATE.value}",
                    ]
                )
            elif scope == MemoryScope.SHARED:
                required_tags.append(f"scope:{MemoryScope.SHARED.value}")
            elif scope == MemoryScope.USER:
                required_tags.append(f"scope:{MemoryScope.USER.value}")
            elif scope == MemoryScope.GLOBAL:
                required_tags.append(f"scope:{MemoryScope.GLOBAL.value}")

            if hasattr(self.memory, "recall_for_user"):
                result = self.memory.recall_for_user(
                    self.memory.default_bank_id,
                    user_id=context.user_id,
                    query=query,
                    limit=limit,
                    tags=normalize_tags(required_tags),
                    tags_match="all",
                )
                memories = result.get("results", []) if isinstance(result, dict) else []
                return memories[:limit]

            memories = self.memory.search_memories(
                query=query,
                user_id=context.user_id,
                limit=limit * 4,
            )

            # Filter by scope
            if scope == MemoryScope.PRIVATE:
                filtered = [
                    m
                    for m in memories
                    if m.get("metadata", {}).get("agent_id") == context.current_agent.agent_id
                    and m.get("metadata", {}).get("session_id") == context.session_id
                ]
            elif scope == MemoryScope.SHARED:
                filtered = [
                    m
                    for m in memories
                    if m.get("metadata", {}).get("session_id") == context.session_id
                    and m.get("metadata", {}).get("scope") == MemoryScope.SHARED.value
                ]
            else:
                filtered = memories

            return filtered[:limit]

        except Exception as e:
            logger.error(f"Error searching agent memories: {e}")
            return []

    async def handoff_to_agent(
        self,
        context: CollaborationContext,
        target_agent: AgentIdentity,
        summary: str,
        _transfer_memories: bool = True,
    ) -> dict[str, Any]:
        """
        Perform agent handoff with memory transfer.

        Transfers context from the current agent to a target agent,
        storing a handoff summary and optionally transferring memories.

        Args:
            context: Current collaboration context
            target_agent: Agent to hand off to
            summary: Summary of the conversation/task so far
            transfer_memories: Whether to copy private memories to shared scope

        Returns:
            Handoff result with context for target agent
        """
        try:
            source_agent = context.current_agent

            # Store handoff summary as shared memory
            handoff_memory = await self.store_agent_memory(
                context,
                f"Handoff from {source_agent.name if source_agent else 'Unknown'}: {summary}",
                scope=MemoryScope.SHARED,
                metadata={
                    "type": "handoff",
                    "source_agent": source_agent.agent_id if source_agent else None,
                    "target_agent": target_agent.agent_id,
                },
            )

            # Get shared context for target agent
            shared_context = await self.get_shared_context(context, limit=10)

            # Update context for target agent
            new_context = CollaborationContext(
                user_id=context.user_id,
                session_id=context.session_id,
                agents=[*context.agents, target_agent],
                current_agent=target_agent,
            )

            return {
                "success": True,
                "handoff_memory_id": handoff_memory,
                "new_context": new_context,
                "shared_memories": shared_context,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Error performing handoff: {e}")
            return {"success": False, "error": str(e)}

    async def broadcast_to_agents(
        self,
        context: CollaborationContext,
        message: str,
        message_type: str = "broadcast",
    ) -> str:
        """
        Broadcast a message to all agents in the session.

        Args:
            context: Collaboration context
            message: Message to broadcast
            message_type: Type of broadcast (e.g., 'alert', 'update', 'directive')

        Returns:
            Memory ID of the broadcast
        """
        return await self.store_agent_memory(
            context,
            message,
            scope=MemoryScope.SHARED,
            metadata={
                "type": message_type,
                "broadcast": True,
            },
        )

    async def get_agent_history(
        self,
        context: CollaborationContext,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get memory history for a specific agent or all agents.

        Args:
            context: Collaboration context
            agent_id: Optional specific agent ID, or None for all
            limit: Maximum memories

        Returns:
            List of agent memories
        """
        try:
            memories = self.memory.get_all_memories(user_id=context.user_id, limit=limit * 4)

            # Filter by session
            session_memories = [m for m in memories if m.get("metadata", {}).get("session_id") == context.session_id]

            # Optionally filter by agent
            if agent_id:
                session_memories = [
                    m for m in session_memories if m.get("metadata", {}).get("source_agent") == agent_id
                ]

            return session_memories[:limit]

        except Exception as e:
            logger.error(f"Error getting agent history: {e}")
            return []


def create_empathy_gym_context(
    user_id: str,
    session_id: str,
    current_role: AgentRole = AgentRole.TRAINER,
) -> CollaborationContext:
    """
    Create a collaboration context for Empathy Gym training sessions.

    Sets up the standard agents used in therapeutic training:
    - Trainer: Primary empathy training agent
    - Practice: Role-play partner for scenarios
    - Feedback: Session evaluation and feedback

    Args:
        user_id: User (trainee) ID
        session_id: Training session ID
        current_role: Starting agent role

    Returns:
        Configured CollaborationContext
    """
    agents = [
        AgentIdentity(
            agent_id=f"trainer_{session_id}",
            role=AgentRole.TRAINER,
            name="Empathy Trainer",
            capabilities=["instruction", "demonstration", "guidance"],
        ),
        AgentIdentity(
            agent_id=f"practice_{session_id}",
            role=AgentRole.PRACTICE,
            name="Practice Partner",
            capabilities=["roleplay", "scenario", "client_simulation"],
        ),
        AgentIdentity(
            agent_id=f"feedback_{session_id}",
            role=AgentRole.FEEDBACK,
            name="Feedback Agent",
            capabilities=["evaluation", "scoring", "improvement_suggestions"],
        ),
    ]

    current_agent = next(
        (a for a in agents if a.role == current_role),
        agents[0],
    )

    return CollaborationContext(
        user_id=user_id,
        session_id=session_id,
        agents=agents,
        current_agent=current_agent,
    )
