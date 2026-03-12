"""
Multi-Agent Memory Layer.

Provides shared memory infrastructure for multi-agent therapeutic workflows,
with clear communication phases and consistent memory access helpers.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from mem0 import MemoryClient
except ImportError:
    try:
        from mem0ai import MemoryClient
    except ImportError:
        MemoryClient = None

logger = logging.getLogger("multi_agent_memory")


class AgentRole(str, Enum):
    """Predefined agent roles for therapeutic workflows."""

    TRAINER = "trainer"
    PRACTICE = "practice"
    FEEDBACK = "feedback"
    SUPERVISOR = "supervisor"
    COORDINATOR = "coordinator"


class MemoryScope(str, Enum):
    """Memory visibility scopes."""

    PRIVATE = "private"
    SHARED = "shared"
    USER = "user"
    GLOBAL = "global"


@dataclass
class AgentIdentity:
    """Identity for an agent in a multi-agent system."""

    agent_id: str
    role: AgentRole
    name: str = ""
    capabilities: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.role.value.title()} Agent"


@dataclass
class CollaborationContext:
    """Context for multi-agent collaboration.

    Tracks the user, session, and participating agents for memory partitioning.
    """

    user_id: str
    session_id: str
    agents: List[AgentIdentity] = field(default_factory=list)
    current_agent: Optional[AgentIdentity] = None

    def get_memory_key(self, scope: MemoryScope) -> Dict[str, str]:
        """Get memory keys for requested scope."""
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
        return {}


@dataclass(frozen=True)
class _HandoffSummary:
    handoff_memory_id: str
    transferred: int
    source_agent: Optional[str]
    target_agent: str


def _normalize_records(raw_result: Any) -> List[Dict[str, Any]]:
    """Normalize MemoryClient responses into a list of record dicts."""
    if raw_result is None:
        return []
    if isinstance(raw_result, list):
        return raw_result
    if isinstance(raw_result, dict):
        return raw_result.get("results", []) or []
    return []


def _extract_memory_id(raw_result: Any) -> str:
    """Extract the first memory id from a write response."""
    records = _normalize_records(raw_result)
    if not records:
        return "stored"
    return records[0].get("id", "stored")


def _scope_filter(
    memories: List[Dict[str, Any]], context: CollaborationContext, scope: MemoryScope
) -> List[Dict[str, Any]]:
    """Filter records by the requested scope."""
    if scope == MemoryScope.PRIVATE:
        if not context.current_agent:
            return []
        agent_id = context.current_agent.agent_id
        return [
            item
            for item in memories
            if item.get("metadata", {}).get("session_id") == context.session_id
            and item.get("metadata", {}).get("scope") == MemoryScope.PRIVATE.value
            and (
                item.get("metadata", {}).get("agent_id") == agent_id
                or item.get("metadata", {}).get("source_agent") == agent_id
            )
        ]
    if scope == MemoryScope.SHARED:
        return [
            item
            for item in memories
            if item.get("metadata", {}).get("session_id") == context.session_id
            and item.get("metadata", {}).get("scope") in (
                MemoryScope.SHARED.value,
                None,
            )
        ]
    if scope == MemoryScope.USER:
        return [
            item for item in memories if item.get("metadata", {}).get("user_id") == context.user_id
        ]
    return memories


def _log_stage(context: CollaborationContext, stage: str, details: Optional[Dict[str, Any]] = None):
    """Log a compact stage message for cross-module observability."""
    payload = {
        "user_id": context.user_id,
        "session_id": context.session_id,
        "stage": stage,
        "agent": context.current_agent.agent_id if context.current_agent else None,
    }
    if details:
        payload.update(details)
    logger.info("multi-agent-memory-stage=%s details=%s", payload["stage"], payload)


class MultiAgentMemory:
    """Shared memory layer for multi-agent therapeutic workflows."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        memory_client: Optional[Any] = None,
    ):
        if memory_client:
            self.memory = memory_client
        elif api_key and MemoryClient:
            self.memory = MemoryClient(api_key=api_key)
        else:
            logger.warning("No memory client available, using null memory")
            self.memory = self._create_null_memory()

    def _create_null_memory(self):
        """Create a null memory shim for testing."""

        class NullMemory:
            def add(self, *args, **kwargs):
                return {"results": [{"id": "null-memory-id"}]}

            def search(self, *args, **kwargs):
                return {"results": []}

            def get_all(self, *args, **kwargs):
                return {"results": []}

            def get(self, *args, **kwargs):
                return None

            def update(self, *args, **kwargs):
                return {"message": "updated"}

            def delete(self, *args, **kwargs):
                return {"message": "deleted"}

        return NullMemory()

    def _call_memory(self, operation: str, *args, **kwargs) -> Any:
        method = getattr(self.memory, operation)
        return method(*args, **kwargs)

    def _collect_memories(self, user_id: str) -> List[Dict[str, Any]]:
        return _normalize_records(self._call_memory("get_all", user_id=user_id))

    async def store_agent_memory(
        self,
        context: CollaborationContext,
        content: str,
        scope: MemoryScope = MemoryScope.SHARED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store memory from an agent and return a stable memory id."""
        try:
            _log_stage(context, "store-start", {"scope": scope.value})
            full_metadata = context.get_memory_key(scope)
            full_metadata["scope"] = scope.value
            if context.current_agent:
                full_metadata["source_agent"] = context.current_agent.agent_id
                full_metadata["source_role"] = context.current_agent.role.value
            if metadata:
                full_metadata.update(metadata)

            result = self._call_memory(
                "add",
                content,
                user_id=context.user_id,
                metadata=full_metadata,
            )
            memory_id = _extract_memory_id(result)
            _log_stage(context, "store-complete", {"memory_id": memory_id, "scope": scope.value})
            return memory_id
        except Exception as e:
            logger.error("Error storing agent memory: %s", e)
            raise

    async def get_shared_context(
        self,
        context: CollaborationContext,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get shared memories for this user/session."""
        try:
            memories = _scope_filter(
                self._collect_memories(context.user_id),
                context,
                MemoryScope.SHARED,
            )
            return memories[:limit]
        except Exception as e:
            logger.error("Error getting shared context: %s", e)
            return []

    async def search_agent_memories(
        self,
        context: CollaborationContext,
        query: str,
        scope: MemoryScope = MemoryScope.SHARED,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memories with scope filtering."""
        try:
            result = self._call_memory(
                "search",
                query,
                user_id=context.user_id,
                limit=limit * 2,
            )
            memories = _normalize_records(result)
            return _scope_filter(memories, context, scope)[:limit]
        except Exception as e:
            logger.error("Error searching agent memories: %s", e)
            return []

    async def handoff_to_agent(
        self,
        context: CollaborationContext,
        target_agent: AgentIdentity,
        summary: str,
        transfer_memories: bool = True,
    ) -> Dict[str, Any]:
        """Perform handoff with optional private memory transfer."""
        try:
            source_agent = context.current_agent
            _log_stage(
                context,
                "handoff-start",
                {
                    "source_agent": source_agent.agent_id if source_agent else None,
                    "target_agent": target_agent.agent_id,
                    "transfer_memories": transfer_memories,
                },
            )

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

            transferred = 0
            if transfer_memories and source_agent:
                source_context = CollaborationContext(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    agents=context.agents,
                    current_agent=source_agent,
                )
                private_memories = _scope_filter(
                    self._collect_memories(context.user_id),
                    source_context,
                    MemoryScope.PRIVATE,
                )
                for memory_record in private_memories:
                    payload = memory_record.get("content") or memory_record.get("memory")
                    if not payload:
                        continue
                    self._call_memory(
                        "add",
                        f"Transferred from {source_agent.agent_id}: {payload}",
                        user_id=context.user_id,
                        metadata={
                            **(memory_record.get("metadata") or {}),
                            "scope": MemoryScope.SHARED.value,
                            "type": "handoff_transfer",
                            "transferred_from": source_agent.agent_id,
                            "transferred_to": target_agent.agent_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    transferred += 1

            shared_context = await self.get_shared_context(context, limit=10)
            new_context = CollaborationContext(
                user_id=context.user_id,
                session_id=context.session_id,
                agents=context.agents + [target_agent],
                current_agent=target_agent,
            )
            handoff_summary = _HandoffSummary(
                handoff_memory_id=handoff_memory,
                transferred=transferred,
                source_agent=source_agent.agent_id if source_agent else None,
                target_agent=target_agent.agent_id,
            )
            _log_stage(context, "handoff-complete", handoff_summary.__dict__)
            return {
                "success": True,
                "handoff_memory_id": handoff_memory,
                "handoff": handoff_summary.__dict__,
                "new_context": new_context,
                "shared_memories": shared_context,
                "summary": summary,
            }

        except Exception as e:
            logger.error("Error performing handoff: %s", e)
            return {"success": False, "error": str(e)}

    async def broadcast_to_agents(
        self,
        context: CollaborationContext,
        message: str,
        message_type: str = "broadcast",
    ) -> str:
        """Broadcast a message to all agents in the session."""
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
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get memory history for a specific agent or all agents in session."""
        try:
            session_memories = [
                item
                for item in self._collect_memories(context.user_id)
                if item.get("metadata", {}).get("session_id") == context.session_id
            ]
            if agent_id:
                session_memories = [
                    item
                    for item in session_memories
                    if item.get("metadata", {}).get("source_agent") == agent_id
                    or item.get("metadata", {}).get("agent_id") == agent_id
                ]
            return session_memories[:limit]
        except Exception as e:
            logger.error("Error getting agent history: %s", e)
            return []


def create_empathy_gym_context(
    user_id: str,
    session_id: str,
    current_role: AgentRole = AgentRole.TRAINER,
) -> CollaborationContext:
    """Create default empathy-gym context with the three core agents."""
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
        (agent for agent in agents if agent.role == current_role),
        agents[0],
    )
    return CollaborationContext(
        user_id=user_id,
        session_id=session_id,
        agents=agents,
        current_agent=current_agent,
    )
