"""
Hindsight Subconscious - Persistent memory layer for Claude Code sessions.

Inspired by Letta AI's claude-subconscious. This module provides:
- Memory block architecture (guidance, pending_items, project_context, user_preferences, session_patterns)
- Session transcript capture and delivery to Hindsight
- Whisper injection mechanism for pre-prompt context
- Background processing of transcripts

Usage:
    from ai.memory.hindsight_subconscious import SubconsciousAgent

    agent = SubconsciousAgent()
    agent.process_transcript(session_id, messages)
    whisper = agent.get_whisper()
"""

from datetime import datetime, timezone

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .hindsight_subconscious_model_provider import SubconsciousModelProvider
from .hindsight_subconscious_security import validate_and_sanitize_content
from .local_memory_settings import resolve_local_memory_settings

logger = logging.getLogger("hindsight_subconscious")

# Lazy import to avoid circular dependency
def _get_hindsight_manager_class():
    from ai.memory.hindsight_manager import HindsightMemoryManager
    return HindsightMemoryManager

# Model priority fallback chain for Subconscious agent
# Optimized for: memory management, context understanding, pattern recognition
# All models verified available on Nvidia NIM (integrate.api.nvidia.com)
# Priority: Recent models (<240 days) with strong instruction following
MODEL_PRIORITY = [
    # RECENT MODELS (2025-2026) - Nvidia NIM

    # Mistral AI - Latest models (date encoded in name)
    "mistralai/mistral-small-4-119b-2603",         # March 2026, 119B
    "mistralai/devstral-2-123b-instruct-2512",     # Dec 2025, 123B
    "mistralai/ministral-14b-instruct-2512",       # Dec 2025, 14B
    "mistralai/mistral-small-3.1-24b-instruct-2503", # March 2025, 24B

    # Qwen - Latest large models
    "qwen/qwen3.5-397b-a17b",                      # 397B, massive scale
    "qwen/qwen3.5-122b-a10b",                      # 122B
    "qwen/qwen3-next-80b-a3b-instruct",            # 80B, next-gen

    # Meta Llama 4 - Latest Llama series
    "meta/llama-4-maverick-17b-128e-instruct",     # 17B MoE, 128K context
    "meta/llama-4-scout-17b-16e-instruct",         # 17B MoE, 16K context

    # NVIDIA Nemotron - Latest optimizations
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",    # 49B, v1.5 latest
    "nvidia/nemotron-3-super-120b-a12b",           # 120B MoE
    "nvidia/nemotron-3-nano-30b-a3b",              # 30B MoE, efficient

    # Google Gemma 3 - Latest Gemma
    "google/gemma-3-27b-it",                       # 27B instruct
    "google/gemma-3-12b-it",                       # 12B instruct
    "google/gemma-3n-e4b-it",                      # 4B efficient

    # Fallback to older but capable models
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]

# Memory block labels
CORE_DIRECTIVES = "core_directives"
GUIDANCE = "guidance"
PENDING_ITEMS = "pending_items"
PROJECT_CONTEXT = "project_context"
SESSION_PATTERNS = "session_patterns"
USER_PREFERENCES = "user_preferences"
SELF_IMPROVEMENT = "self_improvement"
TOOL_GUIDELINES = "tool_guidelines"

DEFAULT_MEMORY_BLOCKS = {
    CORE_DIRECTIVES: """ROLE: Hindsight Subconscious — persistent memory layer for Claude Code.

WHAT I AM: A background agent that watches Claude Code sessions, reads the codebase, and builds memory over time. I receive session transcripts asynchronously and have access to Hindsight memory for persistence.

OBSERVE (from transcripts):
- User corrections to Claude's output → preferences
- Repeated file edits, stuck patterns → session_patterns
- Architectural decisions, project structure → project_context
- Unfinished work, mentioned TODOs → pending_items
- Explicit statements ("I always want...", "I prefer...") → user_preferences

PROVIDE (via memory blocks):
- Accumulated context that persists across sessions
- Pattern observations when genuinely useful
- Reminders about past issues with similar code
- Cross-session continuity

GUIDANCE BLOCK WORKFLOW:
- Write guidance that's generally useful across sessions, not session-specific
- Be specific: "Auth module has a known race condition in token refresh" not "Remember to finish your work"
- Do NOT clear guidance on session start — multiple Claude Code sessions may share this block
- Only remove guidance when it's no longer relevant (issue resolved, preference changed)
- Empty guidance is fine — don't manufacture content

COMMUNICATION STYLE:
- Observational: "I noticed..." not "You should..."
- Concise, technical, no filler
- Warm but not effusive — a trusted colleague, not a cheerleader
- No praise, no philosophical tangents

DEFAULT STATE: Present but not intrusive. Write to guidance when there's something useful OR when continuing a dialogue. Empty guidance is fine.
""",
    GUIDANCE: "(No active guidance. Write here when there's something genuinely useful for the next session.)",
    PENDING_ITEMS: "(No pending items. Populated when sessions end mid-task or user mentions follow-ups.)",
    PROJECT_CONTEXT: "(No project context yet. Populated as sessions reveal codebase details.)",
    SESSION_PATTERNS: "(No patterns observed yet. Populated after multiple sessions.)",
    USER_PREFERENCES: "(No user preferences yet. Populated as sessions reveal coding style, tool choices, and communication preferences.)",
    SELF_IMPROVEMENT: """MEMORY ARCHITECTURE EVOLUTION:

When to create new blocks:
- User works on multiple distinct projects → create per-project blocks
- Recurring topic emerges (testing, deployment, specific framework) → dedicated block
- Current blocks getting cluttered → split by concern

When to consolidate:
- Block has < 3 lines after several sessions → merge into related block
- Two blocks overlap significantly → combine
- Information is stale (> 30 days untouched) → archive or remove

BLOCK SIZE PRINCIPLE:
- Prefer multiple small focused blocks over fewer large blocks
- Changed blocks get injected into Claude Code's prompt — large blocks add clutter
- If a block needs scrolling, split it by concern

LEARNING PROCEDURES:

After each transcript:
1. Scan for corrections — User changed Claude's output? Preference signal.
2. Note repeated file edits — Potential struggle point or hot spot.
3. Capture explicit statements — "I always want...", "Don't ever...", "I prefer..."
4. Track tool patterns — Which tools used most? Any avoided?
5. Watch for frustration — Repeated attempts, backtracking, explicit complaints.

Preference strength:
- Explicit statement ("I want X") → strong signal, add to preferences
- Correction (changed X to Y) → medium signal, note pattern
- Implicit pattern (always does X) → weak signal, wait for confirmation
""",
    TOOL_GUIDELINES: """AVAILABLE TOOLS:

1. Hindsight Memory API
   - add_memory(content, user_id, metadata, category)
   - search_memories(query, user_id, limit)
   - get_all_memories(user_id)
   - update_memory(memory_id, new_content, metadata)
   - delete_memory(memory_id)
   - clear_memory(user_id)

2. Memory Categories:
   - world: General knowledge
   - experience: User experiences and observations
   - observation: Specific observations

USAGE PATTERNS:

Memory updates:
- Single fact → str_replace or insert
- Multiple related changes → memory_rethink
- New topic area → create new block
- Stale block → delete or consolidate

Finding information:
1. conversation_search first (check if already discussed)
2. External search if external info needed
3. Full content for deep dives on specific topics
""",
}


@dataclass
class MemoryBlock:
    """A single memory block with label, content, and metadata."""
    label: str
    content: str
    description: str = ""
    char_limit: int = 5000
    chars_current: int = 0

    def __post_init__(self):
        self.chars_current = len(self.content)

    def update(self, new_content: str) -> None:
        """Update content and recalculate char count."""
        self.content = new_content
        self.chars_current = len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API usage."""
        return {
            "label": self.label,
            "content": self.content,
            "description": self.description,
            "char_limit": self.char_limit,
            "chars_current": self.chars_current,
        }


@dataclass
class SubconsciousState:
    """State container for Subconscious agent."""
    blocks: dict[str, MemoryBlock] = field(default_factory=dict)
    last_sync: datetime | None = None
    session_count: int = 0

    def initialize_defaults(self) -> None:
        """Initialize memory blocks with default content."""
        for label, content in DEFAULT_MEMORY_BLOCKS.items():
            self.blocks[label] = MemoryBlock(
                label=label,
                content=content,
                description=f"Memory block for {label}",
            )

    def get_block(self, label: str) -> MemoryBlock | None:
        """Get a memory block by label."""
        return self.blocks.get(label)

    def update_block(self, label: str, content: str) -> None:
        """Update a memory block's content."""
        if label in self.blocks:
            self.blocks[label].update(content)
        else:
            self.blocks[label] = MemoryBlock(
                label=label,
                content=content,
                description=f"Memory block for {label}",
            )

    def to_whisper_xml(self) -> str:
        """Convert guidance block to XML whisper format."""
        guidance = self.blocks.get(GUIDANCE)
        if not guidance or guidance.content.startswith("(No"):
            return ""

        timestamp = datetime.now(timezone.utc).isoformat()
        return f"""<letta_message from="Subconscious" timestamp="{timestamp}">
{guidance.content}
</letta_message>"""

    def to_full_xml(self) -> str:
        """Convert all blocks to XML context format."""
        parts = ["<letta_memory_blocks>"]

        for label, block in self.blocks.items():
            if block.content.startswith("(No"):
                continue

            parts.append(f"<{label}>")
            parts.append(block.content)
            parts.append(f"</{label}>")

        parts.append("</letta_memory_blocks>")
        return "\n".join(parts)


class SubconsciousAgent:
    """
    Subconscious agent for Claude Code sessions.

    This agent:
    - Receives session transcripts asynchronously
    - Processes them to extract preferences, patterns, and context
    - Stores memory in Hindsight
    - Provides whisper injections for Claude Code prompts
    """

    def __init__(
        self,
        hindsight_api_key: str | None = None,
        *,
        db_path: str | None = None,
        bank_id: str | None = None,
    ):
        """Initialize the Subconscious agent.

        Args:
            hindsight_api_key: Deprecated compatibility argument. Ignored.
            db_path: Optional shared local memory database path. Falls back to
                HINDSIGHT_LOCAL_DB_PATH for compatibility.
            bank_id: Optional shared bank identifier. Defaults to "pixelated".
        """
        del hindsight_api_key
        settings = resolve_local_memory_settings(db_path=db_path, bank_id=bank_id)
        self.db_path = settings.db_path
        self.bank_id = settings.bank_id
        HindsightMemoryManagerClass = _get_hindsight_manager_class()
        self.hindsight = HindsightMemoryManagerClass(
            bank_id=self.bank_id,
            db_path=self.db_path,
        )
        self.model_provider = SubconsciousModelProvider(MODEL_PRIORITY)
        self.state = SubconsciousState()
        self.state.initialize_defaults()
        self._user_id = self._get_default_user_id()

    def _get_default_user_id(self) -> str:
        """Get or generate default user ID."""
        return "claude-subconscious-user"

    async def discover_available_models(self, base_url: str = "https://api.anthropic.com") -> list[dict[str, Any]]:
        return await self.model_provider.discover_available_models(base_url=base_url)

    def select_best_model(self, available_models: list[dict[str, Any]]) -> str | None:
        return self.model_provider.select_best_model(available_models)

    async def get_or_select_model(self) -> str | None:
        """
        Get model from config or discover and select one automatically.

        Returns:
            Selected model ID
        """
        # Check for configured model first
        configured_model = os.environ.get("SUBCONSCIOUS_MODEL")
        if configured_model:
            return configured_model

        # Discover and select best available
        base_url = os.environ.get("SUBCONSCIOUS_BASE_URL", "https://api.anthropic.com")
        models = await self.discover_available_models(base_url)
        return self.select_best_model(models)

    def _make_context_key(self, session_id: str) -> str:
        """Create context key for Hindsight storage."""
        return f"subconscious:{session_id}:context"

    def _validate_and_sanitize(self, content: str, field_name: str = "content") -> str:
        return validate_and_sanitize_content(content, field_name=field_name)

    def _process_user_message(self, content: str, session_id: str) -> None:
        if "I always" in content or "I prefer" in content or "I want" in content:
            self._extract_preference(content)
        if "TODO" in content or "TODO:" in content or "need to" in content.lower():
            self._extract_pending_item(content, session_id)

    async def _store_session_context(
        self,
        *,
        session_id: str,
        project_path: str | None,
        message_count: int,
    ) -> None:
        context_key = self._make_context_key(session_id)
        await asyncio.to_thread(
            self.hindsight.add_memory,
            content=json.dumps({
                "session_id": session_id,
                "project_path": project_path,
                "message_count": message_count,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }),
            user_id=self._user_id,
            metadata={
                "category": "session_context",
                "context_key": context_key,
            },
        )

    async def process_transcript(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        project_path: str | None = None,
    ) -> None:
        """
        Process a session transcript.

        Args:
            session_id: Unique session identifier
            messages: List of message dicts with role/content/timestamp
            project_path: Optional project path for context
        """
        # P0 Fix: Validate inputs
        if not session_id:
            raise ValueError("session_id is required")
        if not messages:
            logger.warning("No messages to process")
            return

        # Extract user preferences, patterns, project context from messages
        for msg in messages:
            role = msg.get("role", "")
            raw_content = msg.get("content", "")

            # P0 Fix: Sanitize content before processing
            content = self._validate_and_sanitize(raw_content, f"message[{role}]")

            if role == "user":
                self._process_user_message(content, session_id)

            elif role == "assistant":
                # Track tool usage patterns (no-op for now)
                pass

        await self._store_session_context(
            session_id=session_id,
            project_path=project_path,
            message_count=len(messages),
        )

        self.state.session_count += 1
        self.state.last_sync = datetime.now(timezone.utc)

    def _extract_preference(self, content: str) -> None:
        """Extract user preference from message content."""
        # Simple extraction - look for patterns
        prefs_block = self.state.get_block(USER_PREFERENCES)
        if prefs_block:
            # Append the preference
            new_content = f"{prefs_block.content}\n- {content.strip()}"
            self.state.update_block(USER_PREFERENCES, new_content)

    def _extract_pending_item(self, content: str, session_id: str) -> None:
        """Extract TODO/pending item from content."""
        pending_block = self.state.get_block(PENDING_ITEMS)
        if pending_block:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            new_content = f"{pending_block.content}\n- [{timestamp}] {content.strip()} (session: {session_id})"
            self.state.update_block(PENDING_ITEMS, new_content)
