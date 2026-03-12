# Pixelated Memory Stack (Mem0 + NVIDIA NIM)

Build memory-powered coaching and agent workflows in three patterns:

- `NIMMem0Manager` for single-assistant memory conversations.
- `MultiAgentMemory` for multi-agent handoffs and shared context.
- `AgentMemoryTools` for exposing memory as async tools inside agent frameworks.

This page is the **customer-first** entrypoint: quick defaults, copy-paste snippets, and clear decision points.

---

## ⚡ Fastest path (recommended)

Run the demo script first to confirm your setup:

```bash
cd ai
python3 core/scripts/test_mem0_nim.py --mode full --user demo_user
```

It will:

- Start a single-agent flow with continuity.
- Show memory telemetry from each turn.
- Run a multi-agent handoff sample.

## 1) Single-agent flow

Best for:

- Conversational assistants
- Chatbots with user context continuity
- Internal tools that need memory recall + generation

```python
from ai.core.memory.mem0_nim import NIMMem0Manager, NIMMem0Config

config = NIMMem0Config(
    nim_api_key="YOUR_NIM_API_KEY",
    user_id="user_123",
)
manager = NIMMem0Manager(config)

result = await manager.get_response("What should I work on this week?")
print(result["response"])
```

### Response payload (what you can trust)

- `response`: assistant message
- `memories_used`: how many memories were incorporated
- `request_id`: traceable request id for logs
- `latency_ms`: response latency
- `store_report`: summary of what was persisted this turn
- `crisis_flagged`: risk-aware safety signal

## 2) Multi-agent flow

Best for:

- Trainer / practice / feedback agent handoff loops
- Coordinated agent teams
- Session-level context that must be shared intentionally

```python
from ai.core.memory.mem0_nim import (
    AgentIdentity,
    AgentRole,
    MemoryScope,
    MultiAgentMemory,
    create_empathy_gym_context,
)

memory = MultiAgentMemory(api_key="YOUR_MEM0_KEY")
context = create_empathy_gym_context(user_id="user_123", session_id="session_abc")

await memory.store_agent_memory(
    context,
    "Opening coaching protocol loaded.",
    scope=MemoryScope.SHARED,
)

handoff_target = AgentIdentity(
    agent_id="feedback_session_abc",
    role=AgentRole.FEEDBACK,
    name="Feedback Agent",
)
handoff = await memory.handoff_to_agent(
    context,
    handoff_target,
    "Transitioning to scoring + feedback phase",
)
print(handoff["handoff"]["transferred"])
```

## 3) Agent SDK tool usage

Best for:

- OpenAI Agent SDK workflows
- LangChain tool patterns
- Framework-native memory tools

```python
from ai.core.memory.mem0_nim import AgentContext, AgentMemoryTools

tools = AgentMemoryTools(api_key="YOUR_MEM0_KEY")
context = AgentContext(
    user_id="user_123",
    session_id="session_abc",
    agent_id="coach",
)

await tools.add_to_memory(context, "User is practicing active listening.")
recent = await tools.search_memory(context, "active listening", limit=5)
```

## 4) Runbook (copy/paste examples)

```bash
cd ai

# Single-user flow
python3 core/scripts/test_mem0_nim.py \
  --mode single \
  --user user_123 \
  --nim-key "$NIM_API_KEY"

# Multi-agent flow
python3 core/scripts/test_mem0_nim.py \
  --mode multi \
  --user user_123 \
  --session session_abc \
  --mem0-key "$MEM0_API_KEY"

# Full UX smoke test (single + multi)
python3 core/scripts/test_mem0_nim.py --mode full --user user_123

# Guided interactive mode for non-CLI users
python3 core/scripts/test_mem0_nim.py --interactive
```

## 5) Observability and metadata conventions

If you read/write memory directly, preserve these metadata fields:

- `user_id` (mandatory)
- `session_id` (session scoping)
- `scope` (`private` or `shared`)
- `source_agent`
- `source_role`
- `timestamp` (iso format recommended)

If no API key is available, the modules fallback to an in-memory test shim to keep samples working during onboarding.

## 6) Quick comparison

| Use case | Pick |
| --- | --- |
| You want a quick chatbot memory experience | `NIMMem0Manager` |
| You want agent-tool abstractions in pipelines | `AgentMemoryTools` |
| You coordinate multiple specialist agents | `MultiAgentMemory` |

## 7) Expected setup quality bar

- Keep `NIM_API_KEY` + `MEM0_API_KEY` in env for local tests.
- Use stable IDs (`user_`, `session_`, `agent_`) for repeatable sessions.
- Prefer `--mode full` when validating a new workstation.
