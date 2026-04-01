# Subconscious Memory Injection - Usage Guide

## TL;DR - It Just Works

Add this **one line** at the top of your main application file:

```python
import ai.memory.subconscious_autopatch  # That's it.
```

Or set environment variable:
```bash
export SUBCONSCIOUS_AUTOACTIVATION=true  # Default
export SUBCONSCIOUS_USER_ID="pixelated"  # Default user ID
```

**All LLM calls now automatically have subconscious context injected.** No code changes needed.

---

## How It Works

The subconscious module **patches** the OpenAI and Anthropic clients at import time. When you call `client.chat.completions.create()` or `client.messages.create()`, the subconscious:

1. **Intercepts** the call before it reaches the LLM
2. **Queries** the reflection subagent for relevant memories/patterns
3. **Prepends** XML context tags to the prompt
4. **Passes** the injected prompt to the LLM

**Before (without subconscious):**
```python
response = client.chat.completions.create(
    model="qwen/qwen3.5-397b-a17b",
    messages=[{"role": "user", "content": "Help with anxiety"}]
)
```

**After (with subconscious auto-patch):**
```python
# The LLM actually receives:
"""
<subconscious_context>
<relevant_memories>
- User mentioned anxiety 3 times this week
- Previous session: boundary-setting worked
</relevant_memories>
</subconscious_context>

Help with anxiety
"""
response = client.chat.completions.create(...)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBCONSCIOUS_ENABLED` | `true` | Enable/disable injection |
| `SUBCONSCIOUS_AUTOACTIVATION` | `true` | Auto-activate on import |
| `SUBCONSCIOUS_USER_ID` | `pixelated` | Default user ID for memory lookup |
| `NVIDIA_API_KEY` | (required) | Your Nvidia API key |

### Manual Activation

If you don't want auto-activation:

```python
# Disable auto-activation
export SUBCONSCIOUS_AUTOACTIVATION=false

# Then manually activate
from ai.memory import activate_subconscious
activate_subconscious(user_id="pixelated")
```

---

## Supported LLM Clients

The auto-patch supports:

- **OpenAI** (`openai.OpenAI`)
- **Anthropic** (`anthropic.Anthropic`)
- **Nvidia NIM** (via OpenAI-compatible API)

Any code using these clients gets subconscious injection automatically.

---

## Disabling

```python
# Temporarily disable
from ai.memory import deactivate_subconscious
deactivate_subconscious()

# Or via environment
export SUBCONSCIOUS_ENABLED=false
```

---

## What Gets Injected

The subconscious injects XML-formatted context based on:

1. **Crisis indicators** - If crisis detected
2. **Relevant memories** - From past conversations
3. **Pattern observations** - Recurring themes
4. **Therapeutic goals** - User-specific goals

Example injection:
```xml
<subconscious_context>
<crisis_alert>Active: suicidal ideation</crisis_alert>
<relevant_memories>
- User responded well to DBT techniques
- Previous crisis: grounding exercises helped
</relevant_memories>
<pattern_observations>
- Pattern: anxiety spikes at night
- Consider: sleep hygiene discussion
</pattern_observations>
</subconscious_context>
```

---

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────┐
│  Subconscious Patch     │  ← intercepts LLM call
│  - Query reflection     │
│  - Build context        │
│  - Prepend to prompt    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  LLM (qwen3.5)          │  ← sees injected prompt
└───────────┬─────────────┘
            │
            ▼
    Response to user
```

---

## Files

| File | Purpose |
|------|---------|
| `subconscious_autopatch.py` | Auto-patch infrastructure |
| `subconscious_wrapper.py` | Manual wrapper (for custom use) |
| `subconscious_example.py` | Usage examples |
| `reflection_bootstrap.py` | Bootstrap for reflection subagent |
| `reflection_factory.py` | Factory functions |

---

## Troubleshooting

**"Subconscious not active"**
- Check `SUBCONSCIOUS_ENABLED=true`
- Verify `NVIDIA_API_KEY` is set
- Check logs for activation errors

**"No context being injected"**
- Verify reflection subagent is running
- Check user_id matches memory records
- Ensure conversation context is provided

**Performance concerns**
- Subconscious queries are async
- Context building is cached
- Typical overhead: <100ms
