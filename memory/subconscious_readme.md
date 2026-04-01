# Subconscious Memory Injection

The supported path is `ai.memory.v3`.

Older wrapper-based and v2 approaches have been removed so the repository only
has one active subconscious architecture. Use the explicit v3 client/context
APIs instead of import-time autopatching or manual wrappers.

## Recommended Usage

```python
from ai.memory.v3 import SubconsciousClient, SubconsciousConfig

config = SubconsciousConfig.from_env()
client = await SubconsciousClient.create(config, user_id="pixelated")

response = await client.chat(
    [{"role": "user", "content": "Help me think through the deployment issue."}]
)

await client.close()
```

## Contextvars Usage

```python
from ai.memory.v3 import set_subconscious, get_subconscious, reset_subconscious
from ai.memory.v3.config import SubconsciousConfig

config = SubconsciousConfig.from_env()
token = set_subconscious(config, user_id="pixelated")

state = get_subconscious()
if state:
    enriched = await state.enrich("What changed in the memory service?")

await reset_subconscious(token)
```

## Current Model

- v3 is the only supported subconscious implementation
- local Hindsight-backed providers are the only supported memory backend here
- no import-time monkey patching
- no legacy wrapper layer

## Files

| File | Purpose |
|------|---------|
| `v3/config.py` | v3 configuration |
| `v3/context.py` | contextvars API |
| `v3/client.py` | explicit client wrapper |
| `v3/provider.py` | local memory provider |

## Migration Note

If you still have old references to `subconscious_wrapper.py`, `subconscious_example.py`,
or `ai.memory.v2`, update them to `ai.memory.v3`.
