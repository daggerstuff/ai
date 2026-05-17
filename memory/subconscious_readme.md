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
- the recommended backend is the shared local memory service via `shared_service`
- direct `local_hindsight` remains available only for explicitly local single-host runs
- no import-time monkey patching
- no legacy wrapper layer

## Files

| File | Purpose |
| ------ | --------- |
| `v3/config.py` | v3 configuration |
| `v3/context.py` | contextvars API |
| `v3/client.py` | explicit client wrapper |
| `v3/provider.py` | local and shared-service memory providers |

## Shared Service Configuration

Set these environment variables on callers that should use the shared memory
service instead of local SQLite:

```bash
SUBCONSCIOUS_MEMORY_PROVIDER=shared_service
SUBCONSCIOUS_MEMORY_BASE_URL=http://memory-host:5003
SUBCONSCIOUS_MEMORY_ACTOR_ID=subconscious-staging
SUBCONSCIOUS_MEMORY_ACTOR_SECRET=replace-with-real-secret
HINDSIGHT_BANK_ID=pixelated-staging
```

To run the shared service itself, use:

```bash
source /path/to/pixelated-memory.env
/home/vivi/pixelated/scripts/memory/run-shared-memory-service.sh
```

## Migration Note

If you still have old references to `subconscious_wrapper.py`, `subconscious_example.py`,
or `ai.memory.v2`, update them to `ai.memory.v3`.
