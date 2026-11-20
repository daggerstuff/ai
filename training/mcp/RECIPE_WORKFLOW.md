# Recipe Addition Workflow (P3.3)

Recipes are the only way training work enters the control plane: an agent
**selects** a recipe, it never defines one. `recipes.yaml` is therefore a
reviewed artifact, and this document is the workflow that keeps it that way.

## 1. Why the registry is human-reviewed

A recipe is executable policy: it names an entrypoint, a backend, a cost class,
and a GPU shape. Letting an agent append entries would let it choose what code
runs on fleet GPUs, which backend spends money, and what the budget cap means.
The control plane instead separates:

- **Proposal** — anyone (including an advisor agent) can draft an entry and
  have the MCP validate it.
- **Landing** — a human reviews and merges a PR to
  `ai/training/mcp/recipes.yaml`. Only then does the recipe exist.

## 2. Validate a draft

`validate_recipe` applies the same checks the service applies at start, so a
draft that validates will not break service startup.

```python
entry = {
    "id": "eval-crisis-v3",
    "kind": "eval_paired",  # curate_dataset | train_sft | eval_paired | build_release | other
    "description": "Paired crisis eval on the v3 suite.",
    "backend": "ovhai",  # local | ovhai | wandb-serverless | wandb | r2
    "entrypoint": "python -m training.ovh_v22_eval.submit run --execute --tag <tag> --suite v3",
    "cost_class": "gpu-hour",  # cpu-minute | cpu-hour | gpu-hour | gpu-day
    "est_cost_usd": 2.0,
    "gpu": "l4-1-gpu",
    "timeout": "2h",
    "inputs": {"tag": "result tag"},
    "outputs": {"r2_prefix": "evals/<tag>/", "result_files": ["manifest.json", "results.jsonl"]},
    "invariants": {"r2_only": True, "no_volume_mounts": True},
}
```

Call the tool:

```bash
cd ai && export TOKEN=$(grep -oP '(?<=^TRAINING_INFRA_MCP_TOKEN=).*' training/mcp/.env)
uv run python - <<'PY'
import asyncio, json, os
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

ENTRY = json.load(open("/tmp/recipe-draft.json"))

async def main():
    async with Client("http://127.0.0.1:8766/mcp", auth=BearerAuth(os.environ["TOKEN"])) as c:
        print(await c.call_tool("validate_recipe", {"entry": ENTRY}))

asyncio.run(main())
PY
```

The response is `{"ok": true, "valid": true|false, "errors": [...]}`. Fix every
error before opening the PR.

## 3. Rules the validator enforces

- Required fields: `id`, `kind`, `description`, `backend`, `entrypoint`.
- `kind` ∈ {`curate_dataset`, `train_sft`, `eval_paired`, `build_release`, `other`}.
- `backend` ∈ {`local`, `ovhai`, `wandb-serverless`, `wandb`, `r2`}.
- `cost_class` ∈ {`cpu-minute`, `cpu-hour`, `gpu-hour`, `gpu-day`}.
- `est_cost_usd` must be numeric.
- Every override must declare a `type` unless it is `locked: true`.
- The `id` must not already exist in the registry.
- **Launching recipes must pass `--execute`.** Both remote entrypoints
  (`wandb_sft_v15_execute`, `ovh_v22_eval.submit run`) default to a dry run
  without it, so a `train_sft` or `eval_paired` recipe that omits the flag
  would submit, report `succeeded`, and never launch anything.
- **Every declared input must be substituted somewhere in the entrypoint.**
  A required input the command never receives is a lie about what the plan
  controls: callers now get a warning, so wire it in as `<name>` or drop it.
- **Never accept an input that duplicates recipe identity.** `eval-crisis-v2`
  pins `--suite v2`; a caller passing `suite=v1` is rejected as an unknown
  input rather than silently evaluating the wrong suite under the right label.
- **No multi-node / multi-GPU shapes** (`nodes`, `num_nodes`, `multi_node`,
  `world_size`, or a GPU flavour requesting more than one GPU). Multi-node is
  deliberately deferred — see `MULTI_NODE_DECISION.md`.

## 4. Landing the recipe (human PR)

1. Append the validated entry to `ai/training/mcp/recipes.yaml`.
2. Open a PR. The PR description should state the entrypoint's real path, why
   the cost class is right, and what artifact the recipe produces.
3. CI and review run as usual. A reviewer checks that the entrypoint exists
   under `training/`, that the cost estimate is honest, and that the outputs
   land somewhere a promotion gate can read.
4. No restart is needed: `recipes.yaml` is re-read on every tool call
   (`recipes.load_recipes()`), so the new entry is live as soon as the file
   lands on the service host. (Restart only for `.env` changes.)
5. Verify: `list_recipes()` includes the new id and `get_recipe(id)` returns it.

## 5. Optional advisor proposal path

An advisor agent can produce the draft entry (as JSON) and call
`validate_recipe`. The advisor must then **open a PR** — it never writes
`recipes.yaml` directly. If the advisor has no PR capability, it posts the
validated JSON to the operator, who opens the PR.

## 6. Changing or removing a recipe

Recipes are versioned by id: prefer adding `eval-crisis-v3` over editing
`eval-crisis-v2`, because plan hashes and audit rows point at ids. If an entry
must change in place, treat it as a PR with reviewer sign-off and note it in the
PR description — an in-place change alters what an existing id means.
