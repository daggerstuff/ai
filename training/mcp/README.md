# Training-infra MCP

The training control plane for the Eve fleet: agents propose training work, the
MCP validates it against a reviewed recipe registry, enforces budget caps, holds
the backend credentials, launches the job, and records an audit trail.

- [RUNBOOK.md](./RUNBOOK.md) — operations: which script, config, credentials;
  lifecycle commands, incidents, token rotation, budget caps.
- [RECIPE_WORKFLOW.md](./RECIPE_WORKFLOW.md) — how a recipe is proposed,
  validated, and landed via reviewed PR.
- [MULTI_NODE_DECISION.md](./MULTI_NODE_DECISION.md) — why multi-GPU/multi-node
  recipes are deferred, and what would unblock them.

## Endpoint

`http://127.0.0.1:8766/mcp` (streamable HTTP, localhost only, bearer-token
authed). State in `state/jobs.db` (SQLite/WAL), logs in `state/logs/`.

## Layout

| File             | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `server.py`      | MCP tools + lifespan reconciler + auth/main entry              |
| `config.py`      | Settings, secrets, budget caps, loopback/allow-remote policy    |
| `db.py`          | Job registry, plan cache, audit log (schema + migration)        |
| `plan.py`        | `plan_job` resolution: validate → command → hashes → cost       |
| `recipes.py`     | Recipe registry loader + draft validation (P3.3)                |
| `recipes.yaml`   | The reviewed registry — agents select, never define             |
| `runner.py`      | Subprocess launch, single-flight locks, cancel, credential allowlist |
| `sync.py`        | Import + reconcile: live backend state → registry, lost → `backend_lost` |
| `backends.py`    | OVH AI, W&B, R2 access + `evaluate_gate` (P3.2)                 |
| `foresight.py`   | Terminal-outcome provenance writes to Foresight                 |
| `deploy/`        | systemd user unit                                               |

## Two-call launch contract

1. `plan_job(recipe, inputs, overrides, requested_by)` — no side effects,
   returns `plan_hash` / `command` / `est_cost_usd` / `warnings`; audited.
2. `submit_job(plan_hash, requested_by, idempotency_key)` — binds the approved
   plan, enforces caps, launches; a stale or mismatched hash is rejected and
   audited unless `force=true`.

## Registry truth

The registry is fed from four directions and stays authoritative:

- `sync_jobs` imports recent ovhai jobs, W&B runs, and R2 eval tags, then
  **reconciles**: an active row whose backend job is no longer listed becomes
  `failed` with `error: backend_lost` (PIX-5012). A backend that is
  unconfigured or whose listing failed is skipped, so an unreachable backend is
  never mistaken for a vanished job.
- `import_history(rows=[...])` seeds operator-supplied historical rows for runs
  no backend can enumerate; `include_backends=true` also imports W&B/R2 history.
- Terminal transitions set `finished_at` and write a Foresight memory
  (`training_job_outcome`, tagged `job:<id>` / `status:<status>`) so
  `searchMemories(training_job_id:...)` returns real data even for jobs no agent
  polled. Set `TRAINING_MCP_FORESIGHT=0` to disable.

## Tool surface

Jobs: `plan_job`, `submit_job`, `cancel_job`, `list_jobs`, `get_job`,
`get_job_logs` (captured subprocess log or live ovhai), `sync_jobs`,
`import_history`.
Registry: `list_recipes`, `get_recipe`, `validate_recipe`.
Evals: `eval_results`, `list_eval_tags`, `eval_gate`, `job_metrics` (accepts a
registry `job_id`), `list_wandb_runs`.
Ops: `health`, `security_status`, `list_audit`.

## Development

```bash
cd ai
uv run ruff check training/mcp/ && uv run ruff format --check training/mcp/
uv run pytest tests/training/mcp -q
systemctl --user restart training-infra-mcp.service
```
