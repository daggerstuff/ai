# Training-infra MCP — Operator Runbook

The training-infra MCP is the control plane agents use to propose and run
training work on the Eve fleet. It holds the backend credentials, validates
every request against a reviewed recipe registry, and records an audit trail.

- **Endpoint:** `http://127.0.0.1:8766/mcp` (streamable HTTP, localhost only)
- **Repo path:** `ai/training/control_plane/`
- **Runtime:** Python 3.14 via `ai/.venv` (uv-managed)
- **State:** `ai/training/control_plane/state/jobs.db` (SQLite, WAL) + `state/logs/`

---

## 1. Which script, config, credentials

| Concern        | Where                                                                               |
| -------------- | ----------------------------------------------------------------------------------- |
| Service code   | `ai/training/control_plane/server.py` (tools), `plan.py`, `runner.py`, `sync.py`              |
| Recipes        | `ai/training/control_plane/recipes.yaml` (reviewed PRs only — see `ai/training/control_plane/RECIPE_WORKFLOW.md`)   |
| Secrets        | `ai/training/control_plane/.env` (mode 600, gitignored) — W&B, R2, OVH, `TRAINING_INFRA_MCP_TOKEN` |
| Budget caps    | same `.env`: `TRAINING_MCP_MAX_CONCURRENT_JOBS`, `TRAINING_MCP_DAILY_JOB_CAP`, `TRAINING_MCP_MONTHLY_COST_CAP_USD` |
| systemd unit   | `ai/training/control_plane/deploy/training-infra-mcp.service` (installed at `~/.config/systemd/user/training-infra-mcp.service`) |
| Job registry   | `ai/training/control_plane/state/jobs.db` — inspect with `list_jobs` / `get_job`              |
| Audit log      | same DB, `audit_log` table — inspect with `list_audit`                              |
| Per-job logs   | `ai/training/control_plane/state/logs/<job_id>.log`                                           |
| Backends       | `ai/training/control_plane/backends.py` (ovhai CLI, W&B API, R2 S3)                            |

Agents **never** hold the backend credentials in the rows above
(`WANDB_API_KEY`, `R2_*`, OVH/HF tokens). Of the control plane they hold only
`TRAINING_INFRA_MCP_URL` and `TRAINING_INFRA_MCP_TOKEN`. An agent's own `.env`
does carry its unrelated runtime secrets (Cloudflare Workers AI key, Slack
tokens, MongoDB URI), so the isolation rule is about **backend** secrets — the
ones that can launch or pay for training — not about every value in the file.

---

## 2. Lifecycle commands

```bash
# Status / logs
systemctl --user status training-infra-mcp.service
journalctl --user -u training-infra-mcp.service -n 100 --no-pager

# Restart after any code or .env change (health cache is 60s; restart is required)
systemctl --user restart training-infra-mcp.service

# Enable at login
systemctl --user enable training-infra-mcp.service
```

### Auth smoke test

```bash
TOKEN=$(grep -oP '(?<=^TRAINING_INFRA_MCP_TOKEN=).*' ai/training/control_plane/.env)

# Expect 401 without the token
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8766/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"ops","version":"1"}}}'

# Expect 200 with it
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8766/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"ops","version":"1"}}}'
```

### Inspect from Python (the supported client path)

```bash
cd ai && export TOKEN=$(grep -oP '(?<=^TRAINING_INFRA_MCP_TOKEN=).*' training/control_plane/.env)
uv run python - <<'PY'
import asyncio, os
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

async def main():
    async with Client("http://127.0.0.1:8766/mcp", auth=BearerAuth(os.environ["TOKEN"])) as c:
        print(await c.call_tool("health", {}))
        print(await c.call_tool("security_status", {}))
        print(await c.call_tool("list_audit", {"limit": 5}))

asyncio.run(main())
PY
```

---

## 3. The two-call launch contract

A job is always **plan first, then submit**. The plan hash binds the exact
spec (recipe, inputs, overrides, command, cost) that a human or agent saw.

1. `plan_job(recipe, inputs, overrides, requested_by)` → `plan_hash`, `command`,
   `est_cost_usd`, `warnings`. No side effects; audited as `plan`.
2. Show the plan to the approver.
3. `submit_job(plan_hash, requested_by, idempotency_key)` → launches and audits
   `submit`. The plan is bound to its approval: the submitted `recipe`, `inputs`
   and `overrides` must match the planned spec exactly, and a missing or expired
   `plan_hash` is rejected and audited as `submit.rejected` unless `force=true`.
   `TRAINING_MCP_PLAN_TTL_SECONDS` (default 86400) bounds how long an approval
   stays submittable.

`submit_job` also enforces `max_concurrent_jobs`, `daily_job_cap`, and
`monthly_cost_cap_usd`, and returns the existing job for a repeated
`idempotency_key`.

---

## 4. Daily operations

| Task                        | Action                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Check service health        | `health` tool, or `systemctl --user status training-infra-mcp.service`               |
| Reconcile backend state     | `sync_jobs` (runs automatically every 60s via the lifespan reconciler)               |
| Import historical runs      | `import_history(rows=[...])`, or `include_backends=true` for W&B/R2 history          |
| List running/queued jobs    | `list_jobs(status="running")`                                                       |
| Show one job                | `get_job(job_id)`                                                                   |
| Job logs                    | `get_job_logs(job_id, tail=200)` — captured subprocess log, else live OVH logs       |
| Job metrics                 | `job_metrics(job_id)` — accepts a registry id; queued jobs report no metrics yet     |
| Cancel a job                | `cancel_job(job_id)` — terminates the subprocess and best-effort stops the OVH job   |
| Eval results for a tag      | `eval_results(tag)` / `list_eval_tags()`                                             |
| Promotion gate              | `eval_gate(tag)` — reads real `state_metrics` grades; see "Promotion gate" below      |
| Who asked for what          | `list_audit(action="submit", limit=50)` — actor, spec_hash, plan_hash, remote_addr   |
| Provenance lookup           | Foresight `search_memories` for tags `training_job_outcome` / `job:<id>`             |

---

## 5. Runbook: common incidents

### Service is down or health says `healthy: false`

1. `systemctl --user status training-infra-mcp.service` — read the failure.
2. `journalctl --user -u training-infra-mcp.service -n 100 --no-pager`.
3. `health` lists per-backend checks (`ovhai`, `wandb`, `r2`, `registry_db`) —
   the failing one names the credential or CLI to fix.
4. Restart after fixing `.env`: `systemctl --user restart training-infra-mcp.service`.

### A submit was rejected

Every rejection is an `audit_log` row with `action='submit.rejected'` and a
reason. Read it with `list_audit(action="submit.rejected")`:

- `plan_hash required` / `plan_hash not found or expired` → re-run `plan_job`
  (a plan older than `TRAINING_MCP_PLAN_TTL_SECONDS` is expired).
- `submitted spec differs from the approved plan` → the caller edited recipe,
  inputs or overrides after planning; re-plan and re-approve. The audit
  `detail.fields` names which of the three differed.
- `max concurrent jobs reached` → wait for a running job to finish, or
  `cancel_job` one; raise `TRAINING_MCP_MAX_CONCURRENT_JOBS` only deliberately.
- `daily job cap reached` → raise `TRAINING_MCP_DAILY_JOB_CAP` or wait for UTC
  midnight.
- `monthly cost cap would be exceeded` → raise
  `TRAINING_MCP_MONTHLY_COST_CAP_USD` deliberately, or wait for the next month.

### A job is stuck `running`

1. `get_job(job_id)` — compare `backend_state` with `status`.
2. `sync_jobs` to refresh from the backends. If the backend no longer lists the
   job, reconcile marks it `failed` with `error: backend_lost`.
3. `get_job_logs(job_id)` — serves the captured subprocess log when present,
   otherwise live OVH logs (the file also lives at
   `ai/training/control_plane/state/logs/<job_id>.log`).
4. `job_metrics(job_id)` — W&B metrics once the run has started.
5. `cancel_job(job_id)` if it must stop; the terminal state is written once and
   the monitor will not overwrite it.

### Agents cannot reach the MCP

1. Confirm the token matches everywhere: `ai/training/control_plane/.env` and **every**
   agent that calls the MCP — today `agents/pipeline-agent/.env` and
   `agents/advisor-agent/.env` (`TRAINING_INFRA_MCP_TOKEN`). A token present in
   only one of them fails per-agent, not globally.
2. Confirm the URL: `http://127.0.0.1:8766/mcp` for a host process,
   `http://host.docker.internal:8766/mcp` from a Compose container. Both the
   pipeline-agent and advisor-agent services declare
   `extra_hosts: host.docker.internal:host-gateway`.
3. A 401 means the token is wrong or unset; a connection refused means the unit
   is not running.

### Rotating the shared token

1. Generate a new value (`python3 -c "import secrets; print('timcp_'+secrets.token_urlsafe(32))"`).
2. Set `TRAINING_INFRA_MCP_TOKEN` in `ai/training/control_plane/.env` **and** in every
   agent `.env` that talks to the MCP.
3. `systemctl --user restart training-infra-mcp.service`.
4. Verify: unauthenticated curl →
   `401`, authenticated curl → `200`.

---

## 6. Security posture (verify with `security_status`)

- **Localhost bind.** The service refuses to start on a non-loopback host
  unless `TRAINING_MCP_ALLOW_REMOTE=1` is set deliberately.
- **Token auth.** `TRAINING_INFRA_MCP_TOKEN` is enforced for every request;
  unset means the server runs unauthenticated (only acceptable on a host where
  nothing else can reach port 8766).
- **Credential isolation.** Job subprocesses receive only an allowlist of
  process vars plus the pipeline's own backend secrets; fleet-wide secrets are
  never inherited. `security_status` reports any non-allowlisted key it can see
  in the job environment.
- **Audit.** Every plan, submit, rejection, cancel, and eval-gate decision is a
  row in `audit_log` with actor, hashes, and remote address.

---

## 7. Budget caps

| Variable                                | Default | Effect                                    |
| --------------------------------------- | ------- | ----------------------------------------- |
| `TRAINING_MCP_MAX_CONCURRENT_JOBS`      | `1`     | Reject a submit that would exceed it       |
| `TRAINING_MCP_SINGLE_FLIGHT_WAIT_SECONDS` | `300`  | How long a queued job waits for the single-flight slot before failing with `single-flight timeout` (not the concurrency cap itself) |
| `TRAINING_MCP_DAILY_JOB_CAP`            | `5`     | Reject after this many jobs in a UTC day   |
| `TRAINING_MCP_MONTHLY_COST_CAP_USD`     | `100.0` | Reject when committed cost would exceed it |

Caps are read from `.env` at service start; changing them requires a restart.
Imported history (`requested_by: import`) does not count against the daily cap.

## 8. Promotion gate

`eval_gate(tag, min_delta, min_pass_rate)` decides whether a release may be
promoted. Omit the thresholds to use the operator policy:

| Variable                            | Default | Effect                                        |
| ----------------------------------- | ------- | --------------------------------------------- |
| `TRAINING_MCP_GATE_MIN_DELTA`       | `0.05`  | Minimum `sft - base` pass-rate delta to pass  |
| `TRAINING_MCP_GATE_MIN_PASS_RATE`   | `0.5`   | Minimum absolute `sft` pass rate to pass      |

The defaults are deliberately non-zero: sending `0` (or omitting the older
zero-valued defaults) made the gate a rubber stamp, passing runs where the
fine-tune was no better than the base model. Every verdict is audited with the
thresholds actually applied:

```bash
list_audit(action="eval.gate", limit=10)
# detail: {"tag": "v29-l4e-v2", "passed": true, "min_delta": 0.05,
#          "min_pass_rate": 0.5, "reasons": []}
```

Grading reads `state_metrics` from each `results.jsonl` row: a case passes only
when it carries checks and **all** of them passed. Rows with no metrics are
ungraded; a tag with no graded runs fails closed. Inspect the raw signal with:

```bash
eval_results("v29-l4e-v2")   # runs -> manifest, results_preview, arm_pass_rates
```

Real verdicts as of this writing: `v29-l4e-v2` sft **0.917** vs base **0.250**
(delta 0.667) on both runs — passes; `v28-l4e` **0.833** vs **0.250** — passes;
`v22` carries no graded metrics — fails closed.
