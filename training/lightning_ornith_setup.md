# Lightning.ai + vLLM: self-hosted Ornith for NF dataset generation

Why: bulk generation (50k records ≈ 375k therapist turns ≈ 95–130M output tokens)
on the shared Featherless key fights your other batch jobs for the account
concurrency limit (proven 429 source — see `eval_results/judge_variance_proof.md`).
A single A100 40GB serving Ornith-1.5-9B in bf16 finishes the whole corpus in
roughly a day for tens of dollars, with pinned weights/sampler (reproducible)
and full sampling fidelity (`top_k`, `presence_penalty`, `repetition_penalty`
all honored — Ollama's OpenAI-compat endpoint drops some of these).

## 0. Studio

Create a Lightning.ai studio: **A100 40GB** (or 48GB), PyTorch base image.
Do not use a 24GB card for bf16 — 9B bf16 weights are ~18GB, leaving no KV cache.

## 1. Serve (inside the studio)

```bash
bash lightning_ornith_serve.sh            # foreground; or:
nohup bash lightning_ornith_serve.sh > serve.log 2>&1 &
tail -f serve.log                          # wait for: Application startup complete.
```

If vLLM errors on the model architecture, check the Ornith model card for the
minimum supported vLLM version and `pip install -U "vllm==<ver>"`. Ollama
(GGUF) is the fallback, but it is pilot-only: weaker batching and lossy
sampling-param support.

## 2. Smoke test (inside the studio)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-ai/Ornith-1.5-9B","messages":[{"role":"user","content":"Reply with the single word PONG"}],"temperature":1.0,"top_k":20,"top_p":0.95,"presence_penalty":1.5,"repetition_penalty":1.0,"max_tokens":32}' | head -c 600
```

## 3. Bring eval inputs onto the studio

```bash
git clone <repo> && cd <repo>/ai
# copy eval caches (gitignored) + judge keys:
scp <local>:pixelated/ai/training/eval_results/{client_lines.json,ornith_gen.json,judge.json} training/eval_results/
scp <local>:pixelated/.env .env           # needs FEATHERLESS_API_KEY + CLOUDFLARE_WORKERS_AI_API_KEY (judging only)
```

The caches are what make this cheap: `client_lines.json` (20 byte-identical
openers) and `ornith_gen.json` (existing featherless-ornith generations) are
reused; `judge.json` partial-merges so the `ornith` arm is NOT re-judged.

## 4. Parity run (inside the studio)

```bash
cd <repo>/ai
EVAL_ARMS=ornith,ornith_local python -m training.eval_boulesis_vs_ornith
```

Only `ornith_local` generates (20 local calls, sequential) and only
`ornith_local` gets judged (20 × 4 judge calls). Output:
`training/eval_results/comparison_report.md`.

Remote-pod alternative (run the harness on your laptop instead): forward the
port (`ssh -L 8000:localhost:8000 <lightning-ssh>` or Lightning's public URL),
set `VLLM_URL`, and run the same command locally. Judging then uses your local
`.env` keys.

## 5. Migration gate — flip `NF_BACKEND` only if ALL hold

| Criterion | Threshold |
|---|---|
| Cliche gate | 20/20 pass |
| First-attempt pass | 20/20 (no `empty_or_unfinished` — max_tokens 4096 locally) |
| Extraction cleanliness | zero think-tag residue / label prefixes in report |
| Judge quality (Wayfarer primary) | mean within ±0.03 of the featherless ornith arm |
| Per-call latency | ≤ 60s at max_tokens 4096 (single stream) |

On failure: check vLLM logs for dtype/arch warnings first, then compare raw
`raw_head` fields between arms — divergence there means serving-level
differences, not model differences.

## 6. Bulk run (after the gate passes)

```ini
# .env
NF_BACKEND=vllm
VLLM_URL=http://localhost:8000        # or the tunnel/public URL
# NF_MODEL stays ornith-ai/Ornith-1.5-9B (featherless default resolution
# no longer applies; resolve_backend reads NF_MODEL directly for vllm)
```

Notes for the bulk runner:
- The eval harness generates sequentially; bulk throughput needs a concurrent
  runner (the vLLM server handles 64 concurrent sequences).
- Featherless keeps: pilots, the dual judge (Wayfarer primary), fallback.
  Judging on the shared key still contends with your batch jobs — the
  judge-side hardening items (429 backoff, semaphore 2, dedicated key) remain.
- Stop the studio when idle; GPUs bill while the pod runs.

## 7. Dry run results (CPU studio, 2026-09-13)

A CPU-only studio rehearsed every step except actual model serving (a tiny
mock OpenAI server stood in). Everything below was executed and verified;
commands are copy-ready rather than aspirational.

Result: **full pipeline green** — sync, deps, endpoint override routing,
extraction/gate loop, artifact writes, tunnel transport, and byte-identical
replay (`EVAL_ARMS=ornith --report` reproduced the canonical featherless stats
exactly: 20/20 gate, 18/20 first-attempt, mean quality 0.74).

### Kink ledger

| # | What bit us | Fix / recipe |
|---|---|---|
| K1 | `lightning` CLI's `~/.lightning/credentials.json` silently rotates/stales between invocations → random 401s | Rewrite it from `.env` before **every** CLI call (see below) |
| K2 | Newer auth rejects the legacy `t=` query param on `/setup/ssh-gen`; `lightning ssh configure` fails, keys get generated but never registered | Two-step: Bearer-download keys, then REST-register + confirm pubkey (below) |
| K3 | The `ai-development` base image ships Python/git only — no `pip`, but `/usr/local/bin/uv` is present | Build a project-local venv with uv instead of system installs (below) |
| K4 | `scp remote:relative/path` let files vanish without error; `pkill -f x` over ssh matches its own wrapper shell and kills the session | Absolute paths everywhere for scp targets; never pkill/pgrep patterns that appear inside your own command string |
| K5 | Pre-existing client-opener cache was missing today's stratified pair (`nf_016/nf_062`) because an earlier rebuild baked an era-specific subset → cache clobbered/regenerated forever after | Root cause found during dry run; harness now has merge-on-miss semantics so caches only ever grow a union. No action needed beyond keeping patched harness deployed |
| K6 | Tunnel doesn't die via pidfile kill when SSH ControlMaster mux owns the listener | Teardown uses `ssh -O exit <host>` |

### Recipes

```bash
# --- K1: deterministic Lightning credentials ---------------------------------
python3 - <<'PY'
import json, pathlib, re, os
root = "/home/vivi/pixelated"
key = ""
uid = ""
for ln in open(root + "/.env"):
    m = re.match(r"LIGHTNING_API_KEY\s*=\s*(.+)", ln.strip())
    if m and not key:
        key = m.group(1).strip().strip("'\"").rstrip("\r")
cred = pathlib.Path.home() / ".lightning/credentials.json"
old = {}
if cred.exists():
    try:
        old = json.loads(cred.read_text())
    except Exception:
        pass
uid = old.get("user_id") or uid      # preserve whatever worked previously
out = {"api_key": "", "auth_token": key, "user_id": uid}
pathlib.Path.mkdir(pathlib.Path.home()/".lightning", exist_ok=True)
cred.write_text(json.dumps(out))
print({"has_uid": bool(uid)})
PY

# --- K2: SSH key download + registration (token must be LIGHTNING sk-lit-) --
curl -fsSL https://lightning.ai/setup/ssh-gen \
     -H "Authorization: Bearer $LIGHTNING_API_KEY" > ~/.ssh/lightning_rsa
chmod 600 ~/.ssh/lightning_rsa

# Generated keys are worthless until their PUBLIC half is registered against
# your account; do that over the standard bearer-auth REST API:
REG=$(curl -fsS -X POST https://lightning.ai/api/v1/ssh-keys \
  -H "Authorization: Bearer $LIGHTNING_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$(hostname)-cli\",\"public_key\":\"$(cat ~/.ssh/lightning_rsa.pub)\",\"comment\":\"dryrun-cli\"}")
echo "$REG"                  # expect setupConfirmed:false
ID=$(echo "$REG" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
curl -fsS "https://lightning.ai/api/v1/ssh-keys/$ID/confirm" \
     -H "Authorization: Bearer $LIGHTNING_API_KEY"
ssh -o ConnectTimeout=15 ornith-dryrun echo CONNECTED   # sanity ping
```

(`Host ornith-dryrun` comes from `lightning ssh generate` output pasted into
`~/.ssh/config`; adjust User to match that printout.)

```bash
# --- K3: dependency-free runtime env on the studio --------------------------
ssh ornith-dryrun '
set -e
mkdir -p ~/pixelated && cd ~/pixelated          # resolve-only-safe: use $PWD below
UV=/usr/local/bin/uv
$UV venv .venv                                  # one-time
. .venv/bin/activate                            # resolves under real home automatically
$UV pip install aiohttp==3.14                   # harness import
python -m compileall -q training || true        # cheap syntax tripwire
python -c "from training.eval_boulesis_vs_ornith import main; print(\"harness_import_ok\")"
'

# --- K6/K4 hygiene: process control cheatsheet -------------------------------
ssh -O exit ornith-dryrun                       # closes tunnels opened via this config host
# For anything else start with ps/pgrep listing FIRST (show PIDs), then explicit PID kill:
ssh ornith-dryrun 'ps -eo pid,args | awk '"'"'/pattern_to_find/ && !/awk/'"'"'; read -p "PID? " P; [ -n "$P" ] && kill "$P"'
```
