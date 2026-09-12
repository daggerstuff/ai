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
