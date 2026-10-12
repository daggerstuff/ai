# W&B Aria / Autoresearch Setup State

Updated: 2026-10-12

## Scope

Prepare W&B Aria and Launch for bounded experiments on the existing Pixelated SFT work. This file tracks readiness only; it does not authorize a new training run.

## W&B workspace

- Entity: `wutang`
- Project: `pixelated-empathy-kan28`
- W&B MCP: `https://mcp.withwandb.com/mcp` is configured in MastraCode and Codex.
- Global W&B skills: `wandb-primary`, `wandb-autoresearch`, `wandb-eval-tables`, and `weave-instrument` are installed. Codex has all four enabled in its global skill config; MastraCode can invoke the installed skills.

## Existing training baselines

| Run | Base model | Final training loss | State |
|---|---|---:|---|
| `arc-corpus-qwen36-sft-20261011` | `Qwen/Qwen3.6-35B-A3B` | 0.2419 | finished |
| `pixelated-arc-gemma4-sft-20261011` | `google/gemma-4-26B-A4B-it` | 0.3344 | finished |

Verified shared settings backfilled onto both W&B runs: 2 epochs, batch size 2, peak LR 1e-4, cosine schedule, 0.1 warmup ratio, all assistant turns, 418 training rows, 418 gradient steps. Both runs now carry `arc-corpus`, `serverless-sft`, and `config-backfill-verified` tags. This records verified settings only; it does not imply the runs have relaunchable code.

## Launch readiness

- Queue: `wandb-launch-k8s` (Kubernetes; namespace `wandb-launch`; defaults 1 GPU, 8 CPU, 80 GiB).
- Active Launch agents: **0** at last check. Queue items will wait until an agent is polling it.
- Relaunchable code artifact: `wutang/pixelated-empathy-kan28/arc-corpus-qwen36-sft-launch:v1` (verified job artifact; Python 3.11 slim; entrypoint `python arc_sft_launch.py`). Source is maintained at `ai/training/launch/`.
- Dataset artifact: `wutang/pixelated-empathy-kan28/arc-corpus-sft-data:v0` (verified; contains `train.jsonl` with 418 rows and `val.jsonl` with 80 rows; no test split).
- Local Docker: daemon access verified on 2026-10-12 (`docker run --rm hello-world` passed). No NVIDIA GPU tooling is present. This queue is Kubernetes-backed, so its execution requires an active Launch agent in the cluster; local Docker is not needed for that queue.
- Queue agent: still 0 active agents. The Helm bootstrap must be run by the user in their Kubernetes cluster with `WANDB_SERVICE_ACCOUNT_API_KEY`; setup did not invoke Helm or enqueue a job.
- No SFT training run was enqueued by this setup.

## Before any experiment

1. Start a W&B Launch agent in the user's Kubernetes cluster for `wandb-launch-k8s` using a W&B service-account key stored in the cluster environment. Do not put credentials in queue defaults or source files.
2. The SFT entrypoint and versioned training-data artifact are published and verified.
3. Agree on the experiment metric, search axis, smoke-test setting, and compute/trial budget.
4. Run one deliberately small smoke job, inspect the completed run, then proceed serially.

## Current stop condition

Aria can inspect the existing W&B project. Launch-driven autoresearch has verified code and dataset artifacts, but is not ready to enqueue until a Launch agent is active on `wandb-launch-k8s` and the experiment metric, search axis, smoke-test setting, and compute/trial budget are agreed. No training run has been submitted by this setup work.

## Notes

- The Aria product surface is hosted in W&B; agent configuration is provided by the W&B MCP plus the installed workflow skills.
- Do not use local `wandb.init()` runs as substitutes for Launch jobs.
- Keep experiments focused on the training workflow per owner direction; do not block this setup on evaluation work.
