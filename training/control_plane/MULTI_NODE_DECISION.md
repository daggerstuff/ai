# P3.4 — Multi-GPU / Multi-Node Decision: **Deferred**

**Status:** decided, deferred. Enforced in the registry.
**Ticket:** PIX-5033.

## Decision

The training-infra control plane supports **single-GPU recipes only** for now.
Multi-node and multi-GPU submissions are rejected at recipe-validation time
rather than run best-effort.

## Why

The repo contains a multi-node launcher
(`ai/training/scripts/sprint8/launch_multi_node.py`) from PIX-4348, but it is a
**preflight script**, not a production path:

- `check_nccl()` and `check_ib()` both return **False** on this host, and
  `main()` returns exit code 1 before launching a distributed job. The failure
  is the **host**, not the torch build: the installed interpreter has
  `torch 2.13.0+cu130` with `torch.distributed.is_nccl_available()` True, but
  there is no `libnccl` in `ldconfig -p` and no `/sys/class/infiniband`, so
  there is no NCCL to link and no InfiniBand device to rendezvous over. (The
  launcher's own header comment still says "torch 2.13.0+cpu" — that comment is
  stale, and `main()` fails on the missing host libraries regardless.)
- It shells out to `torchrun` with no rank/rendezvous management, no OVH job
  topology, and no artifact contract. Our backends (`backends.py`) launch
  single-container OVH jobs and W&B Serverless runs; neither models a
  multi-node topology, hostfile, or cross-node rendezvous endpoint.
- Cost accounting is single-job based: `cost_class` and `est_cost_usd`
  describe one job, and the budget caps (`daily_job_cap`,
  `monthly_cost_cap_usd`) count jobs. A multi-node run silently multiplies GPU
  spend, which is exactly what the caps exist to prevent.
- No eval or promotion gate currently distinguishes a multi-node release, so
  provenance and artifact binding would be unverifiable.

Shipping a half-supported multi-node path would violate the control plane's
core invariant: a recipe must describe something that can actually run, at a
cost the caps understand.

## What is enforced

`validate_recipe_entry` / the `validate_recipe` MCP tool reject:

- `nodes`, `num_nodes`, `world_size` above `1`, **or any truthy
  `multi_node`** (the boolean rejects `true`, not just values above 1);
- a GPU flavour requesting more than one GPU (e.g. `l4-4-gpu`);
- any `gpu` value containing `multi`.

Current single-GPU flavours such as `l4-1-gpu` pass unchanged.

## What would unblock it

Revisit when all of these hold:

1. A GPU host with working NCCL and IB, verified by the PIX-4348 preflight.
2. Backend support for multi-node topology in `backends.py` (OVH multi-node job
   spec, or W&B Serverless with explicit node count) plus a cost model that
   scales `est_cost_usd` by node count.
3. Budget caps extended to account for per-node spend in one job.
4. An artifact contract that binds per-node outputs into one release manifest,
   so the eval gate and promotion tools can still verify provenance.

Until then, recipes stay single-GPU and `launch_multi_node.py` stays a manual,
out-of-band diagnostic.
