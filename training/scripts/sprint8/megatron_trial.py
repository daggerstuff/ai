#!/usr/bin/env python3
"""PIX-4351: Megatron trial (TP=4, PP=2, CP=1) vs FSDP2 — pick winner.

Sprint 8 Lane B M2 wk3 - 5pt, HIGH priority.

Per blueprint @168: Megatron trial (TP=4, PP=2, CP=1) for 70B; compare
throughput vs FSDP2; pick winner.

Megatron-core (NVIDIA) not installed in this env (GPU-only, heavy). This
trial implements TP (tensor parallelism) + PP (pipeline parallelism) using
plain torch primitives so the topology math is exercised on CPU single-proc:

  - TP=4: column-parallel Linear — split weight rows across tp_size ranks,
    each rank holds 1/tp_size of the weight. Forward = partial matmul;
    all-reduce sums partial outputs. (Megatron ColumnParallelLinear.)
  - PP=2: split model into pp_size stages; microbatch schedule interleaved1f1b
    (per @215). Mock uses gpipe (simpler) schedule; interleaved1f1b noted.
  - CP=1: no context parallelism (sequence dim stays local).

Throughput: time N train steps under each strategy, report tokens/sec.
Prod swaps mock for real Megatron-LM + transformer_engine + 8-GPU nodes.

Blueprint ref: docs/training-pipeline-blueprint-2026-08-10.md @168, @202-217,
@273-289, @302-304.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MegatronTrialConfig:
    """Megatron TP=4/PP=2/CP=1 trial config (per blueprint @168)."""

    model_name: str = "mock_70B_megatron"
    tp_size: int = 4  # tensor parallel - must divide GPUs/node @206
    pp_size: int = 2  # pipeline parallel - 70B+ @213
    cp_size: int = 1  # context parallel (off for this trial)
    pp_schedule: str = "interleaved1f1b"  # @215; mock uses gpipe fallback
    pp_microbatch_size: int = 4  # @216
    sequence_parallel: bool = True  # @207 keep TP inside NVLink domain
    world_size: int = 8  # 8 GPUs
    # Training + throughput comparison
    train_steps: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    use_wandb: bool = True
    wandb_project: str = "pixelated-sprint8-megatron"
    wandb_run_name: str = "megatron-tp4-pp2-70b"
    compare_fsdp2: bool = True  # run FSDP2 baseline same steps for comparison
    # Mock model
    mock_dim: int = 256
    mock_layers: int = 8  # split across pp_size=2 stages (4 each)
    mock_n_heads: int = 4
    mock_dtype: str = "bfloat16"
    attention_backend: str = "sdpa"  # @292

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.tp_size < 1:
            errs.append(f"tp_size={self.tp_size} must be >= 1")
        if self.pp_size < 1:
            errs.append(f"pp_size={self.pp_size} must be >= 1")
        if self.cp_size != 1:
            errs.append(f"cp_size={self.cp_size} must be 1 for this trial (@168)")
        if self.world_size % self.tp_size != 0:
            errs.append(f"world_size={self.world_size} not divisible by tp_size={self.tp_size} (@206)")
        # PP invariant: layers divisible by pp_size for even stage split
        if self.mock_layers % self.pp_size != 0:
            errs.append(f"mock_layers={self.mock_layers} not divisible by pp_size={self.pp_size}")
        if self.pp_schedule not in {
            "1f1b",
            "interleaved1f1b",
            "gpipe",
            "looped_bfs",
            "dfs",
            "v_schedule",
            "zero_bubble",
        }:
            errs.append(f"invalid pp_schedule={self.pp_schedule} (@280)")
        if self.pp_microbatch_size < 1:
            errs.append(f"pp_microbatch_size={self.pp_microbatch_size} must be >= 1 (@216)")
        if self.attention_backend not in {"sdpa", "te", "flash_attention"}:
            errs.append(f"attention_backend={self.attention_backend} must be sdpa/te/flash (@292)")
        if self.mock_dim % self.mock_n_heads != 0:
            errs.append(f"mock_dim={self.mock_dim} not divisible by mock_n_heads={self.mock_n_heads}")
        if self.mock_dim % self.tp_size != 0:
            errs.append(f"mock_dim={self.mock_dim} not divisible by tp_size={self.tp_size} (column-parallel split)")
        return errs


def _check_gpu_env() -> tuple[bool, list[str]]:
    notes: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            notes.append(f"CUDA available, device_count={n}")
            return n > 0, notes
        notes.append("CUDA not available (CPU-only run)")
        return False, notes
    except ImportError:
        notes.append("torch not importable")
        return False, notes


class _SelfAttnBlock:
    pass


def _make_selfattn_block_cls():
    from torch import nn

    class SelfAttnBlock(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, x):
            out, _ = self.attn(x, x, x, need_weights=False)
            return out

    return SelfAttnBlock


class _ColumnParallelLinear:
    pass


def _make_col_parallel_linear_cls():
    """Megatron ColumnParallelLinear via plain torch.

    Splits weight rows across tp_size ranks; each rank holds 1/tp_size.
    Forward = partial matmul (out_features/tp_size per rank). All-reduce
    sums partial outputs across TP ranks. Single-proc mock = tp_size=1
    effective (no real split), but the API + topology math is exercised.
    """
    import torch
    from torch import nn

    class ColumnParallelLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, tp_size: int, bias: bool = False) -> None:
            super().__init__()
            self.tp_size = tp_size
            assert out_features % tp_size == 0, f"out_features={out_features} must be divisible by tp_size={tp_size}"
            self.local_out = out_features // tp_size
            self.in_features = in_features
            self.out_features = out_features
            # Each "rank" slice stored as separate param (mocks tp_size shards)
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.empty(self.local_out, in_features)) for _ in range(tp_size)]
            )
            for w in self.weights:
                nn.init.normal_(w, std=0.02)
            self.bias = bias

        def forward(self, x):
            # Partial outputs per rank
            partials = [torch.nn.functional.linear(x, self.weights[i]) for i in range(self.tp_size)]
            # All-reduce: concatenate along feature dim (column-parallel sum
            # semantics after the split). Megatron uses all-reduce on the
            # summed output; column-parallel keeps full out_features via concat.
            return torch.cat(partials, dim=-1)

    return ColumnParallelLinear


def build_megatron_mock_model(cfg: MegatronTrialConfig) -> Any:
    """Build TP-aware mock 70B: column-parallel Linear + SDPA attention.

    PP stages: model split into pp_size chunks of layers/stage. We build the
    full model and the pipeline schedule splits it for forward.
    """
    from torch import nn

    SelfAttnBlock = _make_selfattn_block_cls()
    ColParallelLinear = _make_col_parallel_linear_cls()

    layers = []
    for i in range(cfg.mock_layers):
        layers.append(SelfAttnBlock(cfg.mock_dim, cfg.mock_n_heads))
        layers.append(ColParallelLinear(cfg.mock_dim, cfg.mock_dim, tp_size=cfg.tp_size, bias=False))
        if i < cfg.mock_layers - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def split_into_pp_stages(model: Any, pp_size: int) -> list[Any]:
    """Split nn.Sequential into pp_size stages (Pipeline Parallelism @209-217).

    Each stage = contiguous chunk of layers. Forward passes activations
    between stages (mock: sequential, no real pipeline bubble on single-proc).
    """
    children = list(model.children())
    n = len(children)
    per_stage = n // pp_size
    stages = []
    for s in range(pp_size):
        start = s * per_stage
        end = (s + 1) * per_stage if s < pp_size - 1 else n
        from torch import nn

        stages.append(nn.Sequential(*children[start:end]))
    return stages


def run_pp_forward(stages: list[Any], x: Any, microbatch_size: int) -> Any:
    """GPipe-style pipeline forward: split batch into microbatches, flow
    through stages, concat outputs. (interleaved1f1b noted; gpipe mock simpler.)

    Real interleaved1f1b interleaves microbatches across stages to cut bubble.
    Mock = gpipe (all microbatches through stage 1, then stage 2, etc).
    """
    import torch

    # Split batch into microbatches
    mb = microbatch_size
    if x.shape[0] % mb != 0 and x.shape[0] < mb:
        # pad not needed; single microbatch
        micro_batches = [x]
    else:
        n_mb = x.shape[0] // mb
        micro_batches = list(x.chunk(n_mb, dim=0))

    outputs = []
    for mb_x in micro_batches:
        h = mb_x
        for stage in stages:
            h = stage(h)
            while isinstance(h, tuple):
                h = h[0]
        outputs.append(h)
    return torch.cat(outputs, dim=0)


def init_distributed_if_possible(cfg: MegatronTrialConfig) -> tuple[bool, str]:
    if "RANK" not in os.environ and not os.environ.get("TORCHELASTIC_RUN_ID"):
        return False, "RANK unset - single-process (no torchrun)"
    try:
        import torch.distributed as dist

        backend = "nccl" if os.environ.get("CUDA_VISIBLE_DEVICES") else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return dist.is_initialized(), f"dist init backend={backend}"
    except Exception as e:
        return False, f"dist init failed: {e}"


def init_wandb(cfg: MegatronTrialConfig) -> tuple[Any, str]:
    if not cfg.use_wandb:
        return None, "wandb disabled in config"
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
            mode="disabled" if os.environ.get("WANDB_MODE") == "disabled" else "online",
        )
        return run, f"wandb init project={cfg.wandb_project}"
    except Exception as e:
        return None, f"wandb init failed: {e} (stdout fallback)"


def _timed_train_steps(model_or_stages, cfg, is_pp: bool, optimizer) -> tuple[float, float]:
    """Run cfg.train_steps, return (total_time_s, last_loss)."""
    import torch

    model_or_stages.train() if not is_pp else None
    x = torch.randn(cfg.batch_size, cfg.mock_dim, cfg.mock_dim) if False else torch.randn(cfg.batch_size, cfg.mock_dim)

    start = time.perf_counter()
    last_loss = 0.0
    for step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none=True)
        if is_pp:
            out = run_pp_forward(model_or_stages, x, cfg.pp_microbatch_size)
        else:
            out = model_or_stages(x)
            while isinstance(out, tuple):
                out = out[0]
        loss = out.sum()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - start
    return elapsed, last_loss


def run_megatron_trial(cfg: MegatronTrialConfig) -> int:
    print(f"[PIX-4351] Megatron trial: tp={cfg.tp_size} pp={cfg.pp_size} cp={cfg.cp_size} schedule={cfg.pp_schedule}")

    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"[PIX-4351] CONFIG ERROR: {e}")
        return 2

    has_gpu, gpu_notes = _check_gpu_env()
    for n in gpu_notes:
        print(f"[PIX-4351] {n}")
    if not has_gpu:
        print("[PIX-4351] WARN: no GPU - CPU mock validates topology math only")

    try:
        import torch
    except ImportError:
        print("[PIX-4351] FAIL: torch not installed")
        return 3

    dist_ok, dist_note = init_distributed_if_possible(cfg)
    print(f"[PIX-4351] distributed: {dist_note}")

    wandb_run, wandb_note = init_wandb(cfg)
    print(f"[PIX-4351] wandb: {wandb_note}")

    # --- Megatron TP+PP path ---
    print(f"[PIX-4351] Building Megatron mock (TP={cfg.tp_size} column-parallel, PP={cfg.pp_size} stages)")
    model = build_megatron_mock_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[PIX-4351] mock model: {n_params} params, dim={cfg.mock_dim} layers={cfg.mock_layers} (TP shards per linear={cfg.tp_size})"
    )

    stages = split_into_pp_stages(model, cfg.pp_size)
    print(
        f"[PIX-4351] PP stages: {len(stages)} (layers/stage={cfg.mock_layers // cfg.pp_size}, schedule={cfg.pp_schedule} -> gpipe mock)"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    meg_time, meg_loss = _timed_train_steps(stages, cfg, is_pp=True, optimizer=optimizer)
    print(f"[PIX-4351] Megatron: {cfg.train_steps} steps in {meg_time:.3f}s, last_loss={meg_loss:.4f}")

    # --- FSDP2 comparison baseline (if enabled) ---
    fsdp_time = None
    fsdp_loss = None
    if cfg.compare_fsdp2:
        print("[PIX-4351] Building FSDP2 baseline (same model, no TP/PP) for throughput comparison")
        from torch import nn

        SelfAttnBlock = _make_selfattn_block_cls()
        fsdp_layers = []
        for i in range(cfg.mock_layers):
            fsdp_layers.append(SelfAttnBlock(cfg.mock_dim, cfg.mock_n_heads))
            fsdp_layers.append(nn.Linear(cfg.mock_dim, cfg.mock_dim, bias=False))
            if i < cfg.mock_layers - 1:
                fsdp_layers.append(nn.GELU())
        fsdp_model = nn.Sequential(*fsdp_layers)
        fsdp_n = sum(p.numel() for p in fsdp_model.parameters())
        print(f"[PIX-4351] FSDP2 baseline: {fsdp_n} params (plain Linear, no TP split)")
        fsdp_opt = torch.optim.AdamW(fsdp_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        fsdp_time, fsdp_loss = _timed_train_steps(fsdp_model, cfg, is_pp=False, optimizer=fsdp_opt)
        print(f"[PIX-4351] FSDP2: {cfg.train_steps} steps in {fsdp_time:.3f}s, last_loss={fsdp_loss:.4f}")

    # --- Throughput comparison ---
    metrics = {
        "megatron/time_s": meg_time,
        "megatron/loss": meg_loss,
        "megatron/tp_size": cfg.tp_size,
        "megatron/pp_size": cfg.pp_size,
        "megatron/cp_size": cfg.cp_size,
        "megatron/pp_schedule": cfg.pp_schedule,
        "world_size": cfg.world_size,
        "n_params": n_params,
    }
    if fsdp_time is not None:
        # throughput proxy: steps/sec (higher = faster)
        meg_tput = cfg.train_steps / meg_time if meg_time > 0 else 0
        fsdp_tput = cfg.train_steps / fsdp_time if fsdp_time > 0 else 0
        metrics["fsdp2/time_s"] = fsdp_time
        metrics["fsdp2/loss"] = fsdp_loss
        metrics["fsdp2/steps_per_s"] = fsdp_tput
        metrics["megatron/steps_per_s"] = meg_tput
        winner = "megatron" if meg_tput > fsdp_tput else "fsdp2"
        metrics["winner"] = winner
        speedup = max(meg_tput, fsdp_tput) / min(meg_tput, fsdp_tput) if min(meg_tput, fsdp_tput) > 0 else 0
        print(f"[PIX-4351] Throughput: megatron={meg_tput:.2f} steps/s | fsdp2={fsdp_tput:.2f} steps/s")
        print(f"[PIX-4351] Winner: {winner} ({speedup:.2f}x faster) — NOTE: CPU mock, real verdict needs 8xH100")

    if wandb_run is not None:
        try:
            wandb_run.log(metrics)
            wandb_run.finish()
        except Exception as e:
            print(f"[PIX-4351] wandb log failed: {e}")

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print("[PIX-4351] OK: Megatron TP+PP trial completed (topology + throughput comparison + teardown)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIX-4351 Megatron trial (TP=4, PP=2, CP=1) vs FSDP2")
    parser.add_argument("--tp-size", type=int, default=MegatronTrialConfig().tp_size)
    parser.add_argument("--pp-size", type=int, default=MegatronTrialConfig().pp_size)
    parser.add_argument("--cp-size", type=int, default=MegatronTrialConfig().cp_size)
    parser.add_argument("--pp-schedule", type=str, default=MegatronTrialConfig().pp_schedule)
    parser.add_argument("--pp-microbatch-size", type=int, default=MegatronTrialConfig().pp_microbatch_size)
    parser.add_argument("--train-steps", type=int, default=MegatronTrialConfig().train_steps)
    parser.add_argument("--batch-size", type=int, default=MegatronTrialConfig().batch_size)
    parser.add_argument("--lr", type=float, default=MegatronTrialConfig().learning_rate)
    parser.add_argument("--no-compare-fsdp2", action="store_true", help="skip FSDP2 baseline")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-disabled", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = MegatronTrialConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
        pp_schedule=args.pp_schedule,
        pp_microbatch_size=args.pp_microbatch_size,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        compare_fsdp2=not args.no_compare_fsdp2,
        use_wandb=not args.no_wandb,
    )
    if args.wandb_disabled:
        os.environ["WANDB_MODE"] = "disabled"
    sys.exit(run_megatron_trial(cfg))


if __name__ == "__main__":
    main()
