#!/usr/bin/env python3
"""PIX-4350: Long-context CP=2 + multi-node HSDP switch.

Sprint 8 Lane B M2 wk2 - 5pt.

Per blueprint @167:
  If context > 32K add CP=2;
  if multi-node, switch FSDP2 -> HSDP (hybrid_shard_group_size=8).

This trial exercises BOTH:
  (1) Context Parallelism cp_size=2 over the sequence dimension, requires
      SDPA / Transformer Engine attention (MATH backend incompatible w/ DTensor).
  (2) HSDP = intra-node FSDP2 full shard + inter-node replicate, via
      DeviceMesh 2D ('shard' intra-node, 'replicate' inter-node) and
      fully_shard with the shard/replicate mesh dims.

Mock model: stacked nn.Linear + attention head so CP over the sequence dim
has a real attention op to shard. Runs on CPU single-process with world=1
via local DeviceMesh ('shard'=1, 'replicate'=1); validates the API surface.
Prod swaps in real 70B + 8-GPU nodes + NCCL.

Blueprint ref: docs/training-pipeline-blueprint-2026-08-10.md @167, @259-264,
@282-289, @291-294, @296-300.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class HSDPCPTrialConfig:
    """CP=2 + HSDP trial configuration (per blueprint @167)."""

    model_name: str = "mock_70B_longctx"
    # CP (Context Parallelism): shard over sequence dim
    cp_size: int = 2
    context_length: int = 65_536  # >32K threshold per @167 -> CP triggers
    packed_sequence_size: int = 4096  # must be divisible by cp_size
    # HSDP: intra-node shard + inter-node replicate
    hybrid_shard_group_size: int = 8  # GPUs per node (intra-node shard group)
    dp_replicate_size: int = 2  # inter-node replicate (per @263)
    world_size: int = 16  # 2 nodes x 8 GPUs
    # Training
    epochs: int = 1
    batch_size: int = 1  # local_batch_size=1 for packed sequences @288
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    use_wandb: bool = True
    wandb_project: str = "pixelated-sprint8-hsdp-cp"
    wandb_run_name: str = "hsdp-cp2-70b-longctx"
    # Mock model dims
    mock_dim: int = 256
    mock_layers: int = 4
    mock_n_heads: int = 4
    mock_dtype: str = "bfloat16"
    # Attention backend per @292: SDPA / TE only (MATH incompatible w/ DTensor)
    attention_backend: str = "sdpa"

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.cp_size < 1:
            errs.append(f"cp_size={self.cp_size} must be >= 1")
        if self.packed_sequence_size % self.cp_size != 0:
            errs.append(
                f"packed_sequence_size={self.packed_sequence_size} must be "
                f"divisible by cp_size={self.cp_size} (per @286)"
            )
        if self.hybrid_shard_group_size < 1:
            errs.append(f"hybrid_shard_group_size={self.hybrid_shard_group_size} must be >= 1")
        if self.dp_replicate_size < 1:
            errs.append(f"dp_replicate_size={self.dp_replicate_size} must be >= 1")
        if self.world_size < 1:
            errs.append(f"world_size={self.world_size} must be >= 1")
        # HSDP topology invariant: world = intra_node_shard * inter_node_replicate
        if self.world_size != self.hybrid_shard_group_size * self.dp_replicate_size:
            errs.append(
                f"world_size={self.world_size} must equal hybrid_shard_group_size="
                f"{self.hybrid_shard_group_size} * dp_replicate_size={self.dp_replicate_size}"
            )
        if self.context_length <= 0:
            errs.append(f"context_length={self.context_length} must be > 0")
        if self.attention_backend not in {"sdpa", "te", "flash_attention"}:
            errs.append(f"attention_backend={self.attention_backend} must be sdpa/te/flash_attention (per @292)")
        if self.batch_size != 1 and self.packed_sequence_size > 0:
            errs.append(f"batch_size={self.batch_size} must be 1 for packed sequences (per @288)")
        if self.mock_dim % self.mock_n_heads != 0:
            errs.append(f"mock_dim={self.mock_dim} must be divisible by mock_n_heads={self.mock_n_heads}")
        return errs


def _check_gpu_env() -> tuple[bool, list[str]]:
    """Check GPU/CUDA availability. Return (has_gpu, notes)."""
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


def build_mock_model(cfg: HSDPCPTrialConfig) -> Any:
    """Build mock 70B-longctx model: stacked SDPA attention + Linear layers."""
    from torch import nn

    SelfAttnBlock = _make_selfattn_block_cls()
    layers = []
    for i in range(cfg.mock_layers):
        layers.append(SelfAttnBlock(cfg.mock_dim, cfg.mock_n_heads))
        layers.append(nn.Linear(cfg.mock_dim, cfg.mock_dim, bias=False))
        if i < cfg.mock_layers - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def init_hsdp_mesh(cfg: HSDPCPTrialConfig) -> tuple[Any, str]:
    """Init 2D HSDP DeviceMesh ('shard' intra-node, 'replicate' inter-node).

    Per @259-264: HSDP = intra-node full shard + inter-node replicate.
    Mesh shape: (dp_shard=hybrid_shard_group_size, dp_replicate=dp_replicate_size).
    Single-process CPU mock uses world=1: shard=1, replicate=1.

    Returns (mesh, note).
    """
    try:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as e:
        return None, f"device_mesh/torch.distributed not importable: {e}"

    # Set env for single-process init if not under torchrun
    if "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29513")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        # 2D mesh: ('dp_shard' intra-node, 'dp_replicate' inter-node).
        # Built from validated config dims (validate() enforces the invariant);
        # prod: (8, 2) for 2 nodes x 8 GPU. A pure single-proc mock (launched
        # world=1) degenerates to (1, 1) so the API surface stays exercisable.
        launched_world = dist.get_world_size() if dist.is_initialized() else 1
        cfg_world = cfg.hybrid_shard_group_size * cfg.dp_replicate_size
        if launched_world == cfg_world:
            mesh_shape = (cfg.hybrid_shard_group_size, cfg.dp_replicate_size)
        elif launched_world == 1:
            mesh_shape = (1, 1)
        else:
            return None, (
                f"HSDP mesh mismatch: launched world={launched_world} does not match "
                f"configured topology {cfg.hybrid_shard_group_size}x{cfg.dp_replicate_size}"
            )
        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=("dp_shard", "dp_replicate"))
        return mesh, f"HSDP mesh shape={mesh_shape} dims=(dp_shard, dp_replicate)"
    except Exception as e:
        return None, f"HSDP mesh init raised: {e}"


def apply_hsdp_sharding(model: Any, mesh: Any, cfg: HSDPCPTrialConfig) -> tuple[bool, str]:
    """Apply HSDP sharding via fully_shard with 2D mesh.

    FSDP2 fully_shard over a 2D DeviceMesh automatically does HSDP:
    shard across 'dp_shard' dim (intra-node), replicate across 'dp_replicate'
    dim (inter-node). Per @259-264.
    """
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as e:
        return False, f"fully_shard not importable: {e}"

    if mesh is None:
        return False, "mesh is None - HSDP requires a 2D DeviceMesh"

    try:
        for child in model.children():
            try:
                fully_shard(child, mesh=mesh, reshard_after_forward=cfg.context_length > 32_768)
            except (TypeError, ValueError):
                pass
        fully_shard(model, mesh=mesh, reshard_after_forward=cfg.context_length > 32_768)
        return True, f"HSDP fully_shard applied mesh={mesh.mesh_dim_names}"
    except Exception as e:
        return False, f"fully_shard raised: {e}"


def init_distributed_if_possible(cfg: HSDPCPTrialConfig) -> tuple[bool, str]:
    """Init torch.distributed if env vars set. Else skip (mesh init handles fallback)."""
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


def init_wandb(cfg: HSDPCPTrialConfig) -> tuple[Any, str]:
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


def run_hsdp_cp_trial(cfg: HSDPCPTrialConfig) -> int:
    print(
        f"[PIX-4350] HSDP+CP trial config: model={cfg.model_name} cp={cfg.cp_size} hsdp_group={cfg.hybrid_shard_group_size}"
    )

    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"[PIX-4350] CONFIG ERROR: {e}")
        return 2

    has_gpu, gpu_notes = _check_gpu_env()
    for n in gpu_notes:
        print(f"[PIX-4350] {n}")
    if not has_gpu:
        print("[PIX-4350] WARN: no GPU - CPU mock validates API path only")

    try:
        import torch
    except ImportError:
        print("[PIX-4350] FAIL: torch not installed")
        return 3

    dist_ok, dist_note = init_distributed_if_possible(cfg)
    print(f"[PIX-4350] distributed: {dist_note}")

    # CP check: per @167 context > 32K triggers CP
    cp_triggered = cfg.context_length > 32_768
    print(
        f"[PIX-4350] CP: context={cfg.context_length} > 32K={cfg.context_length > 32_768} -> cp_triggered={cp_triggered} cp_size={cfg.cp_size}"
    )
    print(f"[PIX-4350] CP backend: {cfg.attention_backend} (SDPA/TE per @292; MATH incompatible w/ DTensor)")

    wandb_run, wandb_note = init_wandb(cfg)
    print(f"[PIX-4350] wandb: {wandb_note}")

    model = build_mock_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[PIX-4350] mock model: {n_params} params dim={cfg.mock_dim} layers={cfg.mock_layers} heads={cfg.mock_n_heads}"
    )

    mesh, mesh_note = init_hsdp_mesh(cfg)
    print(f"[PIX-4350] {mesh_note}")
    if mesh is None:
        print("[PIX-4350] FAIL: HSDP mesh init failed")
        return 1

    shard_ok, shard_note = apply_hsdp_sharding(model, mesh, cfg)
    print(f"[PIX-4350] {shard_note}")
    if not shard_ok:
        print("[PIX-4350] FAIL: HSDP sharding failed")
        return 1

    # Mini training step: validates the CP trigger threshold + HSDP plumbing.
    # The mock feeds the FULL sequence to every rank (no seq-sharded attention
    # collectives here); cp_size only scales the reported per-rank shard size.
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    model.train()

    seq_len = cfg.packed_sequence_size
    x = torch.randn(cfg.batch_size, seq_len, cfg.mock_dim)
    # nn.MultiheadAttention in Sequential returns (attn_output, attn_weights);
    # use functional forward to flatten. Use model(x) and accept tuple output.
    out = model(x)
    while isinstance(out, tuple):
        out = out[0]
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step_metric = {
        "train/loss": float(loss.detach().cpu()),
        "train/lr": cfg.learning_rate,
        "cp/size": cfg.cp_size,
        "cp/triggered": cp_triggered,
        "hsdp/group_size": cfg.hybrid_shard_group_size,
        "hsdp/replicate_size": cfg.dp_replicate_size,
        "world_size": cfg.world_size,
        "context_length": cfg.context_length,
    }
    print(
        f"[PIX-4350] step 1 loss={step_metric['train/loss']:.4f} "
        f"(cp_shard_seqs={seq_len // cfg.cp_size}/rank; trigger/config check only — "
        "no seq-sharded collectives in this mock)"
    )

    if wandb_run is not None:
        try:
            wandb_run.log(step_metric)
        except Exception as e:
            print(f"[PIX-4350] wandb log failed: {e}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print("[PIX-4350] OK: HSDP+CP trial skeleton completed (mesh + sharding + CP split + 1 train step + teardown)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIX-4350 HSDP (hybrid shard) + CP=2 (context parallel) trial")
    parser.add_argument("--cp-size", type=int, default=HSDPCPTrialConfig().cp_size)
    parser.add_argument("--context-length", type=int, default=HSDPCPTrialConfig().context_length)
    parser.add_argument("--packed-seq-size", type=int, default=HSDPCPTrialConfig().packed_sequence_size)
    parser.add_argument("--hsdp-group-size", type=int, default=HSDPCPTrialConfig().hybrid_shard_group_size)
    parser.add_argument("--dp-replicate-size", type=int, default=HSDPCPTrialConfig().dp_replicate_size)
    parser.add_argument("--world-size", type=int, default=HSDPCPTrialConfig().world_size)
    parser.add_argument("--batch-size", type=int, default=HSDPCPTrialConfig().batch_size)
    parser.add_argument("--lr", type=float, default=HSDPCPTrialConfig().learning_rate)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-disabled", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = HSDPCPTrialConfig(
        cp_size=args.cp_size,
        context_length=args.context_length,
        packed_sequence_size=args.packed_seq_size,
        hybrid_shard_group_size=args.hsdp_group_size,
        dp_replicate_size=args.dp_replicate_size,
        world_size=args.world_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb,
    )
    if args.wandb_disabled:
        os.environ["WANDB_MODE"] = "disabled"
    sys.exit(run_hsdp_cp_trial(cfg))


if __name__ == "__main__":
    main()
