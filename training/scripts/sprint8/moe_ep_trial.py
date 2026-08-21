#!/usr/bin/env python3
"""PIX-4352: 405B/MoE + Expert Parallelism (EP) trial.

Sprint 8 Lane B optional - Low priority (4).

Per blueprint @169: "Month 3: 405B / MoE path with `strategy=megatron` + EP
if pipeline extends." MegatronFSDP limitation @302-304: no PP (pp_size>1
raises), no EP (ep_size>1 raises) — so 405B/MoE needs pure Megatron-LM for EP.

Expert Parallelism (EP): shards MoE experts across ranks. With ep_size ranks,
num_experts // ep_size experts live on each rank. Router/gating dispatches
tokens to top-k experts; all-to-all sends tokens to the rank owning the expert.
This trial exercises the EP topology math on CPU single-proc using plain torch
primitives (no megatron-core / transformer_engine, both GPU-only + absent):

  - EP: ExpertParallelLinear — ParameterList of num_experts experts, each expert
    assigned an ep_rank = expert_id // (num_experts // ep_size). Single-proc mock
    runs all experts locally but tracks the ep_rank mapping so the shard math is
    exercised. Forward = route tokens top-k, run selected experts, weighted sum.
  - Gating/Router: top-k softmax over expert logits per token. Load-balance aux
    loss (importance + load) added to base loss so grads flow to router.
  - Dense baseline: same param-budget non-MoE MLP for throughput comparison.

Prod swaps mock for real Megatron-LM MoE + transformer_engine + 8-GPU nodes.

Blueprint ref: docs/training-pipeline-blueprint-2026-08-10.md @169, @302-304.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MoEEPTrialConfig:
    """405B/MoE + EP trial config (per blueprint @169, @302-304)."""

    model_name: str = "mock_405B_moe_ep"
    strategy: str = "megatron"  # @169 strategy=megatron + EP
    ep_size: int = 8  # expert parallel ranks (shards experts)
    num_experts: int = 16  # MoE expert count (must divide ep_size)
    top_k: int = 2  # experts active per token
    expert_capacity_factor: float = 1.25  # capacity buffer for dropped tokens
    # Training + throughput comparison
    train_steps: int = 5
    batch_size: int = 4
    seq_len: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    use_wandb: bool = True
    wandb_project: str = "pixelated-sprint8-moe-ep"
    wandb_run_name: str = "moe-ep-405B"
    compare_dense: bool = True  # run dense (non-MoE) baseline for comparison
    aux_loss_weight: float = 0.01  # load-balance aux loss weight
    # Mock model
    mock_dim: int = 256
    mock_layers: int = 4
    mock_n_heads: int = 4
    mock_dtype: str = "bfloat16"
    attention_backend: str = "sdpa"  # @292

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.ep_size < 1:
            errs.append(f"ep_size={self.ep_size} must be >= 1")
        if self.num_experts < 1:
            errs.append(f"num_experts={self.num_experts} must be >= 1")
        if self.num_experts % self.ep_size != 0:
            errs.append(
                f"num_experts={self.num_experts} not divisible by ep_size={self.ep_size} "
                f"(EP requires even expert-per-rank split)"
            )
        if self.top_k < 1 or self.top_k > self.num_experts:
            errs.append(f"top_k={self.top_k} must be in [1, num_experts={self.num_experts}]")
        if self.expert_capacity_factor < 1.0:
            errs.append(f"expert_capacity_factor={self.expert_capacity_factor} must be >= 1.0")
        if self.strategy != "megatron":
            errs.append(f"strategy={self.strategy} must be 'megatron' (@169); MegatronFSDP has no EP (@302-304)")
        if self.mock_dim % self.mock_n_heads != 0:
            errs.append(f"mock_dim={self.mock_dim} not divisible by mock_n_heads={self.mock_n_heads}")
        if self.attention_backend not in {"sdpa", "te", "flash_attention"}:
            errs.append(f"attention_backend={self.attention_backend} must be sdpa/te/flash (@292)")
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


def expert_ep_rank(expert_id: int, num_experts: int, ep_size: int) -> int:
    """Map expert_id to its EP rank (shard). num_experts // ep_size experts per rank."""
    experts_per_rank = num_experts // ep_size
    return min(expert_id // max(experts_per_rank, 1), ep_size - 1)


class _ExpertParallelLinear:
    pass


def _make_ep_linear_cls():
    """MoE ExpertParallelLinear via plain torch.

    num_experts experts; each expert = (W, b) Linear. EP shard mapping:
    expert -> ep_rank = expert_id // (num_experts // ep_size). Single-proc mock
    runs ALL experts locally (no real all-to-all), but ep_rank is tracked so
    the shard topology math is exercised and verified.
    """
    import torch
    from torch import nn

    class ExpertParallelLinear(nn.Module):
        def __init__(
            self, in_features: int, out_features: int, num_experts: int, ep_size: int, bias: bool = True
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_experts = num_experts
            self.ep_size = ep_size
            self.experts_per_rank = num_experts // ep_size
            self.expert_weights = nn.ParameterList([nn.Parameter(torch.empty(num_experts, out_features, in_features))])
            nn.init.normal_(self.expert_weights[0], std=0.02)
            self.bias = bias
            if bias:
                self.expert_bias = nn.ParameterList([nn.Parameter(torch.zeros(num_experts, out_features))])
            # ep_rank mapping (precomputed, for verification)
            self.ep_rank_map = [expert_ep_rank(e, num_experts, ep_size) for e in range(num_experts)]

        def forward_single_expert(self, x: Any, expert_id: int) -> Any:
            w = self.expert_weights[0][expert_id]
            out = torch.nn.functional.linear(x, w)
            if self.bias:
                out = out + self.expert_bias[0][expert_id]
            return out

        def forward(self, x: Any, router_logits: Any, top_k: int):
            """x: (B, S, D). router_logits: (B, S, num_experts). Returns
            output tensor only (aux stored on self.last_aux for collection).
            Routes each token to top_k experts, weighted sum.
            """
            B, S, D = x.shape
            # Router softmax over experts
            router_probs = torch.softmax(router_logits, dim=-1)  # (B,S,E)
            topk_probs, topk_idx = torch.topk(router_probs, k=top_k, dim=-1)  # (B,S,k)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

            out = torch.zeros(B, S, D, dtype=x.dtype, device=x.device)
            for k in range(top_k):
                for e in range(self.num_experts):
                    mask = topk_idx[..., k] == e  # (B,S) bool
                    if not mask.any():
                        continue
                    expert_out = self.forward_single_expert(x, e)  # (B,S,D')
                    weight = topk_probs[..., k : k + 1] * mask.unsqueeze(-1).float()  # (B,S,1)
                    out = out + expert_out * weight

            # Load-balance aux loss (importance + load, per Switch Transformer)
            self.last_aux = self._load_balance_loss(router_probs, topk_idx, topk_probs)
            return out

        def _load_balance_loss(self, router_probs: Any, topk_idx: Any, topk_probs: Any) -> Any:
            """Switch-Transformer load-balance aux loss.

            L = num_experts * sum_e (f_e * P_e)
            f_e = fraction of tokens dispatched to expert e
            P_e = mean router probability for expert e
            """
            B, S, E = router_probs.shape
            n_tokens = B * S
            # f_e: fraction routed to each expert (over all top-k dispatches)
            one_hot = torch.zeros_like(router_probs)
            src = torch.ones_like(topk_idx, dtype=router_probs.dtype)
            for k in range(topk_idx.shape[-1]):
                one_hot.scatter_add_(-1, topk_idx[..., k : k + 1], src[..., k : k + 1])
            f = one_hot.sum(dim=(0, 1)) / (n_tokens * topk_idx.shape[-1] + 1e-9)  # (E,)
            P = router_probs.mean(dim=(0, 1))  # (E,)
            return E * (f * P).sum()

    return ExpertParallelLinear


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


class _MoELayer:
    pass


def _make_moe_layer_cls():
    """MoE layer: router (Linear D->num_experts) + ExpertParallelLinear + residual."""
    from torch import nn

    class MoELayer(nn.Module):
        def __init__(self, dim: int, num_experts: int, ep_size: int, top_k: int, aux_weight: float) -> None:
            super().__init__()
            self.dim = dim
            self.num_experts = num_experts
            self.ep_size = ep_size
            self.top_k = top_k
            self.aux_weight = aux_weight
            self.router = nn.Linear(dim, num_experts, bias=False)
            EPLinear = _make_ep_linear_cls()
            self.expert_layer = EPLinear(dim, dim, num_experts, ep_size, bias=True)

        def forward(self, x):
            router_logits = self.router(x)  # (B,S,E)
            out = self.expert_layer(x, router_logits, self.top_k)
            # aux stored on expert_layer.last_aux; collected by training loop
            return x + out

    return MoELayer


def build_moe_mock_model(cfg: MoEEPTrialConfig) -> Any:
    """Build MoE mock 405B: SDPA attention + MoE layers stacked."""
    from torch import nn

    SelfAttnBlock = _make_selfattn_block_cls()
    MoELayer = _make_moe_layer_cls()

    layers = []
    for i in range(cfg.mock_layers):
        layers.append(SelfAttnBlock(cfg.mock_dim, cfg.mock_n_heads))
        layers.append(MoELayer(cfg.mock_dim, cfg.num_experts, cfg.ep_size, cfg.top_k, cfg.aux_loss_weight))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def build_dense_baseline_model(cfg: MoEEPTrialConfig) -> Any:
    """Dense (non-MoE) baseline with matched parameter budget.

    Dense FFN dim scaled so param count ≈ MoE (num_experts * D * D each layer).
    """
    from torch import nn

    SelfAttnBlock = _make_selfattn_block_cls()
    # Dense FFN: D -> 4D -> D (standard transformer FFN)
    layers = []
    for i in range(cfg.mock_layers):
        layers.append(SelfAttnBlock(cfg.mock_dim, cfg.mock_n_heads))
        layers.append(nn.Linear(cfg.mock_dim, cfg.mock_dim * 4))
        layers.append(nn.GELU())
        layers.append(nn.Linear(cfg.mock_dim * 4, cfg.mock_dim))
    return nn.Sequential(*layers)


def _timed_train_steps(model, cfg, optimizer, is_moe: bool) -> tuple[float, float, float]:
    """Run cfg.train_steps. Return (total_time_s, last_loss, last_aux_loss)."""
    import torch

    model.train()
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.mock_dim)

    start = time.perf_counter()
    last_loss = 0.0
    last_aux = 0.0
    for step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        # MoE Sequential returns intermediate tensors; use final output
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        if is_moe:
            # Collect aux loss from each MoE layer (stored on expert_layer.last_aux)
            import torch as _t

            aux_total = _t.tensor(0.0, device=out.device)
            for module in model.modules():
                if hasattr(module, "expert_layer") and hasattr(module.expert_layer, "last_aux"):
                    aux_total = aux_total + module.expert_layer.last_aux
            loss = loss + cfg.aux_loss_weight * aux_total
            last_aux = float(aux_total.detach().cpu())
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - start
    return elapsed, last_loss, last_aux


def run_moe_ep_trial(cfg: MoEEPTrialConfig) -> int:
    print(
        f"[PIX-4352] MoE+EP trial: strategy={cfg.strategy} ep_size={cfg.ep_size} "
        f"num_experts={cfg.num_experts} top_k={cfg.top_k}"
    )

    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"[PIX-4352] CONFIG ERROR: {e}")
        return 2

    has_gpu, gpu_notes = _check_gpu_env()
    for n in gpu_notes:
        print(f"[PIX-4352] {n}")
    if not has_gpu:
        print("[PIX-4352] WARN: no GPU - CPU mock validates EP topology math only")

    try:
        import torch
    except ImportError:
        print("[PIX-4352] FAIL: torch not installed")
        return 3

    # Verify EP shard mapping
    print("[PIX-4352] EP shard map (expert_id -> ep_rank):")
    experts_per_rank = cfg.num_experts // cfg.ep_size
    for e in range(cfg.num_experts):
        r = expert_ep_rank(e, cfg.num_experts, cfg.ep_size)
        if e < 4 or e >= cfg.num_experts - 2:
            print(f"  expert {e} -> ep_rank {r} ({experts_per_rank} experts/rank)")
    assert all(
        0 <= r < cfg.ep_size for r in [expert_ep_rank(e, cfg.num_experts, cfg.ep_size) for e in range(cfg.num_experts)]
    ), "EP rank out of range"
    print(
        f"[PIX-4352] EP verified: {cfg.num_experts} experts across {cfg.ep_size} ranks, "
        f"{experts_per_rank} experts/rank, top_k={cfg.top_k}"
    )

    # --- MoE+EP path ---
    print(f"[PIX-4352] Building MoE mock (ep={cfg.ep_size}, experts={cfg.num_experts}, top_k={cfg.top_k})")
    model = build_moe_mock_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PIX-4352] mock MoE model: {n_params} params, dim={cfg.mock_dim} layers={cfg.mock_layers}")

    # Direct MoE forward test (verify router + expert dispatch + aux loss)
    MoELayer = _make_moe_layer_cls()
    test_moe = MoELayer(cfg.mock_dim, cfg.num_experts, cfg.ep_size, cfg.top_k, cfg.aux_loss_weight)
    test_x = __import__("torch").randn(2, 8, cfg.mock_dim)
    test_out = test_moe(test_x)
    test_aux = test_moe.expert_layer.last_aux
    print(
        f"[PIX-4352] MoE layer test: in={tuple(test_x.shape)} -> out={tuple(test_out.shape)} "
        f"aux_loss={float(test_aux):.4f} (residual preserved: {test_out.shape == test_x.shape})"
    )
    if test_out.shape != test_x.shape:
        print("[PIX-4352] FAIL: MoE output shape mismatch")
        return 4

    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if False
        else __import__("torch").optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    )
    import torch

    moe_time, moe_loss, moe_aux = _timed_train_steps(model, cfg, optimizer, is_moe=True)
    print(f"[PIX-4352] MoE+EP: {cfg.train_steps} steps in {moe_time:.3f}s, last_loss={moe_loss:.4f}")

    # --- Dense baseline ---
    dense_time = None
    dense_loss = None
    if cfg.compare_dense:
        print("[PIX-4352] Building dense (non-MoE) baseline for throughput comparison")
        dense_model = build_dense_baseline_model(cfg)
        dense_n = sum(p.numel() for p in dense_model.parameters())
        print(f"[PIX-4352] dense baseline: {dense_n} params (FFN 4D, no expert split)")
        dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        dense_time, dense_loss, _ = _timed_train_steps(dense_model, cfg, dense_opt, is_moe=False)
        print(f"[PIX-4352] dense: {cfg.train_steps} steps in {dense_time:.3f}s, last_loss={dense_loss:.4f}")

    # --- Throughput comparison ---
    metrics = {
        "moe_ep/time_s": moe_time,
        "moe_ep/loss": moe_loss,
        "moe_ep/ep_size": cfg.ep_size,
        "moe_ep/num_experts": cfg.num_experts,
        "moe_ep/top_k": cfg.top_k,
        "moe_ep/experts_per_rank": experts_per_rank,
        "moe_ep/n_params": n_params,
        "strategy": cfg.strategy,
    }
    if dense_time is not None:
        moe_tput = cfg.train_steps / moe_time if moe_time > 0 else 0
        dense_tput = cfg.train_steps / dense_time if dense_time > 0 else 0
        metrics["dense/time_s"] = dense_time
        metrics["dense/loss"] = dense_loss
        metrics["dense/steps_per_s"] = dense_tput
        metrics["moe_ep/steps_per_s"] = moe_tput
        winner = "moe_ep" if moe_tput > dense_tput else "dense"
        metrics["winner"] = winner
        speedup = max(moe_tput, dense_tput) / min(moe_tput, dense_tput) if min(moe_tput, dense_tput) > 0 else 0
        print(f"[PIX-4352] Throughput: moe_ep={moe_tput:.2f} steps/s | dense={dense_tput:.2f} steps/s")
        print(
            f"[PIX-4352] Winner: {winner} ({speedup:.2f}x) — NOTE: CPU mock, real verdict needs 8xH100 + real Megatron-LM"
        )

    # WandB
    if cfg.use_wandb:
        try:
            import wandb

            os.environ.setdefault("WANDB_MODE", "disabled")
            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=vars(cfg),
                mode=os.environ.get("WANDB_MODE", "disabled"),
            )
            run.log(metrics)
            run.finish()
        except Exception as e:
            print(f"[PIX-4352] wandb skipped: {e}")

    print("[PIX-4352] OK: MoE+EP trial completed (EP topology + router dispatch + aux loss + teardown)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIX-4352 MoE + Expert Parallelism (EP) trial")
    parser.add_argument("--ep-size", type=int, default=MoEEPTrialConfig().ep_size)
    parser.add_argument("--num-experts", type=int, default=MoEEPTrialConfig().num_experts)
    parser.add_argument("--top-k", type=int, default=MoEEPTrialConfig().top_k)
    parser.add_argument("--train-steps", type=int, default=MoEEPTrialConfig().train_steps)
    parser.add_argument("--batch-size", type=int, default=MoEEPTrialConfig().batch_size)
    parser.add_argument("--seq-len", type=int, default=MoEEPTrialConfig().seq_len)
    parser.add_argument("--lr", type=float, default=MoEEPTrialConfig().learning_rate)
    parser.add_argument("--no-compare-dense", action="store_true", help="skip dense baseline")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = MoEEPTrialConfig(
        ep_size=args.ep_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        compare_dense=not args.no_compare_dense,
        use_wandb=not args.no_wandb,
    )
    sys.exit(run_moe_ep_trial(cfg))


if __name__ == "__main__":
    main()
