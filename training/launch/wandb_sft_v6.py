#!/usr/bin/env python3
"""V6 SFT launcher: G04 protocol over the agent-tool-augmented v6 release.

Carries forward the v5 fixes and adds the per-step LR schedule the
Serverless Training API requires when learning_rate is an array:
- batch_size 16 (not "auto", which resolved to 1 packed sequence per step)
- learning_rate = full schedule, len == total_batch = ceil(78681/16) = 4918
  (warmup 5%: 1e-5 -> 5e-5, hold to 60%, cosine decay to 5e-6)
- assistant_turns "all", metric_logging enabled at 50-step targets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wandb_sft_g04 as g04

_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/training/execution-2026-10-21/v6-launch-config.json"
)


def _load_config() -> dict[str, object]:
    with _CONFIG_PATH.open() as f:
        config = json.load(f)["config"]
    if not isinstance(config, dict):
        raise ValueError("v6 launch config must be a JSON object")
    return config


def main() -> int:
    g04.BASELINE_CONFIG.clear()
    g04.BASELINE_CONFIG.update(_load_config())
    return int(g04.main())


if __name__ == "__main__":
    raise SystemExit(main())
