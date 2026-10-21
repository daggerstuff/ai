#!/usr/bin/env python3
"""V5 SFT launcher: G04 protocol with a schema-valid, stable SFT config.

v4 root causes addressed here (data fixes live in the v5 release itself):
- batch_size 16 instead of "auto" (auto resolved to 1 packed sequence per step,
  which let gradient norms spike past 700k on long unshuffled rows)
- use the scalar learning rate supported by the Serverless Training API
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wandb_sft_g04 as g04

V5_CONFIG: dict[str, object] = {
    "batch_size": 16,
    "learning_rate": 1e-05,
    "assistant_turns": "all",
}


def main() -> int:
    g04.BASELINE_CONFIG.clear()
    g04.BASELINE_CONFIG.update(V5_CONFIG)
    return g04.main()


if __name__ == "__main__":
    raise SystemExit(main())
