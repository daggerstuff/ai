#!/usr/bin/env python3
"""Compatibility wrapper for Serverless RL execution on CoreWeave Forge."""

from __future__ import annotations

from training.scripts.train_rl_serverless import main

if __name__ == "__main__":
    raise SystemExit(main())
