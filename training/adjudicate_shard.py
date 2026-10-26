#!/usr/bin/env python3
"""Shard driver for adjudicate_flags.py.

Usage: adjudicate_shard.py <shard_id> <arc1,arc2,...>

Runs adjudicate_flags.main() with --arcs <list> --last 0, redirecting all
outputs to a per-shard directory so parallel shards never interleave writes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_flags as af

shard_id = sys.argv[1]
arcs = sys.argv[2]

out = af.OUT_DIR / f"shard_{shard_id}"
af.OUT_DIR = out
af.JUDGED_PATH = out / "judged_flags.jsonl"
af.REPORT_PATH = out / "report.md"
af.SUMMARY_PATH = out / "summary.json"

sys.argv = ["adjudicate_flags.py", "--arcs", arcs, "--last", "0"]
raise SystemExit(af.main())
