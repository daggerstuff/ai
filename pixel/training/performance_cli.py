"""
Performance Metrics CLI

Reads JSONL with perf events (ts, latency_ms, status, route, model) and emits
per-interval metrics in JSON and Prometheus formats.

Usage:
  uv run python ai/pixel/training/performance_cli.py --input perf.jsonl --json out.json --prom out.prom
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ai.pixel.training.performance_metrics import PerfAggregator


def main() -> None:
    p = argparse.ArgumentParser(description="Performance Metrics CLI")
    p.add_argument("--input", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--prom", required=True)
    args = p.parse_args()

    agg = PerfAggregator()
    agg.import_jsonl(args.input)

    Path(args.json).write_text(agg.export_json(), encoding="utf-8")
    Path(args.prom).write_text(agg.export_prometheus(), encoding="utf-8")

    print(f"{{\"ok\": true, \"json\": \"{args.json}\", \"prom\": \"{args.prom}\"}}")


if __name__ == "__main__":
    main()
